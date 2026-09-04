"""Export the merged model to quantized ONNX for Transformers.js.

Produces the layout Transformers.js expects: tokenizer and config files at the
top level of models/onnx, with the ONNX graphs in an onnx/ subdirectory. Two
quantizations are written - 4-bit (model_q4.onnx, the default the web app loads)
and 8-bit (model_quantized.onnx, the safer fallback) - then each is smoke-checked
for non-degenerate output. Export runs on CPU; ONNX export from MPS is not
supported.
"""

import argparse
import os
import shutil
import sys

from training.config import MERGED, ONNX, SYSTEM_PROMPT

SMOKE_PROMPT = "How often should I water my Venus flytrap?"
MAX_QUANTIZED_BYTES = 500 * 1024 * 1024


def _dir_size(path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def _human(num_bytes: int) -> str:
    return f"{num_bytes / (1024 * 1024):.1f} MB"


def export_fp32() -> None:
    from optimum.exporters.onnx import main_export

    print(f"Exporting {MERGED} to ONNX (CPU, fp32)")
    main_export(
        model_name_or_path=str(MERGED),
        output=str(ONNX),
        task="text-generation-with-past",
        device="cpu",
        dtype="fp32",
    )


def _inline_chat_template() -> None:
    """Copy chat_template.jinja into tokenizer_config.json.

    Transformers saves the template as a standalone .jinja file, but
    Transformers.js only reads the `chat_template` key of tokenizer_config.json.
    Without this the browser raises "tokenizer.chat_template is not set".
    """
    import json

    template_file = ONNX / "chat_template.jinja"
    config_file = ONNX / "tokenizer_config.json"
    if not template_file.exists() or not config_file.exists():
        return

    config = json.loads(config_file.read_text(encoding="utf-8"))
    if config.get("chat_template"):
        return

    config["chat_template"] = template_file.read_text(encoding="utf-8")
    config_file.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Inlined chat_template.jinja into tokenizer_config.json for Transformers.js")


def _relocate_onnx_files() -> None:
    """Move ONNX graphs into the onnx/ subdirectory Transformers.js expects."""
    subdir = ONNX / "onnx"
    subdir.mkdir(exist_ok=True)
    for item in list(ONNX.iterdir()):
        if item.is_file() and (item.suffix == ".onnx" or item.name.endswith(".onnx_data")):
            item.rename(subdir / item.name)


def quantize_q8(source, target) -> None:
    from onnxruntime.quantization import QuantType, quantize_dynamic

    print("Quantizing to 8-bit (model_quantized.onnx)")
    quantize_dynamic(
        model_input=str(source),
        model_output=str(target),
        weight_type=QuantType.QUInt8,
        op_types_to_quantize=["MatMul"],
        per_channel=False,
        reduce_range=False,
        use_external_data_format=any(source.parent.glob("*.onnx_data")),
        extra_options={"EnableSubgraph": True},
    )


def quantize_q4(source, target) -> bool:
    """Return True if 4-bit quantization succeeded."""
    try:
        import onnx
        from onnxruntime.quantization.matmul_nbits_quantizer import (
            DefaultWeightOnlyQuantConfig,
            MatMulNBitsQuantizer,
        )
    except ImportError as exc:
        print(f"WARNING: 4-bit quantization unavailable ({exc}); using 8-bit only.")
        return False

    print("Quantizing to 4-bit (model_q4.onnx)")
    try:
        model = onnx.load(str(source), load_external_data=True)
        quantizer = MatMulNBitsQuantizer(
            model,
            algo_config=DefaultWeightOnlyQuantConfig(block_size=32, is_symmetric=True, bits=4),
        )
        quantizer.process()
        quantizer.model.save_model_to_file(str(target), use_external_data_format=False)
    except Exception as exc:  # noqa: BLE001 - fall back to 8-bit on any failure
        print(f"WARNING: 4-bit quantization failed ({exc}); using 8-bit only.")
        target.unlink(missing_ok=True)
        return False
    return True


def smoke_check(file_name: str) -> bool:
    """Generate from a quantized graph and report whether output is sane."""
    from optimum.onnxruntime import ORTModelForCausalLM
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(ONNX))
    model = ORTModelForCausalLM.from_pretrained(str(ONNX), file_name=file_name)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": SMOKE_PROMPT},
    ]
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
    )
    outputs = model.generate(
        **inputs,
        max_new_tokens=60,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    reply = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    ).strip()
    print(f"  {file_name}: {reply[:200]}")

    words = reply.split()
    degenerate = len(words) < 5 or len(set(words)) < 3
    if degenerate:
        print(f"  WARNING: {file_name} output looks degenerate.", file=sys.stderr)
    return not degenerate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--keep-fp32",
        action="store_true",
        help="keep the unquantized graph (large) after quantizing",
    )
    args = parser.parse_args()

    if not (MERGED / "config.json").exists():
        print(f"ERROR: no merged model in {MERGED}", file=sys.stderr)
        print("Run `make train` then `make merge` first.", file=sys.stderr)
        return 1

    if ONNX.exists():
        shutil.rmtree(ONNX)
    ONNX.mkdir(parents=True, exist_ok=True)

    export_fp32()
    _inline_chat_template()

    # Quantize and smoke check while everything sits at the top level, then move
    # the graphs into onnx/ - Transformers.js wants them there, but resolving a
    # subdirectory during loading is fiddlier than it is worth here.
    fp32_model = ONNX / "model.onnx"
    if not fp32_model.exists():
        print(f"ERROR: expected {fp32_model} after export", file=sys.stderr)
        return 1

    quantize_q8(fp32_model, ONNX / "model_quantized.onnx")
    have_q4 = quantize_q4(fp32_model, ONNX / "model_q4.onnx")

    print("\nSmoke checking quantized graphs:")
    ok_q8 = smoke_check("model_quantized.onnx")
    ok_q4 = smoke_check("model_q4.onnx") if have_q4 else False

    if not args.keep_fp32:
        fp32_model.unlink()
        for extra in ONNX.glob("*.onnx_data"):
            extra.unlink()

    _relocate_onnx_files()
    subdir = ONNX / "onnx"

    default_dtype = "q4" if (have_q4 and ok_q4) else "q8"
    if not ok_q8 and not ok_q4:
        print("ERROR: no quantized graph produced usable output.", file=sys.stderr)
        return 1
    if default_dtype == "q8" and have_q4:
        print("NOTE: 4-bit output looked degenerate; the web app should use q8.")

    # The browser downloads one graph plus the tokenizer and config files, never
    # both graphs, so the delivery budget applies per graph.
    support_bytes = sum(f.stat().st_size for f in ONNX.iterdir() if f.is_file())
    graphs = sorted(subdir.glob("*.onnx"))

    print(f"\nExport complete: {ONNX} (recommended dtype: {default_dtype})")
    for graph in graphs:
        download = graph.stat().st_size + support_bytes
        flag = "" if download <= MAX_QUANTIZED_BYTES else "  OVER BUDGET"
        print(f"  {graph.name}: {_human(graph.stat().st_size)} "
              f"(download {_human(download)}){flag}")
    print(f"  on disk: {_human(_dir_size(ONNX))} (both graphs kept)")

    smallest = min(g.stat().st_size for g in graphs) + support_bytes
    if smallest > MAX_QUANTIZED_BYTES:
        print(
            f"WARNING: even the smallest download is {_human(smallest)}, above the "
            f"{_human(MAX_QUANTIZED_BYTES)} web delivery budget.",
            file=sys.stderr,
        )

    print("Next: make serve")
    return 0


if __name__ == "__main__":
    code = main()
    # onnxruntime's interpreter-shutdown teardown aborts on macOS with
    # "recursive_mutex lock failed" after a successful export, which would fail the
    # make target. Flush and exit without running that teardown.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)
