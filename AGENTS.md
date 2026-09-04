# Agent instructions

Conventions for any coding agent working in this repository — Claude Code, Codex, Gemini CLI,
or otherwise. Read this before making changes.

## Skills

Reusable instructions live in `.claude/skills/<name>/SKILL.md`. They are vendored into this
repository rather than assumed present in your environment, so they apply to anyone who clones
it regardless of harness. Read the relevant `SKILL.md` in full and follow it.

| Skill | Use when |
| --- | --- |
| `test-driven-development` | Implementing any feature or bugfix, **before** writing implementation code |
| `openspec-propose` | Starting a change: produces proposal, design, specs and tasks |
| `openspec-apply-change` | Implementing an approved change |
| `openspec-archive-change` | Folding a completed change into the baseline specs |
| `openspec-explore` | Orienting in the existing specs |

Harnesses that auto-discover skills (Claude Code) will surface these on their own. Harnesses
that do not should treat the table above as the index and read the files directly.

## Test-driven development

This project follows the iron law in `.claude/skills/test-driven-development/SKILL.md`:

```
NO PRODUCTION CODE WITHOUT A FAILING TEST FIRST
```

Write the test, watch it fail for the right reason, then write the minimum code to pass. If you
find yourself verifying behaviour with a throwaway script after the fact, you have skipped the
step that matters — a check you ran once and discarded is not a test.

## Spec-driven development

Changes are tracked with [OpenSpec](https://github.com/Fission-AI/OpenSpec). Baseline
capability specs live in `openspec/specs/`; completed changes are in `openspec/changes/archive/`.

Non-trivial work should start as a change proposal and end with `openspec archive`, which folds
the deltas into the baseline. Run `openspec validate --all --strict` before considering a change
done.

## Environment

- Python lives in `.venv`. Always invoke it explicitly: `.venv/bin/python`, `.venv/bin/pip`.
  Do not assume an activated shell.
- `make` is the entry point for everything. Prefer adding a target over documenting a raw
  command. `make help` lists them.
- Training knobs are make variables: `make train EPOCHS=10 TRAIN_ARGS="--assistant-loss"`.

## Things that will bite you

- **`training/train.py` deletes `models/checkpoints/` at startup.** Anything you want to survive
  a run must live elsewhere — this is why run metrics go to `models/runs/`.
- **Heavy imports are lazy on purpose.** `torch`, `transformers`, `trl` and `peft` are imported
  inside functions, not at module level, so tooling and `--help` stay fast and the modules stay
  importable without the training stack. Keep it that way.
- **`training/chat_template_assistant_mask.jinja` is training-only.** It must render identically
  to the base model's stock template; it exists solely to add `{% generation %}` markers. It must
  not reach the merged or exported model.
- **Don't retrain to check a code change.** A run costs real time. Most behaviour here is
  verifiable without one.

## Style

Match the surrounding code: no type annotations where the file has none, comments that explain
*why* rather than restating the line, and prose in docs that says what something does rather
than listing adjectives.
