"""Path invariants that other stages depend on."""

from training import config


def test_run_history_survives_a_new_training_run():
    """train.py wipes CHECKPOINTS at startup, so run history must live elsewhere.

    Storing it under the checkpoint directory would destroy the previous run's
    metrics exactly when you want to compare against them.
    """
    assert config.RUNS.resolve() != config.CHECKPOINTS.resolve()
    assert config.CHECKPOINTS.resolve() not in config.RUNS.resolve().parents


def test_stage_directories_are_distinct():
    stages = {config.CHECKPOINTS, config.MERGED, config.ONNX, config.RUNS}
    assert len(stages) == 4


def test_training_chat_template_is_shipped():
    """--assistant-loss depends on this file; the base model does not provide one."""
    assert config.TRAIN_CHAT_TEMPLATE.exists()
    assert "{% generation %}" in config.TRAIN_CHAT_TEMPLATE.read_text()


def test_system_prompt_is_set():
    assert config.SYSTEM_PROMPT.strip()
