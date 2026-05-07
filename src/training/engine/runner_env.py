"""
Deprecated stub for the retired env-based training launcher.

Legacy implementation archived at:
    src/training/legacy/runner_env_legacy.py
"""


def run_train_with_env(*args, **kwargs):
    raise RuntimeError(
        "src.training.engine.runner_env.run_train_with_env is deprecated. "
        "Use the unique official training entry: python src/scripts/train_phase4_end2end.py --config <yaml>"
    )
