"""
Deprecated stub for the retired cfg-based training launcher.

Legacy implementation archived at:
    src/training/legacy/runner_cfg_legacy.py
"""


def run_train_with_cfg(*args, **kwargs):
    raise RuntimeError(
        "src.training.engine.runner_cfg.run_train_with_cfg is deprecated. "
        "Use the unique official training entry: python src/scripts/train_phase4_end2end.py --config <yaml>"
    )
