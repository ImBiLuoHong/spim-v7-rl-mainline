"""
Deprecated compatibility shims for retired training runner entrypoints.

Official training entry:
    python src/scripts/train_phase4_end2end.py --config <yaml>

This module intentionally no longer provides an alternate runnable training path.
"""

from src.training.engine.scheduler import build_scheduler
from src.training.utils import select_device


_OFFICIAL_ENTRY = "python src/scripts/train_phase4_end2end.py --config <yaml>"


def _deprecated_runner_error(api_name: str) -> RuntimeError:
    return RuntimeError(
        f"{api_name} has been retired during training mainline closure. "
        f"Use the unique official training entry instead: {_OFFICIAL_ENTRY}"
    )


def run_train_with_env(*args, **kwargs):
    raise _deprecated_runner_error("src.training.runner.run_train_with_env")


def run_train_with_cfg(*args, **kwargs):
    raise _deprecated_runner_error("src.training.runner.run_train_with_cfg")


__all__ = [
    "run_train_with_env",
    "run_train_with_cfg",
    "build_scheduler",
    "select_device",
]
