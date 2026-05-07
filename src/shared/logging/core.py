"""
Lightweight aggregator for logging utilities.

This module re-exports decoupled functions from src.shared.logging.* submodules
to preserve the original public API while keeping this file compact (<500 lines).

For any functions not yet migrated, this module provides a lazy fallback
via __getattr__ to utils.log_utils_legacy to ensure backward compatibility.
"""

# Decoupled helpers (public when imported here)
from src.shared.logging.logging.common import _get, _to_serializable

# Config snapshots and summaries
from src.shared.logging.logging.cfg_snapshot import (
    write_effective_config_snapshot,
    write_config_snapshot,
    write_config_summary,
    append_summary_lines,
    append_applied_overrides_to_summary,
)
from src.shared.logging.logging.cfg_snapshot_hash import write_config_snapshot_with_hash

# Training CSV and logs
from src.shared.logging.logging.csv_train import (
    ensure_train_log_csv_format,
    append_log,
    append_grad_log,
    append_ce_kl,
)

# Diagnostics and statistics
from src.shared.logging.logging.diagnostics import (
    write_selector_mask_csv,
    log_tv_stats,
    write_coverage_curve_csv,
    write_krho_stats_csv,
    write_eval_caps_csv,
    write_tau_hist_csv,
    write_diag_report_md,
    write_diag_gates_json,
    write_grads_csv,
)

# Environment snapshot
from src.shared.logging.logging.env_snapshot import write_env_snapshot

# Status and events
from src.shared.logging.logging.status_events import (
    write_heartbeat,
    write_finished_ok,
    append_interruption_event,
    write_paused_ok,
    should_stop,
)

# Model architecture doc and hardware statistics
from src.shared.logging.logging.arch_doc import write_model_architecture_doc
from src.shared.logging.logging.hardware import write_hardware_stats


__all__ = [
    # helpers
    '_get', '_to_serializable',
    # cfg snapshot & summary
    'write_effective_config_snapshot', 'write_config_snapshot', 'write_config_summary',
    'append_summary_lines', 'append_applied_overrides_to_summary', 'write_config_snapshot_with_hash',
    # train csv & logs
    'ensure_train_log_csv_format', 'append_log', 'append_grad_log', 'append_ce_kl',
    # diagnostics
    'write_selector_mask_csv', 'log_tv_stats', 'write_coverage_curve_csv', 'write_krho_stats_csv',
    'write_eval_caps_csv', 'write_tau_hist_csv', 'write_diag_report_md', 'write_diag_gates_json', 'write_grads_csv',
    # env snapshot
    'write_env_snapshot',
    # status & events
    'write_heartbeat', 'write_finished_ok', 'append_interruption_event', 'write_paused_ok', 'should_stop',
    # arch & hardware
    'write_model_architecture_doc', 'write_hardware_stats',
]


def __getattr__(name):
    """
    Lazy fallback for attributes not defined in the new decoupled modules.
    If an attribute is requested that isn't re-exported above, try to resolve it
    from utils.log_utils_legacy to maintain backward compatibility.
    """
    try:
        from utils import log_utils_legacy as _legacy
        if hasattr(_legacy, name):
            return getattr(_legacy, name)
    except Exception:
        pass
    raise AttributeError(f"module 'utils.log_utils' has no attribute '{name}'")