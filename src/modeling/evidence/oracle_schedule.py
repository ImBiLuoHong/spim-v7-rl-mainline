from typing import Any, Dict


def _cfg_get(cfg_obj, key: str, default):
    if cfg_obj is None:
        return default
    if isinstance(cfg_obj, dict):
        return cfg_obj.get(key, default)
    return getattr(cfg_obj, key, default)


def _lerp(start: float, end: float, ratio: float) -> float:
    ratio = max(0.0, min(1.0, float(ratio)))
    return float(start) + (float(end) - float(start)) * ratio


def resolve_evidence_oracle_schedule(cfg: Any) -> Dict[str, float]:
    training_cfg = getattr(cfg, "training", None) if cfg is not None else None
    sched_cfg = _cfg_get(training_cfg, "evidence_oracle_schedule", {}) if training_cfg is not None else {}
    enabled = bool(_cfg_get(sched_cfg, "enabled", False))

    if not enabled:
        return {
            "enabled": 0.0,
            "phase": "static_hybrid",
            "phase_index": 3.0,
            "progress": 0.0,
            "oracle_factor": 1.0,
            "live_factor": 1.0,
        }

    mode = str(_cfg_get(sched_cfg, "mode", "step"))
    if mode == "epoch":
        progress = int(_cfg_get(training_cfg, "current_epoch", 0))
    else:
        progress = int(_cfg_get(training_cfg, "global_step", 0))

    phase_a_steps = max(0, int(_cfg_get(sched_cfg, "phase_a_steps", 0)))
    phase_b_steps = max(0, int(_cfg_get(sched_cfg, "phase_b_steps", 0)))
    phase_c_steps = max(1, int(_cfg_get(sched_cfg, "phase_c_steps", 1)))

    boundary_a = phase_a_steps
    boundary_b = phase_a_steps + phase_b_steps

    if progress < boundary_a:
        phase = "phase_a_oracle_pretrain"
        phase_index = 0.0
        phase_progress = float(progress) / float(max(1, phase_a_steps))
        oracle_factor = float(_cfg_get(sched_cfg, "phase_a_oracle_factor", 1.0))
        live_factor = float(_cfg_get(sched_cfg, "phase_a_live_factor", 0.0))
    elif progress < boundary_b:
        phase = "phase_b_hybrid_warm_start"
        phase_index = 1.0
        local_step = progress - boundary_a
        phase_progress = float(local_step) / float(max(1, phase_b_steps))
        oracle_factor = _lerp(
            _cfg_get(sched_cfg, "phase_b_oracle_factor_start", 1.0),
            _cfg_get(sched_cfg, "phase_b_oracle_factor_end", 0.5),
            phase_progress,
        )
        live_factor = _lerp(
            _cfg_get(sched_cfg, "phase_b_live_factor_start", 0.25),
            _cfg_get(sched_cfg, "phase_b_live_factor_end", 1.0),
            phase_progress,
        )
    else:
        phase = "phase_c_anneal_to_live"
        phase_index = 2.0
        local_step = max(0, progress - boundary_b)
        phase_progress = float(min(local_step, phase_c_steps)) / float(max(1, phase_c_steps))
        oracle_factor = _lerp(
            _cfg_get(sched_cfg, "phase_c_oracle_factor_start", 0.5),
            _cfg_get(sched_cfg, "phase_c_oracle_factor_end", 0.0),
            phase_progress,
        )
        live_factor = _lerp(
            _cfg_get(sched_cfg, "phase_c_live_factor_start", 1.0),
            _cfg_get(sched_cfg, "phase_c_live_factor_end", 1.0),
            phase_progress,
        )

    return {
        "enabled": 1.0,
        "phase": phase,
        "phase_index": phase_index,
        "progress": float(progress),
        "oracle_factor": float(oracle_factor),
        "live_factor": float(live_factor),
    }
