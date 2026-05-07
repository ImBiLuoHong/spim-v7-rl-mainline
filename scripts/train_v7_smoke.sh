#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python -m src.scripts.run_spim_teacher_imitation_rl_pilot \
  --teacher-family hsr_soft_scenario_posterior_v7_7offset \
  --paper-like-alpha 0.55 \
  --runner-version-tag smoke_v7_7offset_seed45 \
  --output-dir artifacts/smoke_runs/train_v7_seed45_n4 \
  --train-full-max-cases 4 \
  --train-full-cache-version train_full_rlpilot_smoke_n4_v1 \
  --bc-epochs 1 \
  --bc-recovery-epochs 0 \
  --rl-epochs 1 \
  --rl-update-epochs 1 \
  --device cpu \
  --save-final-checkpoint artifacts/smoke_runs/train_v7_seed45_n4/checkpoints/rl_student_final.pt
