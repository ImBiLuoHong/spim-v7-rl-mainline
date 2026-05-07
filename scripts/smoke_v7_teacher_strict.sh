#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python -m src.scripts.run_spim_policy_eval_strict \
  --teacher-family hsr_soft_scenario_posterior_v7_7offset \
  --paper-like-alpha 0.55 \
  --policy-mode teacher \
  --policy-name smoke_v7_teacher \
  --output-dir artifacts/smoke_runs/strict_v7_teacher_val1 \
  --split val \
  --case-limit 1 \
  --trace-case-limit 1 \
  --trace-step-limit 1 \
  --device cpu
