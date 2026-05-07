#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python -m src.scripts.run_spim_policy_eval_strict \
  --teacher-family hsr_soft_scenario_posterior_v3 \
  --policy-mode rl \
  --policy-name smoke_existing_rl_v3 \
  --checkpoint artifacts/spim_set_level_rl_mainline/20260420_v3_early_set_credit_v1/stage1_c1_r2_seed45/checkpoints/rl_student_final.pt \
  --output-dir artifacts/smoke_runs/strict_existing_rl_v3_val1 \
  --split val \
  --case-limit 1 \
  --trace-case-limit 1 \
  --trace-step-limit 1 \
  --hidden-dim 128 \
  --policy-arch separate_heads \
  --policy-mlp-depth 2 \
  --value-mlp-depth 3 \
  --value-head-width-mult 2.0 \
  --arch-backbone baseline_mlp \
  --slate-size 10 \
  --slate-top-posterior-k 8 \
  --slate-high-disagreement-k 1 \
  --slate-novelty-k 1 \
  --early-stage-round-cutoff 2 \
  --device cpu
