# SPIM V7 + RL — Mainline

Set-level Policy Imitation via Soft Posterior + Reinforcement Learning.

This repository is the SPIM+RL mainline with all runtime source code, tools,
and experiment scripts. Large data assets (graph data, LMDB caches, checkpoints)
are distributed separately.

## What Is Included

- SPIM runtime and evaluation scripts:
  - `src/scripts/run_spim_teacher_imitation_rl_pilot.py` — RL teacher-imitation training
  - `src/scripts/run_spim_policy_eval_strict.py` — strict policy evaluation
  - `src/scripts/run_spim_family_sweep.py` — teacher family sweep
- Full `src/` dependency tree (modeling, evidence, data, training, evaluation)
- HSR baseline agents under `tools/legacy/src_baselines_archive`
- Paper-analysis and figure-rendering tools under `tools/` and `src/scripts/paper_analysis`
- Data split files: `data/train.txt`, `data/val.txt`, `data/test.txt`

## Quick Start

See **[SETUP.md](SETUP.md)** for full Windows/Linux deployment instructions.

```bash
# 1. Create virtual environment
python -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows

# 2. Install dependencies
pip install -r requirements.txt
```

**Data must be obtained separately** — see [SETUP.md](SETUP.md#4-数据部署) for details.

Once data is in place, verify the environment:

```bash
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
```

## Main Fact

`SPIM v7` refers to teacher family `hsr_soft_scenario_posterior_v7_*offset`.
The best teacher configuration from sweeps:

```
hsr_soft_scenario_posterior_v7_7offset, alpha=0.55
```

## Layout

```
src/
├── config/         — unified configuration (SSOT)
├── data/v6/        — data loading pipeline (dataset, collate, loader, topology)
├── evaluation/     — evaluator, metrics, reporting
├── modeling/       — navigators, reasoners, belief updaters, evidence, physics
├── scripts/        — runnable entry points (train, eval, audit, paper analysis)
├── shared/         — shared utilities (auditing, changelog, logging, diagnostics)
├── tools/          — LMDB converter
├── training/       — training engine (driver, loop, scheduler, etc.)
└── utils/          — hash, metrics, profiling, hardware optimization

tools/
├── legacy/src_baselines_archive/  — HSR / ZJU baseline agents
└── *.py                           — rendering, analysis, and comparison tools

scripts/
└── *.sh                           — smoke-test shell templates
```

## Data Dependencies

The following directories are **required at runtime** but are NOT versioned here:

| Directory | Approx. Size | Description |
|-----------|-------------|-------------|
| `datanew/production_data/foundation_20260114_164946_86d5023e/` | ~8.9 GB | Production graph, metadata, subgraph samples |
| `data/cache_lmdb/` | ~428 MB | Pre-built LMDB caches for train/val/test |

See [SETUP.md](SETUP.md) for how to obtain and place these.
