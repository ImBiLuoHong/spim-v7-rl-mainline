# Training Mainline Closure

## 1. Formal mainline definition

This closure round treats the project as a pure support-led training system.

- mainline:
  - `support_score`
  - `observation_validity`
  - `uncertainty_gap`
- optional / diagnostic:
  - `suspect_pool`
  - `topology_gate`
  - `coarse_time_gate`
  - `not_ruled_out_gate`
- frozen auxiliary / audit:
  - `contradiction_score`
  - `contradiction_toxic_term`
  - `contradiction_clean_term`
  - `arrival_gate`
- deprecated / removed from mainline:
  - `source_validity`

Code-level mainline contract now matches that definition:

- [schema.py](/root/autodl-tmp/rl_spim_v7_mainline/src/modeling/state/schema.py#L67)
- [builder.py](/root/autodl-tmp/rl_spim_v7_mainline/src/modeling/evidence/builder.py#L96)
- [standard.py](/root/autodl-tmp/rl_spim_v7_mainline/src/modeling/navigators/standard.py#L99)
- [bayesian.py](/root/autodl-tmp/rl_spim_v7_mainline/src/modeling/reasoners/bayesian.py#L93)

## 2. Training entry unification

### Official keep

- `src/scripts/train_phase4_end2end.py`
  - the only official training entry
  - all formal compare configs were validated through this script
- `src/modeling/builders/model_builder.py`
  - active model construction path used by the official script
- `src/scripts/main_cli.py`
  - retained only as a thin shell
  - `train` now redirects to `train_phase4_end2end.py`
  - `train-mem` now exits as deprecated

### Archived paths

Archived legacy implementations:

- `src/training/legacy/runner_env_legacy.py`
- `src/training/legacy/runner_cfg_legacy.py`
- `src/training/legacy/model_builder_legacy.py`
- archive note: [README.md](/root/autodl-tmp/rl_spim_v7_mainline/src/training/legacy/README.md)

### Deleted active code paths

The following old runnable paths were removed from active use and replaced with explicit stubs/errors:

- `src/training/runner.py::run_train_with_env`
- `src/training/runner.py::run_train_with_cfg`
- `src/training/engine/runner_env.py` legacy implementation body
- `src/training/engine/runner_cfg.py` legacy implementation body
- `src/training/engine/model_builder.py` legacy implementation body
- `src/scripts/main_cli.py::cmd_train_mem` as a runnable training path

Result:

- there is no remaining supported path that can still launch the old runner chain
- old imports now hit explicit deprecation errors instead of silently running an alternate training stack

Relevant files:

- [main_cli.py](/root/autodl-tmp/rl_spim_v7_mainline/src/scripts/main_cli.py#L74)
- [runner.py](/root/autodl-tmp/rl_spim_v7_mainline/src/training/runner.py#L1)
- [runner_env.py](/root/autodl-tmp/rl_spim_v7_mainline/src/training/engine/runner_env.py#L1)
- [runner_cfg.py](/root/autodl-tmp/rl_spim_v7_mainline/src/training/engine/runner_cfg.py#L1)
- [model_builder.py](/root/autodl-tmp/rl_spim_v7_mainline/src/training/engine/model_builder.py#L1)

## 3. `source_validity` removal from mainline

### What changed

- `source_validity` was removed from `FIELD_CONTRACTS` as a mainline validity dependency.
- `support_mainline` field group no longer includes `source_validity`.
- builder no longer uses `source_validity` as a mask for:
  - `support_score`
  - `contradiction_score`
  - `reaction_consistency`
- runtime audit no longer requires `source_validity` for mainline evidence checks.

Relevant code:

- [schema.py](/root/autodl-tmp/rl_spim_v7_mainline/src/modeling/state/schema.py#L67)
- [builder.py](/root/autodl-tmp/rl_spim_v7_mainline/src/modeling/evidence/builder.py#L96)
- [audit.py](/root/autodl-tmp/rl_spim_v7_mainline/src/modeling/loop/audit.py#L100)

### Compatibility handling

`source_validity` is still retained as a compatibility field in `EvidenceState`, but it is now explicitly neutralized and ignored in the mainline:

- builder writes it as all-ones compatibility payload
- it is no longer consumed by mainline composition
- `compatibility_gate` is reduced to a deprecated diagnostic stub

Observed support-mainline smoke value:

- `source_validity_unique = [1.0]`

That is intentional after closure: it is no longer a live evidence channel.

## 4. Physics pruning boundary

`source_validity` removal did **not** remove upstream physical pruning.

Still preserved:

- upstream feasible-mask construction:
  - [physics.py](/root/autodl-tmp/rl_spim_v7_mainline/src/modeling/components/physics.py#L29)
- physics module path that can emit `feasible_mask`:
  - [race_consistency.py](/root/autodl-tmp/rl_spim_v7_mainline/src/modeling/physics/race_consistency.py#L27)
- navigator sampling-space intersection with feasible mask:
  - [episode_stepper.py](/root/autodl-tmp/rl_spim_v7_mainline/src/modeling/loop/episode_stepper.py#L572)
- episode runner valid set intersection with feasible mask:
  - [episode_runner.py](/root/autodl-tmp/rl_spim_v7_mainline/src/modeling/loop/episode_runner.py#L192)

Boundary after closure:

- keep:
  - hydraulic pruning
  - candidate-domain shrinking
  - reachability
  - feasible-mask-based upstream sampling restrictions
- remove from mainline evidence semantics:
  - `source_validity` as a live EvidenceState gate/mask

## 5. Validation package

### Contract / closure tests

Run:

```bash
pytest -q tests/test_evidence_state_v1_contract.py tests/training/test_train_runner_cfg.py tests/config/test_effective_cfg_snapshot.py
```

Result:

- `6 passed`

### Official-entry compare smoke

Each of the following was run via the unique official entry `src/scripts/train_phase4_end2end.py` with:

- `--epochs 1`
- `--batch_size 1`
- `--skip_audit`
- `data.max_samples=1`
- `data.num_workers=0`
- `data.pin_memory=false`
- `data.persistent_workers=false`
- `data.preload=false`
- `training.max_train_episodes=1`
- `training.max_eval_episodes=1`
- `training.enable_eval=false`

Configs:

1. `configs/evidence_v1/base_no_evidence.yaml`
2. `configs/evidence_v1/support_mainline.yaml`
3. `configs/evidence_v1/support_plus_soft_suspect.yaml`
4. `configs/evidence_v1/support_plus_contradiction_aux_compare.yaml`

Result:

- all four official-entry smokes exited successfully
- all four completed one epoch of training on the minimal package

### Support mainline single-batch training smoke

Validated path:

1. build one real batch
2. `model(batch, inference_mode=False, max_episodes=1)`
3. extract `trajectory`
4. compute `ModularLossEngine`
5. `loss.backward()`
6. `optimizer.step()`

Observed result:

- `trajectory_len = 1`
- `loss = 3.9475193`
- `classification_shape = (128, 1)`

## 6. Final answers

### 1. Has `train_phase4_end2end.py` become the unique official training entry?

Yes.

- it is the only supported runnable training entry
- `main_cli train` only redirects to it
- old runner paths no longer execute training logic

### 2. Which old chains were archived and which were deleted?

Archived:

- `src/training/legacy/runner_env_legacy.py`
- `src/training/legacy/runner_cfg_legacy.py`
- `src/training/legacy/model_builder_legacy.py`

Deleted from active use:

- legacy env runner logic at `src/training/engine/runner_env.py`
- legacy cfg runner logic at `src/training/engine/runner_cfg.py`
- legacy training-engine model builder logic at `src/training/engine/model_builder.py`
- legacy compatibility training API behavior at `src/training/runner.py`
- legacy `train-mem` runnable path in `src/scripts/main_cli.py`

### 3. Has `source_validity` been formally removed from the mainline?

Yes.

- it no longer participates in mainline contract grouping
- builder no longer depends on it
- runtime audit no longer treats it as required mainline validity
- retained field is compatibility-only and neutral

### 4. After removal, is the support-led mainline still complete and trainable?

Yes.

- schema is aligned
- builder is aligned
- encoder is aligned
- navigator / reasoner are aligned
- official-entry compare smokes pass
- support-mainline backward smoke passes

### 5. Are upstream hydraulic pruning and candidate shrinking preserved?

Yes.

- feasible-mask construction and reachability code were not removed
- sampling-space intersections with `physics_ctx['feasible_mask']` remain intact

### 6. Can the project enter formal training now?

Yes.

This closure round removed the last mainline ambiguity:

- single official entry
- no active old runner chain
- pure support-led mainline
- `source_validity` no longer blocks readiness
- official-entry smoke passes for all four formal compares

### 7. Remaining blockers

None for training readiness.

Remaining items are optional cleanup only:

- broader legacy test cleanup outside the official path
- further documentation tightening outside the main closure docs

## 7. Key modified files

### Entry cleanup

- [main_cli.py](/root/autodl-tmp/rl_spim_v7_mainline/src/scripts/main_cli.py)
- [runner.py](/root/autodl-tmp/rl_spim_v7_mainline/src/training/runner.py)
- [runner_env.py](/root/autodl-tmp/rl_spim_v7_mainline/src/training/engine/runner_env.py)
- [runner_cfg.py](/root/autodl-tmp/rl_spim_v7_mainline/src/training/engine/runner_cfg.py)
- [model_builder.py](/root/autodl-tmp/rl_spim_v7_mainline/src/training/engine/model_builder.py)
- [README.md](/root/autodl-tmp/rl_spim_v7_mainline/src/training/legacy/README.md)

### Mainline semantic decoupling

- [schema.py](/root/autodl-tmp/rl_spim_v7_mainline/src/modeling/state/schema.py)
- [builder.py](/root/autodl-tmp/rl_spim_v7_mainline/src/modeling/evidence/builder.py)
- [audit.py](/root/autodl-tmp/rl_spim_v7_mainline/src/modeling/loop/audit.py)
- [test_evidence_state_v1_contract.py](/root/autodl-tmp/rl_spim_v7_mainline/tests/test_evidence_state_v1_contract.py)
- [evidence_state_v1_closure.md](/root/autodl-tmp/rl_spim_v7_mainline/docs/evidence_state_v1_closure.md)

### Compatibility / stale-path cleanup

- [test_train_runner_cfg.py](/root/autodl-tmp/rl_spim_v7_mainline/tests/training/test_train_runner_cfg.py)
- [test_effective_cfg_snapshot.py](/root/autodl-tmp/rl_spim_v7_mainline/tests/config/test_effective_cfg_snapshot.py)
- [test_value_dependence.py](/root/autodl-tmp/rl_spim_v7_mainline/tests/test_value_dependence.py)

## 8. Bottom line

The system is now closed as:

- pure support-led
- single-entry
- no live `source_validity` dependency in the training mainline
- preserved upstream physics pruning

Recommendation:

- enter formal training immediately using `src/scripts/train_phase4_end2end.py`.
