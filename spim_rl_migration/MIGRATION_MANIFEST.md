# SPIM+RL Migration Manifest

Created on 2026-04-27.

## Scope

This directory is now the SPIM+RL working copy. It contains:

- SPIM runtime, training, strict-eval, sweep and paper-analysis scripts.
- SPIM-native artifacts and experiment records.
- Set-level RL / SPIM policy checkpoints and strict evaluation outputs.
- v6/v7 teacher sweeps, alpha/offset sweeps, sampling comparisons and figures.
- Supporting posterior/belief/reward/teacher/HSR evidence used by the SPIM line.
- Minimal navigator/reasoner dependency artifacts required by the SPIM runtime source-root.

## Copied Artifact Families

Direct SPIM families:

```text
spim_case_heatmap
spim_family_sweep
spim_forensic_audit
spim_mainline_lock_confirmation
spim_regret_reward_alignment_audit
spim_regret_upper_bound_gateability_audit
spim_reward_family_teacher_control
spim_rl_conservative_audit
spim_sampling_compare_20260416
spim_semantic_publication_figures_20260416
spim_set_level_rl_mainline
spim_strict_set_bundle_headroom_audit
spim_teacher5_compare
spim_teacher_imitation_rl_pilot
spim_teacher_precheck
spim_v6_eval
spim_v7_alpha_sweep
spim_v7_offset_sweep
```

Supporting SPIM/RL evidence:

```text
_tmp_reward_family_smoke
authoritative_hsr_baseline
bayesian_belief_reward_sweep
belief_reward_validity_audit
conservative_corrective_rl_v1
fixed_posterior_action_value_audit
fixed_posterior_deterministic_myopic_baseline
frozen_belief_navigator_feasibility
hsr_discrepancy_audit
hsr_legacy_protocol_repro
hsr_paper_equivalence_audit
hsr_predictive_one_step_planner
hsr_step1_closure
hsr_step1_improvement
posterior_like_belief_acceptability_audit
posterior_like_belief_audit
posterior_to_policy_readiness_audit
teacher_quality_sensitivity
teacher_relative_slot3_residual_v1
paper_analysis
paper_analysis_pass2
paper_freeze_pass3
paper_figures
```

Runtime dependency artifacts retained because SPIM scripts read them:

```text
reasoner_same_case_stronger_source_overfit
task_defined_oracle_sampler
reasoner_clean_aligned_online_finish
clean_navigator_v1
```

## Deliberately Not Bulk-Copied

The old navigator RL line is intentionally left in the previous workspace:

```text
artifacts/navigator_rl_pilot
artifacts/navigator_vnext_rl
```

Other broad navigator/reasoner training campaign artifacts were also not bulk-copied unless the SPIM source-root or manifests required them.

## Validation Evidence

These smoke outputs were produced inside this workspace:

```text
artifacts/smoke_runs/strict_v7_teacher_val1
artifacts/smoke_runs/strict_existing_rl_v3_val1
```

The existing strongest copied RL checkpoint strict-loaded successfully with:

```text
sha256=141cdbcd1e579b446c29e5d97399267d5ccb591919693f19d1d3dd948dc539cc
```
