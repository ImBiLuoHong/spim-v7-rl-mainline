# Actor/Critic Audit

## Scope

This audit is for the current clean Navigator stage lane used by the clean mini evidence artifacts, not for the older integrated `navigator_vnext` path.

## What is actually active

- [proven] The active clean stage lane is the standalone `CleanNavigatorV1` policy instantiated in `src/scripts/diagnostics/run_clean_navigator_v1.py`.
- [proven] `configs/evidence_v1/formal_campaign/official_clean_ref.yaml` still names `navigator_vnext`, but the clean stage script uses that config only for data/environment setup and then instantiates `CleanNavigatorV1` directly.
- [proven] Therefore the current clean lane is not actor-only and not the integrated GAE path. It is a simple actor-critic / policy-gradient-with-value-baseline implementation.

## Exact answers to the audit questions

### 1. Actor-only, actor-critic, or policy + value baseline?

- [proven] Policy gradient with a learned scalar value baseline.
- Equivalent description: a simple actor-critic.
- Evidence:
  - `train_epoch()` computes `advantages = returns - values`, policy loss from `log_probs * advantages.detach()`, and value loss from `MSE(values, returns)`.

### 2. Where are actor and critic implemented?

- [proven] Both are in `src/modeling/navigators/clean_v1.py`.
- Actor:
  - shared node encoder `encode()`
  - per-slot heads `self.slot_heads`
  - sequential selection logic in `act()`
- Critic:
  - scalar `self.value_head`
  - evaluated inside `act()`

### 3. Do actor and critic share a backbone?

- [proven] Yes.
- The actor and critic both consume the same GraphSAGE node encoder output `h`.
- The critic pools that shared encoder output with `graph_context = h.mean(dim=0)`.

### 4. What are the actor heads exactly for the 3-slot parallel action semantics?

- [proven] The actor uses `num_slots = action_budget`, which is 3 in the inspected stage artifacts.
- [proven] Each slot has its own `_SlotHead` MLP.
- [proven] The per-node slot input is:
  - node embedding `h_i`
  - pooled graph context
  - graph-level feature vector
  - learned slot embedding
- [proven] Selection is sequential without replacement:
  - slot 0 selects first node
  - that node is masked out
  - slot 1 selects from remaining nodes
  - slot 2 selects from remaining nodes
- [proven] Optional slot-specific role bias is added only to actor logits, not to the critic.

### 5. What exactly is the critic input?

- [proven] Critic input is `torch.cat([graph_context, graph_features], dim=0)`.
- [proven] `graph_context` is the mean-pooled shared node encoder output.
- [proven] `graph_features` are six rollout features:
  - episode index norm
  - remaining episodes norm
  - positive witness count norm
  - safe witness count norm
  - candidate fraction
  - current time norm
- [proven] The critic does not consume role potentials, slot identity, selected-set structure, or privileged labels.

### 6. What exactly is the critic target?

- [proven] The critic target is the discounted Monte Carlo return from per-step clean rollout rewards.
- [proven] Current stage reward per step is:
  - `ignorance_delta + 0.5 * conflict_delta + 0.25 * pair_delta`
- [proven] In training, an optional extra `train_conflict_bonus_weight * conflict_delta` can be added before return computation.
- [proven] In the inspected stage-role artifacts that bonus is effectively `0.0`.

### 7. How are returns and advantages computed?

- [proven] Returns are full backward discounted sums with `gamma`.
- [proven] There is no bootstrap term.
- [proven] There is no GAE in the active clean lane.
- [proven] Advantages are plain `returns - values`.

### 8. Is the critic healthy and actually learning, or decorative / weak / unstable?

- [proven] The critic is not decorative.
- [proven] Training logs for `artifacts/clean_navigator_v1/stage_rolecmp_rolebias/train_history.jsonl` show non-exploding value loss and stable gradient norms:
  - epoch 1: value loss `0.0911`, grad norm `0.5425`
  - epoch 2: value loss `0.0390`, grad norm `0.3847`
  - epoch 3: value loss `0.0493`, grad norm `0.4625`
  - epoch 4: value loss `0.0454`, grad norm `0.5384`
- [proven] Held-out checkpoint audit on `stage_rolecmp_rolebias/clean_navigator_v1_best.pt` shows useful value signal:
  - val: correlation(value, return) `0.496`, MSE `0.100`, zero-baseline MSE `0.164`
  - test: correlation(value, return) `0.676`, MSE `0.093`, zero-baseline MSE `0.177`
- [partially proven] The critic is still coarse:
  - value mean is about `0.103` on both val and test
  - return mean is `0.241` on val and `0.283` on test
  - so the critic systematically underestimates future return magnitude
- Judgment:
  - learning: yes
  - decorative: no
  - strong enough to explain the remaining bottleneck by itself: no

### 9. Training-only or also used at deployment-time inference?

- [proven] Training-only for decision making.
- `select_action()` uses only `selected_indices` from `model.act(...)`.
- The value output is computed during inference but not used to rank or select actions.

### 10. Under the current role-specialized lane, is the critic unchanged or already role-aware?

- [proven] Unchanged and not role-aware.
- `role_potentials` only affect actor logits.
- The critic still sees only pooled encoder output plus graph features.

### 11. Most likely current bottleneck?

- [proven] Primary bottleneck: set-level value / credit issue centered on the frontier/disambiguation role.
- [partially proven] Secondary bottleneck: frontier slot mechanism remains weak.
- [not proven] Critic weakness is the main blocker.
- Evidence:
  - Role specialization improved reward and unresolved reduction substantially without changing the critic.
  - Latest follow-up `stage_sampler_availability10_rolebias_pairfrontier` increased frontier alignment (`0.490 -> 0.551`) but reduced overall test reward (`0.574 -> 0.535`) and unresolved reduction (`0.427 -> 0.400`).
  - That pattern fits mis-priced set-level credit better than a pure actor-representation failure.
  - Optimization looks stable enough that instability is not the first explanation.

## Important drift note

- [proven] Some supporting role-specialization prose is stale relative to the current code.
- The current `compute_slot_role_potentials()` implementation uses witness-pair breadth for frontier potential.
- Older role-analysis artifacts still describe an earlier frontier formula.
- The targeted role pytest currently fails because it expects the obsolete function signature.
- Conclusion:
  - trust current code for the exact current mechanism
  - treat older role-definition prose as historical evidence, not current source of truth
