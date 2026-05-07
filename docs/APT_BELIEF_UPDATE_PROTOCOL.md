# APT Belief Update Protocol (v1.0)

## 1. Overview
The `RecurrentBeliefUpdate` module is designed to handle long-sequence dependencies and information reveal in the APT platform. It treats each iteration of the sequential loop as a single step of a generalized RNN, maintaining a persistent "belief state" across steps.

This module decouples state management from the Orchestrator (`Phase45Model`), ensuring that memory logic is pluggable, auditable, and configuration-driven.

## 2. Key Interfaces
- **BeliefStateBase**: A dataclass container for persistent states (e.g., hidden vectors in GRU). Must support `detach()` and `to(device)`.
- **BeliefUpdaterBase**: The core logic module.
    - `init_state(batch, num_nodes, device)`: Initializes the episode state.
    - `step(state, step_in)`: Updates state and returns `belief_ctx`.

## 3. Feature Firewall (Whitelist)
The `step_in` dictionary is strictly enforced via a whitelist to prevent information leakage (e.g., Poison Labels or Future Observations).

**Allowed Fields**:
- `t_sim`, `step_idx`, `batch`
- `valid_mask`, `anchor_type`, `anchor_time`, `freshness`
- `reasoner_posterior` (from previous step)
- `physics_ctx` (from current step)
- `fov_params` (from current step)
- `action_summary` (behavioral statistics)
- `node_embeddings` (hidden representations only)

**Forbidden Fields**:
- `poison_label`, `gt_source`, `future_obs`

## 4. Implementations
- **none**: No-op updater for ablation studies.
- **global_gru**: Maintains a single hidden vector `h_t [B, H]` per graph, driven by statistical features.
- **node_gru**: Maintains per-node hidden vectors `H_t [N, H]`, driven by node-wise embeddings.

## 5. YAML Configuration (SSOT)
```yaml
model:
  belief_updater:
    type: "global_gru"
    params:
      hidden_dim: 128
      input_keys: ["entropy", "top1_margin"]
      detach_every: 5  # BPTT truncation
      clamp_norm: 5.0  # Gradient stability
  
  # Injection switches
  inject_belief_to_nav: true
  inject_belief_to_reasoner: true
```

## 6. Auditing
Run the centralized audit script to verify compliance:
```bash
python scripts/run_audit.py
```
This script checks:
1. Interface compliance and registration.
2. Minimal CPU forward for all implementations.
3. Feature Firewall enforcement (forbidden key detection).
4. Orchestrator purity (no direct RNN logic in `Phase45Model`).
