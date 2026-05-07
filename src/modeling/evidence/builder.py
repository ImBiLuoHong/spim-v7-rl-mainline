from typing import Optional
import os
import time

import torch
import torch.nn.functional as F
from src.modeling.evidence.contradiction_oracle_v1 import (
    DEFAULT_FRONTIER_SAFE_CLOSE_TAU_MIN,
    DEFAULT_MINED_TOP_K_SAFE_WITNESSES,
    DEFAULT_ORACLE_HISTORY_PHYSCTX_MODE,
    DEFAULT_WITNESS_MINING_MODE,
    PracticalContradictionV2Config,
    compute_oracle_v1_contradiction,
    compute_practical_v2_contradiction,
)
from src.modeling.state.schema import ObservationState, PhysicsContext, EvidenceState
from src.modeling.evidence.reachability import ReachabilityRuleModule
from src.modeling.evidence.dynamic_reachability import DynamicReachabilityRuleModule

class EvidenceBuilder:
    """
    Core module responsible for constructing EvidenceState from ObservationState and PhysicsContext.
    EvidenceState v1 formalization:
    - support is the mainline evidence
    - suspect is kept as a soft prior / diagnostic
    - contradiction is frozen as an auxiliary audit / explanation branch
    """
    # Hyperparameters for Rule Weights
    ALPHA_T = 1.0   # Toxic Support Weight (Primary)
    ALPHA_C = 0.1   # Chlorine Support Weight (Auxiliary - explicitly weak)
    
    BETA_S = 0.2    # Soft Contradiction Weight
    BETA_H = 1.0    # Hard Contradiction Weight
    
    GAMMA_P = 1.0   # Consistency Positive Weight
    GAMMA_N = 1.0   # Consistency Negative Penalty Weight

    # Physics Constants (Approximation)
    AVG_HOP_TIME = 30.0 # Minutes per hop (conservative)

    # Calibration Parameters (Mutable for Audit)
    SUPPORT_BUFFER = 60.0
    SUPPORT_SIGMA = 15.0
    TIME_SCALE_FACTOR = 1.0
    STT_SCALE_FACTOR = 1.0

    def __init__(self, cfg=None):
        self.cfg = cfg
        self.reachability = ReachabilityRuleModule()
        self.dynamic_reachability = DynamicReachabilityRuleModule()
        self._profile_enabled = os.environ.get("EVIDENCE_PROFILE", "").strip().lower() in {"1", "true", "yes", "on"}
        self._profile = {}
        # Small cache for the upstream reverse adjacency used by contradiction /
        # reaction-consistency propagation. This is topology-only, so reusing it
        # across repeated evidence builds preserves semantics.
        self._upstream_adj_rev_cache_key = None
        self._upstream_adj_rev_cache_value = None

    def _profile_add(self, key: str, value: float):
        if not self._profile_enabled:
            return
        self._profile[key] = self._profile.get(key, 0.0) + float(value)

    def reset_profile(self):
        self._profile.clear()
        self.dynamic_reachability.reset_profile()

    def get_profile(self):
        return dict(self._profile)

    def _cfg_get(self, path: str, default):
        current = self.cfg
        for part in path.split("."):
            if current is None:
                return default
            if isinstance(current, dict):
                current = current.get(part)
            else:
                current = getattr(current, part, None)
        return default if current is None else current

    def _resolve_candidate_mask(
        self,
        candidate_mask: Optional[torch.Tensor],
        num_nodes: int,
        device: torch.device,
    ) -> torch.Tensor:
        if candidate_mask is None:
            return torch.ones(num_nodes, device=device)
        return candidate_mask.float().to(device)

    def build_evidence_state(self, observation_state: ObservationState, physics_context: PhysicsContext, t_sim: torch.Tensor = None) -> EvidenceState:
        """
        Based on observation_state and physics_context, generate evidence_state.
        Args:
            t_sim: [B] Tensor of simulation time (minutes). If None, assume infinite/static.
        """
        t_total = time.perf_counter()
        # 0. Pre-compute Reachability Rules
        # This is the unified physical base.
        if physics_context.batch is not None:
            batch = physics_context.batch
        else:
            # Fallback: assume single graph if no batch provided
            batch = torch.zeros(observation_state.observed_flag.size(0), dtype=torch.long, device=observation_state.observed_flag.device)
            
        # [Upgrade] Switch to Dynamic Reachability if STT Series is available
        support_distance_matrix = None
        max_support_seeds = 50
        if physics_context.stt_dynamic is not None:
            # Dynamic Logic
            t_seg = time.perf_counter()
            reach_res, support_distance_matrix = self.dynamic_reachability.compute_reachability_bundle(
                observation_state,
                physics_context,
                t_sim,
                batch,
                max_pos_seeds=max_support_seeds,
            )
            self._profile_add("build/reachability_s", time.perf_counter() - t_seg)
        else:
            # Static Logic (Fallback)
            t_seg = time.perf_counter()
            reach_res = self.reachability.compute_reachability(observation_state, physics_context, t_sim, batch)
            self._profile_add("build/reachability_s", time.perf_counter() - t_seg)
        
        # 1. Suspect Pool (Coarse Prior)
        t_seg = time.perf_counter()
        suspect_pool = self.compute_suspect_pool(observation_state, physics_context, reach_res)
        self._profile_add("build/suspect_pool_s", time.perf_counter() - t_seg)

        # 2. Support Score (Source-wise Evidence)
        # EvidenceState v1 final closure: support is no longer masked by source_validity.
        t_seg = time.perf_counter()
        support_res = self.compute_support_score(
            observation_state,
            physics_context,
            None,
            reach_res,
            t_sim,
            precomputed_distance_matrix=support_distance_matrix,
        )
        support_score = support_res['total']
        self._profile_add("build/support_score_s", time.perf_counter() - t_seg)

        # 3. Contradiction Score (Source-wise Evidence)
        # Contradiction is retained for auxiliary/audit use and no longer drives the mainline.
        t_seg = time.perf_counter()
        contra_res = self.compute_contradiction_score(observation_state, physics_context, None, reach_res, t_sim)
        contradiction_score = contra_res['total']
        self._profile_add("build/contradiction_s", time.perf_counter() - t_seg)

        # 4. Reaction Consistency (Auxiliary)
        t_seg = time.perf_counter()
        cons_res = self.compute_reaction_consistency(observation_state, physics_context, None, t_sim)
        reaction_consistency = cons_res['total']
        self._profile_add("build/reaction_consistency_s", time.perf_counter() - t_seg)

        # 5. Uncertainty Gap (Information Gap)
        t_seg = time.perf_counter()
        uncertainty_gap = self.compute_uncertainty_gap(observation_state, physics_context, suspect_pool)
        observation_validity = torch.ones_like(observation_state.observed_flag)
        deprecated_source_validity = torch.ones_like(observation_state.observed_flag)
        self._profile_add("build/uncertainty_gap_s", time.perf_counter() - t_seg)

        t_seg = time.perf_counter()
        if t_sim is not None:
            t_nodes = t_sim[batch]
        else:
            t_nodes = torch.full_like(observation_state.observed_flag, 1e6)
        dist_hard_neg = reach_res.get('dist_hard_neg')
        if isinstance(dist_hard_neg, torch.Tensor):
            negative_exclusion_slack = torch.relu((t_nodes - 10.0) - dist_hard_neg) * reach_res['hard_reachability_from_neg']
        else:
            negative_exclusion_slack = torch.zeros_like(observation_state.observed_flag)
        evidence_state = EvidenceState(
            suspect_pool=suspect_pool,
            support_score=support_score,
            contradiction_score=contradiction_score,
            reaction_consistency=reaction_consistency,
            uncertainty_gap=uncertainty_gap,
            source_validity=deprecated_source_validity,
            observation_validity=observation_validity,
            
            # Audit Sub-terms
            support_toxic_term=support_res['toxic'],
            support_chlorine_term=support_res['chlorine'],
            
            # New Support Audit Terms
            support_coverage_term=support_res['coverage'],
            support_timing_term=support_res['timing'],
            support_focus_term=support_res['focus'],
            
            # MAPPING: Soft -> Toxic Term (legacy name match), Hard -> Clean Term
            contradiction_toxic_term=contra_res['soft'], 
            contradiction_clean_term=contra_res['hard'],
            
            consistency_positive_term=cons_res['positive'],
            consistency_negative_penalty=cons_res['negative'],
            
            # Gate Audit
            compatibility_gate=None,
            arrival_gate=contra_res['arrival_gate'], 
            
            # New Suspect Pool Gates Audit
            topology_gate=reach_res['topology_reachable'], 
            coarse_time_gate=reach_res['soft_reachability'], 
            not_ruled_out_gate=1.0 - reach_res['hard_reachability_from_neg'], # Pressure inv
            negative_exclusion_slack=negative_exclusion_slack
        )
        self._profile_add("build/pack_s", time.perf_counter() - t_seg)
        self._profile_add("build/total_s", time.perf_counter() - t_total)
        return evidence_state

    def compute_suspect_pool(self, observation_state: ObservationState, physics_context: PhysicsContext, reach_res: dict):
        """
        1. suspect_pool_rule(s) (Redefined v3)
        Semantics: Coarse Candidate Prior = Topo * CoarseTime * (1 - Pressure).
        """
        # A. Topology Score (Reachability)
        s_topo = reach_res['topology_reachable']
             
        # B. Coarse Time Score (Wide Window Compatibility)
        s_time_coarse = reach_res['soft_reachability']

        # C. Negative Pressure (Hard Reachability from Negative)
        # If candidate reaches a Safe node strictly, it is pressured.
        s_pressure = reach_res['hard_reachability_from_neg']
        
        # Combine: Score-based
        # suspect_score = w1 * topo + w2 * time - w3 * pressure
        # w1=1.0, w2=0.8, w3=0.8
        suspect_score = 1.0 * s_topo + 0.8 * s_time_coarse - 0.8 * s_pressure
        
        # Threshold
        # If Topo=1, Time=1, Pressure=0 => 1.8 -> Keep
        # If Topo=1, Time=1, Pressure=1 => 1.0 -> Keep (Borderline? Maybe false neg)
        # If Topo=1, Time=0, Pressure=0 => 1.0 -> Keep
        # If Topo=1, Time=0, Pressure=1 => 0.2 -> Reject
        # If Topo=0 => 0.0 -> Reject
        
        threshold = 0.5
        suspect_pool = (suspect_score > threshold).float()
        
        # Safety: Ensure at least one candidate if possible
        if suspect_pool.sum() < 0.5:
            # Fallback: Just Topo
            suspect_pool = s_topo

        return suspect_pool

    def _get_adj_rev(self, edge_index, num_nodes, device, physics_context=None):
        """
        Construct Reverse Adjacency Matrix for Upstream Propagation.
        A_rev[v, u] = 1 if u -> v (Flow).
        """
        cache_key = (
            int(num_nodes),
            str(device),
            int(edge_index.data_ptr()),
            tuple(edge_index.shape),
        )
        if physics_context is not None:
            cached_key = getattr(physics_context, "_cached_upstream_adj_rev_key", None)
            cached_value = getattr(physics_context, "_cached_upstream_adj_rev", None)
            if cached_key == cache_key and cached_value is not None:
                return cached_value
        if self._upstream_adj_rev_cache_key == cache_key and self._upstream_adj_rev_cache_value is not None:
            return self._upstream_adj_rev_cache_value

        src, dst = edge_index
        # Propagate FROM dst TO src (Upstream)
        # indices = [src, dst]
        
        values = torch.ones_like(src, dtype=torch.float)
        
        adj_rev = torch.sparse_coo_tensor(
            torch.stack([src, dst]), values, (num_nodes, num_nodes)
        ).coalesce()
        if physics_context is not None:
            physics_context._cached_upstream_adj_rev_key = cache_key
            physics_context._cached_upstream_adj_rev = adj_rev
        self._upstream_adj_rev_cache_key = cache_key
        self._upstream_adj_rev_cache_value = adj_rev
        
        return adj_rev

    def _propagate_upstream(self, signal, physics_context, steps=3, decay=0.8):
        """
        Propagate signal from downstream nodes to upstream candidates.
        """
        if int(steps) <= 0:
            return signal.detach().clone()
        edge_index = physics_context.edge_index
        N = signal.size(0)
        device = signal.device
        
        orig_dtype = signal.dtype
        
        with torch.amp.autocast('cuda', enabled=False):
            signal_f32 = signal.float()
            
            adj_rev = self._get_adj_rev(edge_index, N, device, physics_context=physics_context)
            
            current = signal_f32.unsqueeze(1) # [N, 1]
            accumulated = signal_f32.clone()
            
            for _ in range(int(steps)):
                next_val = torch.sparse.mm(adj_rev, current).squeeze(1) * decay
                accumulated += next_val
                current = next_val.unsqueeze(1)
            
            return accumulated.to(orig_dtype)

    def compute_support_score(
        self,
        observation_state: ObservationState,
        physics_context: PhysicsContext,
        suspect_pool: torch.Tensor,
        reach_res: dict,
        t_sim: torch.Tensor = None,
        precomputed_distance_matrix: torch.Tensor = None,
    ):
        """
        2. support_rule(s) (Redefined v4 - Specificity & Hub Bias Mitigation)
        Semantics: Candidate-Relative Specificity.
        Note: the third arg is historically named `suspect_pool`, but EvidenceState v1
        uses it as a generic candidate mask. Mainline builder now passes `None`,
        while audit callsites may still pass explicit compare masks.
        
        Formula:
          support(s) = 0.20 * Base + 0.55 * Specificity + 0.20 * Focus + 0.05 * Chlorine
          
        Rationale:
          - Base: Raw explanation quality (Reachability + Timing).
          - Specificity: Relative explanation power compared to other candidates (Anti-Hub).
          - Focus: Top-k specificity (Robustness).
          - Chlorine: Auxiliary weak signal.
        """
        t_total = time.perf_counter()
        # 0. Setup
        t_seg = time.perf_counter()
        observed = observation_state.observed_flag
        toxic_flag = observation_state.toxic_positive_flag
        chlorine_dev = observation_state.chlorine_deviation
        
        num_nodes = observed.size(0)
        device = observed.device
        candidate_mask = self._resolve_candidate_mask(suspect_pool, num_nodes, device)
        
        # Identify Positive Observations P
        # indices where toxic_positive_flag == 1
        pos_indices = torch.nonzero(toxic_flag).squeeze(1)
        num_pos = pos_indices.size(0)
        self._profile_add("support/setup_s", time.perf_counter() - t_seg)
        
        if num_pos == 0:
            # No positive observations -> No support
            result = {
                'total': torch.zeros(num_nodes, device=device),
                'base': torch.zeros(num_nodes, device=device),
                'specificity': torch.zeros(num_nodes, device=device),
                'focus': torch.zeros(num_nodes, device=device),
                'chlorine': torch.zeros(num_nodes, device=device),
                # Legacy keys
                'coverage': torch.zeros(num_nodes, device=device),
                'timing': torch.zeros(num_nodes, device=device),
                'toxic': torch.zeros(num_nodes, device=device)
            }
            self._profile_add("support/empty_return_s", time.perf_counter() - t_total)
            self._profile_add("support/total_s", time.perf_counter() - t_total)
            return result
            
        # 1. Compute Pairwise Distances T(s, i) for all i in P
        # We limit max P to avoid OOM if graph is huge (though usually P is small)
        MAX_P = 50
        if num_pos > MAX_P:
            # Random sample or just take first? 
            # Deterministic is better for audit. Take first MAX_P.
            pos_indices = pos_indices[:MAX_P]
            num_pos = MAX_P
            if precomputed_distance_matrix is not None and precomputed_distance_matrix.size(1) >= num_pos:
                precomputed_distance_matrix = precomputed_distance_matrix[:, :num_pos]
        if precomputed_distance_matrix is not None and precomputed_distance_matrix.size(1) != num_pos:
            precomputed_distance_matrix = None
            
        # [Upgrade] Dynamic STT Support
        # Check if dynamic weights are available (via stt_dynamic or reach_res)
        # If using DynamicReachability, we should use dynamic distance computation.
        use_dynamic = (physics_context.stt_dynamic is not None)
        
        # Weights for Bellman-Ford / Dijkstra
        if use_dynamic and precomputed_distance_matrix is None:
            # Use dynamic STT slice (Sign handled by module)
            w_soft = torch.abs(physics_context.stt_dynamic.view(-1))
        elif not use_dynamic:
            # Static Fallback
            if physics_context.stt_median is not None:
                w_soft = torch.expm1(physics_context.stt_median)
            elif physics_context.edge_attr is not None and physics_context.edge_attr.size(1) > 0:
                w_soft = torch.expm1(physics_context.edge_attr[:, 0])
            else:
                w_soft = torch.ones_like(physics_context.edge_index[0], dtype=torch.float) * 20.0

        # Calibration: STT Scaling
        if precomputed_distance_matrix is None:
            t_seg = time.perf_counter()
            w_soft = w_soft * self.STT_SCALE_FACTOR
            w_soft = torch.clamp(w_soft, min=0.0)
            self._profile_add("support/weight_prep_s", time.perf_counter() - t_seg)
        
        # Loop to get distances [N, num_pos]
        # T[s, k] is distance from source s to pos_obs k
        t_seg = time.perf_counter()
        if precomputed_distance_matrix is not None:
            T_matrix = precomputed_distance_matrix
        elif use_dynamic:
            T_matrix = self.dynamic_reachability.compute_distance_matrix(pos_indices, physics_context, num_nodes)
        else:
            dists_list = []
            for idx in pos_indices:
                seed = torch.zeros(num_nodes, device=device)
                seed[idx] = 1.0
                d = self.reachability.compute_distance(seed, physics_context, w_soft, num_nodes)
                dists_list.append(d)
            T_matrix = torch.stack(dists_list, dim=1)
        self._profile_add("support/distance_matrix_s", time.perf_counter() - t_seg)
        
        # 2. Prepare Time Info
        t_seg = time.perf_counter()
        if t_sim is not None:
            # Map batch t_sim to nodes
            if physics_context.batch is not None:
                t_nodes = t_sim[physics_context.batch]
            else:
                t_nodes = torch.full((num_nodes,), t_sim[0], device=device)
        else:
            t_nodes = torch.full((num_nodes,), 100.0, device=device) # Default
            
        # Calibration: Time Scaling (Optional)
        t_nodes = t_nodes * self.TIME_SCALE_FACTOR
        
        t_nodes_exp = t_nodes.unsqueeze(1) # [N, 1]
        self._profile_add("support/time_prepare_s", time.perf_counter() - t_seg)
        
        # 3. Soft Reachability R_soft(s, i)
        # Condition: T(s, i) <= t_now + buffer
        t_seg = time.perf_counter()
        buffer_soft = self.SUPPORT_BUFFER
        R_soft = (T_matrix <= (t_nodes_exp + buffer_soft)).float()
        
        # 4. Compute Base Quality q_{s,i}
        # q = R_soft * phi(T)
        sigma = self.SUPPORT_SIGMA
        time_diff = torch.relu(T_matrix - t_nodes_exp)
        phi = torch.exp(- (time_diff**2) / (2 * sigma**2))
        
        q_vals = R_soft * phi # [N, P]
        self._profile_add("support/gate_and_quality_s", time.perf_counter() - t_seg)
        
        # 5. Compute Terms
        
        # A. Support_Base
        # Mean of q_{s,i} over P
        t_seg = time.perf_counter()
        base_scores = q_vals.mean(dim=1)
        
        # B. Support_Specificity
        # Normalize q by sum over candidates (Z_i)
        epsilon = 1e-6
        Z_i = q_vals.sum(dim=0, keepdim=True) + epsilon # [1, P]
        u_vals = q_vals / Z_i # [N, P]
        
        specificity_scores = u_vals.mean(dim=1)
        self._profile_add("support/base_specificity_s", time.perf_counter() - t_seg)
        
        # C. Support_Focus (Specificity-based)
        # Top-k mean of u_{s,i}
        t_seg = time.perf_counter()
        k = min(5, num_pos)
        if k > 0:
            topk_u, _ = torch.topk(u_vals, k, dim=1)
            focus_scores = topk_u.mean(dim=1)
        else:
            focus_scores = torch.zeros(num_nodes, device=device)
        self._profile_add("support/focus_s", time.perf_counter() - t_seg)
            
        # D. Support_Chlorine (Auxiliary)
        t_seg = time.perf_counter()
        delta_c = 0.5 # Background threshold
        c_i_vals = torch.abs(chlorine_dev[pos_indices]) # [P]
        a_i = (c_i_vals > delta_c).float() # [P]
        chlorine_weight = a_i * torch.log1p(c_i_vals) # [P]
        chlorine_weight_exp = chlorine_weight.unsqueeze(0)
        chlorine_scores = (R_soft * chlorine_weight_exp).mean(dim=1)
        self._profile_add("support/chlorine_s", time.perf_counter() - t_seg)
        
        # 6. Weighted Sum
        # New Weights: Base=0.20, Spec=0.55, Focus=0.20, Chlorine=0.05
        t_seg = time.perf_counter()
        w_base = 0.20
        w_spec = 0.55
        w_focus = 0.20
        w_chl = 0.05
        
        total_score = (w_base * base_scores + 
                       w_spec * specificity_scores + 
                       w_focus * focus_scores + 
                       w_chl * chlorine_scores)
        self._profile_add("support/weighted_sum_s", time.perf_counter() - t_seg)
                       
        # Apply candidate mask. In the pure support-led mainline this is neutral (all ones).
        t_seg = time.perf_counter()
        total_score = total_score * candidate_mask
        base_scores = base_scores * candidate_mask
        specificity_scores = specificity_scores * candidate_mask
        focus_scores = focus_scores * candidate_mask
        chlorine_scores = chlorine_scores * candidate_mask
        result = {
            'total': total_score,
            'base': base_scores,
            'specificity': specificity_scores,
            'focus': focus_scores,
            'chlorine': chlorine_scores,
            # Legacy mapping for compatibility
            'coverage': base_scores, 
            'timing': specificity_scores,
            'toxic': total_score 
        }
        self._profile_add("support/mask_and_pack_s", time.perf_counter() - t_seg)
        self._profile_add("support/total_s", time.perf_counter() - t_total)
        
        return result

    def compute_contradiction_score(
        self,
        observation_state: ObservationState,
        physics_context: PhysicsContext,
        suspect_pool: torch.Tensor,
        reach_res: dict,
        t_sim: torch.Tensor = None,
        contradiction_mode: str = "legacy",
        oracle_history_steps=None,
        safe_violation_tau_min: float = 15.0,
        history_phys_ctx_mode: str = DEFAULT_ORACLE_HISTORY_PHYSCTX_MODE,
        practical_v2_config: Optional[PracticalContradictionV2Config] = None,
        witness_mining_mode: str = DEFAULT_WITNESS_MINING_MODE,
        frontier_safe_close_tau_min: float = DEFAULT_FRONTIER_SAFE_CLOSE_TAU_MIN,
        mined_top_k_safe_witnesses: int = DEFAULT_MINED_TOP_K_SAFE_WITNESSES,
    ):
        """
        3. contradiction_rule(s) (Redefined v3)
        A. Soft Contradiction: SoftReach(Neg) (Likely Arrived).
        B. Hard Contradiction: HardReach(Neg) (Definitely Arrived).

        EvidenceState v1 keeps contradiction as a frozen auxiliary branch.
        The `suspect_pool` arg remains for audit/backward compatibility and acts as a
        generic candidate mask.
        """
        candidate_mask = self._resolve_candidate_mask(
            suspect_pool,
            int(observation_state.observed_flag.size(0)),
            observation_state.observed_flag.device,
        )
        if contradiction_mode == "oracle_v1":
            if oracle_history_steps is None:
                raise ValueError("oracle_history_steps must be provided when contradiction_mode='oracle_v1'")
            return compute_oracle_v1_contradiction(
                reachability_module=self.dynamic_reachability,
                history_steps=oracle_history_steps,
                num_nodes=int(observation_state.observed_flag.size(0)),
                safe_violation_tau_min=float(safe_violation_tau_min),
                suspect_pool=candidate_mask,
                phys_ctx_mode=history_phys_ctx_mode,
            )
        if contradiction_mode == "practical_v2":
            if oracle_history_steps is None:
                raise ValueError("oracle_history_steps must be provided when contradiction_mode='practical_v2'")
            return compute_practical_v2_contradiction(
                reachability_module=self.dynamic_reachability,
                history_steps=oracle_history_steps,
                num_nodes=int(observation_state.observed_flag.size(0)),
                safe_violation_tau_min=float(safe_violation_tau_min),
                suspect_pool=candidate_mask,
                phys_ctx_mode=history_phys_ctx_mode,
                config=practical_v2_config,
                witness_mining_mode=witness_mining_mode,
                frontier_safe_close_tau_min=float(frontier_safe_close_tau_min),
                mined_top_k_safe_witnesses=int(mined_top_k_safe_witnesses),
            )
        if contradiction_mode != "legacy":
            raise ValueError(f"Unknown contradiction_mode: {contradiction_mode}")

        # A. Soft Contradiction (Weak Pressure)
        # "Likely should have arrived at a safe node"
        # We use the binary reachability mask from ReachabilityRuleModule
        # reach_res['soft_reachability_from_neg']
        
        # To get magnitude (how many safe nodes?), we can multiply by Propagate(Safe).
        # But ReachabilityRuleModule aggregates "Is there ANY safe node reachable?".
        # If we want magnitude, we can use Propagate.
        
        sig_soft = observation_state.toxic_negative_flag
        # Use generous steps for Soft
        if t_sim is not None:
            avg_t = t_sim.float().mean().item()
            steps_soft = max(3, int(avg_t / 20.0))
        else:
            steps_soft = 5
            
        mag_soft = self._propagate_upstream(sig_soft, physics_context, steps=steps_soft, decay=0.9)
        
        # Gating: Must be physically/temporally reachable (Soft)
        term_soft = mag_soft * reach_res['soft_reachability_from_neg'] * candidate_mask
        
        # B. Hard Contradiction (Strong Pressure)
        # "Definitely should have arrived at a safe node"
        # We use Hard Reachability from Negative (Pressure)
        
        # We can also use Propagate(Safe) but with stricter steps?
        # Or just use the Binary Pressure from ReachabilityModule?
        # Prompt says: "contradiction ... based on safe observation and hard reachability"
        # "contradiction = beta_s * soft + beta_h * hard"
        
        # Let's use the Pressure (Binary/Max) as the base, maybe scaled by magnitude if available.
        # But Pressure is "Is there ANY?".
        # Let's use Propagate(Safe) * HardReach(Neg).
        
        # Hard steps (Conservative)
        if t_sim is not None:
            avg_t = t_sim.float().mean().item()
            steps_hard = max(0, int((avg_t - 10.0) / 30.0))
        else:
            steps_hard = 0
            
        # Clean Safe: Toxic Neg AND Clean Chlorine
        c_i = torch.abs(observation_state.chlorine_deviation)
        is_clean = (c_i < 0.1).float()
        sig_hard = sig_soft * is_clean
        
        mag_hard = self._propagate_upstream(sig_hard, physics_context, steps=steps_hard, decay=1.0)
        
        # Gating: Must be strictly reachable
        term_hard = mag_hard * reach_res['hard_reachability_from_neg'] * candidate_mask
        
        # Total
        total_score = self.BETA_S * term_soft + self.BETA_H * term_hard
        
        # Arrival Gate for Audit
        arrival_gate = reach_res['soft_reachability_from_neg']
        
        return {
            'total': total_score,
            'soft': term_soft,
            'hard': term_hard,
            'arrival_gate': arrival_gate
        }

    def compute_reaction_consistency(self, observation_state: ObservationState, physics_context: PhysicsContext, suspect_pool: torch.Tensor, t_sim: torch.Tensor = None):
        """
        4. reaction_consistency_rule(s)
        Diagnostic-only monitoring.
        """
        toxic_pos = observation_state.toxic_positive_flag
        toxic_neg = observation_state.toxic_negative_flag
        chlorine_dev = observation_state.chlorine_deviation
        candidate_mask = self._resolve_candidate_mask(
            suspect_pool,
            int(observation_state.observed_flag.size(0)),
            observation_state.observed_flag.device,
        )
        
        c_i = torch.abs(chlorine_dev)
        g_c = torch.log1p(c_i)
        
        # A. Positive Consistency
        sig_pos = toxic_pos * g_c
        term_pos = self._propagate_upstream(sig_pos, physics_context, steps=3)
        term_pos = term_pos * candidate_mask
        
        # B. Negative Inconsistency
        sig_neg = toxic_neg * g_c
        term_neg = self._propagate_upstream(sig_neg, physics_context, steps=3)
        term_neg = term_neg * candidate_mask
        
        # Total
        total_score = self.GAMMA_P * term_pos - self.GAMMA_N * term_neg
        
        return {
            'total': total_score,
            'positive': term_pos,
            'negative': term_neg
        }

    def compute_uncertainty_gap(self, observation_state: ObservationState, physics_context: PhysicsContext, suspect_pool: torch.Tensor):
        """
        5. uncertainty_gap_rule(i)
        """
        observed = observation_state.observed_flag
        gap = (1.0 - observed)
        return gap
