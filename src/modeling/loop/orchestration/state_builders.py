import torch
from src.modeling.state.schema import ObservationState, PhysicsContext, ConstraintState
from src.modeling.evidence.builder import EvidenceBuilder

class StateBuilder:
    """
    Responsibilities:
    - Construct ObservationState from raw features
    - Construct PhysicsContext from topology and physics module
    - Construct EvidenceState using EvidenceBuilder
    - Construct ReasonerState and NavState for modules
    """
    def __init__(self, model):
        self.model = model
        self.evidence_builder = EvidenceBuilder(getattr(model, "cfg", None))

    def fuse_observation(self, x_nav_raw, inverse_indices, num_fused_nodes):
        """
        Fuse Raw Observation into Fused Space.
        Rules:
        - CH0 (Chlorine): Min (Negative Deviation)
        - CH1-End: Max (Binary Flags / Freshness)
        """
        from torch_scatter import scatter_max, scatter_min
        
        if x_nav_raw.size(0) == num_fused_nodes:
            return x_nav_raw
            
        x_nav_fused = torch.zeros(num_fused_nodes, x_nav_raw.size(1), device=x_nav_raw.device)
        
        # Ch0 (Chlorine): Min (Negative Deviation)
        x_nav_fused[:, 0] = scatter_min(x_nav_raw[:, 0], inverse_indices, dim=0, dim_size=num_fused_nodes)[0]
        
        # Ch1-End: Max (Binary Flags / Freshness)
        if x_nav_raw.size(1) > 1:
            x_nav_fused[:, 1:] = scatter_max(x_nav_raw[:, 1:], inverse_indices, dim=0, dim_size=num_fused_nodes)[0]
            
        return x_nav_fused

    def build_observation_state(self, x_nav):
        """Construct Observation State from Raw Features (Observation v2 Mapping)"""
        # x_nav schema:
        # 0: Chlorine (CH0)
        # 1: Toxic (CH1 - raw binary)
        # 2: Freshness
        # 3: Observed Flag
        # 4: Anchor (Compatibility / Legacy)
        
        observed_flag = x_nav[:, 3]
        toxic_raw = x_nav[:, 1]
        
        # Calculate Two-Hot Encoding
        is_observed = (observed_flag > 0.5)
        is_toxic = (toxic_raw > 0.5)
        
        # Positive Flag: Observed AND Toxic
        toxic_positive = torch.zeros_like(toxic_raw)
        toxic_positive[is_observed & is_toxic] = 1.0
        
        # Negative Flag: Observed AND Safe (Not Toxic)
        toxic_negative = torch.zeros_like(toxic_raw)
        toxic_negative[is_observed & (~is_toxic)] = 1.0
        
        # [Cleanup Phase C] Anchor is optional/compatibility
        anchor_feat = torch.zeros_like(x_nav[:, 0])
        if x_nav.size(1) > 4:
            anchor_feat = x_nav[:, 4]
        
        return ObservationState(
            observed_flag=observed_flag,
            freshness=x_nav[:, 2],
            chlorine_deviation=x_nav[:, 0],
            toxic_positive_flag=toxic_positive,
            toxic_negative_flag=toxic_negative,
            anchor=anchor_feat
        )

    def build_physics_context(self, curr_edge_index, curr_edge_attr, physics_ctx, device, num_nodes=None, batch=None, stt_dynamic=None):
        """Construct Physics Context from Topology and Physics Module"""
        stt_tensor = curr_edge_attr[:, 3] if curr_edge_attr.size(1) > 3 else torch.zeros(curr_edge_index.size(1), device=device)
        
        # [Added for Reachability Rule Module]
        # Ch0: Log Median STT
        stt_median = curr_edge_attr[:, 0] if curr_edge_attr.size(1) > 0 else torch.zeros(curr_edge_index.size(1), device=device)
        # Ch2: Log Min STT
        stt_min = curr_edge_attr[:, 2] if curr_edge_attr.size(1) > 2 else torch.zeros(curr_edge_index.size(1), device=device)
        
        direction_tensor = torch.zeros(curr_edge_index.size(1), device=device) # Placeholder
        
        # Get Feasible Mask (Source-wise [N])
        if 'feasible_mask' in physics_ctx:
            feasible_mask_tensor = physics_ctx['feasible_mask']
        else:
            # Default to All Feasible if no physics constraint
            # Must match Node count [N]
            if num_nodes is not None:
                feasible_mask_tensor = torch.ones(num_nodes, device=device)
            else:
                # Fallback (Legacy/Risk of mismatch)
                feasible_mask_tensor = torch.ones(stt_tensor.size(0), device=device) # Likely wrong shape [E]

        return PhysicsContext(
            stt=stt_tensor,
            direction=direction_tensor,
            edge_index=curr_edge_index,
            feasible_mask=feasible_mask_tensor,
            stt_median=stt_median,
            stt_min=stt_min,
            stt_dynamic=stt_dynamic, # [E, 1]
            edge_attr=curr_edge_attr,
            batch=batch
        )



    def build_evidence_state(self, obs_state, phys_context, t_sim=None):
        """Build Higher-Order Evidence State"""
        return self.evidence_builder.build_evidence_state(obs_state, phys_context, t_sim)

    def build_constraint_state(self, dynamic_state, num_nodes, device):
        """Construct Runtime Constraint State from dynamic_state with backward-compatible defaults."""
        existing = dynamic_state.get('constraint_state')
        if existing is not None:
            return existing

        sampled_mask = dynamic_state.get('sampled_mask', dynamic_state.get('accumulated_mask'))
        if sampled_mask is None:
            sampled_mask = torch.zeros((num_nodes, 1), device=device)
        else:
            sampled_mask = sampled_mask.clone()

        zeros = torch.zeros_like(sampled_mask)
        confirmed_non_source_mask = dynamic_state.get('confirmed_non_source_mask')
        confirmed_source_mask = dynamic_state.get('confirmed_source_mask')
        no_resample_mask = dynamic_state.get('no_resample_mask')

        return ConstraintState(
            confirmed_non_source_mask=zeros if confirmed_non_source_mask is None else confirmed_non_source_mask.clone(),
            confirmed_source_mask=zeros if confirmed_source_mask is None else confirmed_source_mask.clone(),
            sampled_mask=sampled_mask,
            no_resample_mask=sampled_mask.clone() if no_resample_mask is None else no_resample_mask.clone(),
        )

    def build_nav_state(self, h_fused, x_nav, obs_state, valid_mask, nav_memory_state, explicit_context, nav_state_summary, 
                       current_action_k, current_tau, fused_batch, fov_params, action_policy, 
                       static_ctx, last_logits, curr_edge_index, curr_edge_attr, evidence_state, constraint_state,
                       critic_privileged=None, emit_action_audit_scalars=True, step_index=None):
        
        feature_mode = getattr(self.model.cfg.data, 'feature_mode', 'baseline')
        x_nav_input = x_nav
        if feature_mode == 'no_mask':
            x_nav_input = x_nav.clone()
            if x_nav_input.size(1) > 3:
                x_nav_input[:, 3] = 0.0

        nav_state = {
            'h_fused': h_fused, 
            'x_nav': x_nav_input,
            'observation_state': obs_state, 
            'valid_mask': valid_mask,
            'nav_memory_state': nav_memory_state,
            'explicit_context': explicit_context,
            'nav_state_summary': nav_state_summary,
            'k': current_action_k,
            'tau': current_tau,
            'batch': fused_batch,
            'fov_params': fov_params,
            'action_policy': action_policy,
            'n_id': static_ctx.get('fused_global_ids'),
            'inverse_indices': static_ctx.get('inverse_indices'),
            'fused_edge_index': curr_edge_index,
            'fused_edge_attr': curr_edge_attr,
            'evidence_state': evidence_state,
            'constraint_state': constraint_state,
        }
        if step_index is not None:
            nav_state['episode_index'] = int(step_index)
            nav_state['current_step'] = int(step_index)
        navigator_type = getattr(self.model.cfg.model, 'navigator_type', '')
        if navigator_type == 'navigator_vnext':
            # Navigator-vNext actor contract is runtime/deployable only.
            nav_state = {
                'h_fused': h_fused,
                'observation_state': obs_state,
                'evidence_state': evidence_state,
                'constraint_state': constraint_state,
                'valid_mask': valid_mask,
                'nav_state_summary': nav_state_summary,
                'k': current_action_k,
                'tau': current_tau,
                'batch': fused_batch,
                'fused_edge_index': curr_edge_index,
                'emit_action_audit_scalars': bool(emit_action_audit_scalars),
            }
        if last_logits is not None:
            if navigator_type != 'navigator_vnext':
                nav_state['reasoner_logits'] = last_logits
        vnext_cfg = getattr(self.model.cfg.model, 'navigator_vnext', {})
        if isinstance(vnext_cfg, dict):
            aux_privileged_enabled = bool(vnext_cfg.get('use_aux_privileged_critic', False))
        else:
            aux_privileged_enabled = bool(getattr(vnext_cfg, 'use_aux_privileged_critic', False))
        if critic_privileged is not None and (navigator_type != 'navigator_vnext' or aux_privileged_enabled):
            nav_state['critic_privileged'] = critic_privileged
        return nav_state

    def build_reasoner_state(
        self,
        h_fused,
        x_nav,
        obs_state,
        static_ctx,
        inverse_indices,
        causal_anchors,
        acc_mask_local,
        reasoner_memory_state,
        evidence_state,
        constraint_state,
        valid_mask=None,
        nav_state_summary=None,
    ):
        
        feature_mode = getattr(self.model.cfg.data, 'feature_mode', 'baseline')
        x_nav_input = x_nav
        if feature_mode == 'no_mask':
            x_nav_input = x_nav.clone()
            if x_nav_input.size(1) > 3:
                x_nav_input[:, 3] = 0.0

        reasoner_state = {
            'h_fused': h_fused,
            'x_nav': x_nav_input,
            'observation_state': obs_state,
            'n_id': static_ctx.get('fused_global_ids'),
            'inverse_indices': inverse_indices,
            'causal_anchors': causal_anchors,
            'accumulated_mask': acc_mask_local,
            'memory_state': reasoner_memory_state,
            'evidence_state': evidence_state,
            'constraint_state': constraint_state,
        }
        if valid_mask is not None:
            reasoner_state['valid_mask'] = valid_mask
        if nav_state_summary is not None:
            reasoner_state['nav_state_summary'] = nav_state_summary
        return reasoner_state
