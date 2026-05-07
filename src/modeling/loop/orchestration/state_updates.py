import torch
from torch_scatter import scatter_max, scatter_sum
from src.modeling.state.schema import ConstraintState


def build_runtime_verdict_payload(
    selection_mask,
    fused_batch=None,
    fused_global_ids=None,
    graph_source_global_ids=None,
    explicit_payload=None,
):
    """
    Build an explicit step verdict payload in fused space.

    Priority:
    1. Reuse an already-explicit payload if provided.
    2. Derive source/non-source verdicts from graph-level source global ids.
    3. Report verdict unavailable when neither source exists.
    """
    if selection_mask is None:
        return {
            'verdict_available': False,
            'verdict_source': None,
            'selected_source_mask': None,
            'selected_non_source_mask': None,
            'is_source_hit': None,
            'has_confirmed_non_source': None,
        }

    selection_mask = selection_mask.float()
    zero_mask = torch.zeros_like(selection_mask)

    if explicit_payload is not None:
        payload = dict(explicit_payload)
        payload.setdefault('selected_source_mask', zero_mask.clone())
        payload.setdefault('selected_non_source_mask', zero_mask.clone())
        payload.setdefault('verdict_available', True)
        payload.setdefault('verdict_source', 'explicit_payload')

        if fused_batch is not None:
            batch_flat = fused_batch.view(-1)
            num_graphs = int(batch_flat.max().item()) + 1 if batch_flat.numel() > 0 else 0
            if payload.get('is_source_hit') is None:
                payload['is_source_hit'] = (
                    scatter_max(
                        payload['selected_source_mask'].view(-1).float(),
                        batch_flat,
                        dim=0,
                        dim_size=num_graphs,
                    )[0]
                    > 0.5
                )
            if payload.get('has_confirmed_non_source') is None:
                payload['has_confirmed_non_source'] = (
                    scatter_max(
                        payload['selected_non_source_mask'].view(-1).float(),
                        batch_flat,
                        dim=0,
                        dim_size=num_graphs,
                    )[0]
                    > 0.5
                )
        return payload

    if fused_batch is None or fused_global_ids is None or graph_source_global_ids is None:
        return {
            'verdict_available': False,
            'verdict_source': None,
            'selected_source_mask': zero_mask,
            'selected_non_source_mask': zero_mask.clone(),
            'is_source_hit': None,
            'has_confirmed_non_source': None,
        }

    batch_flat = fused_batch.view(-1).long()
    num_graphs = int(batch_flat.max().item()) + 1 if batch_flat.numel() > 0 else 0
    source_global_tensor = torch.as_tensor(
        graph_source_global_ids,
        device=selection_mask.device,
        dtype=fused_global_ids.dtype,
    ).view(-1)
    if source_global_tensor.numel() == 1 and num_graphs > 1:
        source_global_tensor = source_global_tensor.repeat(num_graphs)
    if source_global_tensor.numel() != num_graphs:
        return {
            'verdict_available': False,
            'verdict_source': None,
            'selected_source_mask': zero_mask,
            'selected_non_source_mask': zero_mask.clone(),
            'is_source_hit': None,
            'has_confirmed_non_source': None,
        }

    selected_bool = selection_mask.view(-1) > 0.5
    source_global_per_node = source_global_tensor[batch_flat]
    source_hit_bool = selected_bool & (fused_global_ids.view(-1).to(source_global_per_node.dtype) == source_global_per_node)

    selected_source_mask = zero_mask.clone()
    selected_non_source_mask = zero_mask.clone()
    selected_source_mask.view(-1)[source_hit_bool] = 1.0
    selected_non_source_mask.view(-1)[selected_bool & (~source_hit_bool)] = 1.0

    is_source_hit = (
        scatter_max(
            selected_source_mask.view(-1),
            batch_flat,
            dim=0,
            dim_size=num_graphs,
        )[0]
        > 0.5
    )
    has_confirmed_non_source = (
        scatter_max(
            selected_non_source_mask.view(-1),
            batch_flat,
            dim=0,
            dim_size=num_graphs,
        )[0]
        > 0.5
    )

    return {
        'verdict_available': True,
        'verdict_source': 'graph_source_global',
        'selected_source_mask': selected_source_mask,
        'selected_non_source_mask': selected_non_source_mask,
        'is_source_hit': is_source_hit,
        'has_confirmed_non_source': has_confirmed_non_source,
    }


class StateUpdater:
    """
    Responsibilities:
    - Update local mask (Observation write-back)
    - Update freshness (Decay)
    - Update observation state (Signal/Poison from Ground Truth)
    - Update causal anchors
    """
    def __init__(self, model):
        self.model = model
        self.cfg = model.cfg

    def update_mask(self, acc_mask_local, confirmation_mask):
        """Update local mask with new confirmations"""
        A_total_mask = (confirmation_mask > 0.5)
        acc_mask_local = torch.max(acc_mask_local, A_total_mask.float())
        return acc_mask_local, A_total_mask

    def update_constraint_state(
        self,
        constraint_state,
        selection_mask,
        verdict_payload=None,
        fused_source_label=None,
        allow_label_fallback=False,
    ):
        """
        Update hard runtime constraints from the latest sampled nodes and explicit
        step verdict payloads. Supervised label fallback is opt-in only.
        """
        selection_mask = selection_mask.float()

        sampled_mask = torch.max(constraint_state.sampled_mask, selection_mask)
        no_resample_mask = torch.max(constraint_state.no_resample_mask, selection_mask)
        confirmed_non_source_mask = constraint_state.confirmed_non_source_mask.clone()
        confirmed_source_mask = constraint_state.confirmed_source_mask.clone()

        selected_source_mask = torch.zeros_like(selection_mask)
        selected_non_source_mask = torch.zeros_like(selection_mask)
        verdict_available = False
        verdict_source = None
        fallback_used = False
        is_source_hit = None
        has_confirmed_non_source = None

        if verdict_payload is not None and verdict_payload.get('verdict_available', False):
            selected_source_mask = verdict_payload.get('selected_source_mask', selected_source_mask).float()
            selected_non_source_mask = verdict_payload.get('selected_non_source_mask', selected_non_source_mask).float()
            verdict_available = True
            verdict_source = verdict_payload.get('verdict_source', 'explicit_payload')
            is_source_hit = verdict_payload.get('is_source_hit')
            has_confirmed_non_source = verdict_payload.get('has_confirmed_non_source')
        elif allow_label_fallback and fused_source_label is not None:
            selected_bool = selection_mask.view(-1) > 0.5
            source_bool = fused_source_label.view(-1) > 0.5

            selected_source_mask.view(-1)[selected_bool & source_bool] = 1.0
            selected_non_source_mask.view(-1)[selected_bool & (~source_bool)] = 1.0
            verdict_available = True
            verdict_source = 'supervised_label_fallback'
            fallback_used = True

        if verdict_available:
            confirmed_source_mask = torch.max(confirmed_source_mask, selected_source_mask)
            confirmed_non_source_mask = torch.max(confirmed_non_source_mask, selected_non_source_mask)

        return ConstraintState(
            confirmed_non_source_mask=confirmed_non_source_mask,
            confirmed_source_mask=confirmed_source_mask,
            sampled_mask=sampled_mask,
            no_resample_mask=no_resample_mask,
        ), {
            'verdict_available': verdict_available,
            'verdict_source': verdict_source,
            'fallback_used': fallback_used,
            'allow_label_fallback': bool(allow_label_fallback),
            'selected_source_mask': selected_source_mask,
            'selected_non_source_mask': selected_non_source_mask,
            'is_source_hit': is_source_hit,
            'has_confirmed_non_source': has_confirmed_non_source,
        }

    def apply_constraint_masks(self, logits, constraint_state, batch=None):
        """
        Apply hard runtime exclusions to live logits.
        Priority:
        1. confirmed_source_mask keeps only confirmed sources per graph.
        2. confirmed_non_source_mask and no_resample_mask are hard exclusions.
        Safety:
        - If hard exclusions empty an entire graph without any confirmed source,
          restore that graph's raw logits to avoid all-`-inf` distributions.
        """
        if logits is None or constraint_state is None:
            return logits

        raw_flat_logits = logits.view(-1)
        masked_flat_logits = raw_flat_logits.clone()

        no_resample_mask = constraint_state.no_resample_mask.view(-1) > 0.5
        confirmed_non_source_mask = constraint_state.confirmed_non_source_mask.view(-1) > 0.5
        confirmed_source_mask = constraint_state.confirmed_source_mask.view(-1) > 0.5

        hard_exclusion = no_resample_mask | confirmed_non_source_mask
        if hard_exclusion.any():
            masked_flat_logits[hard_exclusion] = -float('inf')

        if confirmed_source_mask.any():
            if batch is None:
                masked_flat_logits.fill_(-float('inf'))
                masked_flat_logits[confirmed_source_mask] = raw_flat_logits[confirmed_source_mask]
            else:
                batch_flat = batch.view(-1)
                graph_has_confirmed_source = scatter_max(
                    confirmed_source_mask.float(), batch_flat, dim=0
                )[0] > 0.5
                masked_flat_logits[graph_has_confirmed_source[batch_flat]] = -float('inf')
                masked_flat_logits[confirmed_source_mask] = raw_flat_logits[confirmed_source_mask]

        if batch is None:
            if not torch.isfinite(masked_flat_logits).any() and not confirmed_source_mask.any():
                masked_flat_logits = raw_flat_logits.clone()
            return masked_flat_logits.view_as(logits)

        batch_flat = batch.view(-1)
        num_graphs = int(batch_flat.max().item()) + 1 if batch_flat.numel() > 0 else 0
        if num_graphs == 0:
            return masked_flat_logits.view_as(logits)

        graph_has_confirmed_source = scatter_max(
            confirmed_source_mask.float(), batch_flat, dim=0, dim_size=num_graphs
        )[0] > 0.5
        graph_has_any_finite = scatter_max(
            torch.isfinite(masked_flat_logits).float(), batch_flat, dim=0, dim_size=num_graphs
        )[0] > 0.5
        fully_masked_graph = (~graph_has_any_finite) & (~graph_has_confirmed_source)
        if fully_masked_graph.any():
            masked_flat_logits[fully_masked_graph[batch_flat]] = raw_flat_logits[fully_masked_graph[batch_flat]]

        return masked_flat_logits.view_as(logits)

    def update_observation_state(self, x_nav, acc_mask_local, inverse_indices, x_raw, static_ctx, t_sim, A_total_mask, causal_anchors, anchor_times, fused_batch):
        """
        Update x_nav (Observation State) based on mask, time, and ground truth.
        """
        x_nav_next = x_nav.clone()
        h_fused_device = x_nav.device
        curr_batch_size = static_ctx['curr_batch_size']
        
        # 1. Update Ch3 (Revealed Mask)
        accumulated_mask_view = acc_mask_local[inverse_indices]
        # [Fix] Out-of-place update
        x_nav_next_ch3 = torch.max(x_nav_next[:, 3], accumulated_mask_view.view(-1))
        
        # 2. Update Ch0 (Signal) and Ch1 (Poison) based on Ch3 and t_sim
        poison_at_t_current = None 
        x_nav_next_ch0 = x_nav_next[:, 0]
        x_nav_next_ch1 = x_nav_next[:, 1]
        
        if x_raw is not None:
            view_batch = static_ctx['view_batch']
            
            # Load params
            delta_t = 45.0 # Default
            if hasattr(self.cfg.data.online, 'episode_duration_seconds'):
                 delta_t = float(self.cfg.data.online.episode_duration_seconds) / 60.0
            
            data_resolution_seconds = 900.0
            if hasattr(self.cfg.data.online, 'data_resolution_seconds'):
                 data_resolution_seconds = float(self.cfg.data.online.data_resolution_seconds)
            elif 'step_seconds' in static_ctx and static_ctx['step_seconds'] is not None:
                 s_sec = static_ctx['step_seconds']
                 if isinstance(s_sec, torch.Tensor):
                     data_resolution_seconds = float(s_sec[0].item())
                 else:
                     data_resolution_seconds = float(s_sec)

            trigger_time_step_batch = static_ctx.get('trigger_time_step')
            if trigger_time_step_batch is None:
                 device = x_raw.device
                 trigger_time_step_batch = torch.zeros(curr_batch_size, device=device, dtype=torch.long)
            
            trigger_time_nodes = trigger_time_step_batch[view_batch]
            
            t_sim_seconds = t_sim * 60.0
            t_sim_steps = (t_sim_seconds / data_resolution_seconds).round().long()
            t_sim_steps_nodes = t_sim_steps[view_batch]
            
            t_idx = (trigger_time_nodes + t_sim_steps_nodes).clamp(0, x_raw.size(1) - 1)
            
            indices = torch.arange(x_raw.size(0), device=x_raw.device)
            signal_at_t = x_raw[indices, t_idx, 0]
            poison_threshold = getattr(self.cfg.physics.env, 'sensor_reading_threshold', 0.1)
            poison_at_t = (x_raw[indices, t_idx, 1] > poison_threshold).float()
            
            poison_at_t_current = poison_at_t 

            current_revealed = (x_nav_next_ch3 > 0.5)
            # [Fix] Out-of-place
            x_nav_next_ch0 = torch.where(current_revealed, signal_at_t, x_nav_next[:, 0])
            x_nav_next_ch1 = torch.where(current_revealed, poison_at_t, x_nav_next[:, 1])
        
        # 3. Update Freshness (Ch2)
        decay_rate = getattr(self.cfg.physics.env, 'freshness_decay', 0.8)
        A_total_mask_view = A_total_mask[inverse_indices].view(-1).float()
        # [Fix] Out-of-place
        x_nav_next_ch2 = x_nav_next[:, 2] * decay_rate
        x_nav_next_ch2 = torch.max(x_nav_next_ch2, A_total_mask_view)
        
        # 4. Update Ch4 (Anchors)
        newly_visited = A_total_mask.view(-1).bool()
        
        if newly_visited.any():
            # [Fix] Causal Anchors Contract: Always Fused Space
            # x_nav_next is Raw Space (Visible). We must aggregate to Fused Space.
            physical_poison_raw = x_nav_next_ch1
            
            # Aggregate Raw Poison -> Fused Poison
            # Use scatter_max to preserve positive poison signal (1.0)
            # inverse_indices maps Raw -> Fused
            # Note: We need dim_size to match causal_anchors size (Fused)
            num_fused = causal_anchors.size(0)
            physical_poison_fused, _ = scatter_max(
                physical_poison_raw, 
                inverse_indices, 
                dim=0, 
                dim_size=num_fused
            )
            
            physical_threshold = getattr(self.cfg.physics.env, 'confirmation_threshold', 0.5)
            is_poison = (physical_poison_fused > physical_threshold)
            
            new_anchors_val = torch.where(
                is_poison, 
                torch.tensor(1.0, device=h_fused_device), 
                torch.tensor(-1.0, device=h_fused_device)
            )
            
            # Update Causal Anchors (Fused)
            # [Fix] Out-of-place update for causal_anchors
            # Use mask + add
            causal_anchors = causal_anchors.clone()
            
            # Ensure dimensions match
            if causal_anchors.dim() == 1:
                causal_anchors = causal_anchors.unsqueeze(-1)
                
            # Direct update in Fused Space
            # newly_visited is Fused (from A_total_mask)
            # new_anchors_val is Fused (aggregated)
            
            # Create update mask/tensor
            anchor_update = torch.zeros_like(causal_anchors)
            anchor_update[newly_visited] = new_anchors_val[newly_visited].unsqueeze(-1)
            
            # Mask out old values where newly_visited is true
            # (Though technically we just overwrite, so add works if we zero out first)
            # Or use torch.where
            update_mask_fused = torch.zeros_like(causal_anchors, dtype=torch.bool)
            update_mask_fused[newly_visited] = True
            
            causal_anchors = torch.where(update_mask_fused, anchor_update, causal_anchors)
            
            # Update Anchor Times (Fused)
            anchor_times = anchor_times.clone()
            if anchor_times.dim() == 1:
                anchor_times = anchor_times.unsqueeze(-1)
                
            # t_sim is [B]. fused_batch maps Fused -> B.
            time_update = torch.zeros_like(anchor_times)
            time_update[newly_visited] = t_sim[fused_batch[newly_visited]].unsqueeze(-1)
            
            anchor_times = torch.where(update_mask_fused, time_update, anchor_times)

        # causal_anchors_view is already Fused
        causal_anchors_view = causal_anchors
             
        # Write back to x_nav_next (Raw)
        # We need to broadcast Fused Anchors -> Raw x_nav
        # inverse_indices maps Raw -> Fused.
        # So x_nav[i] gets causal_anchors[inverse_indices[i]]
        
        # Handle -1 in inverse_indices
        valid_raw = (inverse_indices >= 0)
        safe_indices = inverse_indices.clamp(min=0)
        
        # We only write to valid raw nodes
        # Use a temporary tensor for broadcasting
        # [Fix] Out-of-place
        anchors_broadcast = torch.zeros_like(x_nav_next[:, 4])
        anchors_broadcast[valid_raw] = causal_anchors_view[safe_indices[valid_raw]].view(-1)
        
        x_nav_next_ch4 = anchors_broadcast
        
        # [Final Assembly] Out-of-place
        # Reconstruct x_nav_next from columns
        # Assuming 5 channels or more. Check size.
        cols = [x_nav_next_ch0, x_nav_next_ch1, x_nav_next_ch2, x_nav_next_ch3, x_nav_next_ch4]
        if x_nav_next.size(1) > 5:
             cols.append(x_nav_next[:, 5:])
             x_nav_next = torch.cat([c.unsqueeze(1) if c.dim()==1 else c for c in cols], dim=1)
        else:
             x_nav_next = torch.stack(cols, dim=1)
        
        return x_nav_next, causal_anchors, anchor_times, poison_at_t_current
