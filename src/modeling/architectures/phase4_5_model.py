import torch
import torch.nn as nn
import torch.nn.functional as F
import inspect
from torch_geometric.nn import GATv2Conv, SAGEConv
from torch_scatter import scatter_mean, scatter_max, scatter_min
from src.modeling.builders.model_builder import ModelBuilder
from src.modeling.modules.dynamic_gate import DynamicFeatureGate
from src.modeling.evidence.refiner import SharedEvidenceResidualRefiner
from src.modeling.state.schema import ConstraintState
from src.modeling.loop.orchestration.state_updates import build_runtime_verdict_payload

from src.modeling.loop.episode_stepper import EpisodeStepper

class Phase45Model(nn.Module):
    """
    Phase 4.5 Orchestrator: Neural-Physical Hybrid Framework.
    Now fully platformized and configuration-driven.
    """
    # [Debug] Sentinel Version
    SENTINEL_VERSION = "v2_sentinel_debug_fused_fix"

    def _apply_training_mode_freeze(self):
        training_mode = getattr(self.cfg.model, 'training_mode', 'joint')
        if training_mode == 'joint':
            return

        trainable_modules = []
        if training_mode == 'frozen_nav':
            trainable_modules = ['reasoner_module', 'evidence_refiner']
        elif training_mode == 'frozen_reasoner':
            trainable_modules = ['navigator_module']
        else:
            return

        for param in self.parameters():
            param.requires_grad = False

        for module_name in trainable_modules:
            module = getattr(self, module_name, None)
            if module is None:
                continue
            for param in module.parameters():
                param.requires_grad = True

    def __init__(self, cfg, navigator, reasoner, physics, fov_controller=None, topology_engine=None):
        super().__init__()
        self.cfg = cfg
        self.topology_engine = topology_engine
        
        # 1. Modular Components
        self.navigator_module = navigator
        self.reasoner_module = reasoner
        self.physics_module = physics
        
        # [Debug] Check FoV Config
        print(f"[Phase45Model] FoV Config Type: {cfg.fov_controller.get('type', 'MISSING')}")
        
        if cfg.fov_controller['type'] != 'none':
             from src.modeling.controllers.fov_controller import FoVController
             self.fov_controller = FoVController(cfg.fov_controller['type'], cfg.fov_controller['params'])
        else:
             self.fov_controller = None
        
        # 1.5 Dynamic Gating (Phase 4.5.22 Repair)
        self.dynamic_gate = DynamicFeatureGate(cfg)

        evidence_refiner_cfg = getattr(cfg.model, "evidence_refiner", {})
        if isinstance(evidence_refiner_cfg, dict):
            evidence_refiner_enabled = evidence_refiner_cfg.get("enabled", False)
        else:
            evidence_refiner_enabled = getattr(evidence_refiner_cfg, "enabled", False)
        self.evidence_refiner = (
            SharedEvidenceResidualRefiner(cfg) if bool(evidence_refiner_enabled) else None
        )
        
        # 2. Shared Utilities & State
        from src.modeling.architectures.shared import TriggerInvariantFusion
        self.fusion = TriggerInvariantFusion(hidden_dim=getattr(cfg.model, 'hidden_dim', 64))
        
        # 3. Flags & Hyperparams from SSOT
        self.disable_firewall = getattr(cfg.model, 'disable_firewall', False)
        self.disable_rewiring = getattr(cfg.model, 'disable_rewiring', False)
        self.use_physics_bias = getattr(cfg.model, 'use_physics_bias', False)
        self.lambda_physics_bias = getattr(cfg.model, 'lambda_physics_bias', 1.0)
        self.allowed_channels = getattr(cfg.model, 'allowed_channels', [0, 1, 2, 3, 4, 5, 6])
        self.edge_dim = getattr(cfg.model, 'edge_dim', 8) # [SSOT] Unified Edge Dim
        
        # 4. Episode Protocol (Slot 10)
        self.stepper = EpisodeStepper(self)
        
        # 5. Heuristics Engine (Slot X)
        from src.modeling.heuristics_engine import HeuristicsEngine
        self.heuristics_engine = HeuristicsEngine(cfg.heuristics)
        
        # 6. Training Mode Management
        self._apply_training_mode_freeze()

        self.navigator_backbone_accepts_memory_state = False
        self.navigator_backbone_accepts_batch = False
        try:
            sig = inspect.signature(self.navigator_module.backbone.forward)
            self.navigator_backbone_accepts_memory_state = 'memory_state' in sig.parameters
            self.navigator_backbone_accepts_batch = 'batch' in sig.parameters
        except (TypeError, ValueError):
            pass

    def _expand_graph_meta_tensor(self, value, curr_batch_size, device):
        if value is None:
            return None
        try:
            tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value, device=device)
        except Exception:
            return None
        tensor = tensor.to(device).view(-1)
        if tensor.numel() == curr_batch_size:
            return tensor
        if tensor.numel() == 1:
            return tensor.repeat(curr_batch_size)
        return None

    def _resolve_graph_time_indices(self, t_sim, static_ctx, curr_batch_size, device):
        if t_sim is None or static_ctx is None:
            return {}

        t_sim_tensor = self._expand_graph_meta_tensor(t_sim, curr_batch_size, device)
        g_start = self._expand_graph_meta_tensor(static_ctx.get('global_start_step'), curr_batch_size, device)
        t_trig = self._expand_graph_meta_tensor(static_ctx.get('trigger_time_step'), curr_batch_size, device)
        if t_sim_tensor is None or g_start is None or t_trig is None:
            return {}

        t_sim_steps = torch.div(t_sim_tensor.float(), 15.0, rounding_mode='floor').to(torch.long)
        time_idx = (g_start.to(torch.long) + t_trig.to(torch.long) + t_sim_steps).clamp_(min=0, max=287)
        return {graph_idx: int(time_idx[graph_idx].item()) for graph_idx in range(curr_batch_size)}


    def _update_dynamic_topology(self, causal_anchors, anchor_times, fused_global_ids, fused_batch, base_edge_index, base_edge_attr, t_sim=None, static_ctx=None):
        if self.topology_engine is None or self.disable_rewiring:
            return base_edge_index, base_edge_attr
            
        device = causal_anchors.device
        curr_batch_size = int(t_sim.numel()) if isinstance(t_sim, torch.Tensor) else int(fused_batch.max().item()) + 1
        graph_time_idx = self._resolve_graph_time_indices(t_sim, static_ctx, curr_batch_size, device)

        # [CRITICAL FIX] Generate virtual edges for BOTH Positive (1.0) and Negative (-1.0) anchors.
        # Negative anchors are essential for "Clean Path Suppression" (Heuristic Scheme 1).
        anchor_indices = (causal_anchors.abs() > 0.1).nonzero(as_tuple=True)[0]
        
        # [TRACER] Dynamic Topology Check
        if anchor_indices.numel() > 0:
             # Count types
             vals = causal_anchors[anchor_indices].view(-1)
             n_pos = (vals > 0.5).sum().item()
             n_neg = (vals < -0.5).sum().item()
             # print(f"[Model] _update_dynamic_topology: Found {anchor_indices.numel()} anchors. Pos={n_pos}, Neg={n_neg}")
             
        if anchor_indices.numel() == 0:
            # print(f"[Model] _update_dynamic_topology: NO ANCHORS FOUND! Causal Anchors Abs Sum={causal_anchors.abs().sum().item()}")
            return base_edge_index, base_edge_attr
            
        virt_indices = []
        virt_attrs = []
        
        for idx in anchor_indices:
            global_id = fused_global_ids[idx].item()
            anchor_val = causal_anchors[idx].item()
            anchor_time = anchor_times[idx].item()
            b_id = int(fused_batch[idx].item())
            time_idx = graph_time_idx.get(b_id, -1)
            subgraph_mask = (fused_batch == b_id)
            subgraph_global_ids = fused_global_ids[subgraph_mask]
            
            v_idx, v_attr = self.topology_engine.get_virtual_edges_for_subgraph(
                global_id, subgraph_global_ids, anchor_value=anchor_val, anchor_time=anchor_time, time_idx=time_idx
            )
            
            # [Debug] Trace negative anchors if they yield no edges
            if anchor_val < -0.5 and v_idx.numel() == 0:
               print(f"[Model] WARN: Negative Anchor {global_id} (Time {time_idx}) yielded 0 virtual edges!")
            
            if v_idx.numel() > 0:
                subgraph_indices = subgraph_mask.nonzero(as_tuple=True)[0]
                v_idx_fused = torch.stack([subgraph_indices[v_idx[0]], subgraph_indices[v_idx[1]]], dim=0).to(device)
                
                # [SSOT V6.3] Unified Spatiotemporal 8-Channel Blueprint
                # Align with Physical: [LogMed, LogP90, LogMin, Flip, IsPhys, IsVirt, Type, Rsv]
                
                stt_raw = v_attr[:, 0].to(device)
                stt_log = torch.log1p(stt_raw)
                
                v_attr_8d = torch.zeros((v_attr.size(0), 8), device=device)
                v_attr_8d[:, 0] = stt_log # Ch0: LogMed
                v_attr_8d[:, 1] = stt_log # Ch1: LogP90 (Copy)
                v_attr_8d[:, 2] = stt_log # Ch2: LogMin (Copy)
                v_attr_8d[:, 3] = 0.0     # Ch3: Flip (Stable)
                v_attr_8d[:, 4] = 0.0     # Ch4: IsPhys
                v_attr_8d[:, 5] = 1.0     # Ch5: IsVirt
                v_attr_8d[:, 6] = v_attr[:, 2].to(device) # Ch6: Anchor Type
                v_attr_8d[:, 7] = 0.0     # Ch7: Reserved
                
                virt_indices.append(v_idx_fused)
                virt_attrs.append(v_attr_8d)
                
        if virt_indices:
            virt_edge_index = torch.cat(virt_indices, dim=1)
            virt_edge_attr = torch.cat(virt_attrs, dim=0)
            
            # [TRACER] Dynamic Edge Generation
            # print(f"[Model] Generated {virt_edge_index.size(1)} dynamic edges. Type Unique: {torch.unique(virt_edge_attr[:, 6]).tolist()}")
            
            return torch.cat([base_edge_index, virt_edge_index], dim=1), torch.cat([base_edge_attr, virt_edge_attr], dim=0)
            
        return base_edge_index, base_edge_attr

    def forward(self, batch, max_episodes=10, inference_mode=False, return_trajectory=False, **kwargs):
        # 0. Prep
        batch_x = torch.nan_to_num(batch.x.float(), nan=0.0)
        curr_batch_size = batch.num_graphs
        
        # Parse kwargs for Stepper
        # [SSOT] Use sample_budget from config
        sample_budget = kwargs.get('sample_budget', getattr(self.cfg.model, 'sample_budget', 1)) 
        action_policy = kwargs.get('action_policy', self.cfg.model.sampling_policy) # Use Config as Default
        tau = kwargs.get('tau', 1.0) # [Refactor] Gumbel Temperature
        
        # 1. Feature Firewall
        # MUST exclude Ch1 (PoisonLabel) during firewall, but it's passed for dynamic updates.
        # Channels: Signal(0), Poison(1), Freshness(2), Mask(3), Anchor(4), Sensor(5), LogDegree(6)
        # self.allowed_channels is set in __init__
        
        # User Correction: We need to pass the FULL x_nav to the Stepper so it can reveal Ch0/Ch1.
        # V4.5.22 Fix: We will pass the full x_raw time series to the stepper.
        
        # Prepare x_nav for the loop (Mutable State)
        # We start with the masked version from dataset
        x_nav_full = batch_x.clone()
        
        # [Step 1: Light up the flashlight]
        # Force reveal Triggers (Anchor == 1.0) in the initial state.
        # This ensures the Feature Firewall allows their signals to pass.
        # Anchor is Ch4. Mask is Ch3.
        anchor_val = x_nav_full[:, 4]
        is_trigger = (anchor_val > 0.5)
        # Force Mask = 1.0 for Triggers
        x_nav_full[is_trigger, 3] = 1.0
        
        # [Debug] Anchor/Mask Raw Stats (First Batch Only)
        if batch.batch[0] == 0: # Simple check to limit logs
            anchor_sum_raw = is_trigger.float().sum().item()
            mask_init_sum_raw = x_nav_full[:, 3].sum().item()
            # print(f"[Phase45Model] Raw Stats: AnchorSum={anchor_sum_raw}, MaskSum={mask_init_sum_raw}")


        # [Security -> Sentinel]
        # Detect leaks before masking (Audit)
        mask_init_val = x_nav_full[:, 3].view(-1) # [N]
        is_unrevealed = (mask_init_val < 0.5)
        
        leak_signal = x_nav_full[:, 0].abs() * is_unrevealed.float()
        leak_poison = x_nav_full[:, 1].abs() * is_unrevealed.float()
        
        leak_cnt = (leak_signal > 1e-6).sum() + (leak_poison > 1e-6).sum()
        leak_mag = leak_signal.sum() + leak_poison.sum()
        
        firewall_metrics = {
            'firewall/leak_count': leak_cnt.item(),
            'firewall/leak_magnitude': leak_mag.item()
        }
        
        # [Task 2: Refactor Feature Firewall]
        # REFACTORED: Firewall REMOVED. 
        # The model is now a "Blind End" that trusts the physical engine.
        # x_nav_full contains 0.0 for unrevealed nodes (enforced by dataset/stepper).
        # We do NOT apply any extra masking here.
        
        # 1.5 Dynamic Feature Gating (Applied to visible channels)
        x_nav_visible = x_nav_full[..., self.allowed_channels]
        x_nav_visible, gate_info = self.dynamic_gate(x_nav_visible, batch.batch)
        
        # 2. Embedding & Fusion
        # Navigator uses visible channels
        
        # Check if backbone needs special args
        call_kwargs = {}
        if self.navigator_backbone_accepts_memory_state:
            call_kwargs['memory_state'] = None
        if self.navigator_backbone_accepts_batch:
            call_kwargs['batch'] = batch.batch
        
        h_nav_init = self.navigator_module.backbone(x_nav_visible, batch.edge_index, **call_kwargs)
        if isinstance(h_nav_init, tuple): # GRU returns (emb, mem)
            h_nav_init = h_nav_init[0]
            
        h_fused, inverse_indices, _, fused_global_ids, fused_batch = self.fusion(
            h_nav_init, batch.batch, batch.n_id, batch.scenario_id
        )
        
        # 3. Environment Context
        src, dst = batch.edge_index
        
        # [CRITICAL RECONSTRUCTION] Physical-Virtual Split Fusion
        # We must preserve Dataset-level virtual edges while fusing the graph.
        if hasattr(batch, 'edge_attr') and batch.edge_attr is not None:
            # [SSOT Fix] Pad to target edge_dim if needed
            target_dim = self.edge_dim
            if batch.edge_attr.size(1) < target_dim:
                padding = torch.zeros((batch.edge_attr.size(0), target_dim - batch.edge_attr.size(1)), device=batch.edge_attr.device)
                full_edge_attr = torch.cat([batch.edge_attr, padding], dim=1)
            else:
                full_edge_attr = batch.edge_attr
            
            # Identify Physical vs Virtual (Dataset-level)
            # Threshold 0.1 is robust for -1.0/1.0 signals
            # Use Ch5 (IsVirt) if available (index 5)
            if full_edge_attr.size(1) > 5:
                is_dataset_virt = (full_edge_attr[:, 5] > 0.5)
            else:
                # Fallback to Ch6 if available, or assume all physical
                # If target_dim=8, we padded, so Ch5=0. So IsVirt=False. Correct for Physical.
                is_dataset_virt = torch.zeros(full_edge_attr.size(0), dtype=torch.bool, device=full_edge_attr.device)
            
            # Fuse Physical Edges
            phys_mask = ~is_dataset_virt
            p_src, p_dst = src[phys_mask], dst[phys_mask]
            fused_phys_edge_index = torch.stack([inverse_indices[p_src], inverse_indices[p_dst]], dim=0)
            fused_phys_edge_attr = full_edge_attr[phys_mask]
            
            # Fuse Dataset Virtual Edges (Initial Triggers)
            v_src, v_dst = src[is_dataset_virt], dst[is_dataset_virt]
            fused_virt_edge_index = torch.stack([inverse_indices[v_src], inverse_indices[v_dst]], dim=0)
            fused_virt_edge_attr = full_edge_attr[is_dataset_virt]
            
            # Combine for Base Context
            base_edge_index = torch.cat([fused_phys_edge_index, fused_virt_edge_index], dim=1)
            base_edge_attr = torch.cat([fused_phys_edge_attr, fused_virt_edge_attr], dim=0)
        else:
            # Fallback
            fused_phys_edge_index = torch.stack([inverse_indices[src], inverse_indices[dst]], dim=0)
            base_edge_index = fused_phys_edge_index
            base_edge_attr = torch.zeros((base_edge_index.size(1), self.edge_dim), device=h_fused.device)
        
        fused_source_label = scatter_max(batch.y.view(-1), inverse_indices, dim=0, dim_size=h_fused.size(0))[0].unsqueeze(-1) if hasattr(batch, 'y') else None
        has_y = fused_source_label is not None
        
        t_sim = batch.t0.float() if hasattr(batch, 't0') else torch.zeros(curr_batch_size, device=h_fused.device)

        def get_meta_tensor(key, default=None):
            if hasattr(batch, key):
                val = getattr(batch, key)
                if isinstance(val, torch.Tensor):
                    return val.to(h_fused.device)
                if isinstance(val, list):
                    return torch.tensor(val, device=h_fused.device)
                try:
                    return torch.tensor(val, device=h_fused.device)
                except Exception:
                    return None
            return default

        step_seconds = get_meta_tensor('step_seconds')
        global_start_step = get_meta_tensor('global_start_step')
        trigger_time_step = get_meta_tensor('trigger_time_step')
        graph_source_global_ids = get_meta_tensor('global_injection_node')
        allow_constraint_label_fallback = bool(
            getattr(self.cfg.life_support, 'allow_constraint_label_fallback', False)
        )
        
        # 4. Initialize Dynamic State
        # Ch3 (Mask) is obs_valid_mask (static property of sensors), NOT visited_mask.
        # Initialize accumulated_mask from Ch3 (Trigger is already visited/revealed)
        # [Fix] Fuse to Physical Level (Unique Nodes)
        accumulated_mask_view = x_nav_full[:, 3:4].clone()
        accumulated_mask = scatter_max(accumulated_mask_view, inverse_indices, dim=0, dim_size=h_fused.size(0))[0]
        
        # Fuse Ch4 (Anchor) -> causal_anchors
        # [CRITICAL FIX] Use max/min to preserve strong signals (1.0 or -1.0)
        # scatter_mean would dilute the signal to 0.33, causing heuristics to fail thresholds.
        causal_anchors_view = x_nav_full[:, 4:5].clone()
        c_pos = scatter_max(F.relu(causal_anchors_view), inverse_indices, dim=0, dim_size=h_fused.size(0))[0]
        c_neg = scatter_min(-F.relu(-causal_anchors_view), inverse_indices, dim=0, dim_size=h_fused.size(0))[0]
        causal_anchors = c_pos + c_neg # Combine back to [-1, 0, 1] space
        
        # [Debug] Fused Stats & Force Update
        if batch.batch[0] == 0:
            anchor_fused_sum = (causal_anchors > 0.5).float().sum().item()
            mask_fused_sum = accumulated_mask.sum().item()
            # print(f"[Phase45Model] Fused Stats: AnchorSum={anchor_fused_sum}, MaskSum={mask_fused_sum}")
            
            # [Fail-Safe] If Raw Anchor > 0 but Fused Anchor == 0, we have a mapping issue.
            # But here we just enforce the logic requested:
            # "If anchor_sum_raw > 0 but anchor_fused_sum == 0..."
            # We proactively fix it by using causal_anchors to update accumulated_mask
        
        # [CRITICAL FIX 2] Force accumulated_mask to include Triggers in Fused Space
        # This handles cases where scatter_max for mask might have failed or been overwritten
        trigger_fused = (causal_anchors > 0.5).float()
        accumulated_mask = torch.max(accumulated_mask, trigger_fused)

        initial_verdict_payload = build_runtime_verdict_payload(
            accumulated_mask,
            fused_batch=fused_batch,
            fused_global_ids=fused_global_ids,
            graph_source_global_ids=graph_source_global_ids,
        )

        confirmed_source_mask = torch.zeros_like(accumulated_mask)
        confirmed_non_source_mask = torch.zeros_like(accumulated_mask)
        if initial_verdict_payload.get('verdict_available', False):
            confirmed_source_mask = torch.max(
                confirmed_source_mask,
                initial_verdict_payload['selected_source_mask'],
            )
            confirmed_non_source_mask = torch.max(
                confirmed_non_source_mask,
                initial_verdict_payload['selected_non_source_mask'],
            )
        elif allow_constraint_label_fallback and has_y and fused_source_label is not None:
            sampled_bool = accumulated_mask.view(-1) > 0.5
            source_bool = fused_source_label.view(-1) > 0.5
            confirmed_source_mask.view(-1)[sampled_bool & source_bool] = 1.0
            confirmed_non_source_mask.view(-1)[sampled_bool & (~source_bool)] = 1.0

        constraint_state = ConstraintState(
            confirmed_non_source_mask=confirmed_non_source_mask,
            confirmed_source_mask=confirmed_source_mask,
            sampled_mask=accumulated_mask.clone(),
            no_resample_mask=accumulated_mask.clone(),
        )
        
        # [Debug] Post-Fix Mask Sum
        # if batch.batch[0] == 0:
        #      print(f"[Phase45Model] Post-Fix MaskSum={accumulated_mask.sum().item()}")
        
        # [Probe A] Phase45Model.forward (Generation Point)
        if not inference_mode and batch.batch[0] == 0:
             import wandb
             if wandb.run is not None:
                 mask_a = accumulated_mask.view(-1)
                 mask_sum = mask_a.sum().item()
                 mask_first5 = torch.nonzero(mask_a > 0.5)[:5].view(-1).tolist()
                 
                 log_dict = {
                     "probeA/mask_sum": mask_sum,
                     "probeA/mask_first5": str(mask_first5),
                     "probeA/fused_anchor_sum": float((causal_anchors > 0.5).float().sum().item()),
                     # Raw stats might be out of scope if not calculated every batch, but we can recompute or ignore for now.
                 }
                 wandb.log(log_dict, commit=False)

        # [TRACER] Check Anchor Signal

        # print(f"[Model] Causal Anchors: NonZero={causal_anchors.nonzero().numel()}, Max={causal_anchors.max().item()}, Min={causal_anchors.min().item()}")
        
        anchor_times = torch.zeros_like(accumulated_mask)
        
        hit_before_t = torch.zeros(curr_batch_size, dtype=torch.bool, device=h_fused.device)
        graph_success = torch.zeros(curr_batch_size, dtype=torch.bool, device=h_fused.device)
        
        # 5. Pack Context for Stepper
        # [SSOT V6] Pass Time Resolution and Metadata
        # [Critical Fix] Ensure Metadata is Tensor [B] and not None
        # PyG Collation of attributes:
        # If attribute is a list of scalars (one per graph), PyG collates into a list or tensor.
        # We need to robustly extract it.
        
        # [Debug] Verify Metadata
        if not inference_mode and batch.batch[0] == 0:
             # print(f"[Phase45Model] Metadata Check: StepSec={step_seconds}, GlobalStart={global_start_step}, TriggerTime={trigger_time_step}")
             if trigger_time_step is None:
                 print("🚨 CRITICAL ALARM: 'trigger_time_step' is MISSING in batch! Time-Axis will be misaligned (t=0)!")
                 # Attempt fallback from first graph?
                 # No, we must fail or warn loud.
        
        static_ctx = {
            'h_fused': h_fused,
            'x_nav_full': x_nav_full, # Pass FULL tensor for updates
            'x_nav': x_nav_visible,   # Pass GATED/VISIBLE tensor for module inputs
            'fused_batch': fused_batch,
            'fused_global_ids': fused_global_ids,
            'fused_edge_index': base_edge_index,
            'base_edge_attr': base_edge_attr,
            'inverse_indices': inverse_indices,
            'curr_batch_size': curr_batch_size,
            'fused_source_label': fused_source_label,
            'has_y': has_y,
            'view_batch': batch.batch, # [Fix] Pass View Level batch indices for DynamicGate
            # Pass Ground Truth for Revealing (Full Time Series)
            'x_raw': batch.x_raw if hasattr(batch, 'x_raw') else None,
            'stt_dynamic': batch.stt_dynamic if hasattr(batch, 'stt_dynamic') else None,
            # Pass Edge Index for Re-Embedding
            'batch_edge_index': batch.edge_index,
            'batch_n_id': batch.n_id,
            'batch_scenario_id': batch.scenario_id,
            'graph_source_global_ids': graph_source_global_ids,
            'allow_constraint_label_fallback': allow_constraint_label_fallback,
            # [SSOT V6] Pass Time Resolution and Metadata (Robust)
            'step_seconds': step_seconds,
            'global_start_step': global_start_step,
            'trigger_time_step': trigger_time_step
        }
        
        dynamic_state = {
            't_sim': t_sim,
            'accumulated_mask': accumulated_mask,
            'constraint_state': constraint_state,
            'causal_anchors': causal_anchors,
            'anchor_times': anchor_times,
            'hit_before_t': hit_before_t,
            'graph_success': graph_success,
            'reasoner_memory_state': None,
            # [Exp E] Initialize Navigator Memory State
            # We initialize it as None, Backbone will create zeros on first call
            'nav_memory_state': None
        }
        
        # 6. Run Episode
        # [Bridge Probe] Pre-Stepper
        if not inference_mode and batch.batch[0] == 0:
             # print(f"[Bridge] Pre-Stepper MaskSum: {accumulated_mask.sum().item()}")
             # print(f"[Bridge] Pre-Stepper MaskID: {id(accumulated_mask)}")
             # Ensure dynamic_state has the correct reference
             dynamic_state['accumulated_mask'] = accumulated_mask


        result = self.stepper.run_scenario(
            max_episodes, 
            static_ctx, 
            dynamic_state, 
            inference_mode, 
            sample_budget=sample_budget,
            action_policy=action_policy, # [Refactor] Explicitly pass policy
            tau=tau, # [Refactor] Pass temperature
            enable_tracer=kwargs.get('enable_tracer', False),
            profile_rollout=bool(kwargs.get('profile_rollout', False)),
            skip_reasoner_forward=bool(kwargs.get('skip_reasoner_forward', False)),
            store_resume_state=bool(kwargs.get('store_resume_state', False)),
            forced_action_indices_by_step=kwargs.get('forced_action_indices_by_step'),
        )
        result['gate_info'] = gate_info
        if bool(kwargs.get('return_rollout_context', False)):
            result['rollout_context'] = {
                'static_ctx': static_ctx,
                'dynamic_state': dynamic_state,
            }
        
        # [Sentinel] Attach firewall alerts
        if 'step_metrics' not in result: result['step_metrics'] = {}
        result['step_metrics'].update(firewall_metrics)
        
        return result
