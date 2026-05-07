import torch
import torch.nn as nn
from torch_scatter import scatter_max, scatter_mean, scatter_sum
from src.modeling.loop.episode_runner import EpisodeRunner

class EpisodeStepper:
    """
    Slot 10: EpisodeStepper (Protocol Enforcer)
    
    Responsibilities:
    1. Enforce Physics Protocol (T_sim, Top1+K-1, Stop-on-Hit).
    2. Centralize the Episode Loop.
    3. Coordinate Module Calls (Physics -> FoV -> Nav -> Reasoner -> Action).
    
    This component is NOT pluggable. It defines the rules of the environment.
    """
    def __init__(self, model):
        self.model = model
        self.cfg = model.cfg
        self.runner = EpisodeRunner(model)
        
    def run_scenario(self, 
                    max_episodes, 
                    static_ctx, 
                    dynamic_state, 
                    inference_mode=False,
                    sample_budget=1,
                    action_policy='greedy',
                    tau=1.0, # Added tau
                    enable_tracer=False,
                    profile_rollout=False,
                    skip_reasoner_forward=False,
                    store_resume_state=False,
                    forced_action_indices_by_step=None):
        """
        Executes the scenario loop (formerly run_episode).
        Delegates to EpisodeRunner.
        """
        return self.runner.run(
            max_episodes, 
            static_ctx, 
            dynamic_state, 
            inference_mode, 
            sample_budget, 
            action_policy, 
            tau, 
            enable_tracer,
            profile_rollout,
            skip_reasoner_forward,
            store_resume_state,
            forced_action_indices_by_step,
        )
        probe_b_metrics = {}
        if not inference_mode:
             probe_b_metrics["meta/stepper_enter"] = 1
             probe_b_metrics["meta/stepper_id"] = id(self)
             probe_b_metrics["meta/episode_stepper_file"] = __file__
             
             # [Bridge Probe] Stepper Start
             if 'accumulated_mask' in dynamic_state:
                 mask_in = dynamic_state['accumulated_mask']
                 probe_b_metrics["probeB/bridge_mask_sum"] = mask_in.sum().item()
                 probe_b_metrics["probeB/bridge_mask_id"] = str(id(mask_in))
                 probe_b_metrics["probeB/bridge_mask_shape"] = str(list(mask_in.shape))
                 # print(f"[Bridge] Stepper Start MaskSum: {mask_in.sum().item()} ID: {id(mask_in)}")
        
        # Unpack Static Context
        h_fused = static_ctx['h_fused']
        x_raw = static_ctx.get('x_raw') # V4.5.22
        x_nav_full = static_ctx['x_nav_full']
        # x_nav (visible) needs to be initialized for first loop
        x_nav = static_ctx['x_nav'] # This is visible/gated x_nav passed from Model

        if not hasattr(self, 'verified'):
            self.verified = True
            feature_mode = getattr(self.cfg.data, 'feature_mode', 'baseline')
            print(f"=== Input Verification for Feature Mode: {feature_mode} ===")
            input_channels = x_nav.shape[1]
            print(f"Model actual input channel count: {input_channels}")
            if feature_mode == 'no_mask':
                ch3_sum = x_nav[:, 3].sum().item()
                is_zeroed = ch3_sum == 0
                print(f"Ch3 is zeroed: {is_zeroed} (sum: {ch3_sum})")
            if feature_mode == 'explicit_flag':
                if input_channels > 7:
                    ch7_sum = x_nav[:, 7].sum().item()
                    exists_and_written = ch7_sum > 0
                    print(f"Ch7 exists and written: {exists_and_written} (sum: {ch7_sum})")
                else:
                    print("Ch7 does not exist")
            print("Input tensor slice (first 5 nodes, all channels):")
            slice_tensor = x_nav[:5, :]
            print(slice_tensor)
            print("=== End Verification ===")

        
        fused_batch = static_ctx['fused_batch']
        fused_global_ids = static_ctx['fused_global_ids']
        fused_edge_index = static_ctx['fused_edge_index']
        base_edge_attr = static_ctx['base_edge_attr']
        inverse_indices = static_ctx['inverse_indices']
        curr_batch_size = static_ctx['curr_batch_size']
        fused_source_label = static_ctx.get('fused_source_label')
        has_y = static_ctx['has_y']
        
        # Unpack Dynamic State
        t_sim = dynamic_state['t_sim']
        
        # [Refactor] Semantic Split: acc_mask_fused (Input from Dynamic State)
        # SSOT: This is the mask passed from the Model/Dynamic State
        acc_mask_fused = dynamic_state['accumulated_mask'] 
        
        # [Refactor] Semantic Split: acc_mask_local (Working Copy)
        # We clone it to ensure we don't modify the input in place (though dynamic_state is mutable)
        # This variable tracks the mask EVOLUTION during the scenario
        acc_mask_local = acc_mask_fused.clone()
        
        # [Bridge Probe] Check mask after unpack
        if not inference_mode:
             # Use acc_mask_fused for probe
             probe_b_metrics["probeB/mask_sum_after_load"] = acc_mask_fused.sum().item()
             probe_b_metrics["probeB/acc_mask_fused_id"] = str(id(acc_mask_fused))
             probe_b_metrics["probeB/acc_mask_local_id"] = str(id(acc_mask_local))

        # [Align HSR] Critical: Force Trigger Observation Injection at Step 0
        # If the accumulated_mask already has the Trigger node, we MUST ensure x_nav reflects its Signal/Poison.
        # This aligns with verify_data_with_hsr.py: "Force HSR to process the Trigger as a Positive Observation!"
        if x_raw is not None and acc_mask_local.sum() > 0:
             # Identify initial revealed nodes (likely the Trigger)
             # acc_mask_local is [N_fused], so init_revealed is [N_fused]
             init_revealed = (acc_mask_local.view(-1) > 0.5)
             
             # [Fix] Coordinate System Alignment: Fused -> Raw
             # inverse_indices provided by Phase45Model.fusion() maps Raw Nodes to Fused Nodes.
             # Shape: [N_raw]. Value: fused_idx if mapped, -1 if filtered.
             # We want to find the Raw Indices that correspond to our Fused Indices (init_revealed).
             
             # 1. Get Fused Indices that are revealed
             fused_indices_revealed = init_revealed.nonzero().view(-1)
             
             if fused_indices_revealed.numel() > 0:
                 # 2. Find corresponding Raw Indices
                 # Since inverse_indices is [N_raw] -> fused_idx, we need to find raw_idx such that inverse_indices[raw_idx] in fused_indices_revealed.
                 # This is an inverse lookup.
                 # Optimization: Assuming fusion is a subgraph selection (one-to-one), 
                 # we can use the `subset` or `n_id` if available.
                 # But we only have `inverse_indices` in static_ctx.
                 
                 # Let's use boolean masking on inverse_indices.
                 # We need to construct a mask of size [N_fused] then map to [N_raw]? No.
                 # We check: inverse_indices \in fused_indices_revealed.
                 # Since inverse_indices contains values from 0 to N_fused-1 (and -1).
                 # We can use `init_revealed` as a lookup table!
                 # inverse_indices values are indices into `init_revealed`.
                 
                 # Valid Raw Indices are those where inverse_indices != -1 AND init_revealed[inverse_indices] is True.
                 
                 # Handle -1 sentinel in inverse_indices
                 valid_raw_mask = (inverse_indices >= 0)
                 # Safety: clamp -1 to 0 for lookup, then mask out valid_raw_mask
                 safe_lookup_indices = inverse_indices.clamp(min=0)
                 
                 # Check if mapped fused node is revealed
                 # init_revealed is [N_fused]
                 # safe_lookup_indices is [N_raw]
                 # mapped_revealed is [N_raw]
                 mapped_revealed = init_revealed[safe_lookup_indices]
                 
                 # Final Raw Mask: Is Valid Mapping AND Is Revealed in Fused
                 raw_revealed_mask = valid_raw_mask & mapped_revealed
                 init_indices = raw_revealed_mask.nonzero().view(-1)
                 
                 if init_indices.numel() > 0:
                     # Get Trigger Time Step for these nodes
                     view_batch = static_ctx['view_batch']
                     trigger_time_step_batch = static_ctx.get('trigger_time_step')
                     
                     if trigger_time_step_batch is None:
                          device = x_raw.device
                          trigger_time_step_batch = torch.zeros(curr_batch_size, device=device, dtype=torch.long)
                     
                     # Calculate t_idx for Step 0 (t_sim=0)
                     # t_idx = trigger_time + 0
                     # [Fix] Handle batch mismatch: view_batch is [N_raw], init_indices is subset of [N_raw]
                     # So view_batch[init_indices] gives the batch index for each revealed node.
                     # trigger_time_step_batch is [B]. So trigger_time_nodes is [N_revealed].
                     trigger_time_nodes = trigger_time_step_batch[view_batch[init_indices]]
                     t_idx_nodes = trigger_time_nodes.clamp(0, x_raw.size(1) - 1)
                     
                     # Read Sensor Data from x_raw
                     # Note: init_indices are raw indices
                     # x_raw is [N_raw, T, C]
                     # We need advanced indexing: x_raw[node_idx, t_idx, channel]
                     signal_init = x_raw[init_indices, t_idx_nodes, 0]
                     poison_init = x_raw[init_indices, t_idx_nodes, 1]
                     
                     # Inject into x_nav (fused space)
                     # x_nav is [N_fused, C]
                     # We need to map back to fused indices.
                     # We know `inverse_indices[init_indices]` gives the fused index for each raw node.
                     target_fused_indices = inverse_indices[init_indices]
                     
                     # Check if x_nav already has it (it should if Model init is correct, but let's enforce)
                     # This is the "Force" part.
                     # We use scatter_max or just direct assignment if one-to-one.
                     # Assuming one-to-one for subgraph.
                     x_nav[target_fused_indices, 0] = signal_init
                     x_nav[target_fused_indices, 1] = poison_init
                     # Ensure mask channel is set
                     if x_nav.size(1) > 3:
                         x_nav[target_fused_indices, 3] = 1.0
                     
                     # [Exp 3] Explicit Flag
                     feature_mode = getattr(self.cfg.data, 'feature_mode', 'baseline')
                     if feature_mode == 'explicit_flag' and x_nav.size(1) > 7:
                         x_nav[target_fused_indices, 7] = 1.0
                         
                     # if not inference_mode:
                     #    print(f"[Stepper] Step 0 Trigger Injection: {init_indices.numel()} nodes forced. Signal Mean: {signal_init.mean().item():.4f}")


        causal_anchors = dynamic_state['causal_anchors']
        anchor_times = dynamic_state['anchor_times']
        hit_before_t = dynamic_state['hit_before_t']
        graph_success = dynamic_state['graph_success']
        reasoner_memory_state = dynamic_state.get('reasoner_memory_state')
        # [Exp E] Unpack Navigator Memory State
        nav_memory_state = dynamic_state.get('nav_memory_state')
        
        # [Fix Task 1] Check for Pre-Existing Hits (Target Masking Check)
        if has_y and fused_source_label is not None:
             # Check if any source node is already in acc_mask_fused
             pre_hits = (acc_mask_fused.view(-1) > 0.5) & (fused_source_label.view(-1) > 0.5)
             pre_hits_graph = scatter_max(pre_hits.float(), fused_batch, dim=0, dim_size=curr_batch_size)[0]
             
             # Update initial state
             pre_success = (pre_hits_graph > 0.5)
             hit_before_t = hit_before_t | pre_success
             graph_success = graph_success | pre_success
             
             # [DEBUG] Print Pre-Hits
             # if not inference_mode and pre_success.any():
             #    print(f"[Stepper] Detected {pre_success.sum().item()} graphs with pre-revealed targets!")
        
        # [Metric Fix] Hit@1 should use Fused Source Label
        # It already does: fused_source_label is passed to predict_hit_at_1 logic.
        # But let's verify fused_source_label correctness.
        # In Phase45Model, it is created from batch.y via fusion.
        
        # [Refactor] Load delta_t from config or static_ctx, default to 45.0
        # SSOT: Try cfg.data.online.step_seconds first, then static_ctx, then default
        delta_t = 45.0 # Default fallback (minutes)
        data_resolution_seconds = 900.0 # Default data resolution (15 min)
        
        # 1. Try Config (Primary Source)
        if hasattr(self.cfg, 'data') and hasattr(self.cfg.data, 'online'):
            if hasattr(self.cfg.data.online, 'episode_duration_seconds'):
                 step_sec = float(self.cfg.data.online.episode_duration_seconds)
                 delta_t = step_sec / 60.0
            elif hasattr(self.cfg.data.online, 'step_seconds'):
                 step_sec = float(self.cfg.data.online.step_seconds)
                 delta_t = step_sec / 60.0
            
            # Load data resolution if available
            if hasattr(self.cfg.data.online, 'data_resolution_seconds'):
                 data_resolution_seconds = float(self.cfg.data.online.data_resolution_seconds)
                 
        # 2. Try Static Context (Runtime Override)
        elif 'step_seconds' in static_ctx and static_ctx['step_seconds'] is not None:
             # This 'step_seconds' from dataset usually means DATA RESOLUTION
             s_sec = static_ctx['step_seconds']
             if isinstance(s_sec, torch.Tensor):
                 data_resolution_seconds = float(s_sec[0].item())
             else:
                 data_resolution_seconds = float(s_sec)
             
             # If step_seconds is 900 (15 min), and we want 45 min step,
             # we keep delta_t = 45.0.
             # Unless we want to align delta_t to data_resolution?
             # User protocol: "Episode step is 45 minutes".
             delta_t = 45.0

        
        # Parse kwargs for Stepper
        action_k = sample_budget # Map to internal logic
        max_steps = max_episodes # Map to internal logic
        
        trajectory_data = []
        last_logits = None
        
        # Metrics Tracking
        steps_taken = torch.zeros(curr_batch_size, device=h_fused.device)
        budget_used = torch.zeros(curr_batch_size, device=h_fused.device)
        predict_hit_at_1 = torch.zeros(curr_batch_size, dtype=torch.bool, device=h_fused.device)
        # [Fix] Initialize predict_hit_at_5 to avoid UnboundLocalError
        predict_hit_at_5 = torch.zeros(curr_batch_size, dtype=torch.bool, device=h_fused.device)
        predict_hit_valid = torch.zeros(curr_batch_size, dtype=torch.bool, device=h_fused.device)
        
        max_hit_prob = torch.zeros(curr_batch_size, device=h_fused.device)
        
        # [Task 3] Success Tracking Probes
        first_hit_step = torch.full((curr_batch_size,), -1.0, device=h_fused.device)
        hit_in_selected_step = torch.zeros(curr_batch_size, device=h_fused.device)
        
        # [Metric] Global Event Accumulators
        total_poison_hits_per_graph = torch.zeros(curr_batch_size, device=h_fused.device)
        
        # === THE LOOP ===
        from torch_geometric.utils import softmax as gnn_softmax
        for step in range(max_steps):
            # [Debug] Check accumulated_mask at start of Step 0
            if step == 0 and not inference_mode and fused_batch[0] == 0:
                 mask_sum = acc_mask_local.sum().item()
                 probe_b_metrics["probeB/step0_local_mask_sum"] = mask_sum
                 # print(f"[EpisodeStepper] Step 0 Start: Local Mask Sum = {mask_sum}")

            # 1. T_sim Advancement (Monotonic)
            active_mask_t = (~hit_before_t).clone()
            
            # [CRITICAL FIX] Force Step 0 to be Active to prevent "Sleeping at Start" bug
            # Even if the system thinks the source is found (hit_before_t=True), 
            # we MUST run at least one step to compute Loss/Metrics unless it's pure inference.
            # [SSOT] Pacemaker Logic: Force active at step 0 if enabled
            force_active_step0 = getattr(self.cfg.life_support, 'enable_pacemaker', True)
            if step == 0 and not inference_mode and force_active_step0:
                # Force all graphs active at Step 0 for training
                active_mask_t = torch.ones_like(hit_before_t, dtype=torch.bool)
                # Reset steps_taken for these forced graphs (optional, but cleaner)
                # steps_taken[hit_before_t] = 0.0 
            
            # Optimization: If all graphs are done, stop early (in inference)
            if inference_mode and not active_mask_t.any():
                break
            
            # [SSOT] Early Stop Logic
            allow_early_stop = getattr(self.cfg.life_support, 'allow_early_stop', False)
            if not inference_mode and not active_mask_t.any():
                if allow_early_stop:
                    break
                # Else: Continue loop (Pacemaker behavior)
            # But wait, if active_mask_t is all False, then subsequent steps are meaningless?
            # No, if we forced Loss calculation on inactive graphs, we need the trajectory data (logits).
            # If we break here, trajectory stops.
            # So we MUST NOT break.
            
            # Track steps
            # [Fix] Only increment if graph is still active
            steps_taken[active_mask_t] += 1.0

            # 2. Physics Context (Topology & Consistency)
            # [SSOT Fix] Pass t_sim and static_ctx for Time-Aware Topology
            curr_edge_index, curr_edge_attr = self.model._update_dynamic_topology(
                causal_anchors, anchor_times, fused_global_ids, fused_batch, fused_edge_index, base_edge_attr,
                t_sim=t_sim, static_ctx=static_ctx
            )
            
            # [SSOT Audit] Extract Dynamic STT for Reachability (if available)
            stt_dynamic = None
            if 'stt_dynamic' in static_ctx:
                stt_dynamic = static_ctx['stt_dynamic'] # [E, 1]
                # Map to fused edges?
                # static_ctx['stt_dynamic'] is for SUBGRAPH edges (before fusion).
                # curr_edge_index is FUSED edges.
                # If fusion is 1-to-1 (Identity), then it matches.
                # If fusion merges nodes, we have a problem.
                # Currently fusion is Identity (subgraph level).
                # But `curr_edge_index` might have virtual edges added.
                # `curr_edge_attr` has virtual edges.
                # `stt_dynamic` only has physical edges?
                # NpzDatasetV6 loads it for `sub_edge_index`.
                # Virtual edges are appended later in Dataset.
                # Wait, NpzDatasetV6 appends virtual edges to `edge_index` and `edge_attr`.
                # But I added `stt_dynamic` based on `edge_mask`. `edge_mask` is for physical edges.
                # So `stt_dynamic` size corresponds to PHYSICAL edges in subgraph.
                
                # We need `stt_dynamic` for ALL edges in `curr_edge_index`.
                # Virtual edges should have 0 or their own STT.
                # Virtual edges are type 5. Physical are type 4.
                
                # We need to expand `stt_dynamic` to match `curr_edge_index`.
                # `curr_edge_index` comes from `self.model._update_dynamic_topology`.
                # It starts from `fused_edge_index`.
                # `fused_edge_index` comes from `static_ctx['fused_edge_index']`.
                # In Phase45Model, `fused_edge_index` is created from `data.edge_index`.
                # `data.edge_index` includes virtual edges if loaded.
                
                # If `stt_dynamic` is shorter than `edge_index`, we pad it?
                # Physical edges are first?
                # Usually yes.
                # Let's pad with 0s (instant/unknown) or use static fallback for virtual.
                
                num_current_edges = curr_edge_index.size(1)
                if stt_dynamic.size(0) < num_current_edges:
                    padding = torch.zeros(num_current_edges - stt_dynamic.size(0), 1, device=stt_dynamic.device)
                    stt_dynamic = torch.cat([stt_dynamic, padding], dim=0)
                elif stt_dynamic.size(0) > num_current_edges:
                    # Should not happen unless edges removed?
                    stt_dynamic = stt_dynamic[:num_current_edges]
            
            physics_in = {
                't_sim': t_sim,
                'valid_mask': acc_mask_local.view(-1),
                'anchor_type': causal_anchors.view(-1),
                'anchor_time': anchor_times.view(-1),
                'edge_index': curr_edge_index,
                'edge_stt': curr_edge_attr[:, 3] if curr_edge_attr.size(1) > 3 else torch.zeros(curr_edge_index.size(1), device=h_fused.device),
                'batch': fused_batch
            }
            # Slot 4: Physics
            physics_ctx = self.model.physics_module(physics_in)
            
            # 3. FoV Step
            stats = {'entropy': 2.0, 'race_conflict_mean': 0.0, 'max_prob': 0.0, 'top1_margin': 0.0}
            if step > 0 and last_logits is not None:
                stats = self._calculate_step_stats(last_logits, fused_batch, energy=physics_ctx.get('race_energy'))
            
            # [Exp F] Prepare Explicit Context for Navigator
            # Use dynamic total budget if possible
            # Total time budget in steps = budget_K / action_k ? No.
            # User implies: "400+ samples budget per scenario?"
            # If scenario is 10 rounds * 3 samples = 30 samples.
            # Why 400+?
            # Maybe the user means "Total Nodes in Graph" vs "Budget"?
            # Or maybe the log message "Avg_Samples: 423" is wrong?
            # Let's look at the log: 'Avg_Samples': 423.73
            # This is likely the number of nodes in the graph (N).
            # Ah, 'Avg_Samples' in wandb usually means 'Avg Graph Size' or 'Avg Nodes Visited'?
            # In `train_phase4_end2end.py`:
            # log_dict = { ... 'Avg_Samples': avg_num_nodes ... }
            # Yes, it is graph size.
        
            # Back to budget normalization.
            # We should use max_steps * action_k as total budget?
            # Or just max_steps?
            budget_total = float(max_steps)
            budget_norm = (budget_total - steps_taken) / budget_total
            step_norm = float(step) / float(max_steps) if max_steps > 0 else 0.0
        
            # [Fix] Construct 6-dim nav_state_summary to match Config
            # Channels: Entropy, MaxProb, Margin, Conflict, Budget, Step
            nav_state_summary = torch.stack([
                torch.full((curr_batch_size,), float(stats.get('entropy', 0.0)), device=h_fused.device),
                torch.full((curr_batch_size,), float(stats.get('max_prob', 0.0)), device=h_fused.device),
                torch.full((curr_batch_size,), float(stats.get('top1_margin', 0.0)), device=h_fused.device),
                torch.full((curr_batch_size,), float(stats.get('race_conflict_mean', 0.0)), device=h_fused.device),
                budget_norm,
                torch.full((curr_batch_size,), step_norm, device=h_fused.device)
            ], dim=1) # [B, 6]

            explicit_context = nav_state_summary[:, :3] # Legacy 3-dim support if needed
            
            # Slot 5: FoV
            fov_params = {}
            if self.model.fov_controller:
                fov_params = self.model.fov_controller.step(stats)
            
            # [Fix] Apply FoV Dynamic Parameters to Action K and Temperature
            current_action_k = action_k
            current_tau = tau
            if fov_params:
                 if 'candidate_topM' in fov_params:
                     current_action_k = int(fov_params['candidate_topM'])
                 if 'alpha' in fov_params: # Alpha maps to Temperature scaling?
                     # Assuming alpha controls exploration, high alpha -> low tau (greedy)?
                     # Or directly use 'tau' if FoV outputs it.
                     # For now, let's keep tau static unless FoV explicitly overrides.
                     pass
            
            # Temp Graph Wrapper
            class TempGraph:
                def __init__(self, x, edge_index, edge_attr, batch):
                    self.x, self.edge_index, self.edge_attr, self.batch = x, edge_index, edge_attr, batch
            temp_graph = TempGraph(h_fused, curr_edge_index, curr_edge_attr, fused_batch)

            # [Feature] Reasoner Teacher Shortlist Logic
            rea_teacher_cfg = getattr(self.cfg, 'reasoner_teacher', {})
            enable_shortlist = rea_teacher_cfg.get('enable_shortlist', False)
            enable_score = rea_teacher_cfg.get('enable_score', False)
            
            shortlist_mask_global = None
            
            # Log Meta (Step 0)
            if step == 0 and not inference_mode:
                probe_b_metrics["meta/teacher_shortlist_enabled"] = 1.0 if enable_shortlist else 0.0
                probe_b_metrics["meta/teacher_score_enabled"] = 1.0 if enable_score else 0.0

            # Only run if enabled, not inference, has Ground Truth, and using Reasoner (logits exist)
            if (enable_shortlist or enable_score) and not inference_mode and has_y and last_logits is not None:
                candidate_topk = rea_teacher_cfg.get('candidate_topk', 50)
                shortlist_topk = rea_teacher_cfg.get('shortlist_topk', 10)
                
                # Initialize shortlist mask (default all True, we restrict active graphs)
                if enable_shortlist:
                    shortlist_mask_global = torch.ones_like(acc_mask_local.view(-1), dtype=torch.bool)
                
                # Loop over graphs
                for g_idx in range(curr_batch_size):
                    g_mask = (fused_batch == g_idx)
                    if not g_mask.any(): continue
                    
                    # Get active mask for this graph
                    if hit_before_t[g_idx]: 
                        continue
                    
                    # Get logits for this graph
                    g_logits = last_logits[g_mask]
                    g_indices = g_mask.nonzero().view(-1)
                    
                    # Filter visited nodes
                    g_visited = acc_mask_local[g_indices].view(-1) > 0.5
                    # Clone logits to avoid modifying last_logits
                    g_logits_filtered = g_logits.clone()
                    g_logits_filtered[g_visited] = -float('inf')
                    
                    # Select Candidates (Top-K)
                    k_cand = min(candidate_topk, g_logits_filtered.size(0))
                    # Check if we have valid candidates
                    if (g_logits_filtered > -1e9).sum() < k_cand:
                        k_cand = (g_logits_filtered > -1e9).sum().item()
                    
                    if k_cand > 0:
                        _, topk_local_idx = torch.topk(g_logits_filtered, k_cand)
                        candidates_global = g_indices[topk_local_idx]
                        
                        # Calculate Scores
                        scores = self._calculate_teacher_scores(
                            candidates_global,
                            last_logits, # Use full logits for rank calculation
                            x_nav,
                            h_fused,
                            acc_mask_local,
                            static_ctx,
                            t_sim,
                            temp_graph,
                            physics_ctx,
                            reasoner_memory_state,
                            causal_anchors,
                            fused_source_label
                        )
                        
                        # Shortlist Logic
                        if enable_shortlist:
                            # Restrict this graph
                            shortlist_mask_global[g_indices] = False
                            
                            # Sort by score (descending)
                            k_short = min(shortlist_topk, scores.size(0))
                            _, top_score_idx = torch.topk(scores, k_short)
                            shortlist_nodes = candidates_global[top_score_idx]
                            
                            shortlist_mask_global[shortlist_nodes] = True
                            
                            # Logging (G0 only)
                            if g_idx == 0: 
                                best_idx = top_score_idx[0]
                                best_node = candidates_global[best_idx].item()
                                best_score = scores[best_idx].item()
                                probe_b_metrics[f"teacher/step{step}/top1_node"] = best_node
                                probe_b_metrics[f"teacher/step{step}/top1_score"] = best_score
                                probe_b_metrics[f"teacher/step{step}/score_max"] = scores.max().item()
                                probe_b_metrics[f"teacher/step{step}/score_min"] = scores.min().item()
                                probe_b_metrics[f"teacher/step{step}/score_mean"] = scores.mean().item()
                                probe_b_metrics[f"teacher/step{step}/shortlist_size"] = float(k_short)
                                probe_b_metrics[f"teacher/step{step}/candidate_pool"] = float(k_cand)
                                
                                # Check if True Source is in Shortlist
                                s_mask = (fused_source_label > 0.5) & (fused_batch == 0)
                                if s_mask.any():
                                    s_idx = s_mask.nonzero().view(-1)[0].item()
                                    in_shortlist = (s_idx in shortlist_nodes.tolist())
                                    probe_b_metrics[f"teacher/step{step}/source_in_shortlist"] = 1.0 if in_shortlist else 0.0

            # 4. Navigator Step (Slot 6)
            # [Exp E] Pass nav_memory_state
            # [Exp F] Pass explicit_context
            
            # Apply Shortlist Mask to Valid Mask
            valid_mask_final = ~acc_mask_local.view(-1).bool()
            
            # [Diagnosis] Unified Sampling Space: Intersect with Feasible Mask
            if 'feasible_mask' in physics_ctx:
                feasible_mask = physics_ctx['feasible_mask'].view(-1).bool()
                
                # [Log] Step 0 Sizes
                if step == 0 and not inference_mode and fused_batch[0] == 0:
                     probe_b_metrics["probeB/valid_unvisited_size"] = valid_mask_final.float().sum().item()
                     probe_b_metrics["probeB/feasible_mask_size"] = feasible_mask.float().sum().item()
                
                valid_mask_final = valid_mask_final & feasible_mask
                
                if step == 0 and not inference_mode and fused_batch[0] == 0:
                     probe_b_metrics["probeB/unified_mask_size"] = valid_mask_final.float().sum().item()

            if shortlist_mask_global is not None:
                valid_mask_final = valid_mask_final & shortlist_mask_global

            # [Ablation] Input Semantics Experiment
            feature_mode = getattr(self.cfg.data, 'feature_mode', 'baseline')
            x_nav_input = x_nav # Default reference
            
            if feature_mode == 'no_mask':
                x_nav_input = x_nav.clone()
                if x_nav_input.size(1) > 3:
                    x_nav_input[:, 3] = 0.0

            # [Step 2 & 3] Build Evidence State (Refactored)
            # Phase 1: State Construction
            obs_state = self._build_observation_state(x_nav)
            phys_context = self._build_physics_context(curr_edge_index, curr_edge_attr, physics_ctx, h_fused.device)
            evidence_state = self._build_evidence_state(obs_state, phys_context)

            nav_state = {
                'h_fused': h_fused, 
                'x_nav': x_nav_input, 
                'valid_mask': valid_mask_final, # [Feature] Updated valid mask
                'nav_memory_state': nav_memory_state,
                'explicit_context': explicit_context,
                'nav_state_summary': nav_state_summary, # [Fix] Pass 6-dim summary
                'k': current_action_k,   # [Refactor] Pass dynamic K
                'tau': current_tau,      # [Refactor] Pass temperature
                'batch': fused_batch, # [Fix] Pass batch for per-graph sampling
                'fov_params': fov_params, # [Fix] Pass FoV params to Navigator
                'action_policy': action_policy, # [Refactor] Pass policy
                'n_id': static_ctx.get('fused_global_ids'), # [Fix] Pass Global IDs for HSR
                'reasoner_logits': last_logits, # [Fix] Pass Reasoner Logits for Heuristic Navigator
                'fused_edge_index': curr_edge_index, # [Fix] Pass Topology
                'fused_edge_attr': curr_edge_attr,   # [Fix] Pass Topology Attributes
            }
            self._inject_semantic_states(nav_state, evidence_state) # Inject
            
            nav_out = self.model.navigator_module(nav_state, temp_graph, physics_ctx)
            nav_logits = nav_out.get('logits')
            
            # [Exp E] Update Navigator Memory State
            if 'updated_memory_state' in nav_out:
                nav_memory_state = nav_out['updated_memory_state']
            elif 'nav_memory_state' in nav_state:
                # Fallback to input if not updated (Legacy)
                nav_memory_state = nav_state['nav_memory_state']
            
            # 5. Reasoner Step (Slot 7)
            # [Fix] Explicitly pass inverse_indices to Reasoner for Global->Local mapping
            reasoner_state = {
                'h_fused': h_fused,
                'x_nav': x_nav_input, # [Fix] Use ablated input
                'n_id': static_ctx.get('fused_global_ids'), # [Fix] Pass Global IDs
                'inverse_indices': inverse_indices, # [Critical Fix] Pass Raw->Fused Mapping
                'causal_anchors': causal_anchors,
                'accumulated_mask': acc_mask_local,
                'memory_state': reasoner_memory_state
            }
            self._inject_semantic_states(reasoner_state, evidence_state) # Inject
            # [Vis] Capture state for explanation
            vis_reasoner_state = reasoner_state.copy()
            
            reasoner_out = self.model.reasoner_module(reasoner_state, temp_graph, physics_ctx)
            if 'updated_memory_state' in reasoner_out:
                reasoner_memory_state = reasoner_out['updated_memory_state']
            elif 'memory_state' in reasoner_state:
                # Fallback
                reasoner_memory_state = reasoner_state['memory_state']
            
            logits_fused = reasoner_out['logits']
            
            # [Task 1 & 2] Capture Pure Reasoner Logits & Compute Per-Step Metrics
            # Must be done BEFORE any masking or bias injection
            if not inference_mode and has_y:
                with torch.no_grad():
                    # [PROBE] Start of Probe Implementation
                    # =================================================
                    
                    # A. Calculate Posterior & Entropy
                    probs_fused = gnn_softmax(logits_fused.view(-1), fused_batch)
                    log_probs_fused = torch.log(probs_fused + 1e-9)
                    entropy_per_node = -probs_fused * log_probs_fused
                    entropy_per_graph = scatter_sum(entropy_per_node, fused_batch, dim=0, dim_size=curr_batch_size)
                    
                    # B. Get True Source Info
                    source_mask = (fused_source_label.view(-1) > 0.5)
                    source_logits = logits_fused[source_mask].view(-1)
                    source_probs = probs_fused[source_mask]
                    
                    # Map to graph index
                    source_graph_idx = fused_batch[source_mask]
                    
                    # C. Calculate True Source Rank
                    # For each graph, count how many nodes have higher logits than the source
                    # This is slow to do in a loop. Let's vectorize.
                    # We can compare all logits to the source logit of their respective graph.
                    # First, we need to broadcast the source logit to all nodes in its graph.
                    source_logit_per_graph = torch.zeros(curr_batch_size, device=h_fused.device)
                    source_logit_per_graph.scatter_add_(0, source_graph_idx, source_logits)
                    source_logits_broadcasted = source_logit_per_graph[fused_batch] # [N_fused]
                    
                    # Count nodes with higher logits
                    is_higher = (logits_fused.view(-1) > source_logits_broadcasted)
                    # Sum up counts per graph
                    rank_per_graph = scatter_sum(is_higher.float(), fused_batch, dim=0, dim_size=curr_batch_size)
                    
                    # D. Get Top-1 Predicted Info
                    pred_prob_top1, pred_idx_top1_local = scatter_max(probs_fused, fused_batch, dim=0, dim_size=curr_batch_size)
                    
                    # E. Log probe metrics for active graphs
                    # We only want to log for graphs that are still in play
                    active_graphs = active_mask_t.nonzero().view(-1)
                    if active_graphs.numel() > 0:
                        probe_b_metrics[f"probe/step_{step}/active_count"] = active_graphs.numel()
                        probe_b_metrics[f"probe/step_{step}/true_source_rank"] = rank_per_graph[active_graphs].mean().item()
                        probe_b_metrics[f"probe/step_{step}/posterior_entropy"] = entropy_per_graph[active_graphs].mean().item()
                        probe_b_metrics[f"probe/step_{step}/true_source_prob"] = source_probs.mean().item() # This is avg over active sources, ok
                        probe_b_metrics[f"probe/step_{step}/pred_top1_prob"] = pred_prob_top1[active_graphs].mean().item()

                    # =================================================
                    # [PROBE] End of Probe Implementation

                    # 1. Step Loss (Cross Entropy on Pure Logits)
                    # Use scatter_log_softmax for numerical stability over graph
                    step_log_probs = torch.log(gnn_softmax(logits_fused.view(-1), fused_batch) + 1e-9)
                    # Select log_prob of target node
                    step_loss_per_graph = scatter_sum(-step_log_probs * fused_source_label.view(-1), fused_batch, dim=0, dim_size=curr_batch_size)
                    step_loss_val = step_loss_per_graph.mean().item()
                    
                    # 2. Step Hit@K (Pure Logits Ranking)
                    # We need to rank nodes per graph.
                    # Efficient implementation for K=1, 5, 10
                    # Hit@1
                    _, top1_idx_step = scatter_max(logits_fused.view(-1), fused_batch, dim=0, dim_size=curr_batch_size)
                    
                    # Hit@5
                    hit5_accum = torch.zeros(curr_batch_size, dtype=torch.bool, device=h_fused.device)
                    temp_logits_step = logits_fused.view(-1).clone()
                    for _ in range(5):
                        _, idx = scatter_max(temp_logits_step, fused_batch, dim=0, dim_size=curr_batch_size)
                        valid_mask = (idx >= 0) & (idx < fused_source_label.size(0))
                        if valid_mask.any():
                            hit5_accum[valid_mask] |= (fused_source_label[idx[valid_mask]].view(-1) > 0.5)
                            temp_logits_step[idx[valid_mask]] = -float('inf')
                            
                    # Hit@10
                    hit10_accum = hit5_accum.clone()
                    # Continue from temp_logits_step (already masked top 5)
                    for _ in range(5): # 5 more to get 10
                        _, idx = scatter_max(temp_logits_step, fused_batch, dim=0, dim_size=curr_batch_size)
                        valid_mask = (idx >= 0) & (idx < fused_source_label.size(0))
                        if valid_mask.any():
                            hit10_accum[valid_mask] |= (fused_source_label[idx[valid_mask]].view(-1) > 0.5)
                            temp_logits_step[idx[valid_mask]] = -float('inf')
                    
                    # Validate against graph having source
                    has_source_step, _ = scatter_max(fused_source_label.view(-1), fused_batch, dim=0, dim_size=curr_batch_size)
                    valid_graphs_step = (has_source_step > 0.5)
                    
                    # Calculate Rates
                    if valid_graphs_step.any():
                        # Hit@1 Check
                        hit1_check = torch.zeros(curr_batch_size, dtype=torch.bool, device=h_fused.device)
                        valid_top1 = (top1_idx_step >= 0) & (top1_idx_step < fused_source_label.size(0))
                        if valid_top1.any():
                            hit1_check[valid_top1] = (fused_source_label[top1_idx_step[valid_top1]].view(-1) > 0.5)
                        
                        hit1_rate = (hit1_check & valid_graphs_step).float().sum() / valid_graphs_step.float().sum()
                        hit5_rate = (hit5_accum & valid_graphs_step).float().sum() / valid_graphs_step.float().sum()
                        hit10_rate = (hit10_accum & valid_graphs_step).float().sum() / valid_graphs_step.float().sum()
                        
                        # Task 3: Print & Log
                        # print(f"[Step {step}] Loss: {step_loss_val:.4f} | Hit@1: {hit1_rate.item():.1%} | Hit@5: {hit5_rate.item():.1%} | Hit@10: {hit10_rate.item():.1%}")
                        
                        probe_b_metrics[f"step_metrics/step_{step}_loss"] = step_loss_val
                        probe_b_metrics[f"step_metrics/step_{step}_hit1"] = hit1_rate.item()
                        probe_b_metrics[f"step_metrics/step_{step}_hit5"] = hit5_rate.item()
                        probe_b_metrics[f"step_metrics/step_{step}_hit10"] = hit10_rate.item()
            
            # [Task 1] Calculate Pre-Action Probability of Target
            probs_before = gnn_softmax(logits_fused.view(-1), fused_batch)
            prob_before_target = torch.zeros(curr_batch_size, device=h_fused.device)
            if has_y and fused_source_label is not None:
                prob_before_target = scatter_sum(probs_before * fused_source_label.view(-1), fused_batch, dim=0, dim_size=curr_batch_size)
            
            # Calculate Hit Prob Surrogate (for Metrics/Loss)
            step_probs = gnn_softmax(logits_fused.view(-1), fused_batch)
            hit_prob_surrogate = torch.zeros(curr_batch_size, device=h_fused.device)
            if has_y and fused_source_label is not None:
                hit_prob_surrogate = scatter_sum(step_probs * fused_source_label.view(-1), fused_batch, dim=0, dim_size=curr_batch_size)
            else:
                hit_prob_surrogate, _ = scatter_max(step_probs, fused_batch, dim=0, dim_size=curr_batch_size)
            
            # Metric: Predict Hit@1 (Step 0 Snapshot)
            # [Fix] Measure Hit@1 at Step 0 (After observing Trigger, before Action 1)
            # This ensures we capture the model's ability to diagnose immediately.
            if step == 0:
                if has_y:
                    # Hit@1
                    _, top1_idx = scatter_max(logits_fused.view(-1), fused_batch, dim=0, dim_size=curr_batch_size)
                    has_source_in_graph, _ = scatter_max(fused_source_label, fused_batch, dim=0, dim_size=curr_batch_size)
                    predict_hit_valid = (has_source_in_graph.view(-1) > 0.5)
                    
                    # Ensure indices are valid
                    valid_idx_mask = (top1_idx >= 0) & (top1_idx < fused_source_label.size(0))
                    hit_check = torch.zeros(curr_batch_size, dtype=torch.bool, device=h_fused.device)
                    if valid_idx_mask.any():
                        hit_check[valid_idx_mask] = (fused_source_label[top1_idx[valid_idx_mask]].view(-1) > 0.5)
                    predict_hit_at_1 = hit_check & predict_hit_valid

                    # Hit@5
                    predict_hit_at_5_accum = torch.zeros(curr_batch_size, dtype=torch.bool, device=h_fused.device)
                    temp_logits_h5 = logits_fused.view(-1).clone()
                    
                    for _ in range(5):
                        _, top_idx = scatter_max(temp_logits_h5, fused_batch, dim=0, dim_size=curr_batch_size)
                        
                        valid_mask = (top_idx >= 0) & (top_idx < fused_source_label.size(0))
                        
                        current_hit = torch.zeros(curr_batch_size, dtype=torch.bool, device=h_fused.device)
                        if valid_mask.any():
                             current_hit[valid_mask] = (fused_source_label[top_idx[valid_mask]].view(-1) > 0.5)
                        
                        predict_hit_at_5_accum = predict_hit_at_5_accum | current_hit
                        
                        # Mask out for next iteration
                        if valid_mask.any():
                             temp_logits_h5[top_idx[valid_mask]] = -float('inf')
                    
                    predict_hit_at_5 = predict_hit_at_5_accum & predict_hit_valid

                    # [Alignment Check - Step 0 End]
                    if not inference_mode:
                         # Log the final decision for this step
                         for b in range(curr_batch_size):
                             if predict_hit_valid[b]:
                                 p_loc = top1_idx[b].item()
                                 # Get True Local again
                                 _, t_loc_idx = scatter_max(fused_source_label.view(-1), fused_batch, dim=0, dim_size=curr_batch_size)
                                 t_loc = t_loc_idx[b].item()
                                 
                                 # Global
                                 p_glob = fused_global_ids[p_loc].item()
                                 t_glob = fused_global_ids[t_loc].item()
                                 
                                 # Check if active
                                 is_active = active_mask_t[b].item() > 0.5
                                 
                                 # print(f"[对齐核对] Loss 优化的 Local Target: {t_loc} | 模型预测的 Local Index: {p_loc} | 映射后的 Global 预测: {p_glob} | 真实的 Global 真凶: {t_glob} | Active: {is_active}")
                else:
                    predict_hit_valid = torch.zeros(curr_batch_size, dtype=torch.bool, device=h_fused.device)
                    predict_hit_at_1 = torch.zeros(curr_batch_size, dtype=torch.bool, device=h_fused.device)
                    predict_hit_at_5 = torch.zeros(curr_batch_size, dtype=torch.bool, device=h_fused.device)

            if step == 1:
                # [Task 1: Feature X-Ray at Step 1 (After Update)]
                if has_y and not inference_mode:
                     # x_nav is now x_nav_next from Step 0 (Updated)
                     x_check = x_nav
                     
                     target_mask = (fused_source_label.view(-1) > 0.5)
                     
                     # Vectorized Statistics for entire batch
                     if target_mask.any():
                         # 1. Target Stats
                         # [Fix] Coordinate System Alignment: Local (Fused) vs Global (Raw)
                         # x_check is Fused Features [N_fused, C] (x_nav is Fused)
                         # fused_source_label is Fused Label [N_fused]
                         # So target_mask is [N_fused].
                         # This block is safe IF x_nav is fused.
                         
                         # Check x_nav shape vs target_mask shape
                         # x_nav shape: [N_fused, C]
                         # target_mask shape: [N_fused]
                         # No mismatch here if x_nav is indeed fused.
                         
                         # BUT, wait. x_nav comes from static_ctx['x_nav'].
                         # Let's verify if x_nav is fused or raw.
                         # In Phase45Model, x_nav is created from x_raw using fusion.
                         # It is [N_fused, C].
                         
                         # The error message said:
                         # IndexError: The shape of the mask [140940] at index 0 does not match the shape of the indexed tensor [355737, 7] at index 0
                         # 140940 is Fused size. 355737 is Raw size.
                         # This means x_check IS Raw Tensor!
                         # Let's see where x_check comes from:
                         # x_check = x_nav
                         # So x_nav is [355737, 7] ???
                         
                         # Let's trace x_nav.
                         # Line 85: x_nav = static_ctx['x_nav']
                         # In Phase45Model.forward():
                         # x_nav is passed to Navigator.
                         # If Navigator uses GNN on Fused Graph, x_nav MUST be Fused.
                         
                         # However, if 'x_nav' in static_ctx was assigned x_raw by mistake?
                         # Or if Phase45Model passed x_raw as x_nav?
                         
                         # Let's look at Line 1165 (old code):
                         # x_nav = x_nav_next
                         # x_nav_next = x_nav.clone()
                         
                         # Wait, if x_nav was [355737], then x_nav_next is [355737].
                         # But Step 0 injection used `target_fused_indices`.
                         # If x_nav was Raw, then `x_nav[target_fused_indices]` would be wrong if fused indices are smaller?
                         # No, fused indices are 0..N_fused-1. Raw indices are 0..N_raw-1.
                         
                         # If x_nav is Raw, then indexing it with Fused Indices is WRONG unless Fused is subset of Raw (identity mapping).
                         # But here sizes differ.
                         
                         # Re-read error: mask [140940] vs tensor [355737, 7].
                         # Mask is target_mask. It comes from fused_source_label.
                         # fused_source_label is definitely Fused [140940].
                         # So x_check must be [355737, 7].
                         # This implies x_nav is Global/Raw!
                         
                         # IF x_nav is Global, then ALL GNN operations using fused_edge_index (which is Fused) on x_nav (Global) would fail immediately!
                         # Navigator step: h_nav = backbone(x_nav, curr_edge_index)
                         # If x_nav is 355k and edge_index is on 140k nodes, PyG might complain or just work if indices match.
                         # But usually Fused Graph has remapped indices 0..140k.
                         
                         # If x_nav is Raw, it implies Phase45Model is passing Raw Features to Navigator?
                         # Let's assume x_nav IS fused [140k].
                         # Then why did the error say tensor is [355k]?
                         # MAYBE x_check is NOT x_nav?
                         # Code: x_check = x_nav.
                         
                         # Wait, did I shadow x_nav?
                         # Line 83: x_nav_full = static_ctx['x_nav_full']
                         # Line 85: x_nav = static_ctx['x_nav']
                         
                         # Could `static_ctx['x_nav']` be Raw?
                         # If so, how did Navigator run?
                         # Navigator run log: "Minimal CPU Forward successful..."
                         
                         # Maybe the error comes from line 466, but x_check is actually `x_raw`?
                         # No, code says `x_check = x_nav`.
                         
                         # Hypothesis: `x_nav` variable holds the Raw Tensor [355737] but `fused_source_label` holds Fused Label [140940].
                         # If x_nav is Raw, then we have a huge problem in Navigator logic too.
                         
                         # FIX: We need to ensure we use the FUSED version of features for X-Ray.
                         # If x_nav is Raw, we must map it to Fused.
                         # But x_nav SHOULD be Fused.
                         
                         # Let's assume x_nav IS Raw (as per error trace).
                         # Then we need to map it to Fused to match target_mask.
                         # Or map target_mask to Raw.
                         # target_mask is from fused_source_label.
                         
                         # If x_nav is Raw, we can use `inverse_indices` to map Fused -> Raw.
                         # But we want `x_check` to be Fused size for metrics? Or Raw size?
                         # X-Ray usually checks "What the model sees".
                         # If model sees Raw x_nav (via some internal mapping in Navigator?), then we should check Raw.
                         # But Navigator takes `x_nav` and `curr_edge_index`.
                         # `curr_edge_index` is definitely Fused (dynamic topology update returns fused).
                         
                         # If x_nav is Raw [355k] and Edge Index is Fused [140k], PyG Conv will crash unless x_nav is sliced.
                         # `self.model.dynamic_gate(x_nav, view_batch)` might be doing the slicing?
                         # Line 1139: `x_nav_gated, _ = self.model.dynamic_gate(x_nav_visible, view_batch)`
                         # `view_batch` is [N_raw] or [N_fused]?
                         # `view_batch` is usually `batch` vector.
                         
                         # Let's look at `EpisodeStepper` init.
                         # It unpacks `inverse_indices` (Raw->Fused map).
                         
                         # CRITICAL: If x_nav is [355737], it is Raw.
                         # `target_mask` is [140940] (Fused).
                         
                         # Fix: Slice x_check using `inverse_indices`? 
                         # No, `inverse_indices` is [355737].
                         # We want x_check_fused.
                         # x_check_fused = x_nav[???]
                         # We don't have Fused->Raw map easily available (we just computed it in Step 0 fix).
                         
                         # BUT, we have `h_fused` [140940].
                         # Can we use `h_fused` for X-Ray? 
                         # No, X-Ray wants to check Input Features (Signal/Poison state).
                         
                         # Solution: Use `inverse_indices` to expand `target_mask` (Fused) to Raw Space?
                         # target_mask is [140940].
                         # inverse_indices is [355737], values are 0..140939.
                         # We can map Raw nodes to Fused labels.
                         # raw_target_mask = target_mask[inverse_indices] (handling -1).
                         
                         # Handle -1 in inverse_indices
                         # mask valid mappings
                         valid_raw = (inverse_indices >= 0)
                         safe_indices = inverse_indices.clamp(min=0)
                         
                         # Broadcast fused mask to raw
                         raw_target_mask = torch.zeros(inverse_indices.size(0), dtype=torch.bool, device=inverse_indices.device)
                         raw_target_mask[valid_raw] = target_mask[safe_indices[valid_raw]]
                         
                         # Now use raw_target_mask on x_check (Raw)
                         target_feats = x_check[raw_target_mask]
                         
                         # 2. Background Stats (All non-targets)
                         bg_mask = ~raw_target_mask
                         # Filter valid raw nodes only? Or all raw nodes?
                         # Usually we only care about nodes in the subgraph (valid_raw).
                         bg_mask = bg_mask & valid_raw
                         
                         if bg_mask.any():
                             bg_feats = x_check[bg_mask]
                             bg_mean = bg_feats.mean(dim=0)
                             
                             # 1. Target Stats
                             tgt_mean = target_feats.mean(dim=0)
                             
                             # 3. L1 Diff of Centroids
                             l1_diff = (tgt_mean - bg_mean).abs().sum().item()
                             
                             # 4. Log Metrics
                             probe_b_metrics["xray/step1_l1_diff"] = l1_diff
                             
                             # Channel specific (Signal=0, Poison=1, Mask=3, Anchor=4, LogDeg=6)
                             # Use safe indexing in case channels < 7
                             if tgt_mean.size(0) > 0: probe_b_metrics["xray/step1_tgt_ch0_signal"] = tgt_mean[0].item()
                             if tgt_mean.size(0) > 3: probe_b_metrics["xray/step1_tgt_ch3_mask"] = tgt_mean[3].item()
                             if tgt_mean.size(0) > 4: probe_b_metrics["xray/step1_tgt_ch4_anchor"] = tgt_mean[4].item()
                             if tgt_mean.size(0) > 6: probe_b_metrics["xray/step1_tgt_ch6_deg"] = tgt_mean[6].item()
                             
                             if bg_mean.size(0) > 0: probe_b_metrics["xray/step1_bg_ch0_signal"] = bg_mean[0].item()
                             if bg_mean.size(0) > 6: probe_b_metrics["xray/step1_bg_ch6_deg"] = bg_mean[6].item()
                             
                             # x_raw check
                             probe_b_metrics["xray/has_x_raw"] = 1.0 if x_raw is not None else 0.0
                     else:
                         # Fallback if no target feats
                         pass

                if has_y:
                    # Metric Logic moved to Step 0
                    pass
                else:
                    pass
            elif step > 0 and 'predict_hit_at_1' not in locals():
                 # [Fix] Initialize if loop started late or skipped step 0
                 predict_hit_valid = torch.zeros(curr_batch_size, dtype=torch.bool, device=h_fused.device)
                 predict_hit_at_1 = torch.zeros(curr_batch_size, dtype=torch.bool, device=h_fused.device)
                 predict_hit_at_5 = torch.zeros(curr_batch_size, dtype=torch.bool, device=h_fused.device)

            # 6. Bias Injection (Optional, Config-driven)
            # Heuristics Engine (Slot X)
            if hasattr(self.model, 'heuristics_engine'):
                # Reasoner Penalty (Scheme 1)
                rea_penalty = self.model.heuristics_engine.compute_reasoner_penalty(
                    h_fused, curr_edge_index, curr_edge_attr, t_sim, fused_batch
                )
                logits_fused = logits_fused - rea_penalty.view_as(logits_fused)
                
                # Navigator Bias (Scheme 2 & 3)
                nav_bias = self.model.heuristics_engine.compute_navigator_bias(
                    h_fused, curr_edge_index, curr_edge_attr, t_sim, fused_batch
                )
                if nav_logits is not None:
                    # [Fix] Apply bias to nav_logits. 
                    # Note: nav_logits might be flat or [B, N] depending on implementation.
                    # Usually [N, 1] or [N].
                    # nav_bias is [N].
                    if nav_logits.shape == nav_bias.shape:
                         nav_logits = nav_logits + nav_bias
                    elif nav_logits.view(-1).shape == nav_bias.view(-1).shape:
                         nav_logits = nav_logits.view(-1) + nav_bias.view(-1)
                         nav_logits = nav_logits.view_as(nav_out.get('logits'))

                # [TRACER BULLET] Debug Heuristics
                # Print for step 0 and step 5 to see evolution
                if (step == 0 or step == 5 or step == max_steps - 1) and not inference_mode:
                    prefix = f"tracer/step{step}"
                    
                    # 1. Virtual Edges (Type != 0)
                    is_virt = (curr_edge_attr[:, 5] > 0.5)
                    n_virt = is_virt.sum().item()
                    n_total = curr_edge_index.size(1)
                    probe_b_metrics[f"{prefix}/n_virt"] = float(n_virt)
                    probe_b_metrics[f"{prefix}/n_total"] = float(n_total)
                    
                    if n_virt > 0:
                        # [SSOT V6.3] Virtual STT is in Ch0 (Log Med STT)
                        stt = curr_edge_attr[is_virt, 0]
                        probe_b_metrics[f"{prefix}/virt_stt_min"] = stt.min().item()
                        probe_b_metrics[f"{prefix}/virt_stt_max"] = stt.max().item()
                        probe_b_metrics[f"{prefix}/virt_stt_mean"] = stt.mean().item()
                        
                        virt_attr = curr_edge_attr[is_virt]
                        # Expanded Stats for Virtual Edges (Aligned with Physical)
                        for ch in range(min(virt_attr.size(1), 8)):
                            col = virt_attr[:, ch]
                            probe_b_metrics[f"{prefix}/virt_ch{ch}_mean"] = col.mean().item()
                            probe_b_metrics[f"{prefix}/virt_ch{ch}_max"] = col.max().item()
                            probe_b_metrics[f"{prefix}/virt_ch{ch}_min"] = col.min().item()

                    # 2. Physical Edges
                    is_phys = ~is_virt
                    n_phys = is_phys.sum().item()
                    probe_b_metrics[f"{prefix}/n_phys"] = float(n_phys)
                    
                    if n_phys > 0:
                        phys_attr = curr_edge_attr[is_phys]
                        # Expanded Stats for Physical Edges
                        for ch in range(min(phys_attr.size(1), 8)):
                            col = phys_attr[:, ch]
                            probe_b_metrics[f"{prefix}/phys_ch{ch}_mean"] = col.mean().item()
                            probe_b_metrics[f"{prefix}/phys_ch{ch}_max"] = col.max().item()
                            probe_b_metrics[f"{prefix}/phys_ch{ch}_min"] = col.min().item()

                    # 3. Node Features
                    # x_nav is visible features
                    for i in range(min(x_nav.size(1), 7)):
                        f_col = x_nav[:, i]
                        probe_b_metrics[f"{prefix}/node_ch{i}_mean"] = f_col.mean().item()
                        probe_b_metrics[f"{prefix}/node_ch{i}_max"] = f_col.max().item()
                        probe_b_metrics[f"{prefix}/node_ch{i}_min"] = f_col.min().item()
                        probe_b_metrics[f"{prefix}/node_ch{i}_nonzero"] = float(f_col.abs().gt(1e-6).sum().item())
                        
                    # 4. Rea Penalty
                    probe_b_metrics[f"{prefix}/rea_penalty_sum"] = rea_penalty.sum().item()
                    probe_b_metrics[f"{prefix}/rea_penalty_max"] = rea_penalty.max().item()
                    
                    # 5. Nav Bias
                    probe_b_metrics[f"{prefix}/nav_bias_sum"] = nav_bias.sum().item()
                    probe_b_metrics[f"{prefix}/nav_bias_max"] = nav_bias.max().item()
                
                # [DEBUG] Engineering Checkpoints (Merged into Probe Metrics)
                if step == 0 and not inference_mode:
                    try:
                        if fused_source_label is not None:
                            # Graph 0
                            mask_0 = (fused_batch == 0)
                            label_0 = fused_source_label[mask_0]
                            probe_b_metrics["tracer/step0/g0_label_sum"] = label_0.sum().item()
                            probe_b_metrics["tracer/step0/g0_label_argmax"] = float(label_0.argmax().item())
                    except Exception:
                        pass
                        
                    try:
                        # Logits Stats
                        lg = logits_fused.view(-1)
                        probe_b_metrics["tracer/step0/logits_mean"] = lg.mean().item()
                        probe_b_metrics["tracer/step0/logits_std"] = lg.std().item()
                        probe_b_metrics["tracer/step0/logits_max"] = lg.max().item()
                        
                        if fused_source_label is not None:
                             mask_0 = (fused_batch == 0)
                             lg0 = logits_fused.view(-1)[mask_0]
                             probe_b_metrics["tracer/step0/g0_logits_mean"] = lg0.mean().item()
                             probe_b_metrics["tracer/step0/g0_logits_max"] = lg0.max().item()
                    except Exception:
                        pass

            if self.use_physics_bias and 'bias' in physics_ctx:
                logits_fused = logits_fused - self.lambda_physics_bias * physics_ctx['bias'].view_as(logits_fused)
            
            last_logits = logits_fused
            
            # 7. Action Selection (Protocol: Top1 + K-1)
            # Use dynamic K if available
            action_k = current_action_k # Sync with FoV
            
            # [Bridge Probe] Check mask before action
            if step == 0 and not inference_mode:
                 probe_b_metrics["probeB/mask_sum_before_action"] = acc_mask_local.sum().item()
                 
                 # [Probe Logits] Inspect Raw Logits
                 l_mean = logits_fused.mean().item()
                 l_std = logits_fused.std().item()
                 l_min = logits_fused.min().item()
                 l_max = logits_fused.max().item()
                 probe_b_metrics["probeB/logits_fused_mean"] = l_mean
                 probe_b_metrics["probeB/logits_fused_std"] = l_std
                 probe_b_metrics["probeB/logits_fused_min"] = l_min
                 probe_b_metrics["probeB/logits_fused_max"] = l_max

            if inference_mode:
                # Inference Strategy: Top-K Selection
                pass
            
            confirmation_mask = torch.zeros_like(logits_fused)
            temp_logits = logits_fused.view(-1).clone()
            
            # [Probe B] Trace Temp Logits degeneration
            if step == 0 and not inference_mode:
                probe_b_metrics["probeB/temp_logits_min"] = temp_logits.min().item()
                probe_b_metrics["probeB/temp_logits_max"] = temp_logits.max().item()
                probe_b_metrics["probeB/temp_logits_mean"] = temp_logits.mean().item()
                probe_b_metrics["probeB/temp_logits_std"] = temp_logits.std().item()
                probe_b_metrics["probeB/action_k"] = float(action_k)
            
            if action_k > 1:
                # [Probe B] EpisodeStepper.run_scenario (Usage Point)
                if step == 0 and not inference_mode and fused_batch[0] == 0:
                     pass

                # [DEBUG] Check masking
                # if step == 0:
                #    masked_nodes = (acc_mask_local.view(-1) > 0.5).nonzero()
                #    print(f"[Stepper] Step {step}: Masked Nodes: {masked_nodes.view(-1).tolist()}")

                temp_logits[acc_mask_local.view(-1) > 0.5] = -float('inf')
                # Reasoner uses UNMASKED logits_fused below.
                temp_logits[acc_mask_local.view(-1) > 0.5] = -float('inf')
            
            step_selected_indices = []
            y_action = None # [Refactor] Store gradient-carrying action

            # [DEBUG] Force Target Selection at Step 0
            # Ensure this logic only runs when a debug flag is active or temporarily modify the code as requested.
            # The modification should overwrite `step_selected_indices` with the ground truth target indices.
            forced_target_selection = False
            # [SSOT] Oracle Guidance Logic with Annealing
            debug_force_target = getattr(self.cfg.life_support, 'enable_oracle_guidance', False)
            oracle_anneal = getattr(self.cfg.life_support, 'oracle_anneal', {})
            
            oracle_prob = 1.0
            is_annealing = False
            
            if oracle_anneal.get('enabled', False):
                is_annealing = True
                start_p = oracle_anneal.get('start_prob', 1.0)
                end_p = oracle_anneal.get('end_prob', 0.0)
                min_p = oracle_anneal.get('min_prob', 0.0)
                total_steps = oracle_anneal.get('total_steps', 1000)
                mode = oracle_anneal.get('mode', 'step')
                
                # Determine progress
                current_progress = 0
                if mode == 'epoch':
                    # We need current epoch. Not directly available in Stepper unless passed.
                    # But we have `step` which is rollout step.
                    # We need global context.
                    # Assuming `static_ctx` or `model` has epoch info?
                    # Phase45Model doesn't store epoch.
                    # BUT `train_phase4_end2end.py` loop calls model with `tau`.
                    # Let's use `tau` as proxy? No.
                    
                    # Fallback: use global_step if available, or just rollout step (wrong).
                    # Actually `step` here is rollout step (0..9).
                    # Annealing should be over training duration.
                    
                    # Hack: We can inject `epoch` or `global_step` into `dynamic_state` or `static_ctx` from training loop.
                    # `train_phase4_end2end.py` calls `model(batch, ...)`
                    # Phase45Model calls `stepper.run_scenario(...)`
                    # We can add `epoch` arg to `run_scenario` in `Phase45Model`.
                    # But Phase45Model signature is fixed?
                    # Let's check `train_phase4_end2end.py`: `out = model(batch, ... tau=tau ...)`
                    # We can pass `epoch` via kwargs if model accepts it.
                    pass
                else:
                    # Default to step-based if we can't get epoch?
                    # Or just use `tau` which is already annealed?
                    # If we use `tau`, we couple oracle to temperature.
                    pass
                
                # [Fix] Access Global Step from static_ctx if available
                # In `train_phase4_end2end.py`, we don't pass global step to model.
                # However, we can use `tau` as a progress indicator!
                # tau goes from start -> end.
                # progress = (tau_start - tau) / (tau_start - tau_end)
                
                # Better: Let's assume `enable_oracle_guidance` is the Master Switch.
                # If True, we use Oracle.
                # If Anneal is True, we flip a coin based on probability.
                
                # Calculate probability
                # We need an external counter.
                # For this "Minimum Change" task, let's use `tau` as the clock.
                # tau is passed to `run_scenario`.
                # Anneal Prob = P(tau).
                
                # If we don't want to depend on tau, we need to pass epoch.
                # But let's look at `train_phase4_end2end.py`...
                # It calls `model(batch, ... tau=tau ...)`
                
                # Let's use a simpler approach:
                # If oracle_anneal is enabled, we assume `debug_force_target` is True initially.
                # We modify `debug_force_target` based on probability.
                
                # To get progress, we can use `self.model.training_step_count` if it exists? No.
                
                # Let's use `tau` passed to `run_scenario` (line 55).
                # `run_scenario` has `tau` argument.
                # We can define schedule based on tau.
                # P(Oracle) = (tau - tau_end) / (tau_start - tau_end).
                # This aligns Oracle decay with Temperature decay.
                # Simple and robust.
                
                # Get Tau Params from Config to normalize
                t_start = getattr(self.cfg.training.annealing, 'tau_start', 1.5)
                t_end = getattr(self.cfg.training.annealing, 'tau_end', 0.1)
                
                if abs(t_start - t_end) > 1e-6:
                    progress = 1.0 - (tau - t_end) / (t_start - t_end)
                    progress = max(0.0, min(1.0, progress)) # 0.0 (Start) -> 1.0 (End)
                    
                    # We want Prob to go from Start_P to End_P
                    oracle_prob = start_p - progress * (start_p - end_p)
                    oracle_prob = max(min_p, oracle_prob)
                else:
                    oracle_prob = start_p
                
                # Flip Coin
                import random
                if random.random() > oracle_prob:
                    debug_force_target = False
            
            # Log Oracle State
            if not inference_mode and step == 0:
                 probe_b_metrics["meta/oracle_enabled"] = 1.0 if debug_force_target else 0.0
                 probe_b_metrics["meta/oracle_prob"] = oracle_prob
                 probe_b_metrics["meta/oracle_anneal_active"] = 1.0 if is_annealing else 0.0
            
            if debug_force_target and step == 0 and has_y and fused_source_label is not None:
                # Force selection of ground truth targets
                target_mask = (fused_source_label.view(-1) > 0.5)
                target_indices = target_mask.nonzero().view(-1)
                
                if target_indices.numel() > 0:
                    step_selected_indices = [target_indices]
                    # Reset confirmation mask
                    confirmation_mask.fill_(0.0)
                    confirmation_mask[target_indices] = 1.0
                    forced_target_selection = True
                    # y_action remains None (no gradient for forced step)

            # Logic Split: Greedy vs Nav-Guided
            if forced_target_selection:
                pass
            elif action_policy == 'nav_guided':
                # 1. Verification Step (Top-1 Reasoner)
                # [CRITICAL FIX] Reasoner must be able to select visited nodes (Triggers/Target).
                # Use raw logits_fused instead of masked temp_logits.
                m, reasoner_idx = scatter_max(logits_fused.view(-1), fused_batch, dim=0, dim_size=curr_batch_size)
                valid_selection = (m > -1e9) & active_mask_t
                
                if valid_selection.any():
                    selected_nodes = reasoner_idx[valid_selection]
                    
                    if selected_nodes.numel() > 0:
                         mask_valid_idx = (selected_nodes >= 0) & (selected_nodes < temp_logits.size(0))
                         selected_nodes = selected_nodes[mask_valid_idx]

                    if selected_nodes.numel() > 0:
                        confirmation_mask[selected_nodes] = 1.0
                        temp_logits[selected_nodes] = -float('inf')
                        step_selected_indices.append(selected_nodes)
                
                # 2. Exploration Steps (K-1 from Navigator)
                    expl_logits[acc_mask_local.view(-1) > 0.5] = -float('inf')
                # The user requirement was to refactor Navigator sampling mechanism.
                # If we are in 'nav_guided', we should use Navigator's output if available.
                # But 'nav_guided' manually iterates.
                # For now, I will assume 'nav_guided' stays manual/greedy for Navigator part 
                # UNLESS nav_out has 'y_action' AND we want to use it?
                # The prompt implies we are refactoring "Navigator's sampling mechanism".
                # 'nav_guided' uses Navigator for K-1.
                # If I use 'y_action', I need to mask out the Top-1 Reasoner choice first?
                # This is complex. The Gumbel ST samples K nodes.
                # If we want 1 Reasoner + (K-1) Navigator, we should ask Navigator for K-1 samples.
                # I passed `action_k` to Navigator via `nav_state['k']`.
                # If policy is 'nav_guided', maybe I should pass `action_k - 1`?
                # But I passed `action_k`.
                # Let's keep 'nav_guided' legacy logic for now to respect "Constraints: Absolute not destroy Phase 4.5 dual agent logic".
                # The prompt specifically mentioned "Task 1: 重写 Sampler" and "Task 2: 改造...对接".
                # "topk_indices 继续用于更新... 必须收集 y_action".
                # If `nav_guided` is used, we might collect `nav_logits` but `y_action` might not be fully representative of the HYBRID action.
                # But wait, `learned` policy IS the one that uses Navigator fully.
                # The prompt mentions: "Navigator 当前使用的是 based topk_wo_replacement... 导致...".
                # This usually implies the `learned` or pure navigator mode, OR the component used in `nav_guided`.
                # I will focus on `learned` policy refactoring first.
                
                if action_k > 1 and nav_logits is not None:
                    expl_logits = nav_logits.view(-1).clone()
                    # [CRITICAL FIX] Navigator MUST mask visited nodes to avoid loops.
                    # Use unified valid_mask_final to enforce sampling space (Feasible & Unvisited)
                    expl_logits[~valid_mask_final] = -float('inf')
                    expl_logits[confirmation_mask.view(-1) > 0.5] = -float('inf')
                    
                    for _ in range(action_k - 1):
                        m, idx = scatter_max(expl_logits, fused_batch, dim=0, dim_size=curr_batch_size)
                        valid_selection = (m > -1e9) & active_mask_t
                        if not valid_selection.any(): break
                        
                        selected_nodes = idx[valid_selection]
                        
                        if selected_nodes.numel() > 0:
                             mask_valid_idx = (selected_nodes >= 0) & (selected_nodes < expl_logits.size(0))
                             selected_nodes = selected_nodes[mask_valid_idx]
                        
                        if selected_nodes.numel() > 0:
                            confirmation_mask[selected_nodes] = 1.0
                            expl_logits[selected_nodes] = -float('inf')
                            step_selected_indices.append(selected_nodes)
            
            elif action_policy == 'learned':
                # [Refactor] Dynamic Budget Allocation (Cognitive State Machine)
                # Task 1: Calculate Cognitive State Metrics
                logit_max_abs = getattr(self.cfg.training, 'logit_max_abs', 20.0)
                eps_prob = getattr(self.cfg.training, 'eps_prob', 1e-8)
                
                temp_logits = torch.clamp(temp_logits, min=-logit_max_abs, max=logit_max_abs)
                probs = gnn_softmax(temp_logits, fused_batch)
                
                # 1. Graph Max Prob (Confidence)
                p_max, _ = scatter_max(probs, fused_batch, dim=0, dim_size=curr_batch_size)
                
                # 2. Graph Normalized Entropy (Confusion)
                log_p = torch.log(probs + eps_prob)
                entropy_per_node = -probs * log_p
                h_sum = scatter_sum(entropy_per_node, fused_batch, dim=0, dim_size=curr_batch_size)
                
                # Calculate N_graph for normalization
                ones = torch.ones_like(probs)
                n_graph = scatter_sum(ones, fused_batch, dim=0, dim_size=curr_batch_size)
                h_norm = h_sum / torch.log(n_graph + eps_prob) # Normalized Entropy [0, 1]
                
                # Task 2: Phase Mask Generation
                # Thresholds from Config
                csm_cfg = getattr(self.cfg.model, 'cognitive_state_machine', {})
                # Handle DictConfig or dict
                if hasattr(csm_cfg, 'get'):
                    p1_h = csm_cfg.get('phase1_h_threshold', 0.8)
                    p1_p = csm_cfg.get('phase1_p_threshold', 0.3)
                    p3_h = csm_cfg.get('phase3_h_threshold', 0.4)
                    p3_p = csm_cfg.get('phase3_p_threshold', 0.8)
                else:
                    # Fallback if not dict-like (e.g. object access)
                    p1_h = getattr(csm_cfg, 'phase1_h_threshold', 0.8)
                    p1_p = getattr(csm_cfg, 'phase1_p_threshold', 0.3)
                    p3_h = getattr(csm_cfg, 'phase3_h_threshold', 0.4)
                    p3_p = getattr(csm_cfg, 'phase3_p_threshold', 0.8)
                
                # Phase 1 (Explore): H > 0.8 or P < 0.3 -> Nav=3, Rea=0
                # Phase 3 (Greedy): H <= 0.4 or P > 0.8 -> Nav=0, Rea=3
                # Phase 2 (Hybrid): Else -> Nav=2, Rea=1
                
                phase1_mask = (h_norm > p1_h) | (p_max < p1_p)
                phase3_mask = (h_norm <= p3_h) | (p_max > p3_p)
                phase2_mask = (~phase1_mask) & (~phase3_mask)
                
                # Determine budgets per graph
                # Nav Budget: P1=3, P2=2, P3=0
                # Rea Budget: P1=0, P2=1, P3=3
                # Total K=3
                
                # We need to route actions.
                # Since 'topk' cannot handle variable K in batch easily without loops,
                # we use "Masked Muting".
                # Both Agent output K=3 candidates. We filter them.
                
                # 1. Reasoner Actions (K=3)
                # We perform top-3 globally (per graph logic needed)
                # Iterative scatter_max for Reasoner K=3
                rea_candidates = []
                # [CRITICAL FIX] Reasoner sees all nodes (Unmasked).
                temp_rea_logits = logits_fused.view(-1).clone()
                temp_rea_logits = torch.clamp(temp_rea_logits, min=-logit_max_abs, max=logit_max_abs)
                
                for _ in range(3):
                    m, idx = scatter_max(temp_rea_logits, fused_batch, dim=0, dim_size=curr_batch_size)
                    # Mask invalid
                    valid = (m > -1e9)
                    if not valid.any(): break
                    
                    # Store
                    rea_candidates.append(idx) # [B] indices
                    
                    # Mask out for next iteration
                    # Need to handle valid indices only
                    valid_idx = idx[valid]
                    if valid_idx.numel() > 0:
                        # Safety check for bounds
                        valid_idx = valid_idx[valid_idx < temp_rea_logits.size(0)]
                        temp_rea_logits[valid_idx] = -float('inf')
                
                # rea_candidates is list of [B] tensors. 
                # rea_candidates[0] is Top-1, [1] is Top-2, etc.
                
                # 2. Navigator Actions (K=3)
                # Assuming Sampler returns K=3 selected indices (flattened)
                # We need to map them back to graphs to apply phase mask.
                # This is tricky because Sampler output format is flattened [Total_Selected].
                # If we use GumbelTopKSTSampler with K=3, it returns ~3*B indices.
                # We need to know which graph they belong to.
                # 'nav_out' has 'selected_indices'.
                
                # Wait, if we use `action_k=3` in `forward`, Navigator returns 3 candidates.
                # But we might want 0, 2, or 3.
                # If we ask for 3, we can discard some.
                
                nav_indices_flat = nav_out.get('selected_indices')
                nav_y_action = nav_out.get('y_action') # [N] Soft Action
                
                # [Fix] Assign to outer scope y_action for loss calculation
                if nav_y_action is not None:
                    y_action = nav_y_action

                # Task 3: Action Routing & Fusion
                # We construct the final 'step_selected_indices' based on budgets.
                
                # Expand masks to match batch size [B]
                # phaseX_mask is [B]
                
                final_indices_list = []
                
                # --- Phase 1 (Nav=3, Rea=0) ---
                # Take all 3 from Navigator
                # How to identify which nav indices belong to Phase 1 graphs?
                # We can use `fused_batch[nav_indices_flat]` to get graph_id.
                if nav_indices_flat is not None and nav_indices_flat.numel() > 0:
                    nav_batch = fused_batch[nav_indices_flat]
                    p1_nodes = phase1_mask[nav_batch]
                    if p1_nodes.any():
                        final_indices_list.append(nav_indices_flat[p1_nodes])
                
                # --- Phase 3 (Nav=0, Rea=3) ---
                # Take all 3 from Reasoner
                # rea_candidates[i] is [B]. We mask with phase3_mask.
                for i in range(len(rea_candidates)):
                    idx_b = rea_candidates[i]
                    # Filter for Phase 3 graphs
                    # idx_b contains indices for all graphs.
                    # We only keep those where phase3_mask is True AND idx is valid
                    safe_idx_b = idx_b.clamp(0, temp_logits.size(0) - 1)
                    
                    # Need to check sizes before indexing
                    # phase3_mask is [B]. idx_b is [B].
                    # If batch size mismatch (e.g. some graphs dropped), this fails.
                    # But here fused_batch, phase3_mask are derived from same batch.
                    
                    # Safety check for CUDA assertions
                    if idx_b.size(0) != phase3_mask.size(0):
                        # Should not happen in normal flow
                        continue
                        
                    p3_valid = phase3_mask & (idx_b >= 0) & (idx_b < temp_logits.size(0)) & (temp_logits[safe_idx_b] > -1e9)
                    if p3_valid.any():
                        final_indices_list.append(idx_b[p3_valid])
                
                # --- Phase 2 (Nav=2, Rea=1) ---
                # Nav Top-2 + Rea Top-1
                
                # Rea Top-1
                if len(rea_candidates) > 0:
                    idx_b = rea_candidates[0]
                    p2_valid = phase2_mask & (idx_b >= 0) & (idx_b < temp_logits.size(0))
                    if p2_valid.any():
                        final_indices_list.append(idx_b[p2_valid])
                        
                # Nav Top-2
                # We need to identify Top-1 and Top-2 from Navigator.
                # The Sampler returns flattened indices. Order is likely:
                # Graph 0 (K=3), Graph 1 (K=3)... OR Top-1(All), Top-2(All)...
                # My implementation of `GumbelTopKSTSampler` did:
                # for _ in range(k): scatter_max ... append ... cat
                # So it is Top-1(All), Top-2(All), Top-3(All) concatenated.
                # This structure allows easy slicing!
                
                # Batch size B.
                # Sampler returns `selected_indices` of size ~ B*K.
                # Structure: [Top1_G0, Top1_G1... Top1_GB, Top2_G0... ]
                # Actually `scatter_max` returns `idx` of shape [B].
                # So `selected_indices_list` in Sampler is [Indices_Top1, Indices_Top2, Indices_Top3].
                # Each element is [B] (sparse if some graphs exhausted).
                # Wait, Sampler implementation:
                # `selected_indices_list.append(valid_idx)`
                # `valid_idx` is subset of [B].
                # So it's NOT structured as [B]*K. It's a flat list of valid selections.
                # This makes it hard to distinguish Rank 1 vs Rank 3.
                
                # CRITICAL FIX: To support Phase 2 routing, Sampler must return structured indices
                # or we accept that we take ANY 2 from Navigator.
                # Since Gumbel Top-K is unordered (or ordered by noise), taking any 2 is fine.
                # But we need to know which graph they belong to.
                
                if nav_indices_flat is not None and nav_indices_flat.numel() > 0:
                    nav_batch = fused_batch[nav_indices_flat]
                    p2_nodes = phase2_mask[nav_batch]
                    
                    if p2_nodes.any():
                        # We need to iterate over the TENSOR `nav_indices_flat`
                        # Moving to CPU for logic might be slow? 3*128 = 384 iters. Fast.
                        
                        nav_indices_cpu = nav_indices_flat.cpu()
                        nav_batch_cpu = fused_batch[nav_indices_flat].cpu()
                        p2_mask_cpu = phase2_mask.cpu()
                        phase3_mask_cpu = phase3_mask.cpu() # Also need phase3 mask
                        
                        keep_mask_cpu = torch.ones(nav_indices_flat.size(0), dtype=torch.bool)
                        
                        p2_counts = torch.zeros(curr_batch_size, device='cpu') # Count per graph
                        
                        for i, (idx, b) in enumerate(zip(nav_indices_cpu, nav_batch_cpu)):
                            if p2_mask_cpu[b]:
                                if p2_counts[b] < 2:
                                    p2_counts[b] += 1
                                else:
                                    keep_mask_cpu[i] = False
                            # For P1, we keep all 3 (default)
                            # For P3, we discard all (handled below)
                            if phase3_mask_cpu[b.item()]:
                                keep_mask_cpu[i] = False
                                
                        keep_mask = keep_mask_cpu.to(nav_indices_flat.device)
                        final_indices_list.append(nav_indices_flat[keep_mask])
                
                # Flatten final list
                if final_indices_list:
                    all_selected = torch.cat(final_indices_list, dim=0)
                    
                    # Apply to confirmation_mask
                    if all_selected.numel() > 0:
                        # Safety bound check
                        mask_valid_idx = (all_selected >= 0) & (all_selected < logits_fused.size(0))
                        all_selected = all_selected[mask_valid_idx]
                        
                        confirmation_mask[all_selected] = 1.0
                        step_selected_indices.append(all_selected) # Keep as tensor, not list of tensors
                
                # Task 4: Gradient Muting
                    # We need to zero out y_action gradients for Phase 3 graphs (and 3rd choice of Phase 2).
                    # y_action is [N]. We can generate a node-level mask.
                    if y_action is not None:
                        # Node-level phase mask
                        node_phase3 = phase3_mask[fused_batch]
                        # Mask out P3 nodes from y_action (detach gradient)
                        # y_action[node_phase3] = y_action[node_phase3].detach()
                        # Or set to 0? If we set to 0, loss is 0. Correct.
                        # But we want to keep the "Soft" probability for logging?
                        # The prompt says: "Masked Muting... ensuring gradient flow correct".
                        # If Nav is muted (P3), it shouldn't contribute to gradient.
                        # Multiply y_action by ~node_phase3
                        
                        # For P2, we only muted the 3rd choice. 
                        # But y_action is a dense tensor [N] (soft).
                        # It represents the full distribution.
                        # If we only execute 2 nodes, but y_action reflects 3, 
                        # the loss will try to align 3 nodes.
                        # This is acceptable (soft guidance).
                        # Strictly, we should mute P3 graphs completely.
                        
                        # [Fix] Broadcasting bug: y_action is [N], node_phase3 is [N].
                        # Do NOT unsqueeze, or ensure both are compatible.
                        y_action = y_action * (~node_phase3).float()


            else:
                # Default Greedy (All from Reasoner)
                # [CRITICAL FIX] Greedy uses Unmasked Logits (Allows hitting Triggers)
                temp_greedy_logits = logits_fused.view(-1).clone()
                
                for _ in range(action_k):
                    m, reasoner_idx = scatter_max(temp_greedy_logits, fused_batch, dim=0, dim_size=curr_batch_size)
                    
                    valid_selection = (m > -1e9) & active_mask_t
                    
                    if not valid_selection.any():
                        break
                        
                    selected_nodes = reasoner_idx[valid_selection]
                    
                    if selected_nodes.numel() > 0:
                        if selected_nodes.max() >= temp_greedy_logits.size(0) or selected_nodes.min() < 0:
                             mask_valid_idx = (selected_nodes >= 0) & (selected_nodes < temp_greedy_logits.size(0))
                             selected_nodes = selected_nodes[mask_valid_idx]
                    
                    if selected_nodes.numel() > 0:
                        confirmation_mask[selected_nodes] = 1.0
                        temp_greedy_logits[selected_nodes] = -float('inf')
                        step_selected_indices.append(selected_nodes)
                
                # Flatten step_selected_indices if multiple (e.g. iterative greedy)
                if len(step_selected_indices) > 0:
                    if isinstance(step_selected_indices[0], torch.Tensor):
                        # If list of tensors, cat them?
                        # In loop above, we append.
                        # Wait, we want `step_selected_indices` to be a single tensor for logging?
                        # Or list of tensors?
                        # The `trajectory_data` expects `selected_indices` to be accessible.
                        # If we have multiple selections in one step (K>1), we should cat them.
                        pass
            
            # Consolidate selections for this step
            final_step_indices = None
            if len(step_selected_indices) > 0:
                final_step_indices = torch.cat(step_selected_indices, dim=0)
            
            # Update Accumulation (Physical Level)
            A_total_mask = (confirmation_mask > 0.5)
            
            # [Task 3] Track Newly Revealed Count & Poison Hits
            if not inference_mode:
                new_nodes_step = A_total_mask.float().sum().item()
                probe_b_metrics[f"rollout/step{step}/new_revealed"] = new_nodes_step
                
                # Calculate Poison Hits in this step
                # A_total_mask is [N]. x_nav_next is updated below, but we can check x_raw here or wait.
                # Actually, x_nav_next gets updated based on A_total_mask.
                # But we want to know if the NEWLY revealed nodes are poisonous.
                # We need to access the GROUND TRUTH poison at current t_sim.
                # We already calculated `poison_at_t` [N] in the update block below.
                # Let's move the metric calculation to AFTER the update.
                
            # [Refactor] Update local mask
            acc_mask_local = torch.max(acc_mask_local, A_total_mask.float())
            
            # === DYNAMIC FEATURE UPDATE (User Correction) ===
            x_nav_next = x_nav.clone()
            
            # ... (Update logic) ...
            
            # 1. Update Ch3 (Revealed Mask)
            # Use acc_mask_local
            accumulated_mask_view = acc_mask_local[inverse_indices]
            x_nav_next[:, 3] = torch.max(x_nav_next[:, 3], accumulated_mask_view.view(-1))
            
            # 2. Update Ch0 (Signal) and Ch1 (Poison) based on Ch3 and t_sim
            poison_at_t_current = None # Store for metrics
            
            if x_raw is not None:
                # ... (Time calculation) ...
                view_batch = static_ctx['view_batch']
                step_seconds_local = delta_t * 60.0
                
                trigger_time_step_batch = static_ctx.get('trigger_time_step')
                if trigger_time_step_batch is None:
                     device = x_raw.device if x_raw is not None else h_fused.device
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
                
                poison_at_t_current = poison_at_t # Save for metrics

                current_revealed = (x_nav_next[:, 3] > 0.5)
                x_nav_next[:, 0] = torch.where(current_revealed, signal_at_t, x_nav_next[:, 0])
                x_nav_next[:, 1] = torch.where(current_revealed, poison_at_t, x_nav_next[:, 1])
            
            # ... (Rest of update) ...
            decay_rate = getattr(self.cfg.physics.env, 'freshness_decay', 0.8)
            A_total_mask_view = A_total_mask[inverse_indices].view(-1).float()
            x_nav_next[:, 2] = x_nav_next[:, 2] * decay_rate
            x_nav_next[:, 2] = torch.max(x_nav_next[:, 2], A_total_mask_view)
            
            # 4. Update Ch4 (Anchors)
            newly_visited = A_total_mask.view(-1).bool()
            
            if newly_visited.any():
                physical_poison = scatter_max(x_nav_next[:, 1], inverse_indices, dim=0, dim_size=h_fused.size(0))[0]
                physical_threshold = getattr(self.cfg.physics.env, 'confirmation_threshold', 0.5)
                is_poison = (physical_poison > physical_threshold)
                new_anchors_val = torch.where(is_poison, torch.tensor(1.0, device=h_fused.device), torch.tensor(-1.0, device=h_fused.device))
                
                causal_anchors = causal_anchors.clone()
                causal_anchors[newly_visited] = new_anchors_val[newly_visited].unsqueeze(-1)
                anchor_times = anchor_times.clone()
                anchor_times[newly_visited] = t_sim[fused_batch[newly_visited]].unsqueeze(-1)

            # [Metric] Calculate Poison Hits in this step (Nav Efficacy)
            if not inference_mode and poison_at_t_current is not None:
                # Newly revealed nodes that are poisonous
                # Note: A_total_mask is fused level. poison_at_t_current is raw level [N].
                # We need to map A_total_mask to raw level.
                newly_revealed_raw = A_total_mask[inverse_indices].view(-1).bool()
                
                # 1. Environment Ratio (Difficulty Baseline)
                # Calculate ratio of poisonous nodes in the ENTIRE graph at this time step
                # poison_at_t_current is [N] binary mask of all nodes
                # We compute per-graph mean using fused_batch mapped by inverse_indices
                batch_mapping = fused_batch[inverse_indices]
                env_poison_count = scatter_sum(poison_at_t_current, batch_mapping, dim=0, dim_size=curr_batch_size)
                env_total_count = scatter_sum(torch.ones_like(poison_at_t_current), batch_mapping, dim=0, dim_size=curr_batch_size)
                env_poison_ratio = env_poison_count / (env_total_count + 1e-8)
                probe_b_metrics[f"env/step{step}/poison_ratio"] = env_poison_ratio.mean().item()
                
                # 2. Sampled Ratio (Navigator Performance)
                poison_hits_mask = (poison_at_t_current > 0.5) & newly_revealed_raw
                
                # Per-graph hits
                hits_per_graph = scatter_sum(poison_hits_mask.float(), batch_mapping, dim=0, dim_size=curr_batch_size)
                attempts_per_graph = scatter_sum(newly_revealed_raw.float(), batch_mapping, dim=0, dim_size=curr_batch_size)
                
                # Update Global Accumulator
                total_poison_hits_per_graph += hits_per_graph
                
                # Calculate Hit Rate (Sampled Ratio)
                # Avoid div by zero for graphs with no attempts
                valid_attempts_mask = (attempts_per_graph > 0)
                if valid_attempts_mask.any():
                    sampled_ratios = hits_per_graph[valid_attempts_mask] / attempts_per_graph[valid_attempts_mask]
                    avg_sampled_ratio = sampled_ratios.mean().item()
                else:
                    avg_sampled_ratio = 0.0
                
                probe_b_metrics[f"nav/step{step}/sampled_poison_ratio"] = avg_sampled_ratio
                probe_b_metrics[f"nav/step{step}/total_attempts_mean"] = attempts_per_graph.mean().item()

                # Legacy Scalar Metrics (Total Sum)
                poison_hits_total = hits_per_graph.sum().item()
                total_attempts_total = attempts_per_graph.sum().item()
                
                probe_b_metrics[f"rollout/step{step}/poison_hits_total"] = poison_hits_total
                probe_b_metrics[f"rollout/step{step}/total_attempts_total"] = total_attempts_total

            causal_anchors_view = causal_anchors[inverse_indices]
            x_nav_next[:, 4] = causal_anchors_view.view(-1)
            x_nav = x_nav_next
            
            # === DYNAMIC RE-EMBEDDING ===
            x_nav_visible = x_nav
            view_batch = static_ctx['view_batch']
            x_nav_gated, _ = self.model.dynamic_gate(x_nav_visible, view_batch) 
            
            call_kwargs = {}
            import inspect
            sig = inspect.signature(self.model.navigator_module.backbone.forward)
            if 'memory_state' in sig.parameters:
                call_kwargs['memory_state'] = None
            if 'batch' in sig.parameters:
                call_kwargs['batch'] = view_batch

            with torch.no_grad():
                h_nav = self.model.navigator_module.backbone(x_nav_gated, curr_edge_index, **call_kwargs)
                if isinstance(h_nav, tuple): 
                    h_nav = h_nav[0]
            
            with torch.no_grad():
                h_fused_new, _, _, _, _ = self.model.fusion(
                    h_nav, 
                    view_batch, 
                    static_ctx['batch_n_id'], 
                    static_ctx['batch_scenario_id'] 
                )
            
            h_fused = h_fused_new
            # [Fix Task 3] Do NOT overwrite physical state x_nav with gated features!
            # x_nav = x_nav_gated 
            
            # [Task 1] Calculate Post-Action Gain & Loss
            nav_gain_loss = None
            nav_reward = None # [Task 2] Capture Raw Reward for RL Returns
            
            if not inference_mode and y_action is not None and has_y:
                # 1. Run Reasoner on h_fused_next (h_fused_new)
                # Need to use 'curr_edge_index' (updated? No, update happens at start of next loop)
                # Technically, we should use the updated topology if we updated causal anchors.
                # But let's use the same 'curr_edge_index' and 'physics_ctx' as previous step for consistency?
                # The prompt says "utilize h_fused_next... get reasoner_probs_t_plus_1".
                # Reasoner needs: h_fused, causal_anchors, accumulated_mask.
                # These are all updated now (lines 642, 597, 677).
                
                reasoner_state_next = {
                    'h_fused': h_fused_new, # Updated
                    'causal_anchors': causal_anchors, # Updated
                    'accumulated_mask': acc_mask_local, # Updated
                    'memory_state': reasoner_memory_state # Use current memory state (or update? Memory update happens in forward pass)
                    # We shouldn't update memory here, just peek.
                }
                
                # We need to temporarily use the model in eval mode? No, just no_grad?
                # We need gradients for y_action? No, y_action is already computed.
                # We need gradients for reasoner? No, we detach reward.
                # So reasoner call should be no_grad.
                
                with torch.no_grad():
                    reasoner_out_next = self.model.reasoner_module(reasoner_state_next, temp_graph, physics_ctx)
                    logits_next = reasoner_out_next['logits']
                    probs_next = gnn_softmax(logits_next.view(-1), fused_batch)
                    
                    prob_after_target = scatter_sum(probs_next * fused_source_label.view(-1), fused_batch, dim=0, dim_size=curr_batch_size)
                
                # 2. Calculate Gain Reward
                # [Fix] Reward must be based on Source Label
                # We need to map Gain to Source Label alignment.
                # If target is Source, Gain is Positive.
                # If target is Not Source, Gain is Negative?
                # Usually we just use `prob_after_target`.
                # If `prob_after_target` increases, we get positive reward.
                # But if `prob_after_target` is 0.0, we get 0.0.
                
                # [Fix] Time Penalty
                # Reward = Gain - Step_Cost
                # step_cost = 0.05
                # This encourages finding target EARLIER.
                
                step_cost = 0.05 # [Param]
                gain_reward = (prob_after_target - prob_before_target).detach() - step_cost # [B]
                nav_reward = gain_reward
                
                # 3. Calculate Step Loss
                # y_action is [N] (soft selection probability/mask).
                # We need to broadcast gain_reward [B] to nodes [N].
                gain_reward_nodes = gain_reward[fused_batch]
                
                # Handle possible size mismatch if y_action has different size than nodes
                # y_action comes from nav_out['y_action']
                # If y_action is [N], and fused_batch is [N], then it's fine.
                # But check if y_action is None or different size
                
                if y_action is not None and y_action.size(0) == gain_reward_nodes.size(0):
                    # Loss = - sum(y_action * gain_reward)
                    # We calculate per-graph loss for ModularLossEngine
                    step_loss_per_node = - (y_action * gain_reward_nodes)
                    step_loss_per_graph = scatter_sum(step_loss_per_node, fused_batch, dim=0, dim_size=curr_batch_size) # [B]
                    nav_gain_loss = step_loss_per_graph # [B]
                else:
                    nav_gain_loss = torch.zeros(curr_batch_size, device=h_fused.device)
            
            # Track Budget
            # [Fix] budget_used should be total UNIQUE nodes visited
            # Previously: budget_used += step_budget (which is sum of A_total_mask at this step)
            # But A_total_mask is ACCUMULATED mask. So sum(A_total_mask) grows monotonically.
            # If we sum it every step, we are summing the CUMULATIVE count every step!
            # Example: Step 1 (3 nodes), Step 2 (6 nodes).
            # budget_used = 3 + 6 = 9. WRONG. Should be 6.
            
            # Correct Logic: budget_used should be calculated ONCE at the end.
            # Or we track `newly_visited` count.
            
            # Let's fix it by tracking newly visited.
            newly_visited_mask = (A_total_mask > 0.5) & (~(acc_mask_local > 0.5))
            # Wait, acc_mask_local is updated at line 647 BEFORE this.
            # So A_total_mask is subset of acc_mask_local?
            # No, line 647: acc_mask_local = max(acc_mask_local, A_total_mask)
            # So A_total_mask contains ALL visited nodes up to now?
            # Let's check line 646: A_total_mask = (confirmation_mask > 0.5)
            # confirmation_mask is ONLY current step selections.
            # So A_total_mask is CURRENT STEP selections.
            
            # So `budget_used += step_budget` counts current step selections.
            # This seems correct IF `confirmation_mask` only contains NEW selections.
            # But does it?
            # If policy selects already visited node, confirmation_mask will include it.
            # Then we double count.
            
            # To fix "Avg_Total_Samples" being huge:
            # We must count UNIQUE samples.
            # The best way is to sum `acc_mask_local` at the very end of the loop.
            pass
            
            # [Task 3] Success Tracking
            if has_y:
                hits = (A_total_mask.view(-1) > 0.5) & (fused_source_label.view(-1) > 0.5)
                # hits is [N]. Map to [B]
                hits_graph = scatter_max(hits.float(), fused_batch, dim=0, dim_size=curr_batch_size)[0]
                new_success = (hits_graph > 0.5)
                
                # Update first hit step (only if not hit before)
                just_hit = new_success & (~hit_before_t)
                if just_hit.any():
                     first_hit_step[just_hit] = float(step)
                
                # Check if hit was in THIS step's selection (redundant with just_hit? No, just_hit requires ~hit_before)
                # But hit_in_selected_step means "did we pick source this step?"
                # Yes, new_success means exactly that.
                if not inference_mode:
                    probe_b_metrics[f"rollout/step{step}/hit_count"] = new_success.float().sum().item()

                hit_before_t = hit_before_t | new_success
                graph_success = graph_success | new_success
            else:
                new_success = torch.zeros(curr_batch_size, dtype=torch.bool, device=h_fused.device)
            
            t_sim = t_sim + (active_mask_t.float() * delta_t)

            if not inference_mode:
                # [Bridge Probe] Check mask before record
                if step == 0:
                     probe_b_metrics["probeB/mask_sum_before_record"] = acc_mask_local.sum().item()
                     # Force persist to dynamic_state for safety
                     dynamic_state['accumulated_mask'] = acc_mask_local
                     
                     # [Verification Probe] Input Semantics Check
                     if not inference_mode and fused_batch[0] == 0:
                         probe_b_metrics["verify/input_channels"] = x_nav.size(1)
                         probe_b_metrics["verify/ch3_mean"] = x_nav[:, 3].mean().item()
                         if x_nav.size(1) > 7:
                             probe_b_metrics["verify/ch7_mean"] = x_nav[:, 7].mean().item()
                             probe_b_metrics["verify/ch7_max"] = x_nav[:, 7].max().item()
                         else:
                             probe_b_metrics["verify/ch7_exists"] = 0.0
                         
                         # Print Slice
                         # print(f"[Verify] Mode: {getattr(self.cfg.data, 'feature_mode', 'unknown')} | Channels: {x_nav.size(1)}")
                         # print(f"[Verify] First Node: {x_nav[0, :8].tolist()}")

                trajectory_data.append({
                    'reasoner_logits': logits_fused,
                    'nav_logits': nav_logits,
                    'nav_probs': nav_out.get('nav_probs'),
                    'nav_action': y_action, # [Refactor] Capture y_action
                    'nav_candidates': nav_out.get('selected_indices'),
                    'active_mask': active_mask_t,
                    'is_hit': new_success.float() if has_y else torch.zeros_like(active_mask_t).float(),
                    'fused_batch': fused_batch,
                    'fused_source_label': fused_source_label,
                    'hit_prob_surrogate': hit_prob_surrogate,
                    'nav_gain_loss': nav_gain_loss, # [Task 1] Store Step Loss (Myopic)
                    'nav_reward': nav_reward, # [Task 2] Store Raw Reward (for G_t)
                    'h_fused': h_fused, # [Vis] Store h_fused for Saliency
                    'physics_ctx': physics_ctx, # [Vis] Store Physics Context
                    'curr_edge_index': curr_edge_index, # [Vis] Store Edge Index
                    'curr_edge_attr': curr_edge_attr, # [Vis] Store Edge Attr
                    'inverse_indices': inverse_indices, # [Vis] Map fused to raw
                    'reasoner_input_state': vis_reasoner_state, # [Vis] Store Reasoner Input
                    'selected_indices': final_step_indices, # [Vis] Store Selected Indices
                    'dynamic_state': {
                         't_sim': t_sim.clone()
                    }
                })
            else:
                trajectory_data.append({
                    'reasoner_logits': logits_fused,
                    'fused_batch': fused_batch,
                    'fused_source_label': fused_source_label,
                    'hit_prob_surrogate': hit_prob_surrogate,
                    'is_hit': new_success.float() if has_y else torch.zeros_like(active_mask_t).float(),
                    'h_fused': h_fused, # [Vis] Store h_fused for Saliency
                    'physics_ctx': physics_ctx, # [Vis] Store Physics Context
                    'curr_edge_index': curr_edge_index, # [Vis] Store Edge Index
                    'curr_edge_attr': curr_edge_attr, # [Vis] Store Edge Attr
                    'inverse_indices': inverse_indices, # [Vis] Map fused to raw
                    'reasoner_input_state': vis_reasoner_state, # [Vis] Store Reasoner Input
                    'selected_indices': final_step_indices, # [Vis] Store Selected Indices
                    'dynamic_state': {
                         't_sim': t_sim.clone()
                    }
                })
            
            if inference_mode and hit_before_t.all():
                break
        
        # [SSOT Fix] Recalculate Budget Used (Unique Nodes)
        # Use acc_mask_local to get total unique visited nodes
        budget_used = scatter_sum(acc_mask_local.view(-1), fused_batch, dim=0, dim_size=curr_batch_size)
        
        # [Task 3] Finalize Success Metrics
        if not inference_mode:
            probe_b_metrics["probeB/first_hit_step_mean"] = first_hit_step[first_hit_step >= 0].mean().item()
            probe_b_metrics["probeB/hit_rate_final"] = graph_success.float().mean().item()
            probe_b_metrics["probeB/budget_used_mean"] = budget_used.mean().item()
            
            # [Metric] Global Poison Hits
            probe_b_metrics["rollout/global/avg_poison_hits_per_event"] = total_poison_hits_per_graph.mean().item()
            
        if inference_mode:
            pass # print(f"[Stepper DEBUG] Final Budget Used Mean: {budget_used.float().mean().item()}")
        
        return {
            'classification': last_logits[inverse_indices] if last_logits is not None else torch.zeros(curr_batch_size, 2, device=h_fused.device), 
            'trajectory': trajectory_data,
            'final_dynamic_state': {
                'accumulated_mask': acc_mask_local,
                't_sim': t_sim
            },
            'debug_sentinel': 777, # [Probe C2] Sentinel
            'probe_b_metrics': probe_b_metrics, # [Probe B] Return metrics
            'step_metrics': {
                'success': graph_success.float().mean().item(),
                'steps_taken': t_sim.mean().item(), 
                'budget_used': budget_used.mean().item(),
                'raw_success': graph_success.float(),
                'raw_steps': t_sim, 
                'raw_budget': budget_used,
                'raw_predict_hit': predict_hit_at_1.float(),
                'raw_predict_hit_5': predict_hit_at_5.float(), # [Fix] Pass Hit@5
                'raw_predict_hit_valid': predict_hit_valid.float(),
                'raw_rounds': steps_taken, 
                'raw_max_hit_prob': max_hit_prob
            }
        }
