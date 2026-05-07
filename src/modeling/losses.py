import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import softmax as gnn_softmax
from torch_scatter import scatter_mean, scatter_sum
from typing import List, Dict, Tuple, Optional, Any
from src.modeling.evidence.oracle_schedule import resolve_evidence_oracle_schedule
from src.modeling.interfaces.base import LossEngineBase, LossRequirements, TrajectoryStep
from src.modeling.registry import LOSS_REGISTRY

@LOSS_REGISTRY.register("modular_v4_5")
class ModularLossEngine(LossEngineBase):
    """
    Refactored Modular Loss Engine for Phase 4.5.
    Supports Gumbel Top-K Straight-Through Actions.
    """
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        self.weights = config.get('weights', {})
        self.params = config.get('params', {})
        
        # Weights
        self.w_hard = self.weights.get('w_hard', 1.0)
        self.w_soft = self.weights.get('w_soft', 0.3)
        self.w_delta = self.weights.get('w_delta', 0.8)
        self.w_hit = self.weights.get('w_hit', 0.5)
        self.w_mono = self.weights.get('w_mono', 0.1)
        self.w_ent = self.weights.get('w_ent', 0.05)
        self.w_surv = self.weights.get('w_surv', 0.0)
        
        # Params
        self.loss_mode = self.params.get('loss_mode', 'baseline')
        self.gamma = self.params.get('temporal_decay_gamma', 2.0)
        self.temporal_weight_mode = str(self.params.get('temporal_weight_mode', 'late')).lower()
        self.temporal_weight_floor = float(self.params.get('temporal_weight_floor', 0.0))
        self.rho = self.params.get('rho_discount', 0.9)
        self.hit_eps = self.params.get('hit_epsilon', 1e-7)
        self.surv_eps = self.params.get('survival_eps', 1e-6)
        self.surv_use_rho = self.params.get('survival_use_rho', True)
        self.surv_detach_q = self.params.get('survival_detach_q', False)
        self.mono_margin = self.params.get('delta_margin', 0.0)
        self.sigma = self.params.get('label_smoothing_sigma', 1.0)
        self.ent_start = self.params.get('target_entropy_start', 0.8)
        self.ent_end = self.params.get('target_entropy_end', 0.1)
        self.dense_ce = self.params.get('dense_ce', False)
        self.scale_correction = self.params.get('scale_correction', False)
        self.use_old_hit = self.params.get('use_old_hit_surrogate', False)
        self.evidence_oracle_cfg = self.params.get('evidence_oracle', {})

    def requirements(self) -> LossRequirements:
        return {}

    def _masked_mean(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return (x * mask).sum() / (mask.sum() + 1e-9)

    def _temporal_weight(self, step_index: int, total_steps: int) -> float:
        if total_steps <= 0:
            return 1.0
        progress = (float(step_index) + 1.0) / float(total_steps)
        if self.temporal_weight_mode == 'early':
            raw = ((float(total_steps) - float(step_index)) / float(total_steps)) ** float(self.gamma)
        elif self.temporal_weight_mode == 'uniform':
            raw = 1.0
        else:
            raw = progress ** float(self.gamma)
        floor = min(max(float(self.temporal_weight_floor), 0.0), 1.0)
        return floor + (1.0 - floor) * raw

    def _compute_gae(
        self,
        rewards: List[torch.Tensor],
        values: List[torch.Tensor],
        active_masks: List[torch.Tensor],
        gae_lambda: float,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        T = len(rewards)
        advantages: List[torch.Tensor] = [torch.zeros_like(values[0]) for _ in range(T)]
        returns: List[torch.Tensor] = [torch.zeros_like(values[0]) for _ in range(T)]
        gae = torch.zeros_like(values[0])
        next_value = torch.zeros_like(values[0])
        next_nonterminal = torch.zeros_like(values[0])

        for idx in reversed(range(T)):
            nonterminal = active_masks[idx].float()
            delta = rewards[idx] + self.rho * next_value * next_nonterminal - values[idx]
            gae = delta + self.rho * float(gae_lambda) * next_nonterminal * gae
            gae = gae * nonterminal
            advantages[idx] = gae
            returns[idx] = gae + values[idx]
            next_value = values[idx]
            next_nonterminal = nonterminal
        return advantages, returns

    def _step_evidence_state(self, step_data: Dict[str, Any]):
        if 'evidence_state' in step_data:
            return step_data.get('evidence_state')
        reasoner_state = step_data.get('reasoner_input_state')
        if isinstance(reasoner_state, dict):
            return reasoner_state.get('evidence_state')
        return None

    def _compute_support_pairwise_loss(
        self,
        student_scores: torch.Tensor,
        teacher_scores: torch.Tensor,
        batch: torch.Tensor,
        graph_mask: torch.Tensor,
        margin: float,
    ) -> torch.Tensor:
        losses = []
        if batch.numel() == 0:
            return student_scores.new_zeros(())

        num_graphs = int(batch.max().item()) + 1
        for graph_idx in range(num_graphs):
            if graph_mask[graph_idx].item() <= 0.5:
                continue
            node_mask = batch == graph_idx
            if int(node_mask.sum().item()) <= 1:
                continue
            student_g = student_scores[node_mask]
            teacher_g = teacher_scores[node_mask]
            top_idx = int(torch.argmax(teacher_g).item())
            competitor_mask = torch.ones_like(teacher_g, dtype=torch.bool)
            competitor_mask[top_idx] = False
            if not bool(competitor_mask.any()):
                continue
            student_top = student_g[top_idx]
            student_comp = student_g[competitor_mask].max()
            losses.append(F.relu(float(margin) - (student_top - student_comp)))

        if not losses:
            return student_scores.new_zeros(())
        return torch.stack(losses).mean()

    def _compute_alignment_l1(
        self,
        student_scores: torch.Tensor,
        teacher_scores: torch.Tensor,
        batch: torch.Tensor,
        active_mask: torch.Tensor,
    ) -> torch.Tensor:
        node_mask = active_mask[batch].float()
        if node_mask.sum().item() <= 0:
            return student_scores.new_zeros(())
        return ((student_scores - teacher_scores).abs() * node_mask).sum() / node_mask.sum().clamp_min(1.0)

    def _resolve_suspect_oracle_students(
        self,
        evidence_state,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        live_student = evidence_state.suspect_pool.float().view(-1)
        canonical_latent = getattr(evidence_state, 'suspect_canonical_latent', None)
        if canonical_latent is None:
            return live_student, live_student
        canonical_student = self._soft_clamp_unit_interval(canonical_latent.float().view(-1))
        return live_student, canonical_student

    def _resolve_suspect_oracle_latent(
        self,
        evidence_state,
    ) -> torch.Tensor:
        canonical_latent = getattr(evidence_state, 'suspect_canonical_latent', None)
        if canonical_latent is None:
            return evidence_state.suspect_pool.float().view(-1)
        return canonical_latent.float().view(-1)

    def _soft_clamp_unit_interval(self, values: torch.Tensor, beta: float = 64.0) -> torch.Tensor:
        beta = max(float(beta), 1e-6)
        return F.softplus(values, beta=beta) - F.softplus(values - 1.0, beta=beta)

    def _compute_evidence_oracle_loss(
        self,
        trajectory: List[Dict[str, Any]],
        active_mask_list: List[torch.Tensor],
        cfg: Any,
    ):
        device = trajectory[0]['reasoner_logits'].device
        zero = trajectory[0]['reasoner_logits'].sum() * 0.0
        enabled = bool(self.evidence_oracle_cfg.get('enabled', False))
        schedule = resolve_evidence_oracle_schedule(cfg) if enabled else {
            'enabled': 0.0,
            'phase': 'disabled',
            'phase_index': -1.0,
            'progress': 0.0,
            'oracle_factor': 0.0,
            'live_factor': 1.0,
        }
        if not enabled:
            return zero, {}, schedule

        listwise_temperature = max(float(self.evidence_oracle_cfg.get('listwise_temperature', 1.0)), 1e-6)
        support_pairwise_margin = float(self.evidence_oracle_cfg.get('support_pairwise_margin', 0.05))
        support_listwise_weight = float(self.evidence_oracle_cfg.get('support_listwise_weight', 1.0))
        support_pairwise_weight = float(self.evidence_oracle_cfg.get('support_pairwise_weight', 0.25))
        rebuttal_loss_type = str(self.evidence_oracle_cfg.get('rebuttal_loss_type', 'huber')).lower()
        rebuttal_align_weight = float(
            self.evidence_oracle_cfg.get(
                'rebuttal_align_weight',
                float(self.evidence_oracle_cfg.get('suspect_activation_weight', 0.5))
                + float(self.evidence_oracle_cfg.get('suspect_prune_weight', 0.5)),
            )
        )

        support_listwise_terms = []
        support_pairwise_terms = []
        rebuttal_loss_terms = []
        support_align_terms = []
        support_base_align_terms = []
        rebuttal_align_terms = []
        rebuttal_base_align_terms = []
        support_nonzero_ratios = []
        support_graph_active_ratios = []
        rebuttal_nonzero_ratios = []
        rebuttal_max_values = []

        for t, step_data in enumerate(trajectory):
            evidence_state = self._step_evidence_state(step_data)
            oracle_targets = step_data.get('evidence_oracle_targets')
            if evidence_state is None or oracle_targets is None:
                continue

            batch = step_data['fused_batch'].view(-1).to(device=device, dtype=torch.long)
            active_mask = active_mask_list[t].float().to(device=device)

            support_student = evidence_state.support_score.float().view(-1)
            support_teacher = oracle_targets['support_target'].to(device=device, dtype=torch.float).view(-1)
            support_graph_mass = scatter_sum(support_teacher.abs(), batch, dim=0)
            support_graph_mask = (support_graph_mass > 1e-6).float() * active_mask

            if support_graph_mask.sum().item() > 0:
                teacher_probs = gnn_softmax((support_teacher / listwise_temperature).view(-1, 1), batch).view(-1)
                student_log_probs = torch.log(
                    gnn_softmax((support_student / listwise_temperature).view(-1, 1), batch).view(-1) + 1e-9
                )
                support_ce_graph = -scatter_sum(teacher_probs * student_log_probs, batch, dim=0)
                support_listwise_terms.append(self._masked_mean(support_ce_graph, support_graph_mask))
                support_pairwise_terms.append(
                    self._compute_support_pairwise_loss(
                        support_student,
                        support_teacher,
                        batch,
                        support_graph_mask,
                        support_pairwise_margin,
                    )
                )

            support_base = getattr(evidence_state, 'base_support_score', None)
            if support_base is None:
                support_base = support_student.detach()
            else:
                support_base = support_base.float().view(-1)
            support_align_terms.append(
                self._compute_alignment_l1(support_student, support_teacher, batch, active_mask)
            )
            support_base_align_terms.append(
                self._compute_alignment_l1(support_base, support_teacher, batch, active_mask)
            )

            rebuttal_student = evidence_state.contradiction_score.float().view(-1)
            rebuttal_teacher = oracle_targets['rebuttal_target'].to(device=device, dtype=torch.float).view(-1).clamp_min(0.0)
            rebuttal_base = getattr(evidence_state, 'base_contradiction_score', None)
            if rebuttal_base is None:
                rebuttal_base = rebuttal_student.detach()
            else:
                rebuttal_base = rebuttal_base.float().view(-1)
            if rebuttal_loss_type == 'l1':
                rebuttal_pointwise = F.l1_loss(rebuttal_student, rebuttal_teacher, reduction='none')
            else:
                rebuttal_pointwise = F.smooth_l1_loss(rebuttal_student, rebuttal_teacher, reduction='none')
            node_mask = active_mask[batch].float()
            if node_mask.sum().item() > 0:
                rebuttal_loss_terms.append((rebuttal_pointwise * node_mask).sum() / node_mask.sum().clamp_min(1.0))

            rebuttal_align_terms.append(
                self._compute_alignment_l1(rebuttal_student, rebuttal_teacher, batch, active_mask)
            )
            rebuttal_base_align_terms.append(
                self._compute_alignment_l1(rebuttal_base, rebuttal_teacher, batch, active_mask)
            )

            stats = oracle_targets.get('stats', {})
            support_nonzero_ratios.append(float(stats.get('support_nonzero_ratio', 0.0)))
            support_graph_active_ratios.append(float(stats.get('support_graph_active_ratio', 0.0)))
            rebuttal_nonzero_ratios.append(float(stats.get('rebuttal_nonzero_ratio', 0.0)))
            rebuttal_max_values.append(float(stats.get('rebuttal_max', 0.0)))

        def mean_or_zero(values):
            if not values:
                return zero
            return torch.stack(values).mean()

        support_listwise = mean_or_zero(support_listwise_terms)
        support_pairwise = mean_or_zero(support_pairwise_terms)
        rebuttal_loss = mean_or_zero(rebuttal_loss_terms)

        evidence_total = (
            support_listwise_weight * support_listwise
            + support_pairwise_weight * support_pairwise
            + rebuttal_align_weight * rebuttal_loss
        )

        metrics = {
            'loss/evidence_oracle_support_listwise': float(support_listwise.item()),
            'loss/evidence_oracle_support_pairwise': float(support_pairwise.item()),
            'loss/evidence_oracle_rebuttal_align': float(rebuttal_loss.item()),
            'loss/evidence_oracle_total': float(evidence_total.item()),
            'metric/evidence_oracle_support_align_l1': float(mean_or_zero(support_align_terms).item()),
            'metric/evidence_oracle_support_base_align_l1': float(mean_or_zero(support_base_align_terms).item()),
            'metric/evidence_oracle_rebuttal_align_l1': float(mean_or_zero(rebuttal_align_terms).item()),
            'metric/evidence_oracle_rebuttal_base_align_l1': float(mean_or_zero(rebuttal_base_align_terms).item()),
            'metric/evidence_oracle_support_label_nonzero_ratio': float(sum(support_nonzero_ratios) / max(1, len(support_nonzero_ratios))),
            'metric/evidence_oracle_support_graph_active_ratio': float(sum(support_graph_active_ratios) / max(1, len(support_graph_active_ratios))),
            'metric/evidence_oracle_rebuttal_label_nonzero_ratio': float(sum(rebuttal_nonzero_ratios) / max(1, len(rebuttal_nonzero_ratios))),
            'metric/evidence_oracle_rebuttal_max': float(sum(rebuttal_max_values) / max(1, len(rebuttal_max_values))),
        }
        return evidence_total, metrics, schedule

    def forward(self, trajectory: List[Dict], cfg: Any = None, **kwargs) -> Dict[str, Any]:
        """
        Args:
            trajectory: List of dicts from Phase45Model.forward
            cfg: Configuration object (optional, uses self.config if None)
            kwargs: Additional arguments like graph_structure
        """
        graph_structure = kwargs.get('graph_structure', None)
        
        total_loss = 0.0
        loss_dict = {}
        T = len(trajectory)
        if T == 0:
            return {'total_loss': torch.tensor(0.0, requires_grad=True)}

        # 1. Prepare Trajectory-wide data
        log_p_star_list = []
        hit_prob_list = []
        nav_gain_list = [] # Legacy: Myopic Loss
        nav_reward_list = [] # [Task 2] Raw Reward for Returns
        nav_action_list = [] # [Task 2] Action for Returns
        active_mask_list = []
        entropy_list = []
        nav_log_prob_list = []
        nav_value_list = []
        nav_aux_value_list = []
        nav_policy_entropy_list = []
        
        for step_data in trajectory:
            logits = step_data['reasoner_logits']
            batch = step_data['fused_batch']
            target = step_data['fused_source_label'] # [N, 1]
            active_mask = step_data['active_mask'] # [B]
            
            # Reasoner Probabilities
            probs = gnn_softmax(logits, batch)
            log_probs = torch.log(probs + 1e-9)
            
            # [Debug] Check Target Validity
            if self.config.get('debug_alignment', True):
                target_sum = target.sum().item()
                if target_sum < 0.5:
                     print(f"[Loss WARNING] Step {trajectory.index(step_data)}: Target Sum is {target_sum}! Loss will be 0 but meaningless.")
                else:
                     target_idx = target.nonzero(as_tuple=True)[0]
                     # print(f"[Loss Check] Step {trajectory.index(step_data)}: Target Local Index: {target_idx.tolist()}")

            lp_star_per_node = (log_probs * target)
            # [Fix] Use scatter_sum instead of scatter_mean for Target Log Prob.
            # scatter_mean divides by graph size (N), scaling gradient by 1/N.
            # Since target is one-hot (or close), we want the sum over targets.
            lp_star = scatter_sum(lp_star_per_node.sum(dim=-1), batch, dim=0) # [B]
            log_p_star_list.append(lp_star)
            
            # [Task 2] Collect Nav Data
            nav_gain_loss = step_data.get('nav_gain_loss') # [B]
            nav_gain_list.append(nav_gain_loss)
            
            nav_reward = step_data.get('nav_reward') # [B]
            nav_reward_list.append(nav_reward)
            
            nav_action = step_data.get('nav_action') # [N]
            nav_action_list.append(nav_action)
            nav_log_prob_list.append(step_data.get('nav_log_prob'))
            nav_value_list.append(step_data.get('nav_value'))
            nav_aux_value_list.append(step_data.get('nav_aux_value'))
            nav_policy_entropy_list.append(step_data.get('nav_entropy'))

            # Old Hit Prob Logic (kept for compatibility/logging or mixed loss)
            # If we strictly switch to ST, we might not use this for optimization
            # But 'hit_prob_list' is used for 'l_hit' (Surrogate Hit).
            # The prompt says "废弃原来基于期望概率的 Hit Loss...".
            # So I will use 'nav_gain_list' for 'l_hit' if available.
            
            # [Optimization] Avoid broadcasting large tensors if not needed
            if self.use_old_hit:
                 hp_node = 1.0 - (1.0 - probs)**3 
            else:
                 # Only compute if needed for logging or survival
                 # Memory Optimization: Check shapes
                 nav1_p = step_data.get('nav1_probs')
                 if nav1_p is None: nav1_p = step_data.get('nav_probs')
                 
                 # If we don't need hit_prob for loss (because w_hit uses nav_gain),
                 # we can skip this expensive N*N-like op if not logging?
                 # But we need hit_prob_list for Survival Loss and Logging.
                 
                 if nav1_p is not None:
                      nav2_p = step_data.get('nav2_probs')
                      if nav2_p is None: nav2_p = torch.zeros_like(probs)
                      
                      # Avoid creating new large tensors
                      # hp_node = 1.0 - (1.0 - probs) * (1.0 - nav1_p) * (1.0 - nav2_p)
                      # In-place operations where possible?
                      # term1 = (1.0 - probs) # [N]
                      # term2 = (1.0 - nav1_p) # [N]
                      # term3 = (1.0 - nav2_p) # [N]
                      # hp_node = 1.0 - term1 * term2 * term3
                      
                      # Even more optimized:
                      # hp_node = 1.0 - ((1.0-probs) * (1.0-nav1_p) * (1.0-nav2_p))
                      # To avoid allocating intermediates:
                      term = 1.0 - probs
                      term.mul_(1.0 - nav1_p.view_as(probs))
                      term.mul_(1.0 - nav2_p.view_as(probs))
                      hp_node = 1.0 - term
                 else:
                      hp_node = probs # Fallback
            
            # [Fix] Use scatter_sum for Hit Probability (same logic as log_p)
            hp_star = scatter_sum((hp_node * target).sum(dim=-1), batch, dim=0) # [B]
            hit_prob_list.append(hp_star)
            active_mask_list.append(active_mask.float())
            
            ent_node = - (probs * log_probs).sum(dim=-1, keepdim=True)
            ent_graph = scatter_mean(ent_node.squeeze(-1), batch, dim=0) # [B]
            entropy_list.append(ent_graph)

        def masked_mean(x, mask):
            return self._masked_mean(x, mask)

        # Metrics
        active_ratios = [m.mean().item() for m in active_mask_list]
        loss_dict['metric/active_ratio_start'] = active_ratios[0]
        loss_dict['metric/active_ratio_end'] = active_ratios[-1]
        loss_dict['metric/effective_steps'] = sum(active_ratios)

        scale_factor = 1.0
        if self.scale_correction and loss_dict['metric/effective_steps'] > 0:
            scale_factor = T / loss_dict['metric/effective_steps']
        
        # 4.1 Reasoner Hard CE
        if self.w_hard > 0:
            l_hard = 0.0
            for t in range(T):
                alpha_t = self._temporal_weight(t, T)
                
                # [CRITICAL FIX] Ignore active_mask for Hard Loss (Cross Entropy)
                # Even if graph is inactive (found), we must train on the correct label!
                # The model predicted '3' when target was '22'. Loss was 0 because mask was 0.
                # Now we force mask=1 for Hard Loss.
                
                # mask_t = torch.ones_like(active_mask_list[t]) if self.dense_ce else active_mask_list[t]
                # [SSOT] Life Support: Dense Hard Loss Mask
                dense_mask = True
                if cfg and hasattr(cfg, 'life_support'):
                    dense_mask = getattr(cfg.life_support, 'dense_hard_loss_mask', True)
                
                mask_t = torch.ones_like(active_mask_list[t]) if dense_mask else active_mask_list[t]
                
                # Add extra weight to inactive steps to punish "Early Stop with Wrong Answer"?
                # No, just standard CE is enough. If prediction is wrong, CE will be high.
                
                l_hard += alpha_t * masked_mean(-log_p_star_list[t], mask_t)
            l_hard /= T
            total_loss += self.w_hard * l_hard
            loss_dict['loss/hard_ce'] = l_hard.item()

        # 4.2 Posterior Improvement
        if self.w_delta > 0 and T > 1:
            l_delta = 0.0
            for t in range(T - 1):
                rho_t = self.rho ** t
                delta_t = log_p_star_list[t+1] - log_p_star_list[t]
                l_delta += rho_t * masked_mean(-delta_t, active_mask_list[t])
            l_delta /= (T - 1)
            total_loss += self.w_delta * l_delta * scale_factor
            loss_dict['loss/delta_logp'] = l_delta.item()

        # 4.3 Hit Loss (Refactored for ST with Cumulative Returns)
        if self.w_hit > 0:
            # [Task 2] Calculate Discounted Returns (G_t)
            # We need to handle potential None values in lists
            returns_list = [torch.zeros_like(active_mask_list[0])] * T
            running_return = torch.zeros_like(active_mask_list[0])
            
            for t in reversed(range(T)):
                r_t = nav_reward_list[t] if t < len(nav_reward_list) and nav_reward_list[t] is not None else torch.zeros_like(running_return)
                mask_t = active_mask_list[t]
                
                # G_t = R_t + rho * G_{t+1} * mask_{t+1} (implicit in running_return update)
                # running_return tracks G_{t+1}
                # Reset return if graph finished (not active at t) -> actually if not active at t, loss is masked anyway.
                # But for correctness:
                running_return = r_t + self.rho * running_return
                returns_list[t] = running_return.detach() # Returns are targets, detached
            nav_pg_ready = any(
                value is not None and log_prob is not None
                for value, log_prob in zip(nav_value_list, nav_log_prob_list)
            )

            if nav_pg_ready:
                zero_ref = returns_list[0]
                step_rewards = [
                    nav_reward_list[t].view(-1) if nav_reward_list[t] is not None else torch.zeros_like(zero_ref)
                    for t in range(T)
                ]
                value_coef = float(self.params.get('navigator_value_coef', 0.5))
                aux_value_coef = float(self.params.get('navigator_aux_value_coef', 0.0))
                entropy_coef = float(self.params.get('navigator_entropy_coef', 0.01))
                gae_lambda = float(self.params.get('navigator_gae_lambda', 0.95))
                use_gae = bool(self.params.get('navigator_use_gae', True))

                values_for_adv = [
                    value.view(-1) if value is not None else torch.zeros_like(zero_ref)
                    for value in nav_value_list
                ]
                if use_gae:
                    advantages_list, returns_target_list = self._compute_gae(
                        rewards=step_rewards,
                        values=values_for_adv,
                        active_masks=active_mask_list,
                        gae_lambda=gae_lambda,
                    )
                else:
                    returns_target_list = returns_list
                    advantages_list = [
                        ret.view(-1) - value
                        for ret, value in zip(returns_list, values_for_adv)
                    ]

                actor_terms = []
                critic_terms = []
                aux_critic_terms = []
                entropy_terms = []
                reward_means = []
                for t in range(T):
                    mask_t = active_mask_list[t].float()
                    if float(mask_t.sum().item()) <= 0.0:
                        continue
                    log_prob_t = nav_log_prob_list[t]
                    if log_prob_t is None:
                        continue
                    adv_t = advantages_list[t]
                    active_adv = adv_t[mask_t > 0.0]
                    if active_adv.numel() > 1:
                        adv_std = active_adv.std(unbiased=False)
                        if float(adv_std.item()) > 1e-6:
                            adv_t = (adv_t - active_adv.mean()) / (adv_std + 1e-6)
                    actor_terms.append(-((log_prob_t.view(-1) * adv_t.detach()) * mask_t).sum())

                    value_t = nav_value_list[t]
                    if value_t is not None:
                        critic_terms.append(
                            F.mse_loss(value_t.view(-1), returns_target_list[t].detach(), reduction='none').mul(mask_t).sum()
                        )
                    aux_value_t = nav_aux_value_list[t]
                    if aux_value_t is not None:
                        aux_critic_terms.append(
                            F.mse_loss(aux_value_t.view(-1), returns_target_list[t].detach(), reduction='none').mul(mask_t).sum()
                        )
                    entropy_t = nav_policy_entropy_list[t]
                    if entropy_t is not None:
                        entropy_terms.append((entropy_t.view(-1) * mask_t).sum())
                    reward_means.append((step_rewards[t] * mask_t).sum())

                denom = max(sum(float(mask.sum().item()) for mask in active_mask_list), 1.0)
                actor_loss = (
                    sum(actor_terms) / denom if actor_terms else zero_ref.sum() * 0.0
                )
                critic_loss = (
                    sum(critic_terms) / denom if critic_terms else zero_ref.sum() * 0.0
                )
                aux_critic_loss = (
                    sum(aux_critic_terms) / denom if aux_critic_terms else zero_ref.sum() * 0.0
                )
                entropy_bonus = (
                    sum(entropy_terms) / denom if entropy_terms else zero_ref.sum() * 0.0
                )
                l_hit = actor_loss + value_coef * critic_loss + aux_value_coef * aux_critic_loss - entropy_coef * entropy_bonus
                total_loss += self.w_hit * l_hit * scale_factor
                loss_dict['loss/hit_gain'] = float(l_hit.item())
                loss_dict['loss/nav_actor'] = float(actor_loss.item())
                loss_dict['loss/nav_critic'] = float(critic_loss.item())
                loss_dict['loss/nav_aux_critic'] = float(aux_critic_loss.item())
                loss_dict['metric/nav_entropy'] = float(entropy_bonus.item())
                if reward_means:
                    loss_dict['metric/nav_return_mean'] = float((sum(reward_means) / denom).item())
            else:
                l_hit = 0.0
                for t in range(T):
                    rho_t = self.rho ** t
                    
                    y_action = nav_action_list[t]
                    G_t = returns_list[t] # [B]
                    
                    if y_action is not None:
                        batch = trajectory[t]['fused_batch']
                        G_t_nodes = G_t[batch]
                        
                        if y_action.shape == G_t_nodes.shape:
                            loss_per_node = - (y_action * G_t_nodes)
                            loss_per_graph = scatter_sum(loss_per_node, batch, dim=0)
                            l_hit_step = loss_per_graph
                        else:
                            l_hit_step = torch.zeros_like(G_t)
                    elif nav_gain_list[t] is not None:
                        l_hit_step = nav_gain_list[t]
                    else:
                        l_hit_step = -torch.log(hit_prob_list[t] + self.hit_eps)
                    
                    l_hit += rho_t * masked_mean(l_hit_step, active_mask_list[t])
                l_hit /= T
                total_loss += self.w_hit * l_hit * scale_factor
                loss_dict['loss/hit_gain'] = l_hit.item()

        # 4.5 LogP Monotonicity
        if self.w_mono > 0 and T > 1:
            l_mono = 0.0
            for t in range(T - 1):
                diff = log_p_star_list[t] - log_p_star_list[t+1] + self.mono_margin
                l_mono += masked_mean(F.relu(diff), active_mask_list[t])
            l_mono /= (T - 1)
            total_loss += self.w_mono * l_mono
            loss_dict['loss/monotonicity'] = l_mono.item()

        # 4.6 Target Entropy Following
        if self.w_ent > 0:
            l_ent = 0.0
            for t in range(T):
                target_h = self.ent_start - (self.ent_start - self.ent_end) * (t / max(1, T - 1))
                ent_error = (entropy_list[t] - target_h)**2
                l_ent += masked_mean(ent_error, active_mask_list[t])
            l_ent /= T
            total_loss += self.w_ent * l_ent
            loss_dict['loss/entropy_follow'] = l_ent.item()
            
        # 4.7 Survival Loss (Optional)
        if self.w_surv > 0 or self.loss_mode == 'survival':
            l_surv = 0.0
            for t in range(T):
                q_t = hit_prob_list[t] # Keep using hit_prob for survival (it's about success probability)
                
                # [Fix] Clamp and handle NaNs
                if torch.isnan(q_t).any():
                     q_t = torch.nan_to_num(q_t, nan=0.0)
                q_t = torch.clamp(q_t, 0.0, 1.0)
                
                if self.surv_detach_q:
                    q_t = q_t.detach()
                is_hit_t = trajectory[t]['is_hit'].view(-1)
                mask_t = active_mask_list[t]
                term_hit = -torch.log(q_t + self.surv_eps)
                term_miss = -torch.log(1.0 - q_t + self.surv_eps)
                loss_step = is_hit_t * term_hit + (1.0 - is_hit_t) * term_miss
                if self.surv_use_rho:
                    loss_step = (self.rho ** t) * loss_step
                l_surv += masked_mean(loss_step, mask_t)
            l_surv /= T
            weight = self.w_surv if self.w_surv > 0 else (1.0 if self.loss_mode == 'survival' else 0.0)
            total_loss += weight * l_surv
            loss_dict['loss/survival_nll'] = l_surv.item()

        live_total_loss = total_loss
        if not torch.is_tensor(live_total_loss):
            live_total_loss = torch.tensor(float(live_total_loss), device=trajectory[0]['reasoner_logits'].device)

        evidence_oracle_total, evidence_oracle_metrics, evidence_schedule = self._compute_evidence_oracle_loss(
            trajectory=trajectory,
            active_mask_list=active_mask_list,
            cfg=cfg,
        )
        if evidence_oracle_metrics:
            loss_dict.update(evidence_oracle_metrics)
            loss_dict['loss/live_total'] = float(live_total_loss.item())
            loss_dict['schedule/evidence_oracle_phase'] = evidence_schedule['phase']
            loss_dict['schedule/evidence_oracle_phase_index'] = evidence_schedule['phase_index']
            loss_dict['schedule/evidence_oracle_progress'] = evidence_schedule['progress']
            loss_dict['schedule/evidence_oracle_live_factor'] = evidence_schedule['live_factor']
            loss_dict['schedule/evidence_oracle_oracle_factor'] = evidence_schedule['oracle_factor']
            total_loss = (
                live_total_loss * float(evidence_schedule['live_factor'])
                + evidence_oracle_total * float(evidence_schedule['oracle_factor'])
            )
        else:
            total_loss = live_total_loss

        loss_dict['total_loss'] = total_loss
        loss_dict['loss/total'] = total_loss.item()
        
        # DEBUG NaN
        if torch.isnan(total_loss):
            print("NaN Detected in Loss!")
            print(f"w_hard: {self.w_hard}, l_hard: {loss_dict.get('loss/hard_ce')}")
            print(f"w_delta: {self.w_delta}, l_delta: {loss_dict.get('loss/delta_logp')}")
            print(f"w_hit: {self.w_hit}, l_hit: {loss_dict.get('loss/hit_gain')}")
            print(f"w_ent: {self.w_ent}, l_ent: {loss_dict.get('loss/entropy_follow')}")
            print(f"w_mono: {self.w_mono}, l_mono: {loss_dict.get('loss/monotonicity')}")
            print(f"w_surv: {self.w_surv}, l_surv: {loss_dict.get('loss/survival_nll')}")
            # print(f"log_p_star_list: {[t.item() for t in log_p_star_list if not torch.isnan(t).any()]}") # Summary
             
        return total_loss, loss_dict

@LOSS_REGISTRY.register("survival_engine")
class SurvivalLossEngine(ModularLossEngine):
    """
    Specialized Loss Engine focused on Survival NLL.
    """
    def __init__(self, config: Dict):
        # Force survival mode
        if 'params' not in config: config['params'] = {}
        config['params']['loss_mode'] = 'survival'
        if 'weights' not in config: config['weights'] = {}
        config['weights']['w_surv'] = 1.0
        super().__init__(config)

    def requirements(self) -> LossRequirements:
        return {
            'requires_soft_actions': True,
            'requires_active_mask': True,
            'required_fields': ['is_hit', 'active_mask', 'nav_probs']
        }

@LOSS_REGISTRY.register("standard_engine")
class StandardLossEngine(ModularLossEngine):
    """
    Standard multi-objective loss engine.
    """
    def requirements(self) -> LossRequirements:
        return {
            'requires_stepwise_probs': True,
            'required_fields': ['reasoner_logits', 'active_mask']
        }

def build_primary_criterion(cfg):
    loss_cfg = cfg.loss.__dict__ if hasattr(cfg, 'loss') else {}
    loss_type = loss_cfg.get('type', 'modular_v4_5')
    try:
        cls = LOSS_REGISTRY.get(loss_type)
        return cls(loss_cfg)
    except KeyError:
        return ModularLossEngine(loss_cfg)
