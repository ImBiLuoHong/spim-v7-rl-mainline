import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from src.modeling.interfaces.base import ReasonerBase
from src.modeling.registry import REASONER_REGISTRY
from src.modeling.encoders.observation_encoder import ObservationEncoder

@REASONER_REGISTRY.register("bayesian_v4_5")
class BayesianReasoner(ReasonerBase):
    SENTINEL_VERSION = "v2_sentinel_obs_encoder"
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        # SSOT: Derive hidden_dim from config (Navigator or Reasoner)
        nav_cfg = getattr(cfg.model, 'navigator', {})
        if isinstance(nav_cfg, dict):
            nav_dim = nav_cfg.get('hidden_dim', 64)
        else:
            nav_dim = getattr(nav_cfg, 'hidden_dim', 64)
            
        reasoner_cfg = getattr(cfg.model, 'reasoner', {})
        if isinstance(reasoner_cfg, dict):
            reasoner_dim = reasoner_cfg.get('hidden_dim', nav_dim)
            self.use_evidence = reasoner_cfg.get('use_evidence', False)
            self.evidence_scale = reasoner_cfg.get('evidence_scale', 0.1)
            self.evidence_mode = reasoner_cfg.get('evidence_mode', 'bias') # bias or concat
            self.support_weight = reasoner_cfg.get('support_weight', 1.0)
            self.support_consumer_scale = float(reasoner_cfg.get('support_consumer_scale', 1.0))
            self.suspect_weight = reasoner_cfg.get('suspect_weight', 0.0)
            self.contradiction_weight = reasoner_cfg.get('contradiction_weight', 0.0)
            self.contradiction_consumer_scale = float(reasoner_cfg.get('contradiction_consumer_scale', 1.0))
            self.internal_physics_consumer_scale = float(reasoner_cfg.get('internal_physics_consumer_scale', 1.0))
            self.internal_physics_consumer_mode = str(reasoner_cfg.get('internal_physics_consumer_mode', 'quasi_hard')).lower()
            self.internal_physics_hard_threshold = float(reasoner_cfg.get('internal_physics_hard_threshold', -1e3))
            self.internal_physics_hardlike_floor = float(reasoner_cfg.get('internal_physics_hardlike_floor', -3e3))
            self.internal_physics_soft_scale = float(reasoner_cfg.get('internal_physics_soft_scale', 0.0))
            self.internal_physics_positive_timing_scale = reasoner_cfg.get('internal_physics_positive_timing_scale', None)
            self.internal_physics_negative_arrival_soft_scale = reasoner_cfg.get('internal_physics_negative_arrival_soft_scale', None)
            self.internal_physics_negative_arrival_overlap_scale = float(reasoner_cfg.get('internal_physics_negative_arrival_overlap_scale', 1.0))
            self.negative_exclusion_prior_consumer_scale = float(reasoner_cfg.get('negative_exclusion_prior_consumer_scale', 0.0))
            self.negative_exclusion_prior_mode = str(reasoner_cfg.get('negative_exclusion_prior_mode', 'borrowed_overlap')).lower()
            self.negative_exclusion_prior_base_bias = float(reasoner_cfg.get('negative_exclusion_prior_base_bias', 0.0))
            self.negative_exclusion_prior_soft_weight = float(reasoner_cfg.get('negative_exclusion_prior_soft_weight', 0.5))
            self.negative_exclusion_prior_hard_weight = float(reasoner_cfg.get('negative_exclusion_prior_hard_weight', 1.0))
            self.contradiction_consumer_graph_norm = str(reasoner_cfg.get('contradiction_consumer_graph_norm', 'none')).lower()
            self.contradiction_consumer_graph_quantile = float(reasoner_cfg.get('contradiction_consumer_graph_quantile', 0.95))
            self.contradiction_consumer_scale_floor = float(reasoner_cfg.get('contradiction_consumer_scale_floor', 1.0))
            self.contradiction_consumer_clip_max = float(reasoner_cfg.get('contradiction_consumer_clip_max', 0.0))
            self.reaction_weight = reasoner_cfg.get('reaction_weight', 0.0)
            self.concat_fields = reasoner_cfg.get('concat_fields', ['support_score', 'uncertainty_gap'])
        else:
            reasoner_dim = getattr(reasoner_cfg, 'hidden_dim', nav_dim)
            self.use_evidence = getattr(reasoner_cfg, 'use_evidence', False)
            self.evidence_scale = getattr(reasoner_cfg, 'evidence_scale', 0.1)
            self.evidence_mode = getattr(reasoner_cfg, 'evidence_mode', 'bias')
            self.support_weight = getattr(reasoner_cfg, 'support_weight', 1.0)
            self.support_consumer_scale = float(getattr(reasoner_cfg, 'support_consumer_scale', 1.0))
            self.suspect_weight = getattr(reasoner_cfg, 'suspect_weight', 0.0)
            self.contradiction_weight = getattr(reasoner_cfg, 'contradiction_weight', 0.0)
            self.contradiction_consumer_scale = float(getattr(reasoner_cfg, 'contradiction_consumer_scale', 1.0))
            self.internal_physics_consumer_scale = float(getattr(reasoner_cfg, 'internal_physics_consumer_scale', 1.0))
            self.internal_physics_consumer_mode = str(getattr(reasoner_cfg, 'internal_physics_consumer_mode', 'quasi_hard')).lower()
            self.internal_physics_hard_threshold = float(getattr(reasoner_cfg, 'internal_physics_hard_threshold', -1e3))
            self.internal_physics_hardlike_floor = float(getattr(reasoner_cfg, 'internal_physics_hardlike_floor', -3e3))
            self.internal_physics_soft_scale = float(getattr(reasoner_cfg, 'internal_physics_soft_scale', 0.0))
            self.internal_physics_positive_timing_scale = getattr(reasoner_cfg, 'internal_physics_positive_timing_scale', None)
            self.internal_physics_negative_arrival_soft_scale = getattr(reasoner_cfg, 'internal_physics_negative_arrival_soft_scale', None)
            self.internal_physics_negative_arrival_overlap_scale = float(getattr(reasoner_cfg, 'internal_physics_negative_arrival_overlap_scale', 1.0))
            self.negative_exclusion_prior_consumer_scale = float(getattr(reasoner_cfg, 'negative_exclusion_prior_consumer_scale', 0.0))
            self.negative_exclusion_prior_mode = str(getattr(reasoner_cfg, 'negative_exclusion_prior_mode', 'borrowed_overlap')).lower()
            self.negative_exclusion_prior_base_bias = float(getattr(reasoner_cfg, 'negative_exclusion_prior_base_bias', 0.0))
            self.negative_exclusion_prior_soft_weight = float(getattr(reasoner_cfg, 'negative_exclusion_prior_soft_weight', 0.5))
            self.negative_exclusion_prior_hard_weight = float(getattr(reasoner_cfg, 'negative_exclusion_prior_hard_weight', 1.0))
            self.contradiction_consumer_graph_norm = str(getattr(reasoner_cfg, 'contradiction_consumer_graph_norm', 'none')).lower()
            self.contradiction_consumer_graph_quantile = float(getattr(reasoner_cfg, 'contradiction_consumer_graph_quantile', 0.95))
            self.contradiction_consumer_scale_floor = float(getattr(reasoner_cfg, 'contradiction_consumer_scale_floor', 1.0))
            self.contradiction_consumer_clip_max = float(getattr(reasoner_cfg, 'contradiction_consumer_clip_max', 0.0))
            self.reaction_weight = getattr(reasoner_cfg, 'reaction_weight', 0.0)
            self.concat_fields = getattr(reasoner_cfg, 'concat_fields', ['support_score', 'uncertainty_gap'])
        
        hidden_dim = int(reasoner_dim)
        
        # [Refactor] Standardized Observation Encoder
        self.obs_encoder = ObservationEncoder(hidden_dim, self.use_evidence, self.evidence_mode, self.concat_fields)
        
        edge_dim = getattr(cfg.model, 'edge_dim', 8)
        
        # [Decouple] Use hidden_dim as input (from ObsEncoder)
        self.encoder = GATv2Conv(hidden_dim, hidden_dim // 4, heads=4, edge_dim=edge_dim)
        self.prior_proj = nn.Linear(hidden_dim, 1)
        self.like_proj = nn.Linear(hidden_dim, 1)
        self.alpha = nn.Parameter(torch.tensor(0.01))
        self.lambda_prune = nn.Parameter(torch.tensor(1.0))

    def _calibrate_contradiction_signal(self, contradiction: torch.Tensor, batch: torch.Tensor = None) -> torch.Tensor:
        calibrated = contradiction
        norm_mode = str(getattr(self, 'contradiction_consumer_graph_norm', 'none')).lower()
        if norm_mode == 'q95' and batch is not None and contradiction.numel() > 0:
            batch_flat = batch.view(-1)
            contra_flat = contradiction.view(-1)
            calibrated_flat = contra_flat.clone()
            num_graphs = int(batch_flat.max().item()) + 1 if batch_flat.numel() > 0 else 0
            quantile = min(max(float(getattr(self, 'contradiction_consumer_graph_quantile', 0.95)), 0.0), 1.0)
            scale_floor = max(float(getattr(self, 'contradiction_consumer_scale_floor', 1.0)), 1e-6)
            for graph_idx in range(num_graphs):
                graph_mask = batch_flat == graph_idx
                if not bool(graph_mask.any()):
                    continue
                graph_values = contra_flat[graph_mask]
                graph_pos = graph_values[graph_values > 0]
                if graph_pos.numel() == 0:
                    continue
                graph_scale = torch.quantile(graph_pos, graph_pos.new_tensor(quantile))
                graph_scale = torch.clamp(graph_scale, min=scale_floor)
                calibrated_flat[graph_mask] = graph_values / graph_scale
            calibrated = calibrated_flat.view_as(contradiction)

        clip_max = float(getattr(self, 'contradiction_consumer_clip_max', 0.0))
        if clip_max > 0.0:
            calibrated = calibrated.clamp_max(clip_max)

        scale = float(getattr(self, 'contradiction_consumer_scale', 1.0))
        if scale != 1.0:
            calibrated = calibrated * scale
        return calibrated

    def _calibrate_support_signal(self, support: torch.Tensor) -> torch.Tensor:
        scale = float(getattr(self, 'support_consumer_scale', 1.0))
        if scale != 1.0:
            return support * scale
        return support

    def _resolve_internal_physics_positive_timing_scale(self) -> float:
        value = getattr(self, 'internal_physics_positive_timing_scale', None)
        if value is None:
            return float(getattr(self, 'internal_physics_soft_scale', 0.0))
        return float(value)

    def _resolve_internal_physics_negative_arrival_soft_scale(self) -> float:
        value = getattr(self, 'internal_physics_negative_arrival_soft_scale', None)
        if value is None:
            return float(getattr(self, 'internal_physics_soft_scale', 0.0))
        return float(value)

    def _resolve_negative_exclusion_prior_base_bias(self) -> float:
        mode = str(getattr(self, 'negative_exclusion_prior_mode', 'borrowed_overlap')).lower()
        if mode in {'prior_only_floor_soft', 'prior_native_gate_strength', 'prior_native_slack_log1p'}:
            floor = abs(float(getattr(self, 'internal_physics_hardlike_floor', -3e3)))
            if float(getattr(self, 'negative_exclusion_prior_base_bias', 0.0)) > 0.0:
                return float(getattr(self, 'negative_exclusion_prior_base_bias', 0.0))
            return floor * self._resolve_internal_physics_negative_arrival_soft_scale()
        base_bias = float(getattr(self, 'negative_exclusion_prior_base_bias', 0.0))
        return max(base_bias, 0.0)

    def _get_evidence_attr(self, evidence_state, attr_name, default=None):
        if evidence_state is None:
            return default
        if isinstance(evidence_state, dict):
            return evidence_state.get(attr_name, default)
        return getattr(evidence_state, attr_name, default)

    def _prepare_internal_physics_bias(self, bias: torch.Tensor) -> torch.Tensor:
        hardlike_bias, soft_bias = self._split_internal_physics_bias_lanes(bias)
        hardlike_scale = float(getattr(self, 'internal_physics_consumer_scale', 1.0))
        soft_scale = float(getattr(self, 'internal_physics_soft_scale', 0.0))
        return hardlike_bias * hardlike_scale + soft_bias * soft_scale

    def _split_internal_physics_bias_lanes(self, bias: torch.Tensor):
        mode = str(getattr(self, 'internal_physics_consumer_mode', 'quasi_hard')).lower()
        if mode in {'split_floor', 'explicit_factorized_split'}:
            floor = float(getattr(self, 'internal_physics_hardlike_floor', -3e3))
            soft_bias = torch.clamp(bias, min=floor, max=0.0)
            hardlike_bias = bias - soft_bias
            return hardlike_bias, soft_bias

        threshold = float(getattr(self, 'internal_physics_hard_threshold', -1e3))
        is_hardlike = bias < threshold
        hardlike_bias = torch.where(is_hardlike, bias, torch.zeros_like(bias))
        soft_bias = torch.zeros_like(bias)
        return hardlike_bias, soft_bias

    def _zero_like_internal_physics(self, physics_ctx):
        if isinstance(physics_ctx, dict):
            for key in (
                'logit_bias',
                'positive_timing_failure_logit_bias',
                'negative_arrival_pressure_logit_bias',
            ):
                value = physics_ctx.get(key)
                if isinstance(value, torch.Tensor):
                    return torch.zeros_like(value)
        return None

    def _build_negative_arrival_prior_terms(self, negative_raw: torch.Tensor, evidence_state):
        overlap_scale = float(getattr(self, 'internal_physics_negative_arrival_overlap_scale', 1.0))
        not_ruled_out = self._get_evidence_attr(evidence_state, 'not_ruled_out_gate', None)
        if isinstance(not_ruled_out, torch.Tensor):
            if not_ruled_out.dim() == 1:
                not_ruled_out = not_ruled_out.unsqueeze(-1)
            not_ruled_out = not_ruled_out.to(device=negative_raw.device, dtype=negative_raw.dtype)
            residual_gate = not_ruled_out.clamp(0.0, 1.0)
            overlap_gate = (1.0 - residual_gate).clamp(0.0, 1.0)
        else:
            residual_gate = torch.ones_like(negative_raw)
            overlap_gate = torch.zeros_like(negative_raw)

        overlap_like_raw = negative_raw * overlap_gate
        residual_raw = negative_raw * residual_gate
        overlap_suppressed_raw = overlap_like_raw * (1.0 - overlap_scale)
        effective_negative_raw = residual_raw + overlap_like_raw * overlap_scale
        prior_gate_raw = -overlap_gate * self._resolve_negative_exclusion_prior_base_bias()
        arrival_gate = self._get_evidence_attr(evidence_state, 'arrival_gate', None)
        if isinstance(arrival_gate, torch.Tensor):
            if arrival_gate.dim() == 1:
                arrival_gate = arrival_gate.unsqueeze(-1)
            arrival_gate = arrival_gate.to(device=negative_raw.device, dtype=negative_raw.dtype).clamp(0.0, 1.0)
        else:
            arrival_gate = torch.zeros_like(negative_raw)
        structural_strength = (
            float(getattr(self, 'negative_exclusion_prior_soft_weight', 0.5)) * arrival_gate
            + float(getattr(self, 'negative_exclusion_prior_hard_weight', 1.0)) * overlap_gate
        )
        prior_native_structural_raw = -structural_strength * self._resolve_negative_exclusion_prior_base_bias()
        negative_exclusion_slack = self._get_evidence_attr(evidence_state, 'negative_exclusion_slack', None)
        if isinstance(negative_exclusion_slack, torch.Tensor):
            if negative_exclusion_slack.dim() == 1:
                negative_exclusion_slack = negative_exclusion_slack.unsqueeze(-1)
            negative_exclusion_slack = negative_exclusion_slack.to(device=negative_raw.device, dtype=negative_raw.dtype).clamp_min(0.0)
        else:
            negative_exclusion_slack = torch.zeros_like(negative_raw)
        prior_native_slack_raw = -torch.log1p(negative_exclusion_slack)
        return {
            'overlap_gate': overlap_gate,
            'overlap_like_raw': overlap_like_raw,
            'residual_raw': residual_raw,
            'overlap_suppressed_raw': overlap_suppressed_raw,
            'effective_negative_raw': effective_negative_raw,
            'prior_gate_raw': prior_gate_raw,
            'structural_strength': structural_strength,
            'prior_native_structural_raw': prior_native_structural_raw,
            'negative_exclusion_slack': negative_exclusion_slack,
            'prior_native_slack_raw': prior_native_slack_raw,
        }

    def _decompose_internal_physics_terms(self, physics_ctx, evidence_state=None):
        zero = self._zero_like_internal_physics(physics_ctx)
        if zero is None:
            return None

        raw_total = physics_ctx.get('logit_bias')
        if not isinstance(raw_total, torch.Tensor):
            raw_total = zero
        positive_raw = physics_ctx.get('positive_timing_failure_logit_bias')
        if not isinstance(positive_raw, torch.Tensor):
            positive_raw = zero
        negative_raw = physics_ctx.get('negative_arrival_pressure_logit_bias')
        if not isinstance(negative_raw, torch.Tensor):
            negative_raw = zero

        mode = str(getattr(self, 'internal_physics_consumer_mode', 'quasi_hard')).lower()
        if mode in {'explicit_factorized_split', 'explicit_factorized_neg_residualized'} and (
            isinstance(physics_ctx.get('positive_timing_failure_logit_bias'), torch.Tensor)
            or isinstance(physics_ctx.get('negative_arrival_pressure_logit_bias'), torch.Tensor)
        ):
            negative_prior_terms = self._build_negative_arrival_prior_terms(negative_raw, evidence_state)
            effective_negative_raw = negative_raw
            if mode == 'explicit_factorized_neg_residualized':
                effective_negative_raw = negative_prior_terms['effective_negative_raw']

            negative_hardlike_raw, negative_soft_raw = self._split_internal_physics_bias_lanes(effective_negative_raw)
            positive_consumed = positive_raw * self._resolve_internal_physics_positive_timing_scale()
            negative_hardlike = negative_hardlike_raw * float(getattr(self, 'internal_physics_consumer_scale', 1.0))
            negative_soft = negative_soft_raw * self._resolve_internal_physics_negative_arrival_soft_scale()
            negative_consumed = negative_hardlike + negative_soft
            prior_mode = str(getattr(self, 'negative_exclusion_prior_mode', 'borrowed_overlap')).lower()
            if prior_mode == 'borrowed_overlap':
                negative_exclusion_prior_raw = (
                    negative_prior_terms['overlap_like_raw']
                    * self._resolve_internal_physics_negative_arrival_soft_scale()
                )
            elif prior_mode == 'prior_native_gate_strength':
                negative_exclusion_prior_raw = negative_prior_terms['prior_native_structural_raw']
            elif prior_mode == 'prior_native_slack_log1p':
                negative_exclusion_prior_raw = negative_prior_terms['prior_native_slack_raw']
            else:
                negative_exclusion_prior_raw = negative_prior_terms['prior_gate_raw']
            negative_exclusion_prior_consumed = (
                negative_exclusion_prior_raw
                * float(getattr(self, 'negative_exclusion_prior_consumer_scale', 0.0))
            )
            soft_consumed = positive_consumed + negative_soft
            total_consumed = negative_hardlike + soft_consumed + negative_exclusion_prior_consumed
            return {
                'raw_total': raw_total,
                'positive_raw': positive_raw,
                'negative_raw': negative_raw,
                'negative_exclusion_prior_raw': negative_exclusion_prior_raw,
                'negative_exclusion_prior_strength': negative_prior_terms['structural_strength'],
                'negative_exclusion_slack_raw': negative_prior_terms['negative_exclusion_slack'],
                'negative_overlap_like_raw': negative_prior_terms['overlap_like_raw'],
                'negative_residual_raw': negative_prior_terms['residual_raw'],
                'negative_overlap_suppressed_raw': negative_prior_terms['overlap_suppressed_raw'],
                'negative_exclusion_prior_consumed': negative_exclusion_prior_consumed,
                'positive_consumed': positive_consumed,
                'negative_consumed': negative_consumed,
                'hardlike_consumed': negative_hardlike,
                'soft_consumed': soft_consumed,
                'total_consumed': total_consumed,
            }

        hardlike_bias, soft_bias = self._split_internal_physics_bias_lanes(raw_total)
        hardlike_consumed = hardlike_bias * float(getattr(self, 'internal_physics_consumer_scale', 1.0))
        soft_consumed = soft_bias * float(getattr(self, 'internal_physics_soft_scale', 0.0))
        return {
            'raw_total': raw_total,
            'positive_raw': positive_raw,
            'negative_raw': negative_raw,
            'negative_exclusion_prior_raw': zero,
            'negative_exclusion_prior_strength': zero,
            'negative_exclusion_slack_raw': zero,
            'negative_overlap_like_raw': zero,
            'negative_residual_raw': negative_raw,
            'negative_overlap_suppressed_raw': zero,
            'negative_exclusion_prior_consumed': zero,
            'positive_consumed': zero,
            'negative_consumed': zero,
            'hardlike_consumed': hardlike_consumed,
            'soft_consumed': soft_consumed,
            'total_consumed': hardlike_consumed + soft_consumed,
        }

    def forward(self, state, graph, physics_ctx=None):
        # [Decouple] We fetch h_fused only to check graph size (N_fused vs N_raw)
        h_structure = state.get('h_fused', state.get('h_structure'))
        num_fused_nodes = h_structure.size(0)
        
        # 1. Standardized Input Encoding
        # This handles Raw->Fused alignment, Normalization, and Evidence injection (if concat)
        h_obs = self.obs_encoder(state, num_fused_nodes)
        
        edge_index = graph.edge_index
        edge_attr = graph.edge_attr
        
        # Apply Gating from physics_ctx if available
        edge_attr_gated = edge_attr
        if physics_ctx and 'neg_gate' in physics_ctx and physics_ctx['neg_gate'] is not None:
            if edge_attr.size(1) >= 7:
                prefix = edge_attr[:, :6]
                target = edge_attr[:, 6:7]
                suffix = edge_attr[:, 7:]
                target_gated = target * physics_ctx['neg_gate']
                edge_attr_gated = torch.cat([prefix, target_gated, suffix], dim=1)

        # 2. Structure Path (Prior)
        h_structure_refined = self.encoder(h_obs, edge_index, edge_attr=edge_attr_gated)
        logits_prior = self.prior_proj(h_structure_refined)
        
        # 3. Likelihood Path (Data)
        # We reuse h_obs here. This means Likelihood is now "Deep Likelihood" (MLP encoded).
        # Previously it was MLP(Raw Features). Now it is MLP(Encoded Features).
        # obs_encoder includes an MLP projection, so this effectively adds another layer.
        logits_like = self.like_proj(h_obs)
        
        # [Audit 2] Check Variance
        logits = logits_prior + torch.abs(self.alpha) * logits_like
        
        # [Experiment] Evidence Injection (Bias Mode)
        # In 'bias' mode, ObsEncoder excludes evidence from h_obs. We inject it here.
        evidence_bias = None
        if self.use_evidence and self.evidence_mode == 'bias' and 'evidence_state' in state:
             ev = state['evidence_state']
             # [Fix] Handle EvidenceState object (dataclass)
             if ev is not None:
                 def get_ev(attr_name, default=None):
                     if isinstance(ev, dict):
                         return ev.get(attr_name, default)
                     return getattr(ev, attr_name, default)

                 # Check if it has attributes (dataclass) or keys (dict - legacy)
                 support = get_ev('support_score')
                 
                 if support is not None:
                     contra = get_ev('contradiction_score', torch.zeros_like(support))
                     consistency = get_ev('reaction_consistency', torch.zeros_like(support))
                     suspect = get_ev('suspect_pool', torch.zeros_like(support))
                     # Ensure dims [N, 1]
                     if support.dim() == 1: support = support.unsqueeze(-1)
                     if contra.dim() == 1: contra = contra.unsqueeze(-1)
                     if consistency.dim() == 1: consistency = consistency.unsqueeze(-1)
                     if suspect.dim() == 1: suspect = suspect.unsqueeze(-1)

                 if support is not None:
                     # Align to Fused Space
                     # [Cleanup Phase A] Removed redundant Raw->Fused alignment.
                     # EvidenceState MUST be in Fused Space per System v2 contract.
                     if support.size(0) != num_fused_nodes:
                         raise ValueError(f"EvidenceState mismatch: Expected Fused size {num_fused_nodes}, got {support.size(0)}. Check EvidenceBuilder.")
                     
                     aligned_support = self._calibrate_support_signal(support)
                     aligned_contra = contra
                     aligned_consistency = consistency
                     aligned_suspect = suspect
                     if hasattr(graph, 'batch') and graph.batch is not None and graph.batch.numel() == num_fused_nodes:
                         aligned_contra = self._calibrate_contradiction_signal(aligned_contra, graph.batch)
                     else:
                         aligned_contra = self._calibrate_contradiction_signal(aligned_contra, None)
                     
                     # EvidenceState v1 mainline: support is primary. Suspect/contradiction are opt-in.
                     signal = self.support_weight * aligned_support
                     signal = signal + self.suspect_weight * aligned_suspect
                     signal = signal - self.contradiction_weight * aligned_contra
                     signal = signal + self.reaction_weight * aligned_consistency
                     evidence_bias = signal * self.evidence_scale
                     
                     logits = logits + evidence_bias
                     
                     # Trace
                     if not self.training and torch.rand(1).item() < 0.001:
                         print(f"[Evidence] Injected Bias. Scale={self.evidence_scale}, Mean={evidence_bias.mean().item():.4f}")

        # Apply Logit Bias (e.g., Race Energy) from physics_ctx
        energy = None
        if physics_ctx and 'logit_bias' in physics_ctx:
            # [Audit 3] Logit Bias Trace & Soft Masking
            bias = physics_ctx['logit_bias']
            
            # Trace stats before applying
            if False and torch.rand(1).item() < 0.01:
                print(f"[Bias Trace] Bias Mean: {bias.mean().item():.4f}, Std: {bias.std().item():.4f}, Min: {bias.min().item():.4f}, Max: {bias.max().item():.4f}")

            bias_terms = self._decompose_internal_physics_terms(physics_ctx, state.get('evidence_state'))
            if bias_terms is not None:
                logits = logits + bias_terms['total_consumed']
            else:
                bias = self._prepare_internal_physics_bias(bias)
                logits = logits + bias

            if 'race_energy' in physics_ctx:
                energy = physics_ctx['race_energy']
        
        # [Audit 3] Head Trace
        if False and torch.rand(1).item() < 0.01: # Sample 1% of steps to avoid spam
             print(f"[Head Trace] Logits Mean: {logits.mean().item():.4f}, Std: {logits.std().item():.4f}")
             print(f"[Head Trace] Prior Mean: {logits_prior.mean().item():.4f}, Std: {logits_prior.std().item():.4f}")
             print(f"[Head Trace] Like Mean: {logits_like.mean().item():.4f}, Std: {logits_like.std().item():.4f}")
             print(f"[Head Trace] Alpha: {self.alpha.item():.4f}")

        return {
            'logits': logits,
            'logits_like': logits_like,
            'logits_prior': logits_prior,
            'energy': energy
        }
