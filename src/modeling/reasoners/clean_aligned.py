from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, SAGEConv
from torch_scatter import scatter_mean

from src.modeling.clean_aligned_features import (
    GRAPH_FEATURE_DIM,
    NODE_FEATURE_DIM,
    build_clean_aligned_feature_payload,
)
from src.modeling.interfaces.base import ReasonerBase, ReasonerCapabilities
from src.modeling.registry import REASONER_REGISTRY


EVIDENCE_CORE_INDICES = (0, 1, 2, 3, 4, 6)
EVIDENCE_CORE_FIELDS = (
    "support_score",
    "contradiction_score",
    "support_bounded",
    "contradiction_bounded",
    "live_plausibility",
    "ignorance_mass",
)
ALL_NODE_FEATURE_INDICES = tuple(range(NODE_FEATURE_DIM))
AUXILIARY_FEATURE_INDICES = tuple(idx for idx in ALL_NODE_FEATURE_INDICES if idx not in EVIDENCE_CORE_INDICES)
EVIDENCE_CONTRAST_BASE_INDICES = tuple(range(7))


@REASONER_REGISTRY.register("clean_aligned_reasoner_mainline")
class CleanAlignedReasoner(ReasonerBase):
    """
    Reasoner mainline that consumes the same clean state-to-feature contract used
    by the frozen clean navigator bridge, while keeping a simple source-logit head.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        reasoner_cfg = getattr(cfg.model, "reasoner", {})
        if isinstance(reasoner_cfg, dict):
            hidden_dim = int(reasoner_cfg.get("hidden_dim", getattr(cfg.model, "hidden_dim", 128)))
            num_layers = int(reasoner_cfg.get("aligned_num_layers", 2))
            self.frontier_mode = str(reasoner_cfg.get("aligned_frontier_mode", "unresolved_without_pair"))
            self.enable_evidence_core_contrast_adapter = bool(reasoner_cfg.get("enable_evidence_core_contrast_adapter", False))
            self.evidence_core_contrast_mode = str(reasoner_cfg.get("evidence_core_contrast_mode", "residual"))
        else:
            hidden_dim = int(getattr(reasoner_cfg, "hidden_dim", getattr(cfg.model, "hidden_dim", 128)))
            num_layers = int(getattr(reasoner_cfg, "aligned_num_layers", 2))
            self.frontier_mode = str(getattr(reasoner_cfg, "aligned_frontier_mode", "unresolved_without_pair"))
            self.enable_evidence_core_contrast_adapter = bool(getattr(reasoner_cfg, "enable_evidence_core_contrast_adapter", False))
            self.evidence_core_contrast_mode = str(getattr(reasoner_cfg, "evidence_core_contrast_mode", "residual"))

        self.node_proj = nn.Sequential(
            nn.Linear(NODE_FEATURE_DIM, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.convs = nn.ModuleList(
            [SAGEConv(hidden_dim, hidden_dim, aggr="mean") for _ in range(max(num_layers, 1))]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(max(num_layers, 1))])
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2 + GRAPH_FEATURE_DIM, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        if self.enable_evidence_core_contrast_adapter:
            self.evidence_core_contrast_adapter = nn.Linear(28, len(EVIDENCE_CONTRAST_BASE_INDICES))

    def _compute_evidence_core_contrast(
        self,
        node_features: torch.Tensor,
        valid_mask: torch.Tensor | None,
        batch_index: torch.Tensor,
    ) -> torch.Tensor:
        core = node_features[:, EVIDENCE_CONTRAST_BASE_INDICES].float()
        valid_mask = (
            valid_mask.view(-1).bool().to(device=node_features.device)
            if isinstance(valid_mask, torch.Tensor)
            else torch.ones(core.size(0), dtype=torch.bool, device=node_features.device)
        )
        contrast = torch.zeros(core.size(0), len(EVIDENCE_CONTRAST_BASE_INDICES) * 3, device=node_features.device, dtype=core.dtype)
        graph_count = int(batch_index.max().item()) + 1 if batch_index.numel() > 0 else 0
        for graph_id in range(graph_count):
            graph_mask = batch_index == int(graph_id)
            if not bool(graph_mask.any().item()):
                continue
            graph_core = core[graph_mask]
            graph_valid = valid_mask[graph_mask]
            if not bool(graph_valid.any().item()):
                continue
            valid_core = graph_core[graph_valid]
            means = valid_core.mean(dim=0, keepdim=True)
            stds = valid_core.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)
            maxs = valid_core.max(dim=0, keepdim=True).values
            percentile_cols = []
            for dim_idx in range(valid_core.size(1)):
                sorted_vals = torch.sort(valid_core[:, dim_idx])[0]
                ranks = torch.searchsorted(sorted_vals, graph_core[:, dim_idx].contiguous(), right=True).float() / max(int(sorted_vals.numel()), 1)
                percentile_cols.append(ranks.view(-1, 1))
            percentile_rank = torch.cat(percentile_cols, dim=1)
            z_score = (graph_core - means) / stds
            gap_to_max = graph_core - maxs
            graph_contrast = torch.cat([percentile_rank, z_score, gap_to_max], dim=1)
            graph_contrast[~graph_valid] = 0.0
            contrast[graph_mask] = graph_contrast
        return contrast

    def _apply_evidence_core_contrast_adapter(
        self,
        node_features: torch.Tensor,
        valid_mask: torch.Tensor | None,
        batch_index: torch.Tensor,
    ) -> torch.Tensor:
        if not self.enable_evidence_core_contrast_adapter:
            return node_features
        adapted = node_features.float().clone()
        core = adapted[:, EVIDENCE_CONTRAST_BASE_INDICES].float()
        tail = adapted[:, len(EVIDENCE_CONTRAST_BASE_INDICES) :].float()
        contrast = self._compute_evidence_core_contrast(adapted, valid_mask, batch_index)
        adapter_input = torch.cat([core, contrast], dim=1)
        delta = self.evidence_core_contrast_adapter(adapter_input)
        if self.evidence_core_contrast_mode == "replace":
            adapted_core = delta
        else:
            adapted_core = core + delta
        return torch.cat([adapted_core, tail], dim=1)

    def _encode(self, node_features: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        node_features = node_features.float().clone()
        first_two = torch.sign(node_features[:, :2]) * torch.log1p(node_features[:, :2].abs())
        node_features = torch.cat([first_two, node_features[:, 2:]], dim=1)
        h = self.node_proj(node_features)
        for conv, norm in zip(self.convs, self.norms):
            residual = h
            h = conv(h, edge_index)
            h = norm(F.relu(h) + residual)
        return h

    def forward(self, state: Dict[str, Any], graph: Any, physics_ctx: Dict[str, Any] = None) -> Dict[str, Any]:
        if hasattr(graph, "batch") and graph.batch is not None:
            batch_index = graph.batch.view(-1).long()
        else:
            node_count = int(state["observation_state"].observed_flag.numel())
            batch_index = torch.zeros(node_count, dtype=torch.long, device=graph.edge_index.device)

        node_features = state.get("clean_aligned_node_features")
        graph_features_by_graph = state.get("clean_aligned_graph_features")
        if node_features is None or graph_features_by_graph is None:
            payload = build_clean_aligned_feature_payload(
                state,
                batch_index=batch_index,
                edge_index=graph.edge_index.view(2, -1).long(),
                physics_ctx=physics_ctx,
                frontier_mode=self.frontier_mode,
            )
            node_features = payload["node_features"].float()
            graph_features_by_graph = payload["graph_features_by_graph"].float()
            input_contract = "clean_aligned_navigator_rl_features"
        else:
            node_features = node_features.float()
            graph_features_by_graph = graph_features_by_graph.float()
            input_contract = "precomputed_clean_aligned_bank_features"

        node_features = self._apply_evidence_core_contrast_adapter(
            node_features,
            state.get("valid_mask"),
            batch_index,
        )
        h = self._encode(node_features, graph.edge_index)
        graph_context = scatter_mean(
            h,
            batch_index,
            dim=0,
            dim_size=graph_features_by_graph.size(0),
        )[batch_index]
        graph_features = graph_features_by_graph[batch_index]
        logits = self.head(torch.cat([h, graph_context, graph_features], dim=1))

        firewall_report = {
            "input_contract": input_contract,
            "shared_feature_builder": "src.modeling.clean_aligned_features.build_clean_aligned_feature_payload",
            "legacy_bias_bypassed": True,
            "num_nodes": int(node_features.size(0)),
            "evidence_core_contrast_adapter_enabled": bool(self.enable_evidence_core_contrast_adapter),
            "evidence_core_contrast_mode": str(self.evidence_core_contrast_mode),
        }
        return {
            "logits": logits,
            "probs": F.softmax(logits, dim=0),
            "firewall_report": firewall_report,
            "updated_memory_state": None,
        }

    def capabilities(self) -> ReasonerCapabilities:
        return {
            "supports_memory": False,
            "requires_memory": False,
            "supports_dense_supervision": True,
            "supports_physics_ctx": True,
        }


@REASONER_REGISTRY.register("clean_aligned_reasoner_evidence_gat_v1")
class EvidenceCenteredGATReasoner(ReasonerBase):
    """
    Minimal consumer-only upgrade on top of the clean-aligned feature contract.
    Support and contradiction drive a dedicated evidence path that is preserved
    through GATv2 propagation via per-layer evidence residual injection.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        reasoner_cfg = getattr(cfg.model, "reasoner", {})
        if isinstance(reasoner_cfg, dict):
            hidden_dim = int(reasoner_cfg.get("hidden_dim", getattr(cfg.model, "hidden_dim", 128)))
            self.frontier_mode = str(reasoner_cfg.get("aligned_frontier_mode", "unresolved_without_pair"))
            num_layers = int(reasoner_cfg.get("evidence_gat_num_layers", 2))
            heads = int(reasoner_cfg.get("evidence_gat_heads", 4))
            dropout = float(reasoner_cfg.get("evidence_gat_dropout", 0.0))
        else:
            hidden_dim = int(getattr(reasoner_cfg, "hidden_dim", getattr(cfg.model, "hidden_dim", 128)))
            self.frontier_mode = str(getattr(reasoner_cfg, "aligned_frontier_mode", "unresolved_without_pair"))
            num_layers = int(getattr(reasoner_cfg, "evidence_gat_num_layers", 2))
            heads = int(getattr(reasoner_cfg, "evidence_gat_heads", 4))
            dropout = float(getattr(reasoner_cfg, "evidence_gat_dropout", 0.0))

        self.hidden_dim = hidden_dim
        self.evidence_indices = torch.tensor(EVIDENCE_CORE_INDICES, dtype=torch.long)
        self.auxiliary_indices = torch.tensor(AUXILIARY_FEATURE_INDICES, dtype=torch.long)

        self.evidence_proj = nn.Sequential(
            nn.Linear(len(EVIDENCE_CORE_INDICES), hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.aux_proj = nn.Sequential(
            nn.Linear(len(AUXILIARY_FEATURE_INDICES), hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.fuse = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )

        if hidden_dim % heads != 0:
            raise ValueError(
                f"clean_aligned_reasoner_evidence_gat_v1 requires hidden_dim divisible by heads, got {hidden_dim=} and {heads=}."
            )

        self.convs = nn.ModuleList(
            [
                GATv2Conv(
                    hidden_dim,
                    hidden_dim // heads,
                    heads=heads,
                    concat=True,
                    dropout=dropout,
                )
                for _ in range(max(num_layers, 1))
            ]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(max(num_layers, 1))])
        self.evidence_residuals = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim, bias=False) for _ in range(max(num_layers, 1))])
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 4 + GRAPH_FEATURE_DIM, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def _prepare_inputs(self, node_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        node_features = node_features.float().clone()
        first_two = torch.sign(node_features[:, :2]) * torch.log1p(node_features[:, :2].abs())
        node_features = torch.cat([first_two, node_features[:, 2:]], dim=1)
        evidence_idx = self.evidence_indices.to(device=node_features.device)
        auxiliary_idx = self.auxiliary_indices.to(device=node_features.device)
        evidence_features = node_features.index_select(1, evidence_idx)
        auxiliary_features = node_features.index_select(1, auxiliary_idx)
        return evidence_features, auxiliary_features

    def _encode(self, node_features: torch.Tensor, edge_index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        evidence_features, auxiliary_features = self._prepare_inputs(node_features)
        evidence_embedding = self.evidence_proj(evidence_features)
        auxiliary_embedding = self.aux_proj(auxiliary_features)
        h = self.fuse(torch.cat([evidence_embedding, auxiliary_embedding], dim=1))
        for conv, norm, evidence_residual in zip(self.convs, self.norms, self.evidence_residuals):
            residual = h
            propagated = F.elu(conv(h, edge_index))
            h = norm(residual + propagated + evidence_residual(evidence_embedding))
        return evidence_embedding, h

    def forward(self, state: Dict[str, Any], graph: Any, physics_ctx: Dict[str, Any] = None) -> Dict[str, Any]:
        if hasattr(graph, "batch") and graph.batch is not None:
            batch_index = graph.batch.view(-1).long()
        else:
            node_count = int(state["observation_state"].observed_flag.numel())
            batch_index = torch.zeros(node_count, dtype=torch.long, device=graph.edge_index.device)

        node_features = state.get("clean_aligned_node_features")
        graph_features_by_graph = state.get("clean_aligned_graph_features")
        if node_features is None or graph_features_by_graph is None:
            payload = build_clean_aligned_feature_payload(
                state,
                batch_index=batch_index,
                edge_index=graph.edge_index.view(2, -1).long(),
                physics_ctx=physics_ctx,
                frontier_mode=self.frontier_mode,
            )
            node_features = payload["node_features"].float()
            graph_features_by_graph = payload["graph_features_by_graph"].float()
            input_contract = "clean_aligned_navigator_rl_features"
        else:
            node_features = node_features.float()
            graph_features_by_graph = graph_features_by_graph.float()
            input_contract = "precomputed_clean_aligned_bank_features"

        evidence_embedding, hidden = self._encode(node_features, graph.edge_index)
        hidden_graph_context = scatter_mean(
            hidden,
            batch_index,
            dim=0,
            dim_size=graph_features_by_graph.size(0),
        )[batch_index]
        evidence_graph_context = scatter_mean(
            evidence_embedding,
            batch_index,
            dim=0,
            dim_size=graph_features_by_graph.size(0),
        )[batch_index]
        graph_features = graph_features_by_graph[batch_index]
        logits = self.head(
            torch.cat(
                [
                    evidence_embedding,
                    hidden,
                    hidden_graph_context,
                    evidence_graph_context,
                    graph_features,
                ],
                dim=1,
            )
        )

        firewall_report = {
            "input_contract": input_contract,
            "shared_feature_builder": "src.modeling.clean_aligned_features.build_clean_aligned_feature_payload",
            "legacy_bias_bypassed": True,
            "num_nodes": int(node_features.size(0)),
            "consumer_architecture": "evidence_centered_gat_v1",
            "evidence_core_fields": list(EVIDENCE_CORE_FIELDS),
        }
        return {
            "logits": logits,
            "probs": F.softmax(logits, dim=0),
            "firewall_report": firewall_report,
            "updated_memory_state": None,
        }

    def capabilities(self) -> ReasonerCapabilities:
        return {
            "supports_memory": False,
            "requires_memory": False,
            "supports_dense_supervision": True,
            "supports_physics_ctx": True,
        }
