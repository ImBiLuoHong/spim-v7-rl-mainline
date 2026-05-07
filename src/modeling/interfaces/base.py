from typing import Dict, Any, List, Optional, Union, TypedDict
import torch
import torch.nn as nn

class NavigatorCapabilities(TypedDict, total=False):
    supports_soft_actions: bool
    provides_khot_mask: bool
    supports_without_replacement: bool
    supports_candidate_topM_masking: bool
    output_fields: List[str]

class ReasonerCapabilities(TypedDict, total=False):
    supports_memory: bool
    supports_bayesian_fusion: bool
    output_fields: List[str]

class LossRequirements(TypedDict, total=False):
    requires_soft_actions: bool
    requires_stepwise_probs: bool
    requires_active_mask: bool
    required_fields: List[str]

class PhysicsCapabilities(TypedDict, total=False):
    provides_race_energy: bool
    provides_bias: bool
    provides_neg_gating: bool
    output_fields: List[str]

class TrajectoryStep(TypedDict, total=False):
    # Step identifiers
    t_sim: float
    step_idx: int
    
    # Model Outputs
    reasoner_logits: torch.Tensor
    reasoner_probs: torch.Tensor
    nav_logits: Optional[torch.Tensor]
    nav_probs: Optional[torch.Tensor]
    soft_khot_mask: Optional[torch.Tensor]
    selected_indices: Optional[torch.Tensor]  # Hard action indices
    value: Optional[torch.Tensor]
    
    # Context & Masks
    active_mask: torch.Tensor  # [B]
    candidate_topM_mask: Optional[torch.Tensor]
    physics_ctx_summary: Optional[Dict[str, Any]]
    
    # Metadata for debugging/audit
    entropy: Optional[torch.Tensor]
    
    # Ground Truth (fused during training/eval)
    fused_source_label: Optional[torch.Tensor]
    fused_batch: Optional[torch.Tensor]
    is_hit: Optional[torch.Tensor]

class NavigatorBackboneBase(nn.Module):
    """
    Backbone Interface: N-to-N Graph Encoder.
    """
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Returns: Node embeddings [N, D]
        """
        raise NotImplementedError

class NavigatorHeadBase(nn.Module):
    """
    Head Interface: Maps node embeddings to logits.
    """
    def forward(self, node_embeddings: torch.Tensor, state: Dict[str, Any]) -> torch.Tensor:
        """
        Returns: Logits [N, 1] or [N, D]
        """
        raise NotImplementedError
    
    def capabilities(self) -> NavigatorCapabilities:
        return {}

class ActionSamplerBase(nn.Module):
    """
    Sampler Interface: Converts logits to discrete or soft actions.
    """
    def forward(self, logits: torch.Tensor, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Args:
            logits: Output from Head [N, 1]
            state: Context containing 'valid_mask', 'k_explore', etc.
        Returns: Dict containing:
            - 'selected_indices': [K] (hard actions)
            - 'soft_khot_mask': [N] (optional)
            - 'action_audit': Dict (entropy, budget_ok, etc.)
        """
        raise NotImplementedError

    def capabilities(self) -> NavigatorCapabilities:
        return {}

class NavigatorBase(nn.Module):
    """
    Navigator Interface: Responsible for selecting next nodes to sample.
    """
    def forward(self, state: Dict[str, Any], graph: Any, physics_ctx: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Returns: Dict containing 'logits', 'probs', and optionally 'value'.
        """
        raise NotImplementedError

    def capabilities(self) -> NavigatorCapabilities:
        return {}

class ReasonerBackboneBase(nn.Module):
    """
    Backbone Interface for Reasoner: Node Embeddings (N x D)
    """
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: Optional[torch.Tensor] = None) -> torch.Tensor:
        raise NotImplementedError

class ReasonerHeadBase(nn.Module):
    """
    Head Interface for Reasoner: Node Embeddings + (Optional Memory) -> Logits
    """
    def forward(self, node_embeddings: torch.Tensor, memory_state: Optional[torch.Tensor] = None) -> torch.Tensor:
        raise NotImplementedError

class MemoryBase(nn.Module):
    """
    Memory Interface for Reasoner: Recursive Belief Update (RNN-like)
    """
    def init_state(self, batch_size: int, device: torch.device) -> Any:
        raise NotImplementedError
        
    def step_update(self, current_state: Any, step_ctx: Dict[str, Any]) -> Any:
        raise NotImplementedError

class ReasonerBase(nn.Module):
    """
    Reasoner Interface: Responsible for estimating the source posterior.
    """
    def forward(self, state: Dict[str, Any], graph: Any, physics_ctx: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Args:
            state: Current hidden state / evidence encoding.
            graph: PyG graph object.
            physics_ctx: Context from PhysicsConsistency module.
        Returns:
            Dict containing 'logits', 'probs', 'top1_idx', and 'updated_memory_state' if applicable.
        """
        raise NotImplementedError

    def capabilities(self) -> ReasonerCapabilities:
        return {}

class PhysicsConsistencyBase(nn.Module):
    """
    PhysicsConsistency Interface: Translates hydraulic rules to soft constraints.
    """
    def compute(self, graph: Any, t_sim: Any) -> Dict[str, Any]:
        """
        Legacy interface for Orchestrator.
        """
        raise NotImplementedError

    def __call__(self, physics_in: Dict[str, Any]) -> Dict[str, Any]:
        # Firewall: Only allowed fields
        allowed = {
            't_sim', 'valid_mask', 'anchor_type', 'anchor_time', 
            'edge_index', 'edge_stt', 'batch', 'reasoner_posterior'
        }
        for k in physics_in.keys():
            if k not in allowed:
                raise ValueError(f"Physics Firewall Violation: Field '{k}' is NOT allowed in physics_in.")
        return super().__call__(physics_in)

    def forward(self, physics_in: Dict[str, Any]) -> Dict[str, Any]:
        """
        Args:
            physics_in: White-listed input fields:
                - 't_sim': torch.Tensor [B]
                - 'valid_mask': torch.Tensor [N]
                - 'anchor_type': torch.Tensor [N] (+1: pos, -1: neg, 0: none)
                - 'anchor_time': torch.Tensor [N]
                - 'edge_index': torch.Tensor [2, E]
                - 'edge_stt': torch.Tensor [E] (median/p90/min STT)
                - 'batch': torch.Tensor [N]
                - 'reasoner_posterior': Optional[torch.Tensor] [N]
        Returns:
            physics_ctx: Dict containing:
                - 'race_energy': [N]
                - 'bias': [N] (to be subtracted from logits)
                - 'neg_gating_summary': [N] or [E]
                - 'audit': Dict with t_sim, num_anchors, stats, etc.
        """
        raise NotImplementedError

    def capabilities(self) -> PhysicsCapabilities:
        """
        Returns: PhysicsCapabilities declaring what this module provides.
        """
        return {}

class RaceEnergyBase(nn.Module):
    """
    RaceEnergy Interface: Calculates E_t(s) consistency energy.
    """
    def forward(self, physics_in: Dict[str, Any]) -> torch.Tensor:
        raise NotImplementedError

class NegativeGatingBase(nn.Module):
    """
    NegativeGating Interface: Calculates time-conditioned gating g(t).
    """
    def forward(self, physics_in: Dict[str, Any]) -> torch.Tensor:
        raise NotImplementedError

class FoVControllerBase:
    """
    FoVController Interface: Dynamic Field-of-View regulation.
    """
    def step(self, stats_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Args:
            stats_dict: Entropy, conflict energy, evidence density, etc.
        Returns:
            fov_params: candidate_topM, message_passing_depth, etc.
        """
        raise NotImplementedError

class LossEngineBase(nn.Module):
    """
    LossEngine Interface: Centralized credit assignment.
    """
    def forward(self, trajectory: List[TrajectoryStep], cfg: Any = None) -> Dict[str, Any]:
        """
        Returns: loss_dict: 'total_loss' + individual components.
        """
        raise NotImplementedError

    def requirements(self) -> LossRequirements:
        return {}

class CurriculumSchedulerBase:
    """
    Curriculum Interface: Manages training difficulty knobs.
    """
    def get_params(self, epoch: int) -> Dict[str, Any]:
        """
        Returns: max_train_steps, fov_strength, loss_weights, etc.
        """
        raise NotImplementedError

class MetricsBase:
    """
    Metrics Interface: Unified evaluation statistics.
    """
    def update(self, batch_out: Dict[str, Any], batch_gt: Any):
        raise NotImplementedError

    def compute(self) -> Dict[str, Any]:
        raise NotImplementedError
