from typing import Dict, Any, List, Optional, Union, TypedDict
import torch
import torch.nn as nn
from dataclasses import dataclass

@dataclass
class BeliefStateBase:
    """
    Abstract container for persistent cross-step states.
    """
    def detach(self):
        """Used for BPTT truncation or debugging."""
        raise NotImplementedError

    def to(self, device: torch.device):
        """Supports device transfer."""
        raise NotImplementedError

class BeliefUpdaterCapabilities(TypedDict, total=False):
    provides_node_belief: bool
    provides_global_belief: bool
    provides_memory_bank: bool
    output_fields: List[str]

class BeliefUpdaterBase(nn.Module):
    """
    Core interface for Recursive Belief Update (Generalized RNN).
    """
    def init_state(self, batch_size: int, num_nodes: Optional[int] = None, device: Optional[torch.device] = None) -> BeliefStateBase:
        """Initialize the hidden state for a new episode."""
        raise NotImplementedError

    def step(self, state: BeliefStateBase, step_in: Dict[str, Any]) -> tuple[BeliefStateBase, Dict[str, Any]]:
        """
        Perform one step of belief update.
        Args:
            state: Current hidden state.
            step_in: White-listed input fields:
                - 't_sim': torch.Tensor [B]
                - 'valid_mask': torch.Tensor [N]
                - 'anchor_type': torch.Tensor [N]
                - 'anchor_time': torch.Tensor [N]
                - 'freshness': torch.Tensor [N]
                - 'reasoner_posterior': Optional[torch.Tensor] [N]
                - 'physics_ctx': Dict from physics module
                - 'fov_params': Dict from FoV controller
                - 'action_summary': Dict of sampling behavior
                - 'node_embeddings': Optional[torch.Tensor] [N, D]
        Returns:
            new_state: Updated hidden state.
            belief_ctx: Audit and feature dictionary.
        """
        self._validate_input(step_in)
        return self._step_impl(state, step_in)

    def _step_impl(self, state: BeliefStateBase, step_in: Dict[str, Any]) -> tuple[BeliefStateBase, Dict[str, Any]]:
        raise NotImplementedError

    def capabilities(self) -> BeliefUpdaterCapabilities:
        return {}

    def _validate_input(self, step_in: Dict[str, Any]):
        """Feature Firewall: Whitelist enforcement."""
        allowed_keys = {
            't_sim', 'valid_mask', 'anchor_type', 'anchor_time', 'freshness',
            'reasoner_posterior', 'physics_ctx', 'fov_params', 'action_summary',
            'node_embeddings', 'step_idx', 'batch',
            'evidence_state', 'constraint_state', 'reasoner_logits',
            'node_features', 'graph_features',
            'history', 'global_node_ids', 'topology', 'episode_duration_min',
            'source_local'
        }
        forbidden_keys = {'poison_label', 'gt_source', 'future_obs'}
        
        for key in step_in.keys():
            if key in forbidden_keys or key not in allowed_keys:
                raise ValueError(f"BeliefUpdater: Forbidden or non-whitelisted key detected in step_in: {key}")
