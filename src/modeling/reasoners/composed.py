import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional
from src.modeling.interfaces.base import ReasonerBase, ReasonerCapabilities
from src.modeling.registry import REASONER_REGISTRY
from src.modeling.encoders.observation_encoder import ObservationEncoder

@REASONER_REGISTRY.register("composed_reasoner_v4_5")
class ComposedReasoner(ReasonerBase):
    """
    Modular Reasoner: Composes Backbone x Head x Memory.
    """
    def __init__(self, backbone, head, memory, cfg):
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.memory = memory
        self.cfg = cfg
        reasoner_cfg = getattr(cfg.model, 'reasoner', {})
        if isinstance(reasoner_cfg, dict):
            hidden_dim = int(reasoner_cfg.get('hidden_dim', getattr(cfg.model, 'hidden_dim', 64)))
            self.use_evidence = reasoner_cfg.get('use_evidence', False)
            self.evidence_mode = reasoner_cfg.get('evidence_mode', 'concat')
            self.concat_fields = reasoner_cfg.get('concat_fields', ['support_score', 'uncertainty_gap'])
        else:
            hidden_dim = int(getattr(reasoner_cfg, 'hidden_dim', getattr(cfg.model, 'hidden_dim', 64)))
            self.use_evidence = getattr(reasoner_cfg, 'use_evidence', False)
            self.evidence_mode = getattr(reasoner_cfg, 'evidence_mode', 'concat')
            self.concat_fields = getattr(reasoner_cfg, 'concat_fields', ['support_score', 'uncertainty_gap'])

        self.obs_encoder = ObservationEncoder(hidden_dim, self.use_evidence, self.evidence_mode, self.concat_fields)
        backbone_input_dim = getattr(self.backbone, 'input_dim', hidden_dim)
        self.input_bridge = nn.Identity() if backbone_input_dim == hidden_dim else nn.Linear(hidden_dim, backbone_input_dim)
        
        # Internal state for memory (managed by Orchestrator usually, but kept here for convenience)
        self.current_memory_state = None

    def forward(self, state: Dict[str, Any], graph: Any, physics_ctx: Dict[str, Any] = None, belief_ctx: Dict[str, Any] = None) -> Dict[str, Any]:
        # 1. Contracted encoding through ObservationState / EvidenceState.
        h_structure = state.get('h_fused')
        if h_structure is not None:
            num_fused_nodes = h_structure.size(0)
        else:
            obs_state = state.get('observation_state')
            if obs_state is None:
                raise ValueError("ComposedReasoner requires observation_state or h_fused for size alignment")
            num_fused_nodes = obs_state.observed_flag.size(0)
        encoded = self.obs_encoder(state, num_fused_nodes)
        x_reasoner = self.input_bridge(encoded)
        firewall_report = {
            'input_contract': 'ObservationState/EvidenceState',
            'legacy_adapter_bypassed': True,
            'num_nodes': int(num_fused_nodes),
        }
        
        # 2. Backbone
        h_nodes = self.backbone(x_reasoner, graph.edge_index, getattr(graph, 'edge_attr', None))
        
        # 3. Recurrent Belief Injection (SSOT driven)
        # We replace the old self.memory with belief_ctx managed by Orchestrator
        belief_h = None
        belief_injection_enabled = getattr(self.cfg.model, 'inject_belief_to_reasoner', False)
        if belief_injection_enabled and belief_ctx is not None:
            if 'node_h' in belief_ctx:
                belief_h = belief_ctx['node_h']
            elif 'global_h' in belief_ctx:
                belief_h = belief_ctx['global_h'][graph.batch]
        
        # 4. Head
        logits = self.head(h_nodes, belief_h)

        updated_memory_state = None
        if self.memory is not None:
            batch = getattr(graph, 'batch', None)
            if batch is None:
                batch = torch.zeros(h_nodes.size(0), dtype=torch.long, device=h_nodes.device)
            batch_size = int(batch.max().item()) + 1 if batch.numel() > 0 else 1
            current_memory_state = state.get('memory_state', self.current_memory_state)
            if current_memory_state is None and hasattr(self.memory, 'init_state'):
                current_memory_state = self.memory.init_state(batch_size, h_nodes.device)
            if hasattr(self.memory, 'step_update'):
                updated_memory_state = self.memory.step_update(current_memory_state, {
                    'node_embeddings': h_nodes,
                    'batch': batch,
                })
                self.current_memory_state = updated_memory_state
        
        # 5. Physics Bias (Gating/Energy)
        if physics_ctx is not None:
            if 'logit_bias' in physics_ctx:
                logits = logits + physics_ctx['logit_bias']
            
        return {
            'logits': logits,
            'probs': F.softmax(logits, dim=0), # Note: per-graph softmax is usually done by Orchestrator
            'firewall_report': firewall_report,
            'updated_memory_state': updated_memory_state,
        }

    def capabilities(self) -> ReasonerCapabilities:
        return {
            'supports_memory': self.memory is not None,
            'requires_memory': self.memory is not None and not isinstance(self.memory, nn.Identity),
            'supports_dense_supervision': True,
            'supports_physics_ctx': True
        }
