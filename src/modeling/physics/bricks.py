import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Optional
from src.modeling.interfaces.base import PhysicsConsistencyBase, RaceEnergyBase, NegativeGatingBase, PhysicsCapabilities
from src.modeling.registry import PHYSICS_REGISTRY, RACE_ENERGY_REGISTRY, NEG_GATING_REGISTRY

# -----------------------------------------------------------------------------
# Sub-modules: Energy and Gating (Standard Implementations)
# -----------------------------------------------------------------------------

@RACE_ENERGY_REGISTRY.register("standard_race_energy")
class StandardRaceEnergy(RaceEnergyBase):
    def __init__(self, cfg):
        super().__init__()
        # Use cfg.physics.rea_rules
        phys_cfg = getattr(cfg.physics, 'rea_rules', {})
        if isinstance(phys_cfg, object): # It's a dataclass
             self.energy_form = getattr(phys_cfg, 'energy_form', 'softplus')
        else: # Fallback
             self.energy_form = 'softplus'
        
    def compute_components(self, physics_in: Dict[str, Any]):
        t_sim = physics_in['t_sim']          # [B]
        batch = physics_in['batch']          # [N]
        anchor_type = physics_in['anchor_type'] # [N]
        anchor_time = physics_in['anchor_time'] # [N]
        edge_index = physics_in['edge_index']   # [2, E]
        edge_stt = physics_in['edge_stt']       # [E]
        
        if edge_index.numel() == 0:
            zeros = torch.zeros_like(anchor_type)
            return zeros, zeros
            
        src, dst = edge_index
        t_curr = t_sim[batch[src]]
        
        # (t - t_v) - STT(u->v)
        delta = (t_curr - anchor_time[dst]) - edge_stt
        
        is_pos = (anchor_type[dst] > 0.5).float()
        is_neg = (anchor_type[dst] < -0.5).float()
        
        if self.energy_form == 'softplus':
            e_pos_edge = is_pos * F.softplus(-delta)
            e_neg_edge = is_neg * F.softplus(delta)
        elif self.energy_form == 'margin':
            margin = 1.0
            e_pos_edge = is_pos * F.relu(margin - delta)
            e_neg_edge = is_neg * F.relu(delta + margin)
        else:
            e_pos_edge = torch.zeros_like(delta)
            e_neg_edge = torch.zeros_like(delta)
            
        # Aggregate to nodes
        num_nodes = anchor_type.size(0)
        positive_timing_failure = torch.zeros((num_nodes, 1), device=anchor_type.device, dtype=anchor_type.dtype)
        negative_arrival_pressure = torch.zeros((num_nodes, 1), device=anchor_type.device, dtype=anchor_type.dtype)
        positive_timing_failure.scatter_add_(0, src.unsqueeze(-1), e_pos_edge.unsqueeze(-1))
        negative_arrival_pressure.scatter_add_(0, src.unsqueeze(-1), e_neg_edge.unsqueeze(-1))
        return positive_timing_failure, negative_arrival_pressure

    def forward(self, physics_in: Dict[str, Any]) -> torch.Tensor:
        positive_timing_failure, negative_arrival_pressure = self.compute_components(physics_in)
        energy = positive_timing_failure + negative_arrival_pressure
        return energy

@NEG_GATING_REGISTRY.register("standard_negative_gating")
class StandardNegativeGating(NegativeGatingBase):
    def __init__(self, cfg):
        super().__init__()
        phys_cfg = getattr(cfg.physics, 'rea_rules', {})
        if isinstance(phys_cfg, object):
            self.beta = getattr(phys_cfg, 'beta', 1.0)
        else:
            self.beta = 1.0
        
    def forward(self, physics_in: Dict[str, Any]) -> torch.Tensor:
        t_sim = physics_in['t_sim']
        batch = physics_in['batch']
        anchor_type = physics_in['anchor_type']
        anchor_time = physics_in['anchor_time']
        edge_index = physics_in['edge_index']
        edge_stt = physics_in['edge_stt']
        
        if edge_index.numel() == 0:
            return torch.ones((anchor_type.size(0), 1), device=anchor_type.device, dtype=anchor_type.dtype)
            
        src, dst = edge_index
        t_curr = t_sim[batch[src]]
        
        # g_{u->v}(t) = sigmoid( ((t - t_v) - STT(u->v)) / beta )
        delta = (t_curr - anchor_time[dst]) - edge_stt
        is_neg = (anchor_type[dst] < -0.5).float()
        
        gate = torch.sigmoid(delta / (abs(self.beta) + 1e-6))
        
        # Unified gate: 1.0 for non-neg, gate for neg
        final_gate = torch.where(is_neg > 0.5, gate, torch.ones_like(gate))
        
        return final_gate.unsqueeze(-1)

# -----------------------------------------------------------------------------
# Main PhysicsConsistency Implementations (Pluggable Bricks)
# -----------------------------------------------------------------------------

class BaseBrick(PhysicsConsistencyBase):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def _prepare_input(self, graph: Any, t_sim: torch.Tensor) -> Dict[str, Any]:
        # Extract fields from graph (SubGraph or PyG batch)
        # 7D edge_attr: [..., STT:3, anchor_time:4, Euclidean:5, anchor_type:6]
        edge_attr = graph.edge_attr
        edge_index = graph.edge_index
        
        # For anchor_type/time, we need node-level info. 
        # In our Phase 4.5 model, these are stored in edge_attr for virtual edges to anchors.
        # However, the interface expects node-level anchor_type/time.
        # We'll use a simplified mapping for the bricks.
        num_nodes = graph.x.size(0)
        device = graph.x.device
        
        # Mock node-level anchors from edge-level virtual attributes if possible
        # In a real scenario, these should be passed in reasoner_state
        # For now, we'll extract them from edge_attr where dst is an anchor
        anchor_type = torch.zeros(num_nodes, device=device)
        anchor_time = torch.zeros(num_nodes, device=device)
        
        if edge_attr is not None and edge_attr.size(1) >= 7:
            src, dst = edge_index
            # Use scatter_max to get anchor info if multiple edges point to same anchor
            # Actually, virtual edges point FROM model nodes TO anchors or vice-versa.
            # Based on StandardRaceEnergy, it expects anchor_type[dst].
            # So virtual edges should be (model_node -> anchor_node).
            anchor_type.scatter_(0, dst, edge_attr[:, 6])
            anchor_time.scatter_(0, dst, edge_attr[:, 4])
            
        return {
            't_sim': t_sim,
            'batch': graph.batch,
            'anchor_type': anchor_type,
            'anchor_time': anchor_time,
            'edge_index': edge_index,
            'edge_stt': edge_attr[:, 3] if edge_attr is not None else torch.zeros(edge_index.size(1), device=device),
            'valid_mask': torch.ones(num_nodes, device=device)
        }

    def compute(self, graph: Any, t_sim: torch.Tensor) -> Dict[str, Any]:
        physics_in = self._prepare_input(graph, t_sim)
        return self.forward(physics_in)

@PHYSICS_REGISTRY.register("physics_none")
class PhysicsNone(BaseBrick):
    def forward(self, physics_in: Dict[str, Any]) -> Dict[str, Any]:
        num_nodes = physics_in['anchor_type'].size(0)
        device = physics_in['anchor_type'].device
        dtype = physics_in['anchor_type'].dtype
        return {
            "race_energy": torch.zeros((num_nodes, 1), device=device, dtype=dtype),
            "logit_bias": torch.zeros((num_nodes, 1), device=device, dtype=dtype),
            "neg_gate": None,
            "energy_mean": 0.0,
            "energy_p95": 0.0,
            "gate_mean": 1.0,
            "conflict_rate": 0.0
        }
        
    def capabilities(self) -> PhysicsCapabilities:
        return {"provides_bias": False, "provides_neg_gating": False}

@PHYSICS_REGISTRY.register("physics_neg_gate")
class PhysicsNegGate(BaseBrick):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.gating_module = StandardNegativeGating(cfg)
        
    def forward(self, physics_in: Dict[str, Any]) -> Dict[str, Any]:
        gate = self.gating_module(physics_in)
        num_nodes = physics_in['anchor_type'].size(0)
        device = gate.device
        
        return {
            "race_energy": torch.zeros((num_nodes, 1), device=device),
            "logit_bias": torch.zeros((num_nodes, 1), device=device),
            "neg_gate": gate,
            "gate_mean": gate.mean().item()
        }
        
    def capabilities(self) -> PhysicsCapabilities:
        return {"provides_bias": False, "provides_neg_gating": True}

@PHYSICS_REGISTRY.register("physics_race_energy_v1")
class PhysicsRaceEnergyV1(BaseBrick):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.energy_module = StandardRaceEnergy(cfg)
        phys_cfg = getattr(cfg.physics, 'rea_rules', {})
        if isinstance(phys_cfg, object):
            self.lambda_energy = getattr(phys_cfg, 'lambda_energy', 1.0)
        else:
            self.lambda_energy = 1.0
        
    def forward(self, physics_in: Dict[str, Any]) -> Dict[str, Any]:
        energy = self.energy_module(physics_in)
        num_nodes = energy.size(0)
        
        return {
            "race_energy": energy,
            "logit_bias": -self.lambda_energy * energy,
            "neg_gate": None,
            "energy_mean": energy.mean().item(),
            "energy_p95": torch.quantile(energy.float(), 0.95).item() if energy.numel() > 0 else 0.0,
            "conflict_rate": (energy > 0.1).float().mean().item()
        }
        
    def capabilities(self) -> PhysicsCapabilities:
        return {"provides_bias": True, "provides_neg_gating": False}

@PHYSICS_REGISTRY.register("physics_race_energy_v2")
class PhysicsRaceEnergyV2(PhysicsRaceEnergyV1):
    def forward(self, physics_in: Dict[str, Any]) -> Dict[str, Any]:
        res = super().forward(physics_in)
        energy = res['race_energy']
        # steeper for E > 1, smoother for E < 1
        energy_v2 = torch.where(energy > 1.0, energy**1.5, energy**0.8)
        res['race_energy'] = energy_v2
        res['logit_bias'] = -self.lambda_energy * energy_v2
        res['energy_mean'] = energy_v2.mean().item()
        return res

@PHYSICS_REGISTRY.register("physics_hybrid")
class PhysicsHybrid(BaseBrick):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.gating_module = StandardNegativeGating(cfg)
        self.energy_module = StandardRaceEnergy(cfg)
        phys_cfg = getattr(cfg.physics, 'rea_rules', {})
        if isinstance(phys_cfg, object):
            self.lambda_energy = getattr(phys_cfg, 'lambda_energy', 1.0)
        else:
            self.lambda_energy = 1.0
        
    def forward(self, physics_in: Dict[str, Any]) -> Dict[str, Any]:
        gate = self.gating_module(physics_in)
        positive_timing_failure, negative_arrival_pressure = self.energy_module.compute_components(physics_in)
        energy = positive_timing_failure + negative_arrival_pressure
        
        return {
            "race_energy": energy,
            "positive_timing_failure_energy": positive_timing_failure,
            "negative_arrival_pressure_energy": negative_arrival_pressure,
            "logit_bias": -self.lambda_energy * energy,
            "positive_timing_failure_logit_bias": -self.lambda_energy * positive_timing_failure,
            "negative_arrival_pressure_logit_bias": -self.lambda_energy * negative_arrival_pressure,
            "neg_gate": gate,
            "energy_mean": energy.mean().item(),
            "gate_mean": gate.mean().item()
        }
        
    def capabilities(self) -> PhysicsCapabilities:
        return {"provides_bias": True, "provides_neg_gating": True}
