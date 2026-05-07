import torch
import torch.nn as nn
from torch_scatter import scatter_max, scatter_mean
from src.modeling.state.schema import ObservationState, EvidenceState

class ObservationEncoder(nn.Module):
    """
    Standardized Observation Encoder.
    Responsibility:
    1. Extract Raw Features from ObservationState
    2. Extract Evidence Features from EvidenceState (if enabled)
    3. Normalize and Pre-process Features (SymLog, etc.)
    4. Align Raw Features to Fused Space (using inverse_indices)
    5. Encode into a unified node embedding
    """
    def __init__(self, hidden_dim, use_evidence=False, feature_mode='concat', evidence_fields=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_evidence = use_evidence
        self.feature_mode = feature_mode
        
        # Channel Config (SSOT)
        # 0: Signal (Cl)
        # 1: Toxic Positive Flag (Two-Hot A)
        # 2: Toxic Negative Flag (Two-Hot B)
        # 3: Freshness
        # 4: Observed
        # 5: Anchor (Compatibility)
        self.raw_channels = 6
        
        if use_evidence and feature_mode == 'concat':
            default_fields = ['support_score', 'uncertainty_gap']
            self.evidence_fields = list(evidence_fields or default_fields)
        else:
            self.evidence_fields = []
        
        self.evidence_channels = len(self.evidence_fields)
        
        self.input_dim = self.raw_channels + self.evidence_channels
        
        # Projection Layer
        self.projector = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )
        
    def _symlog(self, x):
        return torch.sign(x) * torch.log1p(x.abs())

    def _normalize_evidence_feature(self, attr_name, value):
        if attr_name in {'support_score', 'contradiction_score', 'uncertainty_gap'}:
            return self._symlog(value)
        return value

    def forward(self, state, num_fused_nodes):
        """
        Args:
            state: Dict containing 'observation_state', 'evidence_state', 'inverse_indices'
            num_fused_nodes: Int, target size for alignment
        Returns:
            h_obs: [N_fused, hidden_dim]
        """
        # [Contract Assertion] Check ObservationState existence
        if 'observation_state' not in state:
            raise ValueError("ObservationState missing in state dict. Ensure StateBuilder populates it.")
            
        obs_state: ObservationState = state['observation_state']
        device = obs_state.chlorine_deviation.device
        
        # 1. Extract & Normalize Raw Features
        # [Contract Assertion] Check Tensor Dimensions
        if obs_state.chlorine_deviation.dim() != 1:
            # Allow 2D [N, 1] but squeeze/unsqueeze carefully
            pass 
            
        def ensure_2d(t):
            return t.unsqueeze(1) if t.dim() == 1 else t

        # Stack channels: [Cl, ToxicPos, ToxicNeg, Fresh, Obs, Anchor]
        # Apply SymLog to Signal (Cl)
        # Keep others as is (0-1 range)
        
        # [Contract Assertion] Anchor existence (Compatibility)
        # [Cleanup Phase A] Anchor is strictly compatibility-only now.
        # It is no longer part of the core observation semantics (Obs v2).
        # If anchor is None, we use zeros for compatibility channel.
        if obs_state.anchor is None:
             anchor_feat = torch.zeros_like(obs_state.chlorine_deviation)
        else:
             # TODO: Fully deprecate this path in future phases
             anchor_feat = obs_state.anchor

        # [Contract Assertion] Two-Hot Flags existence
        if not hasattr(obs_state, 'toxic_positive_flag') or not hasattr(obs_state, 'toxic_negative_flag'):
             raise ValueError("ObservationState missing 'toxic_positive_flag' or 'toxic_negative_flag'. Check schema version.")

        raw_feats = torch.cat([
            self._symlog(ensure_2d(obs_state.chlorine_deviation)),
            ensure_2d(obs_state.toxic_positive_flag),
            ensure_2d(obs_state.toxic_negative_flag),
            ensure_2d(obs_state.freshness),
            ensure_2d(obs_state.observed_flag),
            ensure_2d(anchor_feat)
        ], dim=1)
        
        # 2. Extract Evidence Features (Optional)
        ev_feats = None
        if self.use_evidence and self.feature_mode == 'concat':
            # [Contract Assertion] EvidenceState existence
            ev_state = state.get('evidence_state')
            if ev_state is None:
                 # Only allow None if use_evidence is explicitly False, but here it is True.
                 # Actually, allow graceful handling if not ready? No, user said "Formalize".
                 raise ValueError("EvidenceState missing but use_evidence=True.")
            
            # Check type
            if not isinstance(ev_state, EvidenceState):
                 # Fallback for legacy dict (should be removed after migration)
                 # raise TypeError(f"Expected EvidenceState object, got {type(ev_state)}")
                 # For now, let's assume object as per plan.
                 pass

            # Helper for extraction
            def get_ev(attr_name):
                if isinstance(ev_state, dict):
                    val = ev_state.get(attr_name)
                else:
                    val = getattr(ev_state, attr_name, None)
                if val is None:
                    raise ValueError(f"EvidenceState attribute {attr_name} is None.")
                return ensure_2d(val)
            ev_feature_tensors = [
                self._normalize_evidence_feature(field_name, get_ev(field_name))
                for field_name in self.evidence_fields
            ]
            if ev_feature_tensors:
                ev_feats = torch.cat(ev_feature_tensors, dim=1)
            
        # 3. Combine in Raw Space
        if ev_feats is not None:
            # [Contract Assertion] Dimension Alignment
            if ev_feats.size(0) != raw_feats.size(0):
                raise ValueError(f"Evidence/Raw dimension mismatch: {ev_feats.size(0)} vs {raw_feats.size(0)}")
            
            combined_raw = torch.cat([raw_feats, ev_feats], dim=1)
        else:
            combined_raw = raw_feats
            
        # 4. Align to Fused Space
        if combined_raw.size(0) == num_fused_nodes:
            fused_input = combined_raw
        elif 'inverse_indices' in state:
            inv_idx = state['inverse_indices']
            
            # [Contract Assertion] Inverse Indices Validity
            if inv_idx.max() >= num_fused_nodes:
                raise ValueError(f"inverse_indices max ({inv_idx.max()}) >= num_fused_nodes ({num_fused_nodes})")
            
            # Filter valid mappings
            valid_mask = (inv_idx >= 0)
            
            if valid_mask.any():
                valid_feat = combined_raw[valid_mask]
                valid_idx = inv_idx[valid_mask]
                
                # Use scatter_max to preserve strongest signals
                fused_input, _ = scatter_max(valid_feat, valid_idx, dim=0, dim_size=num_fused_nodes)
            else:
                fused_input = torch.zeros(num_fused_nodes, self.input_dim, device=device)
        else:
            # [Contract Assertion] Alignment Failure
            raise ValueError(f"Cannot align raw ({combined_raw.size(0)}) to fused ({num_fused_nodes}) without inverse_indices.")
            
        # 5. Project to Hidden Dim
        h_obs = self.projector(fused_input)
        
        return h_obs
