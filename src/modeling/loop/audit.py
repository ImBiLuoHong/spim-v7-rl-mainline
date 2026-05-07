import torch
from src.modeling.state.schema import ObservationState, EvidenceState

class SystemAuditor:
    """
    Lightweight runtime audit for System v2 Skeleton.
    Checks data integrity and contract assertions.
    """
    def __init__(self, cfg):
        self.cfg = cfg
        system_cfg = getattr(cfg, 'system', None)
        if system_cfg:
            self.enabled = getattr(system_cfg, 'enable_audit', True)
        else:
            self.enabled = True # Default enabled

    def audit_step(self, step, state_dict):
        """
        Audit critical states at each step.
        """
        if not self.enabled:
            return

        try:
            self._check_observation(state_dict.get('observation_state'))
            self._check_evidence(state_dict.get('evidence_state'))
            self._check_alignment(state_dict)
        except Exception as e:
            # Log error but don't crash unless strict mode
            if getattr(self.cfg.system, 'strict_audit', False):
                raise e
            print(f"[Audit Warning] Step {step}: {e}")

    def _check_observation(self, obs_state):
        if obs_state is None:
            raise ValueError("ObservationState is None")
        if not isinstance(obs_state, ObservationState):
            raise TypeError(f"Expected ObservationState, got {type(obs_state)}")
        
        # Check critical fields (Observation v2 Two-Hot)
        required_fields = ['chlorine_deviation', 'observed_flag', 'freshness', 'toxic_positive_flag', 'toxic_negative_flag', 'anchor']
        for field in required_fields:
            if not hasattr(obs_state, field):
                raise ValueError(f"ObservationState missing field: {field}")
            val = getattr(obs_state, field)
            if val is None:
                raise ValueError(f"ObservationState.{field} is None")
            if not isinstance(val, torch.Tensor):
                raise TypeError(f"ObservationState.{field} must be a Tensor")

        # Check Dimensions
        N = obs_state.chlorine_deviation.size(0)
        for field in required_fields:
            val = getattr(obs_state, field)
            if val.size(0) != N:
                 raise ValueError(f"Dimension mismatch in ObservationState.{field}: {val.size(0)} vs {N}")

        # Check Values (Contract)
        # Flags must be 0 or 1
        if obs_state.toxic_positive_flag.max() > 1.0 or obs_state.toxic_positive_flag.min() < 0.0:
             raise ValueError("toxic_positive_flag must be 0 or 1")
        if obs_state.toxic_negative_flag.max() > 1.0 or obs_state.toxic_negative_flag.min() < 0.0:
             raise ValueError("toxic_negative_flag must be 0 or 1")

        # Check explicit exclusion of sensor_type
        if hasattr(obs_state, 'sensor_type'):
             raise ValueError("ObservationState must NOT contain 'sensor_type'. Remove it from schema/builder.")
        
        # Check explicit exclusion of toxic_state (scalar)
        if hasattr(obs_state, 'toxic_state'):
             raise ValueError("ObservationState must NOT contain 'toxic_state'. Use two-hot flags.")

    def _check_evidence(self, ev_state):
        if ev_state is None:
            # EvidenceState v2 requires evidence_state to be present if use_evidence is True.
            # But here we just check structure if it exists.
            return

        if not isinstance(ev_state, EvidenceState):
             raise TypeError(f"Expected EvidenceState, got {type(ev_state)}")

        # Check Fields Completeness (EvidenceState v2)
        required_fields = ['suspect_pool', 'support_score', 'contradiction_score', 'reaction_consistency', 'uncertainty_gap']
        for field in required_fields:
            if not hasattr(ev_state, field):
                raise ValueError(f"EvidenceState missing field: {field}")
            val = getattr(ev_state, field)
            if val is None:
                raise ValueError(f"EvidenceState.{field} is None")
            if not isinstance(val, torch.Tensor):
                raise TypeError(f"EvidenceState.{field} must be a Tensor")
        
        # Check Dimensions
        N = ev_state.suspect_pool.size(0)
        for field in required_fields:
            val = getattr(ev_state, field)
            if val.size(0) != N:
                 raise ValueError(f"Dimension mismatch in EvidenceState.{field}: {val.size(0)} vs {N}")

        if ev_state.observation_validity is None:
            raise ValueError("EvidenceState.observation_validity is required to mark node-wise evidence support")
        if ev_state.observation_validity.size(0) != N:
            raise ValueError("EvidenceState.observation_validity must align with uncertainty_gap")
        if ev_state.source_validity is not None and ev_state.source_validity.size(0) != N:
            raise ValueError("EvidenceState.source_validity must align when present")

        contracts = getattr(EvidenceState, 'FIELD_CONTRACTS', {})
        for field in ['suspect_pool', 'support_score', 'contradiction_score', 'reaction_consistency', 'uncertainty_gap']:
            contract = contracts.get(field, {})
            validity_name = contract.get('validity')
            if validity_name:
                validity = getattr(ev_state, validity_name, None)
                if validity is None:
                    raise ValueError(f"EvidenceState.{field} requires validity tensor '{validity_name}'")
                if validity.size(0) != N:
                    raise ValueError(f"EvidenceState.{validity_name} must align with {field}")

        # Check Value Ranges
        # suspect_pool: 0.0 to 1.0 (Mask/Prob)
        if ev_state.suspect_pool.min() < 0.0 or ev_state.suspect_pool.max() > 1.0:
             # Allow small float error
             if ev_state.suspect_pool.min() < -1e-5 or ev_state.suspect_pool.max() > 1.00001:
                  raise ValueError(f"suspect_pool out of range [0, 1]: min={ev_state.suspect_pool.min()}, max={ev_state.suspect_pool.max()}")

        # contradiction_score: 0.0 to 1.0 (as constructed)
        if ev_state.contradiction_score.min() < 0.0:
             raise ValueError("contradiction_score must be non-negative")
        
        # support_score: >= 0.0
        if ev_state.support_score.min() < 0.0:
             raise ValueError("support_score must be non-negative")

        # uncertainty_gap: 0.0 to 1.0
        if ev_state.uncertainty_gap.min() < 0.0 or ev_state.uncertainty_gap.max() > 1.0:
             if ev_state.uncertainty_gap.min() < -1e-5 or ev_state.uncertainty_gap.max() > 1.00001:
                  raise ValueError("uncertainty_gap out of range [0, 1]")

    def _check_alignment(self, state_dict):
        # Check Raw vs Fused sizes
        obs = state_dict.get('observation_state')
        inv_idx = state_dict.get('inverse_indices')
        num_fused = state_dict.get('num_fused_nodes')
        
        if obs and inv_idx is not None and num_fused:
            obs_size = obs.chlorine_deviation.size(0)
            raw_count = inv_idx.size(0)
            
            is_raw = (obs_size == raw_count)
            is_fused = (obs_size == num_fused)
            
            if not (is_raw or is_fused):
                 raise ValueError(f"Observation size ({obs_size}) matches neither Raw ({raw_count}) nor Fused ({num_fused}) node count")
            
            if inv_idx.max() >= num_fused:
                 raise ValueError(f"inverse_indices max ({inv_idx.max()}) >= num_fused_nodes ({num_fused})")
