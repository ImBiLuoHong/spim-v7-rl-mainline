from dataclasses import dataclass
from typing import ClassVar, Dict
import torch

@dataclass
class ObservationState:
    """
    Data structure to store node and edge observation facts.
    Observation v2 Schema (Revised Two-Hot Encoding).
    """
    # 1. Meta-info: Is this node observed?
    observed_flag: torch.Tensor  # 0 or 1

    # 2. Fact: Chlorine Deviation (CH0)
    # Value for chlorine deviation. 0 does NOT mean unobserved.
    chlorine_deviation: torch.Tensor  

    # 3. Fact: Toxic Positive Flag (Two-Hot Part A)
    # 1 = Observed & Toxic
    toxic_positive_flag: torch.Tensor     

    # 4. Fact: Toxic Negative Flag (Two-Hot Part B)
    # 1 = Observed & Safe (No Toxin)
    toxic_negative_flag: torch.Tensor

    # 5. Meta-info: Freshness
    freshness: torch.Tensor      # 0 to 1, where 1 means most recent

    # --- Deprecated / Compatibility ---
    # Anchor information (CH4)
    # DEPRECATED: Only kept for compatibility.
    anchor: torch.Tensor = None  
    
    # NOTE: sensor_type is explicitly removed.
    # NOTE: toxic_state (scalar 3-state) is explicitly removed.

@dataclass
class PhysicsContext:
    """
    Data structure to organize physical information independently.
    """
    stt: torch.Tensor  # Legacy/Default STT (Likely Flip Rate in current code, kept for compatibility)
    direction: torch.Tensor  # Directional info for flow
    edge_index: torch.Tensor  # Graph structure (topology)
    feasible_mask: torch.Tensor  # Feasible mask for physical reachability
    
    # [Added for Reachability Rule Module]
    stt_median: torch.Tensor = None # Log Median STT (Ch0)
    stt_min: torch.Tensor = None    # Log Min STT (Ch2)
    stt_dynamic: torch.Tensor = None # Dynamic STT [E, 1] or [E] for current time step
    edge_attr: torch.Tensor = None  # Full Edge Attributes
    batch: torch.Tensor = None      # Batch indices [N]


@dataclass
class ConstraintState:
    """
    Runtime hard constraints derived from source verdicts and sampling history.
    These masks are not soft evidence and should be consumed as hard exclusions.
    """
    confirmed_non_source_mask: torch.Tensor
    confirmed_source_mask: torch.Tensor
    sampled_mask: torch.Tensor
    no_resample_mask: torch.Tensor = None

    def __post_init__(self):
        if self.no_resample_mask is None:
            self.no_resample_mask = self.sampled_mask


@dataclass
class EvidenceStateMini:
    """
    Minimal navigator evidence payload with exactly two evidence channels.
    """
    support_score: torch.Tensor
    contradiction_score: torch.Tensor


@dataclass
class EvidenceState:
    """
    Data structure to store higher-order evidence derived from observation and physics.
    EvidenceState v1 formalizes a support-led mainline, with suspect as soft prior and
    contradiction frozen as an auxiliary audit/explanation branch.
    """
    ROLE_MAINLINE: ClassVar[str] = "mainline"
    ROLE_AUXILIARY: ClassVar[str] = "auxiliary"
    ROLE_DIAGNOSTIC_ONLY: ClassVar[str] = "diagnostic_only"

    FIELD_CONTRACTS: ClassVar[Dict[str, Dict[str, str]]] = {
        'suspect_pool': {
            'axis': 'source-wise',
            'space': 'fused',
            'provenance': 'EvidenceBuilder.compute_suspect_pool',
            'branch': 'suspect',
            'role': ROLE_AUXILIARY,
            'consumer_default': 'optional_soft_prior',
        },
        'support_score': {
            'axis': 'source-wise',
            'space': 'fused',
            'provenance': 'EvidenceBuilder.compute_support_score',
            'branch': 'support',
            'role': ROLE_MAINLINE,
            'consumer_default': 'navigator,reasoner,training',
        },
        'contradiction_score': {
            'axis': 'source-wise',
            'space': 'fused',
            'provenance': 'EvidenceBuilder.compute_contradiction_score',
            'branch': 'contradiction',
            'role': ROLE_AUXILIARY,
            'consumer_default': 'optional_aux_compare',
        },
        'reaction_consistency': {
            'axis': 'source-wise',
            'space': 'fused',
            'provenance': 'EvidenceBuilder.compute_reaction_consistency',
            'branch': 'diagnostic',
            'role': ROLE_DIAGNOSTIC_ONLY,
            'consumer_default': 'diagnostic_only',
        },
        'uncertainty_gap': {
            'axis': 'node-wise',
            'space': 'fused',
            'validity': 'observation_validity',
            'provenance': 'EvidenceBuilder.compute_uncertainty_gap',
            'branch': 'context',
            'role': ROLE_MAINLINE,
            'consumer_default': 'navigator',
        },
    }
    FIELD_GROUPS: ClassVar[Dict[str, tuple]] = {
        'support_mainline': ('support_score', 'observation_validity', 'uncertainty_gap'),
        'suspect_soft_prior': ('suspect_pool', 'topology_gate', 'coarse_time_gate', 'not_ruled_out_gate'),
        'contradiction_auxiliary': ('contradiction_score', 'contradiction_toxic_term', 'contradiction_clean_term', 'arrival_gate'),
        'diagnostic_only': ('reaction_consistency', 'consistency_positive_term', 'consistency_negative_penalty'),
    }

    # 1. Suspect Pool (Candidate-Domain Prior)
    # Space: Source-wise / Fused-wise [N]
    # Semantics: Soft prior indicating if a node looks source-like under coarse gates.
    # Role: Auxiliary / diagnostic. Not a hard mainline gate in EvidenceState v1.
    suspect_pool: torch.Tensor       

    # 2. Support Score (Source-wise Evidence)
    # Space: Source-wise / Fused-wise [N]
    # Semantics: Degree to which observations support this node being the source.
    # Role: Mainline ranking evidence consumed by navigator / reasoner / training.
    support_score: torch.Tensor      

    # 3. Contradiction Score (Source-wise Evidence)
    # Space: Source-wise / Fused-wise [N]
    # Semantics: Degree to which observations contradict this node being the source.
    # Role: Frozen auxiliary branch for audit / explanation / optional compare.
    contradiction_score: torch.Tensor 

    # 4. Reaction Consistency (Rule-based MVP)
    # Space: Source-wise / Fused-wise [N]
    # Semantics: Auxiliary source-relative consistency term derived from observation reactions.
    # Role: Diagnostic only in EvidenceState v1; not part of the default main loss.
    reaction_consistency: torch.Tensor 

    # 5. Uncertainty Gap (Information Gap)
    # Space: Node-wise / Fused-wise [N]
    # Semantics: Quantifies the lack of information for a node.
    # Role: Mainline exploration context, especially for navigator.
    uncertainty_gap: torch.Tensor

    # 6. Explicit validity/provenance contract
    # source_validity is deprecated in the support-led training mainline and retained only
    # as an ignored compatibility field for old audit payloads.
    source_validity: torch.Tensor = None
    observation_validity: torch.Tensor = None
    schema_version: str = "EvidenceState_v1"
    provenance_tag: str = "src.modeling.evidence.builder.EvidenceBuilder.build_evidence_state[v1_support_mainline_pure]"

    # 7. Gate Audit (Optional)
    compatibility_gate: torch.Tensor = None # [N], deprecated/diagnostic compatibility stub
    arrival_gate: torch.Tensor = None # [N]
    
    # New Suspect Pool Gates
    topology_gate: torch.Tensor = None # [N]
    coarse_time_gate: torch.Tensor = None # [N]
    not_ruled_out_gate: torch.Tensor = None # [N]
    negative_exclusion_slack: torch.Tensor = None # [N]
    # --- Audit Sub-terms (Optional, for detailed auditing) ---
    support_toxic_term: torch.Tensor = None
    support_chlorine_term: torch.Tensor = None
    
    # New Support Terms (Audit Only)
    support_coverage_term: torch.Tensor = None
    support_timing_term: torch.Tensor = None
    support_focus_term: torch.Tensor = None
    
    contradiction_toxic_term: torch.Tensor = None
    contradiction_clean_term: torch.Tensor = None
    consistency_positive_term: torch.Tensor = None
    consistency_negative_penalty: torch.Tensor = None

    # Residual refiner audit track (optional)
    base_support_score: torch.Tensor = None
    base_suspect_pool: torch.Tensor = None
    base_contradiction_score: torch.Tensor = None
    support_score_delta: torch.Tensor = None
    suspect_pool_delta: torch.Tensor = None
    contradiction_score_delta: torch.Tensor = None
    suspect_canonical_latent: torch.Tensor = None

    @property
    def support_main_score(self):
        return self.support_score

    @property
    def suspect_prior(self):
        return self.suspect_pool

    @property
    def contradiction_aux_score(self):
        return self.contradiction_score
