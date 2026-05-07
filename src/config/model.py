from dataclasses import dataclass
from src.config.interaction import InteractionConfig

@dataclass
class ModelConfig:
    """模型配置类 - 管理模型架构相关的超参数"""
    
    def __init__(self):
        self.architecture = 't2i_bdas' # Options: 'spatiotemporal', 'autosampling', 't2i_bdas'
        self.hidden_dim = 128     # [SSOT Upgrade] 64 -> 128 (Deep Capacity)
        self.num_gnn_layers = 5   # [SSOT Upgrade] 2/3 -> 5 (Deep Capacity)
        
        # [SSOT] Feature Firewall & Gating
        self.disable_firewall = True  # [User Request] Default to True to abolish firewall
        self.allowed_channels = [0, 1, 2, 3, 4, 5, 6] # V6 Schema
        
        # [SSOT] Cognitive State Machine (Dynamic Budgeting)
        # [User Request] Baseline Mode: Force Phase 1 (Navigator Only)
        self.cognitive_state_machine = {
            'phase1_h_threshold': -1.0, # Always True -> Force Phase 1
            'phase1_p_threshold': 2.0,  # Always True -> Force Phase 1
            'phase3_h_threshold': 0.4,
            'phase3_p_threshold': 0.8
        }
        
        self.num_tcn_layers = 4
        self.tcn_kernel_size = 3
        self.dropout_rate = 0.3
        self.disable_dropout = False
        self.posenc_dim = 16
        self.attention_heads = 4
        self.use_dual_branch = True
        self.enable_feature_mixer = True
        self.temporal_backbone = 'tcn'
        self.spatial_backbone = 'gatv2'
        self.pooling_type = 'none'
        self.input_dim = 7  # V4.5.22 Spec: 7 Channels
        self.edge_dim = 8   # SSOT V6.3: Unified 8-Channel Spatiotemporal Blueprint
        # [Moved to PhysicsConfig] freshness_decay
        
        # Phase 4.5: Dynamic Feature Gating (Reasoner Repair)
        self.dynamic_gate = {
            'mode': 'off', # off, fixed, learnable
            'target_indices': [1, 2, 3], # Indices in x_nav corresponding to Freshness(2), Mask(3), Anchor(4)
            'context_dim': 16,
            'regularization': {
                'l1': 0.0,
                'entropy': 0.0
            }
        }
        
        # Plan B: Overfitting Countermeasures
        self.navigator_noise_std = 1.0       # Strong Gaussian Noise on Navigator Logits
        self.reasoner_dropout_mask_prob = 0.1 # DropMask Regularization
        self.mask_dropout_prob = 0.3 # Task 1: Mask Dropout Probability
        
        # Phase 3 Platform Configs
        # [Legacy] 冗余开关，目前的架构强依赖 Navigator 存在
        self.navigator_enabled = True
        # [Legacy] 冗余开关，目前的架构强依赖 Reasoner 存在
        self.reasoner_enabled = True
        self.sample_budget = 3 # [SSOT Change] Default sample budget per episode
        self.sampling_policy = 'learned' # [True Full Blood] Activate Cognitive State Machine
        self.training_mode = 'joint' # joint, frozen_nav, frozen_reasoner
        self.oracle_mode = False
        self.nav_state_summary = {
            'enabled': True, # [True Full Blood] Global Awareness
            'dim': 6
        }
        self.proxy_ig = {
            'enabled': False,
            'loss': 'huber',
            'delta': 1.0,
            'weight': 0.1,
            'target': 'gap'
        }
        
        # Phase 4.5: Navigator Config (SSOT)
        self.navigator_type = 'standard_v4_5'
        self.navigator_mode = 'learned' # Options: 'learned', 'random', 'heuristic_stt'
        self.heuristic_sub_mode = 'stt_var' # 'stt_var', 'time_split', 'hybrid'
        self.heuristic_hybrid_alpha = 0.5
        self.heuristic_stt_topk_sources = 50 # For heuristic_stt mode
        
        # [Diagnosis] Reasoner Decoupling
        self.reasoner_decoupled = True # Master switch for Reasoner independent encoder
        self.reasoner_type = 'bayesian_v4_5'

        self.evidence = {
            'version': 'evidence_state_v1',
            'support_role': 'mainline',
            'suspect_role': 'soft_prior',
            'contradiction_role': 'auxiliary_frozen',
            'reaction_role': 'diagnostic_only',
        }
        self.evidence_refiner = {
            'enabled': False,
            'hidden_dim': 32,
            'support_delta_scale': 1.0,
            'suspect_delta_scale': 0.25,
            'suspect_activation_scale': 1.0,
            'suspect_prune_scale': 2.0,
        }
        
        self.navigator = {
            'enabled': True,
            'backbone_type': 'sage_backbone', # [SSOT Rollback] gru_backbone -> sage_backbone (Exp B)
            'hidden_dim': 128,    # Keep Deep Capacity (Exp B)
            'layers': 5,          # Keep Deep Capacity (Exp B)
            # [System v2] Policy Interface
            'use_evidence': False, # Master switch for Policy v2
            'evidence_mode': 'bias', # bias / concat (bias only for Phase 3)
            'support_bias_scale': 1.0, # Strength of support_score bias
            'suspect_bias_scale': 0.0, # Optional soft suspect prior
            'contradiction_bias_scale': 0.0, # Optional auxiliary compare only
            'uncertainty_bias_scale': 0.5, # Strength of uncertainty_gap bias
            'source_like_bonus_scale': 0.0,
            'source_like_bonus_later_round_multiplier': 1.0,
            'flat_drift_penalty_scale': 0.0,
        }
        self.navigator_vnext = {
            'semantic_hidden_dim': 128,
            'actor_hidden_dim': 128,
            'critic_hidden_dim': 128,
            'use_h_fused_skip': True,
            'use_reasoner_logits': False,
            'support_plausible_delta': 0.25,
        }

        # [Experiment] EvidenceState Injection
        self.reasoner = {
            'enabled': True,
            'hidden_dim': 128,
            'use_evidence': False, # [Experiment] Master switch
            'evidence_mode': 'bias', # bias or concat
            'evidence_scale': 0.1,
            'support_weight': 1.0,
            'suspect_weight': 0.0,
            'contradiction_weight': 0.0,
            'reaction_weight': 0.0,
            'concat_fields': ['support_score', 'uncertainty_gap'],
        }

        # [Feature] Reasoner Teacher Shortlist
        self.reasoner_teacher = {
            'enable_score': False,          # Master switch for calculating score
            'enable_shortlist': False,      # Master switch for applying shortlist to Navigator
            'candidate_topk': 50,           # Number of candidates to evaluate (from current posterior)
            'shortlist_topk': 10,           # Number of candidates to keep in shortlist
            'metric': 'rank_improve'        # Scoring metric: 'rank_improve' or 'kl_div'
        }
        
        # Phase 4: Shrink Mechanism
        self.shrink = {} # Dict for flexibility
        self.auto_focus = {}
        self.soft_expand = {}
        self.residual_bg = {}

        # [Moved to PhysicsConfig] Hydraulic Race Soft Pruning
        # physics_type, physics, use_physics_bias, lambda_physics_bias moved to cfg.physics
        
        # Phase 3.1 Interaction Configs
        self.interaction = InteractionConfig()
