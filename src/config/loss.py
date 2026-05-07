from dataclasses import dataclass

@dataclass
class LossConfig:
    """损失配置 (极端基线版：只测 Reasoner 硬实力，剥离所有 RL 干扰)"""
    def __init__(self):
        # 1. Weights: 暴切所有花里胡哨的动态奖励，只保留绝对的硬核分类！
        self.weights = {
            'w_hard': 1.0,        # [唯一核心] Reasoner 认凶手的硬交叉熵 (Cross Entropy)
            'w_soft': 0.0,        # 彻底关闭
            'w_delta': 0.0,       # 彻底关闭 (斩断导致负数 Loss 和作弊的内鬼！)
            'w_hit': 0.0,         # 彻底关闭 (不再给提前命中发奖金，看最终硬实力)
            'w_mono': 0.0,        # 彻底关闭
            'w_ent': 0.0,         # 彻底关闭
            'w_physical': 0.0,    # 彻底关闭
            'w_surv': 0.0         # 彻底关闭
        }
        
        # 2. Parameters (由于上面的权重全砍为0了，下面这些参数其实已经不生效了，保持最简即可)
        self.params = {
            'loss_mode': 'baseline',      
            'temporal_decay_gamma': 1.0,  # 去掉时间衰减，每一步的推理权重一样
            'temporal_weight_mode': 'late',
            'temporal_weight_floor': 0.0,
            'delta_margin': 0.0,          
            'label_smoothing_sigma': 1.0, 
            'rho_discount': 1.0,          
            'target_entropy_start': 0.8,  
            'target_entropy_end': 0.1,    
            'dense_ce': False,            # 只算最后一步(或选中步)的CE即可
            'scale_correction': False,    
            'use_old_hit_surrogate': False, 
            'survival_eps': 1e-6,         
            'survival_use_rho': False,    
            'survival_detach_q': False,
            'evidence_oracle': {
                'enabled': False,
                'listwise_temperature': 1.0,
                'support_listwise_weight': 1.0,
                'support_pairwise_weight': 0.25,
                'support_pairwise_margin': 0.05,
                'suspect_target_mode': 'absolute',
                'suspect_activation_focus_scale': 0.0,
                'suspect_activation_deficit_weight': 0.0,
                'suspect_activation_occurrence_weight': 0.0,
                'suspect_activation_weight': 0.5,
                'suspect_prune_weight': 0.5,
            },
        }
