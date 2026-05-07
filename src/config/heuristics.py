from dataclasses import dataclass, field

@dataclass
class HeuristicConfig:
    """
    启发式规则配置 (Heuristics SSOT)
    
    这里存放所有的“非物理铁律”，而是基于专家经验、直觉或统计规律的“拇指规则”。
    这些规则通常用于加速搜索、剪枝或作为一种软性的归纳偏置。
    
    与 PhysicsConfig 的区别：
    - PhysicsConfig: 描述客观世界的不可违背的规律（如水流方向、时间流逝）。
    - HeuristicConfig: 描述智能体的主观策略偏好（如“优先查上游”、“优先查大节点”）。
    """
    
    # ==========================================
    # 1. 拓扑结构启发 (Topological Heuristics)
    # ==========================================
    # 枢纽节点偏好：是否倾向于选择度数高的节点
    enable_hub_preference: bool = False
    hub_bias_strength: float = 0.0 # 偏置强度
    
    # 上游优先：是否倾向于选择上游节点（基于拓扑排序或深度）
    enable_upstream_priority: bool = False
    upstream_bias_strength: float = 0.0
    
    # ==========================================
    # 2. 贝叶斯/统计启发 (Statistical Heuristics)
    # ==========================================
    # 清白证人阻断 (Clean Path Veto) 的软化版本
    # 如果一个路径上有清白传感器，是否降低上游嫌疑？
    enable_clean_path_suppression: bool = False
    clean_path_suppression_factor: float = 0.5 # 抑制因子 (0.0=完全屏蔽, 1.0=无影响)
    
    # 稀释惩罚 (Mass Dilution)
    # 污染物在传播过程中会稀释，距离越远，源头可能性越低（在同等读数下）
    enable_dilution_penalty: bool = False
    dilution_penalty_weight: float = 0.0
    
    # ==========================================
    # 3. 搜索策略启发 (Search Heuristics)
    # ==========================================
    # 线性冗余屏蔽：如果一段管网是线性的（无分支），中间节点是否跳过？
    enable_linear_redundancy_skip: bool = False
    
    # 阴影区域屏蔽：如果一个区域被证明“绝对安全”，是否直接Mask掉？
    enable_hydraulic_shadow_mask: bool = False

    # ==========================================
    # 4. 新增：基于虚拟边的高级启发式 (Tensor Heuristics)
    # ==========================================
    
    # 方案一：针对 Reasoner 的“时空感知清白抑制” (Time-Aware Clean Path Suppression)
    # 利用 ReLU(delta_t - STT) 对超时未到的阴性路径进行惩罚
    enable_time_aware_suppression: bool = False
    suppression_lambda: float = 1.0 # 惩罚系数
    suppression_epsilon: float = 0.1 # 时间容差 (1+epsilon)
    
    # 方案二：针对 Navigator 的“波前冲浪”偏置 (Wave Front Chasing Bias)
    # 利用 Gaussian(delta_t - STT) 奖励刚好流到的波前节点
    enable_wave_front_bias: bool = False
    wave_front_lambda: float = 1.0 # 奖励系数
    wave_front_sigma: float = 15.0 # 波前宽度 (分钟)
    
    # 方案三：针对 Navigator 的“水力枢纽引力” (Hub Attractor Bias)
    # 奖励处于多个虚拟边交汇点的枢纽
    enable_hub_attractor: bool = False
    hub_attractor_lambda: float = 1.0 # 奖励系数

    # ==========================================
    # 5. Diagnostic Navigator Heuristics
    # ==========================================
    # STT Variance Based Selection
    stt_variance_topk: int = 50
    stt_variance_sigma: float = 1.0 # Softmax Temperature for Heuristic Score
