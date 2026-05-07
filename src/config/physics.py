from dataclasses import dataclass, field
from typing import Dict, Any

@dataclass
class EnvironmentPhysicsConfig:
    """第一层：客观世界与仿真环境 (World & Simulation)"""
    data_resolution_seconds: int = 900      # [SSOT] 数据本身的时间分辨率 (15min)
    simulation_step_seconds: int = 2700     # [SSOT] 物理引擎单步推进时长 (45min)
    
    # Compatibility
    time_resolution: int = 900              # [Deprecated] Alias for data_resolution_seconds
    sensor_reading_threshold: float = 0.1   # 传感器异常阈值
    freshness_decay: float = 0.8            # 信息时效衰减率
    diffusion_sigma: float = 900.0          # 真实源头时间高斯平滑度
    confirmation_threshold: float = 0.5     # GT判定阈值
    label_type: str = 'time-gaussian'       # 标签类型

@dataclass
class NavigatorPhysicsConfig:
    """第二层：探索手脚的物理启发 (Active Sensing Rules)"""
    pass # 启发式规则已迁移至 HeuristicConfig

@dataclass
class ReasonerPhysicsConfig:
    """第三层：推理大脑的因果律 (Causal Inference Rules)"""
    use_bias: bool = True                   # (旧参收编) 启用物理偏置
    energy_model: str = 'physics_hybrid'    # (旧参收编) 能量模型类型
    lambda_energy: float = 2.0              # (旧参收编) 物理惩罚权重
    beta: float = 1.0                       # (旧参收编) 软剪枝温度
    energy_form: str = 'softplus'           # (旧参收编) 能量函数形式
    lambda_bias: float = 1.0                # (旧参收编) 物理偏置权重
    unreachable_penalty: float = 30.0       # (旧参收编) 物理不可达硬惩罚 (取代旧的离散屏蔽)
    # enable_clean_path_veto 已迁移至 HeuristicConfig
    # mass_dilution_penalty_weight 已迁移至 HeuristicConfig

@dataclass
class PhysicsConfig:
    """全局物理层总控 (Root Physics Node)"""
    enabled: bool = True
    env: EnvironmentPhysicsConfig = field(default_factory=EnvironmentPhysicsConfig)
    nav_rules: NavigatorPhysicsConfig = field(default_factory=NavigatorPhysicsConfig)
    rea_rules: ReasonerPhysicsConfig = field(default_factory=ReasonerPhysicsConfig)
