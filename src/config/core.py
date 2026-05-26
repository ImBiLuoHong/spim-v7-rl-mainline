"""
统一配置中心 (SSOT V6)
====================================

本模块实现“单一事实来源（SSOT）”的集中化配置：
1. 所有行为与超参仅来源于 cfg（代码默认值 + 运行时内存态覆盖）
2. 严禁在核心业务代码中读取环境变量；不再提供环境变量优先级
3. 训练主流程只读取配置对象，不做隐式行为变更
4. 提供一致的控制台摘要与文件摘要，确保可审计与可复现

使用方式:
    from src.config.core import Config
    cfg = Config()
    
    # 如需在代码中应用运行时覆盖：
    cfg.apply_overrides({
        'training': {'batch_size': 128},
        'model': {'hidden_dim': 128},
    })
    
    # 访问路径
    print(cfg.paths.foundation_path)
    
    # 访问超参数
    print(cfg.training.batch_size)
    print(cfg.model.hidden_dim)

配置来源（从高到低）：
1. 代码中的默认值（本文件内定义）
2. 运行时内存态 dict 覆盖（通过 cfg.apply_overrides 应用）

重要说明：
- 禁止将环境变量或磁盘文件作为配置源；如需变更，请在代码中调用 cfg.apply_overrides 或修改默认值
- 每次运行会写入“冻结有效配置快照”与“人类可读摘要”，用于审计与复现
"""

import os
import torch
from dataclasses import dataclass
from typing import Optional, List
from datetime import datetime
from src.config.paths import ensure_unique_run_name
from src.config.efficiency import EfficiencyConfig
from src.config.loss import LossConfig
from src.config.model import ModelConfig
from src.config.data import DataConfig
from src.config.interaction import InteractionConfig
from src.config.physics import PhysicsConfig
from src.config.heuristics import HeuristicConfig


@dataclass
class PathConfig:
    """路径配置类 - 集中管理所有路径相关配置 (V6)"""
    
    def __init__(self, root_dir: str):
        """
        初始化路径配置
        
        Args:
            root_dir: 项目根目录路径
        """
        self.root_dir = root_dir
        
        # 实验目录配置
        self.experiments_dir = os.path.join(root_dir, 'runs')
        try:
            self.run_dir = os.path.join(self.experiments_dir, datetime.now().strftime('exp-%Y%m%d-%H%M%S'))
        except Exception:
            self.run_dir = os.path.join(self.experiments_dir, 'exp-unknown')
            
        # V6 新增核心路径
        # 指向 datanew/production_data/foundation_...
        self.foundation_path = os.path.join(root_dir, 'datanew', 'production_data', 'foundation_20260114_164946_86d5023e')
        # 指向压缩后的 V12 indexer subgraphs；V11 dense 仅作为离线转换源。
        self.samples_path = os.path.join(self.foundation_path, 'subgraph_v12')
        
        # Explicitly define split_dir to point to the splits in data/
        self.split_dir = os.path.join(root_dir, 'data')
        
        # 工件策略与目录
        self.artifacts_strategy = 'global'
        self.artifacts_dir = os.path.join(root_dir, 'data', 'artifacts')
        
        # 日志/检查点目录
        self.logs_dir = os.path.join(self.run_dir, 'logs')
        self.checkpoints_dir = os.path.join(self.run_dir, 'checkpoints')
        
        # Artifact 子路径
        self.posenc_path = os.path.join(self.artifacts_dir, 'posenc_k16.npy')
        
        # Mapping Paths (V6)
        self.node_map_path = os.path.join(self.foundation_path, 'node_mapping.csv')
        self.pipe_map_path = os.path.join(self.foundation_path, 'pipe_map.csv')
        
        # [SSOT] Dedicated Cache Directory
        self.cache_dir = os.path.join(root_dir, 'data', 'cache_lmdb')

@dataclass
class LifeSupportConfig:
    """保命机制配置 (SSOT) - 管理 Oracle 与 Pacemaker"""
    def __init__(self):
        # Oracle / God View
        # Default: False (All God View logic DISABLED)
        self.enable_oracle_guidance = False
        
        # Oracle Annealing (Linear Decay)
        # Modify these values directly to enable "Teaching Mode"
        self.oracle_anneal = {
            'enabled': False,   # Set to True to enable linear decay (Teaching Mode)
            'start_prob': 1.0,  # 100% guidance at start
            'end_prob': 0.0,    # 0% guidance at end
            'total_steps': 1000, 
            'min_prob': 0.0,
            'mode': 'step'      # 'step' uses tau schedule
        }
        
        # Pacemaker / Forced Active
        # Default: True (Current file state: break is disabled, active_mask_t forced to True at step 0)
        self.enable_pacemaker = True 
        
        # Early Stop
        # Default: False (Current file state: break is disabled)
        self.allow_early_stop = False
        
        # Dense Hard Loss Mask
        # Default: True (Current file state: mask_t = torch.ones_like(...) in losses.py)
        self.dense_hard_loss_mask = True

        # Constraint verdict fallback
        # Default: False. Runtime constraint updates must prefer explicit step verdict payloads.
        # Enable this only for supervised/debug fallback when explicit verdict payload is unavailable.
        self.allow_constraint_label_fallback = False

        # Profile Name (You can ignore this if you modify above values directly)
        self.profile = "custom_direct_edit"

    def set_profile(self, profile_name: str):
        if profile_name == "current_life_support":
            self.enable_oracle_guidance = False
            self.enable_pacemaker = True
            self.allow_early_stop = False
            self.dense_hard_loss_mask = True
            self.profile = "current_life_support"
            
        elif profile_name == "oracle_soft_anneal_v1":
            # Soft mode: Try to relax some constraints
            self.enable_oracle_guidance = False
            self.enable_pacemaker = False # Let it die if finished?
            self.allow_early_stop = True # Allow break
            self.dense_hard_loss_mask = False # Mask out finished graphs
            self.profile = "oracle_soft_anneal_v1"
            
        elif profile_name == "oracle_teaching_v1":
            # Teaching Mode: Oracle starts strong and fades out
            self.enable_oracle_guidance = True # Master switch ON
            self.enable_pacemaker = True # Keep safety net
            self.allow_early_stop = False
            self.dense_hard_loss_mask = True
            
            self.oracle_anneal = {
                'enabled': True,
                'start_prob': 1.0, # 100% Oracle at start
                'end_prob': 0.0,   # 0% Oracle at end
                'min_prob': 0.0,
                'mode': 'step'     # Follows tau schedule
            }
            self.profile = "oracle_teaching_v1"
            
        else:
            raise ValueError(f"Unknown profile: {profile_name}")


@dataclass
class TrainingConfig:
    """训练配置类 - 管理所有训练相关的超参数"""
    
    def __init__(self):
        self.run_name = "finally_rollback" # Explicitly named retry
        self.select_best = "recall@30"
        self.test_best = False
        self.resume = False
        self.resume_checkpoint = None
        self.init_checkpoint = None
        self.init_checkpoint_strict = True
        self.enable_eval = True
        self.train_only = False
        self.val_every_n_epochs = 100    # Balanced evaluation frequency for extreme throughput
        self.enable_wandb = True
        self.log_every_n_steps = 1
        # Detailed rollout/evidence step metrics are useful for audits,
        # but they add noticeable CPU overhead during large formal runs.
        self.collect_detailed_step_metrics = True
        # Force legacy runtime probe/tracer behavior even when throughput-oriented
        # settings would normally disable it.
        self.force_runtime_probes = False

        self.learning_rate = 5e-4
        self.weight_decay = 1e-4
        self.seed = 45 # [SSOT] Changed seed to force new hash
        self.force_gpu = True
        self.gradient_clip_norm = 1.0 # [SSOT] Global gradient clipping
        self.grad_accum_steps = 1
        self.num_epochs = 100
        
        # [SSOT] Numerical Stability
        self.eps_prob = 1e-8 # Log(0) protection
        self.logit_max_abs = 20.0 # Gumbel softmax protection
        
        # [SSOT] Annealing Parameters
        self.annealing = {
            'tau_start': 1.5,
            'tau_end': 0.1,
            'anneal_end_ratio': 1.0 # Anneal over full 100 epochs (was 0.7)
        }
        
        # Overfitting & Debugging
        self.overfit_one_batch = False
        self.overfit_steps = 500
        self.ofb_save_every = 100
        self.ofb_fixed_batch_cpu = False
        self.diag_print_grad = False
        self.diag_print_top1 = False
        self.print_ce_kl = True
        
        # Evaluation
        self.max_eval_episodes = 10 # [SSOT] Standard evaluation horizon (Episodes)
        self.formal_eval_batch_size = 1
        self.max_train_episodes = 10    # [User Request] Fixed to 10 episodes
        self.rank_ks = [1, 3, 5, 10, 30]
        self.ndcg_k = 5
        
        # Phase 3 Task Configs
        self.hard_only_training = False
        self.hard_weight = 1.0
        self.use_trigger_feature = True
        self.use_two_stage = False
        
        # Curriculum Learning
        self.curriculum = {
            'enabled': False, # [True Full Blood] Activate Curriculum Learning
            'horizon_schedule': [(0, 3), (10, 5), (20, 8), (30, 10)], # Gradual difficulty increase
            'strength_schedule': [(0, 1.0)] # Always Full Strength
        }
        self.evidence_oracle_schedule = {
            'enabled': False,
            'mode': 'step',
            'phase_a_steps': 0,
            'phase_b_steps': 0,
            'phase_c_steps': 1,
            'phase_a_oracle_factor': 1.0,
            'phase_a_live_factor': 0.0,
            'phase_b_oracle_factor_start': 1.0,
            'phase_b_oracle_factor_end': 0.5,
            'phase_b_live_factor_start': 0.25,
            'phase_b_live_factor_end': 1.0,
            'phase_c_oracle_factor_start': 0.5,
            'phase_c_oracle_factor_end': 0.0,
            'phase_c_live_factor_start': 1.0,
            'phase_c_live_factor_end': 1.0,
        }


@dataclass
class FeaturesConfig:
    """功能开关"""
    def __init__(self):
        # [Unused] 在代码中未生效
        self.enable_subgraph_loss = False
        self.subgraph_target_k = 0
        
        # [Audit Only] 仅用于审计日志，不影响训练逻辑
        self.contract = {'audit': {'enabled': True}}
        
        # [Active] 上下游剪枝逻辑
        self.prune_upstream = {'enabled': False, 'window_s': 3600}
        # [Moved to PhysicsConfig] evidence_hint


@dataclass
class LabelsConfig:
    """标签配置 (Deprecated: Use PhysicsConfig.diffusion_sigma)"""
    def __init__(self):
        self.type = 'time-gaussian'
        self.normalize_sum = False
        self.sigma = 900.0 # Kept for backward compatibility, but prefer cfg.physics.diffusion_sigma


@dataclass
class SystemConfig:
    """System v2 Configuration"""
    def __init__(self):
        self.enable_audit = True
        self.strict_audit = False

class Config:
    """统一配置中心 (V6)"""
    
    VERSION = 6
    
    def __init__(self, root_dir: Optional[str] = None):
        if root_dir is None:
            root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        self.root_dir = root_dir
        self.paths = PathConfig(root_dir)
        self.training = TrainingConfig()
        self.system = SystemConfig()
        self.model = ModelConfig()
        self.data = DataConfig()
        self.efficiency = EfficiencyConfig()
        self.physics = PhysicsConfig()
        self.heuristics = HeuristicConfig()
        self.life_support = LifeSupportConfig()
        
        # Propagate efficiency profile to DataConfig
        if self.efficiency.hardware_profile == 'RTX_5090_EXTREME':
            self.data.num_workers = self.efficiency.num_workers
            self.data.prefetch_factor = self.efficiency.prefetch_factor
            self.data.pin_memory = self.efficiency.pin_memory
            self.data.persistent_workers = (
                bool(self.efficiency.persistent_workers)
                and int(self.efficiency.num_workers) > 0
            )
            self.data.preload = self.efficiency.preload
            self.data.non_blocking_transfer = self.efficiency.non_blocking_transfer
        
        self.features = FeaturesConfig()
        self.loss = LossConfig()
        self.labels = LabelsConfig()
        self.interaction = InteractionConfig()
        
        # Phase 4.5: Adaptive FoV Controller
        self.fov_controller = {
            'type': 'none', # none, entropy_driven, conflict_driven
            'strength': 1.0,
            'params': {
                'M_min': 20,
                'M_max': 100,
                'entropy_hi': 2.0,
                'entropy_lo': 0.5,
                'L_min': 1,
                'L_max': 3,
                'energy_hi': 10.0,
                'energy_lo': 1.0
            }
        }
        
        self._ensure_directories()

    def _ensure_directories(self):
        dirs_to_create = [
            self.paths.artifacts_dir,
            self.paths.logs_dir,
            self.paths.checkpoints_dir,
        ]
        if self.paths.experiments_dir:
            dirs_to_create.append(self.paths.experiments_dir)
        if self.paths.run_dir:
            dirs_to_create.append(self.paths.run_dir)
        
        for dir_path in dirs_to_create:
            os.makedirs(dir_path, exist_ok=True)
            
    def apply_overrides(self, overrides: dict):
        # 简化版 override 应用
        def _apply(obj, d):
            for k, v in d.items():
                if hasattr(obj, k):
                    try:
                        attr = getattr(obj, k)
                        if isinstance(attr, dict) and isinstance(v, dict):
                            attr.update(v)
                        elif hasattr(attr, '__dict__') and isinstance(v, dict):
                            # Recursively apply to nested config object
                            _apply(attr, v)
                        else:
                            setattr(obj, k, v)
                    except Exception:
                        pass
                        
        if 'paths' in overrides: _apply(self.paths, overrides['paths'])
        if 'training' in overrides: _apply(self.training, overrides['training'])
        if 'system' in overrides: _apply(self.system, overrides['system'])
        if 'model' in overrides: _apply(self.model, overrides['model'])
        if 'data' in overrides: _apply(self.data, overrides['data'])
        if 'efficiency' in overrides: _apply(self.efficiency, overrides['efficiency'])
        if 'interaction' in overrides: _apply(self.interaction, overrides['interaction'])
        if 'features' in overrides: _apply(self.features, overrides['features'])
        if 'loss' in overrides: _apply(self.loss, overrides['loss'])
        if 'loss_engine' in overrides: _apply(self.loss, overrides['loss_engine'])
        if 'physics' in overrides: _apply(self.physics, overrides['physics'])
        if 'life_support' in overrides: 
            ls_overrides = overrides['life_support']
            if 'profile' in ls_overrides:
                self.life_support.set_profile(ls_overrides['profile'])
            _apply(self.life_support, ls_overrides)
        if 'heuristics' in overrides: _apply(self.heuristics, overrides['heuristics'])
        if 'fov_controller' in overrides: _apply(self, {'fov_controller': overrides['fov_controller']})
        
        return self

    def save_snapshot(self, save_dir=None, filename='config_snapshot.json'):
        """保存当前配置快照"""
        try:
            import json
            def _to_dict(obj):
                if hasattr(obj, '__dict__'):
                    return {k: _to_dict(v) for k, v in obj.__dict__.items() if not k.startswith('_')}
                return obj
                
            snapshot = {
                'version': self.VERSION,
                'paths': _to_dict(self.paths),
                'training': _to_dict(self.training),
                'system': _to_dict(self.system),
                'model': _to_dict(self.model),
                'data': _to_dict(self.data),
                'efficiency': _to_dict(self.efficiency),
                'features': _to_dict(self.features),
                'loss': _to_dict(self.loss),
                'labels': _to_dict(self.labels),
                'interaction': _to_dict(self.interaction),
                'physics': _to_dict(self.physics),
                'heuristics': _to_dict(self.heuristics),
                'life_support': _to_dict(self.life_support),
                'timestamp': datetime.now().isoformat(),
                # Hash for verification (placeholder)
                'hash': 'N/A',
                'seed': self.training.seed
            }
            
            if save_dir is None:
                save_dir = self.paths.run_dir
                
            path = os.path.join(save_dir, filename)
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(snapshot, f, indent=2, ensure_ascii=False)
                
            snapshot['path'] = path
            return snapshot
        except Exception as e:
            print(f"[CFG][WARN] Failed to save snapshot: {e}")
            return {}

    def print_summary(self):
        print("=" * 60)
        print("配置摘要 (SSOT V6)")
        print("=" * 60)
        print(f"根目录: {self.root_dir}")
        print(f"Foundation: {self.paths.foundation_path}")
        print(f"Samples: {self.paths.samples_path}")
        print()
        print(f"Model Input Dim: {self.model.input_dim}")
        print(f"Model Edge Dim: {self.model.edge_dim}")
        print("=" * 60)
