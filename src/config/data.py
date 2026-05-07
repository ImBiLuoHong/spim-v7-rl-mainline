from dataclasses import dataclass

@dataclass
class OnlineConfig:
    """在线模式配置类 (Phase 3.1)"""
    def __init__(self):
        self.keep_time_series = True
        self.episode_duration_seconds = 2700 # [SSOT] Standard physical episode duration (45 mins)
        self.step_seconds = 2700             # [Deprecated] Alias for episode_duration_seconds
        self.data_resolution_seconds = 900   # [SSOT] Data Resolution (15 mins)
        self.t0 = 0
        self.t_max = 127
        self.default_sampling_delay_steps = 2
        self.obs_mode = "instant" # instant, window
        self.window_W = 4
        # [Moved to PhysicsConfig] poison_threshold
        # [Moved to PhysicsConfig] physical_threshold


@dataclass
class DataConfig:
    """数据配置类 - 管理数据处理和加载相关的参数 (V6)"""
    
    def __init__(self):
        self.use_dataloader_v2 = False # [Deprecated] 废弃旧V2
        self.use_dataloader_v6 = True  # [Active] 启用V6
        
        self.window_size = 12
        self.num_workers = 16 
        self.pin_memory = True
        self.prefetch_factor = 4 
        self.persistent_workers = True
        self.preload = False # SSOT: Disable blocking preload by default
        self.non_blocking_transfer = True # SSOT: Default to True for performance
        
        # Phase 3.1 Task Mode
        self.task_mode = "forensics" # forensics, online
        self.online = OnlineConfig()
        
        # V6 Feature Flags (Replacing legacy include_* flags)
        self.use_chlorine_val = True    # Channel 0
        self.use_poison_bin = True      # Channel 1
        self.use_is_revealed = True     # Channel 2
        self.use_is_trigger = True      # Channel 3
        self.use_is_sensor = True       # Channel 4
        self.use_relative_stt = True    # Channel 5
        self.use_relative_euc = True    # Channel 6
        self.use_subgraph_centrality = True # Channel 7 (if index 7)
        
        self.use_signed_stt_edge = True # [Deprecated] Legacy
        
        # Phase 4: Precomputed Edge Features
        self.use_edge_attr = True      # [True Full Blood] Enable loading edge_attr_summary (Physics STT)
        self.use_virtual_edges = False   # [User Request] Enable virtual edges by default
        self.filter_no_source = True    # [User Request] Filter groups where source is not in subgraph
        
        # [SSOT V6.3] Unified Spatiotemporal 8-Channel Blueprint
        self.edge_dim = 8
        self.edge_channels = {
            0: "log_med_stt",
            1: "log_p90_stt",
            2: "log_min_stt",
            3: "flip_rate",
            4: "is_physical",
            5: "is_virtual",
            6: "anchor_type",
            7: "reserved"
        }
        
        # Input Semantic Diagnostics
        self.feature_mode = "baseline" # baseline, no_mask, explicit_flag
        self.cache_version = "v1" # v1, baseline, explicit, etc.
        self.rebuild_cache = False # If True, delete LMDB before loading
        self.skip_lmdb = False # Default to LMDB-backed loading; only disable for targeted debugging
        self.max_samples = None # Limit dataset size for fast experiments
        
        # Plan B: Data Scale Up & Augmentation
        # self.augmentation_factor = 8     # 6k -> ~50k
        # self.aug_noise_std = 0.05        # Injection noise
        # self.aug_mask_prob = 0.15        # Sensor failure simulation
        
        # 归一化
        self.normalize = True
        # [Moved to PhysicsConfig] data_resolution_seconds
