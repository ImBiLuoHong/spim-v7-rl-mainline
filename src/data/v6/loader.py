import os
import torch
import logging
from torch.utils.data import DataLoader
from src.data.v6.dataset import NpzDatasetV6
from src.data.v6.collate import v6_collate_fn
from src.data.v6.lmdb_dataset import LmdbDatasetV6
from src.tools.convert_to_lmdb import convert_to_lmdb
from src.utils.hash_utils import generate_dir_fingerprint

logger = logging.getLogger(__name__)

# [CPU AFFINITY HACK] Force PyTorch to use all cores
# Moved to module level for pickling support in 'spawn' mode
def _worker_init_fn(worker_id):
    try:
        import os
        # Get all available cores (0-255 usually on 5090 machines)
        all_cpus = os.sched_getaffinity(0)
        # Force current process to use ALL cores
        os.sched_setaffinity(0, all_cpus)
    except Exception:
        pass

def check_lmdb_validity(cache_dir, source_dir, mode, cache_version='v1'):
    """
    检查 LMDB 是否存在且 Hash 匹配。
    """
    lmdb_path = os.path.join(cache_dir, f"v6_dataset_{mode}_{cache_version}.lmdb")
    hash_path = os.path.join(cache_dir, f"v6_dataset_{mode}_{cache_version}.hash")
    
    if not os.path.exists(lmdb_path) or not os.path.exists(hash_path):
        return False
        
    try:
        current_hash = generate_dir_fingerprint(source_dir)
        with open(hash_path, 'r') as f:
            old_hash = f.read().strip()
        return current_hash == old_hash
    except Exception as e:
        logger.warning(f"Hash check failed: {e}")
        return False

def create_dataloaders(
    data_root,
    node_mapping_path=None,
    pipe_mapping_path=None,
    batch_size=32,
    eval_batch_size=None,
    normalize=True,
    build_soft_labels=False,
    window_size=12,
    trace_time_limit=12,
    cfg=None,
    audit_mode=None, # [AUDIT]
    skip_lmdb=False, # [TEST] Force Raw Dataset
    train_only=False,
):
    """
    Creates V6 DataLoaders (LMDB Optimized)
    """
    logger.info(f"[V6Loader] Creating dataloaders from {cfg.paths.samples_path} (Audit: {audit_mode}, SkipLMDB: {skip_lmdb})")

    # 1. Split Config
    split_dir = getattr(cfg.paths, 'split_dir', None)
    if not split_dir:
        split_dir = getattr(cfg.paths, 'data_split_dir', None)
    if not split_dir:
        split_dir = data_root
        
    logger.info(f"[V6Loader] Using split_dir: {split_dir}")
    
    # Common Args for Raw Dataset
    # 强制 preload=False 以加速初始化
    # 强制 keep_raw=True 以便 convert_to_lmdb 能读取到 raw data (然后删除)
    # 这里的 keep_raw 只是告诉 NpzDatasetV6 加载 raw data。
    # convert_to_lmdb 会负责删除它。
    
    # 提取参数
    use_edge_attr = bool(getattr(cfg.data, 'use_edge_attr', False))
    use_virtual_edges = bool(getattr(cfg.data, 'use_virtual_edges', False))
    filter_no_source = bool(getattr(cfg.data, 'filter_no_source', False))
    task_mode = getattr(cfg.data, 'task_mode', 'forensics')
    feature_mode = getattr(cfg.data, 'feature_mode', 'baseline')
    max_samples = getattr(cfg.data, 'max_samples', None)
    cache_version = getattr(cfg.data, 'cache_version', 'v1')
    rebuild_cache = getattr(cfg.data, 'rebuild_cache', False)
    
    # [SSOT] Use Dedicated Cache Dir
    cache_dir = getattr(cfg.paths, 'cache_dir', os.path.join(data_root, 'cache_lmdb'))
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir, exist_ok=True)
    
    online_config = vars(cfg.data.online) if hasattr(cfg.data, 'online') else {}
    
    # [SSOT V6.3] Edge Configuration
    edge_config = {
        'dim': getattr(cfg.data, 'edge_dim', 8),
        'channels': getattr(cfg.data, 'edge_channels', {})
    }
    
    # [SSOT Fix] Enforce data resolution from Physics Config (SSOT Source)
    # Map 'data_resolution_seconds' (900) to 'step_seconds' for Dataset consumption
    if hasattr(cfg, 'physics') and hasattr(cfg.physics, 'data_resolution_seconds'):
        online_config['step_seconds'] = cfg.physics.data_resolution_seconds
    elif hasattr(cfg.data.online, 'data_resolution_seconds'):
         online_config['step_seconds'] = cfg.data.online.data_resolution_seconds
    
    # 实例化 Raw Datasets (轻量级，因为 preload=False)
    # 我们需要实例化它们来处理 split 和 filtering 逻辑
    def create_raw(mode):
        return NpzDatasetV6(
            samples_dir=cfg.paths.samples_path,
            foundation_dir=cfg.paths.foundation_path,
            mode=mode,
            window_size=window_size,
            split_dir=split_dir,
            preload=False, # 强制关闭 preload
            keep_raw=True, # 必须为 True 以便读取数据
            task_mode=task_mode,
            online_config=online_config,
            use_edge_attr=use_edge_attr,
            use_virtual_edges=use_virtual_edges,
            filter_no_source=filter_no_source,
            num_workers=0, # 强制单进程初始化
            audit_mode=audit_mode,
            log_normalize=normalize,
            edge_config=edge_config, # [SSOT] Pass Edge Config
            feature_mode=feature_mode,
            max_samples=max_samples
        )

    train_dataset = create_raw('train')
    val_dataset = None if train_only else create_raw('val')
    test_dataset = None if train_only else create_raw('test')
    
    # Handle Random Split if needed
    if len(train_dataset) == 0:
        pass

    if (not train_only) and train_dataset.sample_files == val_dataset.sample_files and len(train_dataset) > 0:
        logger.info("[V6Loader] Split files not found. Performing random split 70/15/15.")
        full_files = train_dataset.sample_files
        import random
        random.seed(42)
        random.shuffle(full_files)
        
        n_total = len(full_files)
        n_train = int(n_total * 0.7)
        n_val = int(n_total * 0.15)
        
        train_files = full_files[:n_train]
        val_files = full_files[n_train:n_train+n_val]
        test_files = full_files[n_train+n_val:]
        
        # 更新 Raw Datasets 的文件列表
        # NpzDatasetV6 会在 __getitem__ 中使用 self.groups，而 groups 是基于 sample_files 生成的。
        # 我们需要重新 group。
        # NpzDatasetV6 没有公开 re-group 的接口，但我们可以手动调用内部方法。
        # 或者更简单：修改 sample_files 后重新调用 _group_files
        
        def update_dataset(ds, files):
            ds.sample_files = files
            ds.groups = ds._group_files(files)
            # 重新过滤 (filter_no_source)
            if ds.filter_no_source:
                 # 这里为了简单，我们假设之前已经 filter 过了 (基于 cache)。
                 # 但如果 split 变了，我们需要重新 filter 吗？
                 # filter 是针对每个 group 的，跟 split 无关。
                 # 但是 ds.groups 变了，我们需要重新应用 filter。
                 # 由于 filter 很慢，我们希望复用 cache。
                 # 现有的 filter 逻辑是基于 self.groups 的。
                 # 我们可以再次调用 filter 逻辑，但如果 cache 存在，它会读取 cache (全量)。
                 # 这会导致 split 失效（因为 cache 是全量的）。
                 # 所以我们必须手动 filter。
                 
                 # 幸好，我们在初始化时已经 filter 过一次全量了 (如果 cache 存在)。
                 # 但是 wait，train_dataset 初始化时 filter 了全量。
                 # 现在我们将 files 缩减了。
                 # 我们只需要保留在 files 里的 groups。
                 
                 # 简单的做法：
                 # 遍历当前的 groups，只保留那些文件在 new files 里的。
                 new_groups = []
                 file_set = set(files)
                 for g in ds.groups:
                     if g[0] in file_set: # 只要检查第一个文件
                         new_groups.append(g)
                 ds.groups = new_groups
        
        update_dataset(train_dataset, train_files)
        update_dataset(val_dataset, val_files)
        update_dataset(test_dataset, test_files)
        
    if train_only:
        logger.info(f"[V6Loader] Raw Dataset Sizes - Train: {len(train_dataset)} (train_only=True)")
    else:
        logger.info(f"[V6Loader] Raw Dataset Sizes - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    # === LMDB 拦截与转换 ===
    
    def get_lmdb_dataset(raw_dataset, mode):
        # 1. 检查有效性
        samples_dir = cache_dir # Use cache_dir instead of samples_path
        source_dir = getattr(raw_dataset, 'samples_dir', cfg.paths.samples_path)
        
        lmdb_path = os.path.join(samples_dir, f"v6_dataset_{mode}_{cache_version}.lmdb")
        hash_path = os.path.join(samples_dir, f"v6_dataset_{mode}_{cache_version}.hash")
        
        if rebuild_cache:
             if os.path.exists(lmdb_path):
                 import shutil
                 if os.path.isdir(lmdb_path): shutil.rmtree(lmdb_path)
                 else: os.remove(lmdb_path)
             if os.path.exists(hash_path): os.remove(hash_path)
             logger.info(f"[Loader] Force Rebuild: Deleted old cache {lmdb_path}")

        if not check_lmdb_validity(samples_dir, source_dir, mode, cache_version=cache_version):
            logger.warning(f"[Loader] LMDB for {mode} (Ver: {cache_version}) is missing or outdated. Rebuilding in {samples_dir}...")
            # 阻塞式转换
            success = convert_to_lmdb(raw_dataset, mode=mode, output_dir=samples_dir, cache_version=cache_version)
            if not success:
                raise RuntimeError(f"Failed to create LMDB for {mode}")
        
        # 2. 返回 LMDB Dataset，并保留诊断/回放依赖的静态图资产
        lmdb_dataset = LmdbDatasetV6(lmdb_path)
        for attr_name in (
            'topology',
            'global_edge_index',
            'stt_dynamic_series',
            'num_nodes',
            'samples_dir',
            'foundation_dir',
            'mode',
        ):
            attr_value = getattr(raw_dataset, attr_name, None)
            if attr_value is not None:
                setattr(lmdb_dataset, attr_name, attr_value)
        return lmdb_dataset

    # 替换为 LMDB Dataset
    # 注意：如果 audit_mode 开启，我们可能不想转换？
    # 或者 audit mode 也需要 LMDB？
    # 如果是 audit mode，可能不需要转换，直接用 raw 也可以。
    # 但为了一致性，我们尽量用 LMDB。
    # 除非是 fast audit，我们不想等待转换。
    
    if audit_mode == 'fast' or skip_lmdb:
        logger.info(f"[V6Loader] Audit Mode ({audit_mode}) or SkipLMDB ({skip_lmdb}): Skipping LMDB check/conversion, using Raw Dataset.")
        # 直接使用 Raw Dataset
        train_ds = train_dataset
        val_ds = val_dataset
        test_ds = test_dataset
    else:
        train_ds = get_lmdb_dataset(train_dataset, 'train')
        val_ds = None if train_only else get_lmdb_dataset(val_dataset, 'val')
        test_ds = None if train_only else get_lmdb_dataset(test_dataset, 'test')

    # DataLoader optimization config should come from the active SSOT config.
    num_workers = int(
        getattr(
            cfg.data,
            'num_workers',
            getattr(cfg.efficiency, 'num_workers', 12),
        )
    )
    pin_memory = bool(
        getattr(
            cfg.data,
            'pin_memory',
            getattr(cfg.efficiency, 'pin_memory', True),
        )
    )
    persistent_workers = bool(
        getattr(
            cfg.data,
            'persistent_workers',
            getattr(cfg.efficiency, 'persistent_workers', num_workers > 0),
        )
    )
    prefetch_factor = getattr(
        cfg.data,
        'prefetch_factor',
        getattr(cfg.efficiency, 'prefetch_factor', 2),
    )
    if audit_mode:
        num_workers = 0
    if num_workers <= 0:
        num_workers = 0
        persistent_workers = False
        prefetch_factor = None
    elif prefetch_factor is None:
        prefetch_factor = 2

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=v6_collate_fn,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        worker_init_fn=_worker_init_fn if num_workers > 0 else None
    )

    pos_weight = 30.0
    imbalance_info = {
        'class_weights_vec': [1.0, pos_weight],
    }

    if train_only:
        return train_loader, None, None, imbalance_info

    if eval_batch_size is None:
        eval_batch_size = batch_size

    val_loader = DataLoader(
        val_ds,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=v6_collate_fn,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        worker_init_fn=_worker_init_fn if num_workers > 0 else None
    )
    
    test_loader = DataLoader(
        test_ds,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=v6_collate_fn,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        worker_init_fn=_worker_init_fn if num_workers > 0 else None
    )
    
    return train_loader, val_loader, test_loader, imbalance_info
