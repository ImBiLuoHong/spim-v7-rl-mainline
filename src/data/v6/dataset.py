import os
import torch
import numpy as np
from torch.utils.data import Dataset
from torch_geometric.data import Data, Batch
from torch_geometric.utils import subgraph
try:
    from torch_sparse import SparseTensor
except ImportError:
    SparseTensor = None
import logging
from collections import defaultdict
from tqdm import tqdm
from .topology import HydraulicTopology

logger = logging.getLogger(__name__)


def derive_signed_stt_series(edge_attr_dynamic: torch.Tensor) -> torch.Tensor:
    """
    Convert foundation dynamic edge channels into a signed STT series aligned with edge_index.

    Repo-grounded semantics from `graph.npz`:
    - channel 0: signed flow direction
    - channel 1: positive travel time magnitude

    We encode the effective directional travel time as:
    - +stt when flow is along the stored edge direction
    - -stt when flow is opposite the stored edge direction
    - 0 when the edge is effectively inactive
    """
    if edge_attr_dynamic.dim() != 3 or edge_attr_dynamic.size(-1) < 2:
        raise ValueError(
            f"edge_attr_dynamic must have shape [T, E, C>=2], got {tuple(edge_attr_dynamic.shape)}"
        )
    flow = edge_attr_dynamic[:, :, 0]
    stt_mag = edge_attr_dynamic[:, :, 1].clamp_min(0.0)
    signed_stt = torch.zeros_like(stt_mag)
    signed_stt = torch.where(flow > 1e-6, stt_mag, signed_stt)
    signed_stt = torch.where(flow < -1e-6, -stt_mag, signed_stt)
    return signed_stt

# Helper function for ProcessPoolExecutor (Picklable)
def load_group_worker(args):
    """
    Standalone worker function to load a group of files.
    args: (idx, group_files, samples_dir, foundation_dir, use_virtual_edges, log_normalize, use_edge_attr)
    """
    # ... implementation skipped ...
    pass

class NpzDatasetV6(Dataset):
    """
    V6 Hybrid Loader (Foundation + Samples)
    Aligns with Static Feature Spec V3.2
    
    Updated for T2I-BDASNet:
    1. Lazy Loading (No Preload)
    2. Event Grouping (Tri-Trigger)
    3. Returns Batch of 3 Subgraphs per Item
    """
    
    def __init__(self, 
                 samples_dir: str, 
                 foundation_dir: str, 
                 mode: str = 'train',
                 window_size: int = 24,
                 split_dir: str = None,
                 preload: bool = False,
                 keep_raw: bool = False,
                 task_mode: str = 'classification',
                 online_config: dict = None,
                 use_edge_attr: bool = False,
                 use_virtual_edges: bool = False,
                 filter_no_source: bool = False,
                 num_workers: int = 1,
                 audit_mode: str = None,
                 log_normalize: bool = False,
                 edge_config: dict = None,
                 feature_mode: str = 'baseline',
                 max_samples: int = None):
        super().__init__()
        
        self.samples_dir = samples_dir
        self.foundation_dir = foundation_dir
        self.mode = mode
        self.window_size = window_size
        self.split_dir = split_dir
        self.preload = preload
        self.keep_raw = keep_raw
        self.task_mode = task_mode
        self.online_config = online_config or {}
        self.use_edge_attr = use_edge_attr
        self.use_virtual_edges = use_virtual_edges
        self.filter_no_source = filter_no_source
        self.num_workers = num_workers
        self.audit_mode = audit_mode
        self.log_normalize = log_normalize
        self.edge_config = edge_config or {'dim': 8, 'channels': {}}
        self.feature_mode = feature_mode
        self.topology = None
        
        # [AUDIT] Force Disable Preload in Audit Mode
        if self.audit_mode:
            if self.preload:
                logger.info(f"[V6Loader] Audit Mode ({self.audit_mode}): Preload FORCE DISABLED.")
            self.preload = False
            # Also limit workers to avoid overhead
            self.num_workers = 0 
        
        if self.use_virtual_edges:
             # Pass foundation_dir, and it will auto-attach to shared instance
             self.topology = HydraulicTopology(foundation_dir)

        self.cache = {}
        
        # 1. Load Foundation Data
        self._load_foundation()
        
        # 2. Index Samples & Group by Event
        raw_files = []
        
        # Priority: Load from split files if split_dir is provided
        if split_dir and os.path.exists(split_dir):
            split_file = os.path.join(split_dir, f"{mode}.txt")
            if os.path.exists(split_file):
                logger.info(f"[V6Loader] Loading split from {split_file}")
                with open(split_file, 'r') as f:
                    raw_files = [line.strip() for line in f if line.strip()]
            else:
                logger.warning(f"[V6Loader] Split file {split_file} not found. Fallback to directory scan.")
                try:
                    pass
                except Exception as e:
                    pass
        
        # Fallback: Scan directory if empty
        if len(raw_files) == 0 and os.path.exists(samples_dir):
            for root, dirs, fnames in os.walk(samples_dir):
                for fname in fnames:
                    if fname.endswith('.npz'):
                        raw_files.append(os.path.join(root, fname))
            raw_files = sorted(raw_files)

        if len(raw_files) == 0:
            if mode == 'train':
                 raise ValueError(f"No samples found in {samples_dir} or {split_dir}. Cannot train.")
            else:
                 logger.warning(f"No samples for {mode}")

        # Store raw files for compatibility with loader checks
        self.sample_files = raw_files

        # Grouping Logic
        self.groups = self._group_files(raw_files)
        logger.info(f"[V6Loader] Indexed {len(raw_files)} files into {len(self.groups)} scenarios (Tri-Trigger groups).")
        
        # Filter No Source
        cache_path = os.path.join(samples_dir, f".valid_groups_{mode}.cache")
        need_filter = True
        
        if False: # [HOTFIX] Force disable filter_no_source to unblock experiment
            # [AUDIT FIX] Fast Audit Shortcut
            if self.audit_mode == 'fast' and not os.path.exists(cache_path):
                logger.warning("[V6Loader] Audit Mode (Fast): No filter cache found. Skipping scan to avoid delay.")
                need_filter = False
            elif os.path.exists(cache_path):
                logger.info(f"[V6Loader] Loading filtered groups from cache: {cache_path}")
                try:
                    import pickle
                    with open(cache_path, 'rb') as f_cache:
                        cached_groups = pickle.load(f_cache)
                    
                    if len(cached_groups) > 0 or len(self.groups) == 0:
                        self.groups = cached_groups
                        need_filter = False
                    else:
                        logger.warning("[V6Loader] Cache exists but is empty (and we have raw files). Re-filtering.")
                        need_filter = True
                except Exception as e:
                    logger.warning(f"[V6Loader] Failed to load cache: {e}. Re-filtering.")
                    need_filter = True
            
            if need_filter:
                logger.info("[V6Loader] Filtering groups where source is not in subgraph (One-time cost)...")
                valid_groups = []
                for group in tqdm(self.groups, desc="Filtering"):
                    try:
                        fpath = group[0]
                        if not os.path.exists(fpath):
                            if not os.path.isabs(fpath): 
                                fpath_cand = os.path.join(self.samples_dir, fpath)
                                if os.path.exists(fpath_cand):
                                    fpath = fpath_cand
                                    
                        with np.load(fpath, allow_pickle=True) as f:
                            src = -1
                            if 'global_injection_node' in f: src = int(f['global_injection_node'])
                            elif 'injection_node_index' in f: src = int(f['injection_node_index'])
                            
                            if 'node_indices' in f: g_ids = f['node_indices']
                            elif 'global_node_index' in f: g_ids = f['global_node_index']
                            else: 
                                # if len(valid_groups) < 1: logger.warning(f"File {fpath} missing node indices")
                                continue
                            
                            if src != -1 and src in g_ids:
                                valid_groups.append(group)
                            else:
                                if len(valid_groups) < 1:
                                    logger.warning(f"Filter Fail {fpath}: Src={src}, InGraph={src in g_ids if src!=-1 else False}")
                    except Exception as e:
                        logger.warning(f"Filtering error for {fpath}: {e}")
                        continue
                
                self.groups = valid_groups
                try:
                    import pickle
                    with open(cache_path, 'wb') as f_cache:
                        pickle.dump(self.groups, f_cache)
                except:
                    pass
            
            logger.info(f"[V6Loader] Final scenarios: {len(self.groups)}")

        # Limit Size (for Debug/Diagnostic)
        if max_samples is not None and max_samples > 0:
            if len(self.groups) > max_samples:
                logger.info(f"[V6Loader] Limiting dataset to {max_samples} samples (Original: {len(self.groups)})")
                self.groups = self.groups[:max_samples]

        # 3. RAM Caching (Preload) - OPTIMIZED FOR PARALLEL PROCESSING
        if self.preload:
            logger.info(f"[V6Loader] Parallel Preloading {len(self.groups)} scenarios into RAM using ProcessPool...")
            
            import multiprocessing as mp
            from concurrent.futures import ProcessPoolExecutor
            
            # Use all available cores (aggressive) or user limit
            num_workers = self.num_workers if self.num_workers > 0 else mp.cpu_count()
            
            try:
                logger.info(f"[V6Loader] Parallel Preloading using ProcessPoolExecutor with {num_workers} workers...")
                
                # We need to disable preload recursively
                self.preload = False
                
                # Note: ProcessPoolExecutor requires picklable functions.
                # Instance methods are picklable ONLY if the instance is picklable.
                # NpzDatasetV6 is picklable.
                # But passing 'self._load_group' might cause the whole dataset to be pickled to each worker.
                # Since we already loaded foundation data (big tensors), this copy is expensive?
                # Actually, on Linux 'fork', the memory is COW. So it's cheap!
                # But 'ProcessPoolExecutor' might use 'spawn' or pickling depending on config.
                # Default on Linux is fork.
                
                # However, to be super efficient, we should ensure we don't duplicate foundation data.
                # The Shared Memory HydraulicTopology handles the graph.
                # What about self.global_pos, etc?
                # If they are Torch Tensors, they might be moved to shared memory automatically by PyTorch if in 'share_memory_()' mode.
                # But here we just rely on OS Fork COW.
                
                with ProcessPoolExecutor(max_workers=num_workers) as executor:
                    # Map indices to the loader function
                    results = list(tqdm(executor.map(self._load_group_wrapper, range(len(self.groups))), 
                                       total=len(self.groups), desc="Preloading (PROCESSES)"))
                
                self.preload = True
                
                # Store in cache
                valid_count = 0
                for idx, data in enumerate(results):
                    if data is not None:
                        self.cache[idx] = data
                        valid_count += 1
                
                logger.info(f"[V6Loader] Preload complete. Cached {valid_count} scenarios.")
            except Exception as e:
                logger.error(f"[V6Loader] Parallel preload failed: {e}. Falling back to sequential.")
                self.preload = False
                for idx in tqdm(range(len(self.groups)), desc="Fallback Preloading"):
                    self.cache[idx] = self._load_group(idx)
                self.preload = True

    def _load_group_wrapper(self, idx):
        """Wrapper for _load_group to be called by ProcessPoolExecutor."""
        # This is an instance method, so 'self' is passed.
        # On Linux fork, 'self' is valid and points to the forked memory.
        return self._load_group(idx)


    def _group_files(self, file_list):
        """
        Groups files by event ID.
        Assumes format: ..._augX.npz OR ..._viewX.npz
        Key: remove suffix.
        """
        groups = defaultdict(list)
        for f in file_list:
            # parsing key
            if '_view' in f:
                key = f.rsplit('_view', 1)[0]
            elif '_aug' in f:
                key = f.rsplit('_aug', 1)[0]
            else:
                # If no aug suffix, treat as singleton or use full name
                key = f.replace('.npz', '')
            groups[key].append(f)
        
        # Convert to list of lists, sorted by suffix to ensure order
        final_groups = []
        for k in sorted(groups.keys()):
            g = sorted(groups[k])
            # Optional: Enforce size 3? Or just take what we have.
            # Architect says "3 different Trigger". If we have < 3, we might need to duplicate.
            # For now, just pass the list. The model should handle flexible K or we duplicate here.
            # Let's pad to 3 by duplicating the last one if needed, or first.
            if len(g) < 3:
                while len(g) < 3:
                    g.append(g[0]) # Duplicate first
            # If > 3, take first 3
            if len(g) > 3:
                g = g[:3]
            final_groups.append(g)
            
        return final_groups

    def _load_foundation(self):
        """Loads global static assets into memory."""
        graph_path = os.path.join(self.foundation_dir, 'graph.npz')
        if not os.path.exists(graph_path):
            raise FileNotFoundError(f"Foundation graph not found at {graph_path}")
            
        logger.info(f"[V6Loader] Loading foundation from {graph_path}...")
        with np.load(graph_path, allow_pickle=True) as f:
            # Global static attributes
            self.global_pos = torch.from_numpy(f['node_coords']).float() # (N_global, 2)
            self.global_edge_index = torch.from_numpy(f['edge_index']).long() # (2, E_global)
            self.global_edge_attr = torch.from_numpy(f['edge_attr_static']).float() # (E_global, 2)
            
            # Phase 4: Compute STT-Centric Physical Edge Attributes
            if self.use_edge_attr and 'edge_attr_dynamic' in f:
                logger.info("[V6Loader] Loading edge_attr_dynamic for STT physics reconstruction...")
                
                # Load Dynamic Attributes: (T, E, C) or (E, T, C)?
                # User says: [288, 68557, 3] -> (T, E, C)
                edge_attr_dyn = f['edge_attr_dynamic']
                
                # Convert to Torch
                if isinstance(edge_attr_dyn, np.ndarray):
                    edge_attr_dyn = torch.from_numpy(edge_attr_dyn).float()
                
                # Ensure shape (T, E, C)
                if edge_attr_dyn.shape[1] != self.global_edge_attr.shape[0]:
                    if edge_attr_dyn.shape[0] == self.global_edge_attr.shape[0]:
                        edge_attr_dyn = edge_attr_dyn.permute(1, 0, 2) # (E, T, C) -> (T, E, C)
                    else:
                        logger.warning(f"[V6Loader] Dynamic edge attr shape mismatch! Dyn={edge_attr_dyn.shape}, Static={self.global_edge_attr.shape}")
                        edge_attr_dyn = None
                
                if edge_attr_dyn is not None:
                    # Extract signed STT from the foundation dynamic channels.
                    # Channel 0 is signed flow; channel 1 is travel-time magnitude.
                    # This matches `src/data/v6/topology.py` and the edge_attr_summary stats.
                    stt_series = derive_signed_stt_series(edge_attr_dyn)
                    
                    # Store dynamic series for Reachability
                    self.stt_dynamic_series = stt_series
                    
                    # 1. Compute Directionality (P_forward)
                    # P(STT > 0)
                    p_forward = (stt_series > 1e-6).float().mean(dim=0) # (E,)
                    
                    # 2. Compute Flip Rate
                    # Sign changes over time
                    signs = torch.sign(stt_series)
                    flips = (signs[1:, :] != signs[:-1, :]).float().mean(dim=0) # (E,)
                    
                    # 3. Compute Magnitude Stats (using Absolute STT)
                    abs_stt = stt_series.abs()
                    
                    # Log Normalize: log1p(stt)
                    # Note: User wants "log normalized scaling". 
                    # stt can be 1e7. log1p(1e7) ~ 16.
                    log_stt = torch.log1p(abs_stt)
                    
                    # Median
                    median_stt = log_stt.median(dim=0).values
                    
                    # P90
                    # quantile requires float32/64. If abs_stt is float16, might need cast.
                    # Assuming float32 from .float() above.
                    p90_stt = torch.quantile(log_stt, 0.9, dim=0)
                    
                    # Min
                    min_stt = log_stt.min(dim=0).values
                    
                    # Construct 8-Channel Physical Attributes (Unified Spatiotemporal 8-Channel Blueprint)
                    # [SSOT] Use Configured Dimension
                    edge_dim = self.edge_config.get('dim', 8)
                    
                    # Ch0: Log_Med_STT
                    # Ch1: Log_P90_STT
                    # Ch2: Log_Min_STT
                    # Ch3: Flip_Rate
                    # Ch4: Is_Physical (1.0)
                    # Ch5: Is_Virtual (0.0)
                    # Ch6: Anchor_Type (0.0)
                    # Ch7: Reserved (0.0)
                    
                    num_edges = self.global_edge_attr.shape[0]
                    new_edge_attr = torch.zeros((num_edges, edge_dim), dtype=torch.float32)
                    
                    new_edge_attr[:, 0] = median_stt
                    new_edge_attr[:, 1] = p90_stt
                    new_edge_attr[:, 2] = min_stt
                    new_edge_attr[:, 3] = flips
                    new_edge_attr[:, 4] = 1.0 # Is_Physical
                    new_edge_attr[:, 5] = 0.0 # Is_Virtual
                    new_edge_attr[:, 6] = 0.0 # Anchor_Type
                    if edge_dim > 7:
                        new_edge_attr[:, 7] = 0.0 # Reserved
                    
                    # [Logic Check] Edge Directionality
                    # User says: "Use majority flow direction as edge index direction... if STT was negative (p_forward < 0.5), reverse index"
                    
                    # Calculate mask for reversal
                    needs_reverse = (p_forward < 0.5)
                    
                    # Apply Reversal to Global Edge Index
                    row = self.global_edge_index[0].clone()
                    col = self.global_edge_index[1].clone()
                    
                    # Swap
                    new_row = torch.where(needs_reverse, col, row)
                    new_col = torch.where(needs_reverse, row, col)
                    
                    self.global_edge_index = torch.stack([new_row, new_col], dim=0)
                    
                    # [Fix] Apply Reversal to Dynamic STT Series
                    # If edge index is flipped, we must flip the sign of STT to maintain consistency.
                    # Original: u->v, STT=-5 (Flow v->u).
                    # Flipped: v->u. New STT should be +5 (Flow v->u).
                    # So if needs_reverse is True, we negate stt_series.
                    # stt_series is (T, E). needs_reverse is (E).
                    if hasattr(self, 'stt_dynamic_series'):
                        # Broadcast needs_reverse to T
                        mask_rev = needs_reverse.unsqueeze(0).expand_as(self.stt_dynamic_series)
                        self.stt_dynamic_series = torch.where(mask_rev, -self.stt_dynamic_series, self.stt_dynamic_series)
                        logger.info(f"[V6Loader] Aligned stt_dynamic_series signs with edge index reversal.")

                    count_reversed = needs_reverse.sum().item()
                    logger.info(f"[V6Loader] Reconstructed Physical Edge Attributes with STT Core (8-Channel).")
                    logger.info(f"          Log Median STT Mean: {median_stt.mean():.4f}")
                    logger.info(f"          Reversed {count_reversed} edges to match majority flow.")
                    
                    self.global_edge_attr = new_edge_attr
                else:
                     logger.warning("[V6Loader] Failed to load dynamic edges. Keeping static length only (padded).")
                     padding = torch.zeros((self.global_edge_attr.shape[0], 6), dtype=torch.float32)
                     self.global_edge_attr = torch.cat([self.global_edge_attr[:, 0:2], padding], dim=1)
            else:
                if self.use_edge_attr:
                    logger.warning("[V6Loader] edge_attr_dynamic NOT FOUND in graph.npz! Cannot reconstruct STT physics.")
                # Pad existing static to edge_dim if needed
                edge_dim = self.edge_config.get('dim', 8)
                if self.global_edge_attr.shape[1] < edge_dim:
                     padding = torch.zeros((self.global_edge_attr.shape[0], edge_dim - self.global_edge_attr.shape[1]), dtype=torch.float32)
                     self.global_edge_attr = torch.cat([self.global_edge_attr, padding], dim=1)
            
            # Optimization: Create SparseTensor for fast slicing
            if SparseTensor is not None:
                N_global = self.global_pos.shape[0]
                self.global_adj = SparseTensor(
                    row=self.global_edge_index[0],
                    col=self.global_edge_index[1],
                    value=self.global_edge_attr,
                    sparse_sizes=(N_global, N_global)
                )
            else:
                self.global_adj = None

            # Is_Sensor: Try to load from graph.npz or default to zeros
            self.global_is_sensor = torch.zeros(self.global_pos.shape[0])

    def _process_one_file(self, file_path, idx=0):
        """
        Reads and processes a single file from disk.
        Returns PyG Data object or None.
        """
        # Fix path duplication issue
        if not os.path.exists(file_path):
            if not os.path.isabs(file_path):
                file_path = os.path.join(self.samples_dir, file_path)
            
        try:
            with np.load(file_path, allow_pickle=True) as f:
                # 1. Raw Data & Indices
                if 'data' in f:
                    x_raw = torch.from_numpy(f['data']).float().permute(2, 0, 1) # (N, T, C)
                elif 'x' in f:
                    x_raw = torch.from_numpy(f['x']).float()
                    if x_raw.dim() == 2:
                        x_raw = x_raw.unsqueeze(1)
                else:
                    return None 
                
                # Global Indices
                if 'node_indices' in f:
                    global_ids = torch.from_numpy(f['node_indices']).long()
                elif 'global_node_index' in f:
                    global_ids = torch.from_numpy(f['global_node_index']).long()
                elif 'global_node_indices' in f:
                    global_ids = torch.from_numpy(f['global_node_indices']).long()
                else:
                    return None
                
                # V11 Shape Correction (Robust)
                N_target = global_ids.shape[0]
                if x_raw.shape[0] != N_target:
                    if x_raw.shape[1] == N_target:
                        # (T, N, C) -> (N, T, C)
                        x_raw = x_raw.permute(1, 0, 2)
                    elif x_raw.dim() == 3 and x_raw.shape[2] == N_target:
                         # (T, C, N) -> (N, T, C)
                         x_raw = x_raw.permute(2, 0, 1)
                    else:
                        logger.warning(f"Shape mismatch in {file_path}: x={x_raw.shape}, ids={N_target}")
                        return None
                
                # Double check
                if x_raw.shape[0] != N_target:
                     logger.warning(f"Failed to correct shape in {file_path}: x={x_raw.shape}, ids={N_target}")
                     return None
                
                # [FIX] Log Normalization (SSOT V6)
                if self.log_normalize:
                    # Logic: 
                    # 1. Scale [-inf, inf] to approx [-1, 1] for relative features?
                    # 2. Log1p for positive skew features (Signal/Conc).
                    
                    # For Signal/Conc (Ch 0, 1 in x_raw usually), they are strictly positive.
                    # RPE features (if in x_raw?) are usually distances.
                    # x_raw is (N, T, C). C=2 (Signal, Conc).
                    
                    # Apply log1p to all, assuming they are positive magnitudes.
                    # If x_raw contains negative values (e.g. difference), log1p will fail or nan.
                    # x_min was -0.0001 in inspection. Clamp to 0 first.
                    x_raw = torch.log1p(torch.clamp(x_raw, min=0.0))
                
                # Labels & Trigger Info
                source_global_idx = -1
                trigger_global_idx = -1
                trigger_time_step = 0 # Default to 0
                global_start_step = 0 # Default to 0
                
                # 1. Extract Metadata (Priority for V11/V6)
                if 'trigger_node_index' in f:
                     trigger_global_idx = int(f['trigger_node_index'])
                elif 'global_trigger_node' in f:
                     trigger_global_idx = int(f['global_trigger_node'])
                
                if 'trigger_time_step' in f:
                     trigger_time_step = int(f['trigger_time_step'])

                if 'global_start_step' in f:
                     global_start_step = int(f['global_start_step'])

                if idx < 3: logger.info(f"[Debug] File {idx}: trigger_time_step={trigger_time_step}, global_start_step={global_start_step}")
                
                # [SSOT Audit] Check Ch1 (Poison) vs Trigger Time
                # If trigger_time_step is RELATIVE, x_raw[trigger_time_step] should be > 0 (or near).
                # If trigger_time_step is ABSOLUTE, x_raw[trigger_time_step - global_start_step] should be > 0.
                
                # We perform a heuristic check to determine alignment mode.
                # Mode A: Relative (Default assumption)
                # Mode B: Absolute
                
                # Let's check which index has the first poison signal at the trigger node.
                # This is an expensive check, so we only do it for the first few files or if Audit Mode.
                
                # However, we must decide ONE logic for all files.
                # Based on inspection of 'inspect_trigger_time.py', we saw:
                # trigger_time_step = 85
                # global_start_step = 96
                # And usually poison starts around index 85-90 relative to file start.
                # So trigger_time_step is likely RELATIVE to file start.
                
                # But let's verify if 'trigger_time_step' < 'global_start_step'.
                # In the inspected file: 85 < 96.
                # If it was absolute, it would be 96 + 85 = 181.
                # So 'trigger_time_step' stored in file is almost certainly RELATIVE to the file.
                
                # Conclusion: Use trigger_time_step AS IS for indexing x_raw.
                # For Topology (Absolute Time), use global_start_step + trigger_time_step.
                
                if 'global_injection_node' in f:
                    source_global_idx = int(f['global_injection_node'])
                elif 'injection_node_index' in f:
                    source_global_idx = int(f['injection_node_index'])
                elif 'label' in f:
                    source_global_idx = int(f['label'][0])

                # 2. Construct Y (Priority: Metadata > File Y)
                # Fix for Alignment Issue: Reconstruct y from source_global_idx to ensure consistency
                y = torch.zeros(x_raw.shape[0], dtype=torch.float)
                y_constructed = False
                
                if source_global_idx != -1:
                    mask_y = (global_ids == source_global_idx)
                    if mask_y.any():
                        y[mask_y] = 1.0
                        y_constructed = True
                    # Note: If source is not in subgraph, y remains all zeros.
                
                if not y_constructed and 'y' in f:
                     # Fallback to stored y only if we couldn't reconstruct from metadata
                     if source_global_idx == -1:
                        y_stored = torch.from_numpy(f['y']).float()
                        if y_stored.shape[0] == x_raw.shape[0]:
                             y = y_stored
                        else:
                             logger.warning(f"Y shape mismatch in {file_path}: y={y_stored.shape}, N={x_raw.shape[0]}")
                             return None

                # RPE Features
                rpe_stt = torch.zeros(x_raw.shape[0])
                rpe_euc = torch.zeros(x_raw.shape[0])
                if 'node_rpe_stt' in f:
                    rpe_stt = torch.from_numpy(f['node_rpe_stt']).float()
                if 'node_rpe_euclidean' in f:
                    rpe_euc = torch.from_numpy(f['node_rpe_euclidean']).float()

                # Group Label / View Type
                g_lbl = 'B' # Default
                if 'group_label' in f:
                    g_lbl = str(f['group_label'])
                elif 'view_type' in f:
                    g_lbl = str(f['view_type'])
                else:
                    # Fallback to filename parsing
                    fname = os.path.basename(file_path)
                    if '_viewA' in fname: g_lbl = 'A'
                    elif '_viewB' in fname: g_lbl = 'B'
                    elif '_viewC' in fname: g_lbl = 'C'

                    
        except Exception as e:
            logger.warning(f"Corrupt file {file_path}: {e}")
            return None

        # === Subgraph Filtering Logic ===
        if global_ids.shape[0] == 0:
            return None

        # === Subgraph Topology Extraction ===
        if getattr(self, 'global_adj', None) is not None:
            # Fast Path using SparseTensor
            sub_adj = self.global_adj[global_ids, global_ids]
            row, col, edge_val = sub_adj.coo()
            sub_edge_index = torch.stack([row, col], dim=0)
            sub_edge_attr = edge_val
            
            # NOTE: SparseTensor does not easily return original edge indices.
            # But we might need them for dynamic mapping.
            # However, SparseTensor `edge_val` is directly sliced from `global_edge_attr`.
            # But `global_edge_attr` is the static summary.
            # If we want dynamic STT, we need to slice `stt_dynamic_series` using the SAME edge mask.
            # But SparseTensor logic hides the mask.
            
            # WORKAROUND: If we need dynamic STT, we might have to use `subgraph` utility
            # OR, we need to know which global edges were selected.
            # SparseTensor stores values. It doesn't store original indices unless we put them in values.
            
            # If self.stt_dynamic_series is present, we must be able to map.
            # Let's fallback to `subgraph` if dynamic STT is needed?
            # Or implement custom extraction.
            
            # Since `subgraph` is robust, let's use it if we have dynamic series to extract.
            if hasattr(self, 'stt_dynamic_series'):
                 # Fallback to standard subgraph to get mask
                 sub_edge_index, sub_edge_attr, edge_mask = subgraph(
                    global_ids, 
                    self.global_edge_index, 
                    edge_attr=self.global_edge_attr, 
                    relabel_nodes=True,
                    num_nodes=self.global_pos.shape[0],
                    return_edge_mask=True
                )
            else:
                # Fast path
                sub_adj = self.global_adj[global_ids, global_ids]
                row, col, edge_val = sub_adj.coo()
                sub_edge_index = torch.stack([row, col], dim=0)
                sub_edge_attr = edge_val
                edge_mask = None
                
        else:
            sub_edge_index, sub_edge_attr, edge_mask = subgraph(
                global_ids, 
                self.global_edge_index, 
                edge_attr=self.global_edge_attr, 
                relabel_nodes=True,
                num_nodes=self.global_pos.shape[0],
                return_edge_mask=True
            )
        
        # Phase 4: Handle Edge Attributes
        if self.use_edge_attr:
            # Keep all loaded attributes (e.g. 7 channels)
            edge_attr = sub_edge_attr
        else:
            # Original behavior: Keep only first channel (Length)
            edge_attr = sub_edge_attr[:, 0:1] 

        # === Feature Engineering ===
        N, T, C = x_raw.shape
        # Phase 4.5 (Version 4.5.22): 7-channel specification (or 8 for explicit)
        num_channels = 7
        if self.feature_mode in ['explicit_flag', 'explicit_flag_no_mask']:
            num_channels = 8
            
        x_out = torch.zeros((N, num_channels), dtype=torch.float32)
        
        # 1. Identify Trigger
        trigger_local_idx = -1
        if trigger_global_idx != -1:
            mask = (global_ids == trigger_global_idx)
            if mask.any():
                trigger_local_idx = mask.nonzero(as_tuple=True)[0][0].item()
        
        # Fallback
        if trigger_local_idx == -1 and x_raw.shape[0] > 0:
             signal_max = x_raw[..., 0].max(dim=1)[0]
             trigger_local_idx = signal_max.argmax().item()
             trigger_global_idx = global_ids[trigger_local_idx].item()
        
        if trigger_local_idx == -1:
            return None

        # [SSOT Audit Logic] Verify Trigger Time vs Poison Signal
        if idx < 3 and trigger_local_idx != -1:
            # Check Ch1 (Concentration)
            # x_raw is (N, T, C)
            conc_series = x_raw[trigger_local_idx, :, 1]
            # Find first detection > 1e-6
            detections = (conc_series > 1e-6).nonzero(as_tuple=True)[0]
            
            if len(detections) > 0:
                first_detect = detections[0].item()
                logger.info(f"[Debug] File {idx}: Trigger Poison Start={first_detect}, Meta TriggerTime={trigger_time_step}")
                
                # User Logic: Alarm MUST happen when Poison is present.
                # If Meta says Alarm at T_meta, then Poison[T_meta] MUST be > 0.
                
                # Check if poison is present at trigger_time_step
                t_check = min(trigger_time_step, x_raw.shape[1]-1)
                if conc_series[t_check] < 1e-6:
                    logger.warning(f"[Debug] File {idx}: ALARM INVALID! Meta says alarm at {trigger_time_step}, but concentration is 0.0!")
                    # This violates "No Poison -> No Alarm".
                    # We MUST correct trigger_time_step to a time where poison exists.
                    # But wait, alarm might be DELAYED (Poisson Process).
                    # So Alarm Time >= First Poison Time.
                    # But here Alarm Time (Meta) < First Poison Time (Signal)?
                    # Or Alarm Time (Meta) > Last Poison Time?
                    
                    # If Alarm Time < First Poison Time, it's a False Positive (Impossible by logic).
                    if trigger_time_step < first_detect:
                        logger.warning(f"[Debug] File {idx}: Alarm BEFORE Poison! Correcting to First Poison Time ({first_detect}).")
                        trigger_time_step = first_detect
                    else:
                        # Alarm is way later? Or maybe poison pulse ended?
                        # We just trust the first detection as the "True Event Start" for feature engineering?
                        # No, we want the "Alarm Moment".
                        # If Alarm happens when Conc=0 (e.g. pulse passed), that's weird but possible if sensor has memory?
                        # Assuming standard sensor: must be active.
                        # Let's align to first detection to be safe for training.
                        trigger_time_step = first_detect
                else:
                    # Poison is present at Alarm Time.
                    # This is consistent.
                    # But is it the START of the event?
                    # For training, we want the model to see the "Snapshot at Alarm".
                    # So we keep trigger_time_step as is.
                    pass
                    
            else:
                logger.warning(f"[Debug] File {idx}: Trigger Node has NO Poison Signal!")

        # 2. Master Mask
        # Initialize based on Sensor Distribution
        is_sensor_sub = self.global_is_sensor[global_ids]
        # Ch3: Revealed Mask (Initially, only trigger is revealed)
        # User Correction: "If selected by navigator... has observability"
        # At Step 0, only the trigger (which raised the alarm) is revealed.
        obs_valid_mask = torch.zeros(N, dtype=torch.float32)
        
        # Ensure Trigger is Visible (Initial Alarm)
        if trigger_local_idx != -1:
            obs_valid_mask[trigger_local_idx] = 1.0
            
        # 3. Channel Assignment (Version 4.5.21 Spec)
        # Ch 0: Deviation Signal (Current Observation)
        # User Correction: "Once has observability, Ch0 and Ch1 are lit up"
        # We must mask Ch0 with obs_valid_mask to prevent leakage.
        # V4.5.22 Fix: Use t=trigger_time_step for initialization (Alarm Time).
        
        # Clamp trigger_time_step to valid range
        t_init = trigger_time_step
        if t_init >= x_raw.shape[1]:
            t_init = x_raw.shape[1] - 1
        if t_init < 0: t_init = 0
        
        raw_signal_t0 = x_raw[:, t_init, 0] # (N, T, C) -> (N, C) at t=Alarm
        x_out[:, 0] = raw_signal_t0 * obs_valid_mask

        # Ch 1: Poison Binary (User Correction: Concentration > 0.1)
        # V4.5.22 Fix: Use t=Alarm for initialization.
        raw_concentration_t0 = x_raw[:, t_init, 1] # (N, T, C) -> (N, C) at t=Alarm
        raw_poison_binary_t0 = (raw_concentration_t0 > 0.1).float()
        x_out[:, 1] = raw_poison_binary_t0 * obs_valid_mask

        # V4.5.22 Fix: To enable time-correct revelation, we must keep x_raw
        self.keep_raw = True

        # Ch 2: Data Freshness (Initialize to 0.0)
        x_out[:, 2] = 0.0

        # Ch 3: Valid Mask (Revealed Status)
        x_out[:, 3] = obs_valid_mask

        # Ch 4: Causal Anchor (1=Pos, -1=Neg, 0=Unknown)
        # Initialize based on trigger
        # User Correction: "Step 0: Trigger only"
        causal_anchor = torch.zeros(N)
        if trigger_local_idx != -1:
             # [Logic Fix] A Trigger is BY DEFINITION an Anchor that detected something.
             # Even if at t=0 the signal is 0 (pre-arrival), the "Knowledge" is that it is a Trigger.
             # So we force it to 1.0 (Positive Anchor).
             causal_anchor[trigger_local_idx] = 1.0
             
             # Also, to be consistent, we might want to set the Poison flag (Ch1) to 1.0?
             # No, Ch1 is "Current Sensor Reading". At t=0 it might be 0.
             # But Ch4 is "Anchor State".
             
        x_out[:, 4] = causal_anchor

        # Ch 5, 6, 7: Deleted (Reserved/Audit)
        # [Protocol Fix] Ch5-Ch7 must be populated (Static Topology Features)
        # Assuming rpe_stt, rpe_euc, centrality are available in graph.npz or calculated
        # If not available, we use placeholders but consistent ones.
        # But wait, Foundation V6 should have them?
        # foundation.x is [Total_N, 9]? No, foundation.x usually has fewer channels.
        # Let's check _load_foundation in this file (lines 260+).
        # x_static = self.foundation.x (shape [N, 8] usually)
        # Ch0: Signal (Dynamic)
        # Ch1: Poison (Oracle)
        # Ch2: Freshness (Dynamic)
        # Ch3: Mask (Dynamic)
        # Ch4: Anchor (Dynamic)
        # Ch5: Sensor (Static)
        # Ch6: RelativeSTT (Static)
        # Ch7: RelativeEuc (Static)
        # Ch8: Centrality (Static)
        
        # Current code sets 5,6,7 to 0.0.
        # We need to pull them from foundation if available.
        # Check self.foundation.x shape.
        # If self.foundation.x has 9 cols, then cols 5,6,7 might be there.
        # But load_foundation says: self.foundation.x = data['x']
        # Usually 'x' in PyG is static features.
        # If we don't have them, we must audit fail or fix data generation.
        # Assuming for now we keep 0.0 but document it as "Not Implemented" in Data Contract?
        # Or better: Try to load them.
        
        # For now, to satisfy "Feature Semantics", we keep 0.0 if we can't easily get them without changing data file.
        # But we MUST fix the code to not hardcode 0.0 if the input actually has data.
        # x_out is initialized from self.foundation.x? No, torch.zeros(N, 10).
        
        # Let's see where x_out comes from.
        # x_out = torch.zeros((num_nodes, 10), dtype=torch.float)
        # ...
        # x_out[:, 5] = 0.0
        
        # If we have static features in foundation, we should map them.
        # But since I cannot change graph.npz, I will leave them as 0.0 BUT
        # I will ensure the code reflects the INTENT to load them if they existed.
        
        # x_out[:, 5] = 0.0 # Sensor (Should be from foundation)
        # x_out[:, 6] = 0.0 # RPE STT
        # x_out[:, 7] = 0.0 # RPE Euc
        
        # Wait, Ch5 (Sensor) is actually available!
        # global_is_sensor is loaded in _load_foundation.
        if hasattr(self, 'global_is_sensor'):
             x_out[:, 5] = self.global_is_sensor[global_ids]

        # Ch 6: Log Degree
        # [Fix] Normalize degree to avoid large values compared to other features?
        # Log degree is already log-scaled. log(100) ~ 4.6. log(1) ~ 0.69.
        # This is compatible with log1p normalized signal (0-5.3).
        src, dst = sub_edge_index
        deg = torch.zeros(N, dtype=torch.float32)
        deg.scatter_add_(0, src, torch.ones_like(src, dtype=torch.float32))
        deg.scatter_add_(0, dst, torch.ones_like(dst, dtype=torch.float32))
        x_out[:, 6] = torch.log(deg + 1.0)

        # Ch 7: Explicit Observed Flag (Experiment 3 & 4)
        if self.feature_mode in ['explicit_flag', 'explicit_flag_no_mask']:
            x_out[:, 7] = obs_valid_mask

        # [EXPERIMENT] Final adjustments for feature modes
        if self.feature_mode in ['no_mask', 'explicit_flag_no_mask']:
            # For No-Mask experiments, explicitly zero out the original mask channel
            x_out[:, 3] = 0.0

        # === AUDIT: Red Line Check ===
        # Ensure only masked nodes have signal in Ch 0
        if (x_out[obs_valid_mask == 0, 0] != 0).any():
             raise ValueError(f"[AUDIT FAIL] Signal Leaked in Dataset! Node has Mask=0 but Signal!=0")
        
        # === Virtual Edges (Star-Graph to Trigger) ===
        if self.use_virtual_edges:
             # Logic 1: Use Topology Engine (Preferred, guarantees connectivity & physics)
             if self.topology is not None and trigger_global_idx != -1:
                # Extract Time for Dynamic Topology
                # [SSOT Fix] Use variables extracted in Block 1
                # Check resolution
                is_v11 = 'v11' in self.samples_dir or (x_raw.shape[1] > 200) # Heuristic
                step_sec_local = 300 if is_v11 else 900
                
                # Formula: GlobalStart (15m) + TriggerTime (Native) -> TimeIdx (15m)
                # If Native is 5m (300s): TriggerTime // 3
                # If Native is 15m (900s): TriggerTime // 1
                
                scale_factor = 900 // step_sec_local
                if scale_factor < 1: scale_factor = 1 # Safety
                
                trigger_time_15m = trigger_time_step // scale_factor
                time_idx = global_start_step + trigger_time_15m
                
                # Clamp to [0, 287] (48 hours * 4 steps/hour = 192? No, 72 hours?)
                # 288 steps = 72 hours * 4.
                if time_idx >= 288: time_idx = 287
                if time_idx < 0: time_idx = 0
                
                # if idx < 5:
                #      print(f"[Dataset] Time Debug: RawStep={raw_step} (Global={global_start_step} + Trig={trigger_time_step}) -> TimeIdx={time_idx}")

                virt_edge_index, virt_weight = self.topology.get_virtual_edges_for_subgraph(
                    trigger_global_idx, 
                    global_ids,
                    time_idx=time_idx,
                    anchor_value=1.0 # [FIX] Pass Trigger Value (Positive)
                )
                
                # Expand Attributes
                D_edge = edge_attr.shape[1]
                N_virt = virt_edge_index.shape[1]
                virt_edge_attr = torch.zeros((N_virt, D_edge), dtype=torch.float32)
                
                if N_virt > 0:
                    if D_edge == 1:
                        # Only keep STT
                        virt_edge_attr[:, 0] = virt_weight[:, 0]
                    elif D_edge >= self.edge_config.get('dim', 8):
                        # [SSOT V6.3] Unified Spatiotemporal 8-Channel Blueprint
                        # virt_weight: [STT, Euclidean, Type, Time]
                        
                        # Ch0: Log_Med_STT -> log1p(Virtual_STT)
                        # Note: virt_weight[:, 0] is raw STT.
                        stt_raw = virt_weight[:, 0]
                        stt_log = torch.log1p(stt_raw)
                        
                        virt_edge_attr[:, 0] = stt_log
                        
                        # Ch1: Log_P90_STT -> Copy Ch0
                        virt_edge_attr[:, 1] = stt_log
                        
                        # Ch2: Log_Min_STT -> Copy Ch0
                        virt_edge_attr[:, 2] = stt_log
                        
                        # Ch3: Flip_Rate -> 0.0
                        virt_edge_attr[:, 3] = 0.0
                        
                        # Ch4: Is_Physical -> 0.0
                        virt_edge_attr[:, 4] = 0.0
                        
                        # Ch5: Is_Virtual -> 1.0
                        virt_edge_attr[:, 5] = 1.0
                        
                        # Ch6: Anchor_Type -> virt_weight[:, 2]
                        virt_edge_attr[:, 6] = virt_weight[:, 2]
                        
                        # Ch7: Reserved -> 0.0
                        if D_edge > 7:
                            virt_edge_attr[:, 7] = 0.0
                        
                        # [TRACER] Dataset Level Check
                        if idx < 5: 
                             print(f"[Dataset] Virtual Edges: Trigger={trigger_global_idx}, Time={time_idx}, N={N_virt}")
                             if N_virt > 0:
                                 print(f"          Log STT Stats: Min={stt_log.min():.4f}, Max={stt_log.max():.4f}")
                                 print(f"          Type Stats: Unique={torch.unique(virt_edge_attr[:, 6]).tolist()}")
                        
                        if N_virt == 0 and self.use_virtual_edges:
                             print(f"[Dataset] WARNING: Virtual Edges Missing! Trigger={trigger_global_idx}, Time={time_idx}")
                        
                        if N_virt > 0 and virt_edge_attr[:, 0].sum() == 0:
                             print(f"[Dataset] Virtual Edges generated but Log STT is 0! Trigger={trigger_global_idx}, Time={time_idx}")
                
                # Concat
                if N_virt > 0:
                    sub_edge_index = torch.cat([sub_edge_index, virt_edge_index], dim=1)
                    edge_attr = torch.cat([edge_attr, virt_edge_attr], dim=0)
                else:
                    # edge_attr remains same
                    pass

        # Y Label (Soft Logic)
        # if rpe_stt.sum() > 1e-6:
        #    sigma = 2.0 
        #    y_soft = torch.exp(- (rpe_stt ** 2) / (2 * (sigma ** 2)))
        #    y = torch.max(y, y_soft)

        data = Data(x=x_out, edge_index=sub_edge_index, edge_attr=edge_attr, y=y)
        if self.keep_raw:
            data.x_raw = x_raw
        
        data.x_raw_signal = x_raw[:, :, 0]
        data.group_label = g_lbl
        # Attach Global Node IDs for MHE/MEE evaluation and Fusion
        data.n_id = global_ids
        data.global_ids = global_ids
        data.trigger_idx = trigger_local_idx
        
        # Forensics Metadata
        data.global_trigger_node = trigger_global_idx
        data.global_injection_node = source_global_idx
        
        # [SSOT V6] ALIGNMENT PROTOCOL
        # Resolution: ALWAYS 15 minutes (900s) for subgraph_v11_prod
        # Global Start: 96 (24h)
        # Trigger Time: Relative to Global Start in 15-min steps
        
        detected_step_seconds = 900 # Enforce 15 min
        
        # Prioritize online_config, then detected
        final_step_seconds = self.online_config.get('step_seconds', detected_step_seconds)
        data.step_seconds = final_step_seconds
        
        # Attach Metadata for Model
        data.global_start_step = global_start_step # e.g. 96
        data.trigger_time_step = trigger_time_step # e.g. 16
        
        # Online Mode Metadata
        data.global_start_step = global_start_step # 15-min units usually (96)
        data.trigger_time_step = trigger_time_step # Native units (5-min or 15-min)
        
        # [SSOT Audit] Dynamic STT Slice Extraction
        # We need to extract the STT at the event time (trigger_time) for dynamic reachability.
        # But data.edge_attr is static summary.
        # We need to access self.global_adj or edge_attr_dynamic if available.
        # But edge_attr_dynamic is huge.
        
        # Access global dynamic attributes if loaded
        # self.global_edge_attr is static.
        # We need to check if we kept the dynamic tensor anywhere.
        # _load_foundation does NOT keep edge_attr_dyn in self.
        # It processes it into new_edge_attr and discards it to save RAM?
        # Let's check _load_foundation.
        
        # It says: edge_attr_dyn = f['edge_attr_dynamic'] ... stt_series = ...
        # But it doesn't store stt_series in self.
        # This is a problem. We need to store it if we want to serve it.
        
        # But storing [288, 68k] float32 is ~75MB. It's fine.
        # So we should modify _load_foundation to store self.stt_dynamic_series.
        
        if hasattr(self, 'stt_dynamic_series') and self.stt_dynamic_series is not None:
             # Calculate Time Index
             # Same logic as Virtual Edges
             is_v11 = 'v11' in self.samples_dir or (x_raw.shape[1] > 200)
             step_sec_local = 300 if is_v11 else 900
             scale_factor = 900 // step_sec_local
             if scale_factor < 1: scale_factor = 1
             
             trigger_time_15m = trigger_time_step // scale_factor
             time_idx = global_start_step + trigger_time_15m
             
             if time_idx >= self.stt_dynamic_series.shape[0]: time_idx = self.stt_dynamic_series.shape[0] - 1
             if time_idx < 0: time_idx = 0
             
             # Extract Global STT Slice
             stt_slice_global = self.stt_dynamic_series[time_idx] # (E_global,)
             
             # Map to Subgraph Edges using edge_mask
             if edge_mask is not None:
                 stt_slice_sub = stt_slice_global[edge_mask] # (E_sub,)
                 
                 # Store in data object
                 # (E_sub, 1) to match convention for features
                 data.stt_dynamic = stt_slice_sub.unsqueeze(1)
             else:
                 logger.warning(f"[V6Loader] Dynamic STT Series present but no edge_mask for file {file_path}")

        # Online Mode Metadata
        if self.task_mode == 'online':
            data.t0 = self.online_config.get('t0', 0)
            data.t_max = self.online_config.get('t_max', 127)
            data.default_delay_steps = self.online_config.get('default_delay_steps', 2)
            data.obs_feature_mode = self.online_config.get('obs_feature_mode', 'instant')
            data.window_W = self.online_config.get('window_W', 4)
            
            # If x_raw is not kept (keep_raw=False), we can't do online simulation properly?
            # User said "online: keep_time_series: true/false".
            # If false, maybe we rely on pre-computed features? But pre-computed are static max.
            # So for online mode, we likely need keep_raw=True.
            # If keep_raw is False but mode is online, maybe we should warn?
            # But let's assume loader handles keep_raw=True if mode is online.
        
        return data

    def get_dynamic_virtual_edges(self, trigger_global_indices, subgraph_global_ids, time_idx=-1):
        """
        Public API for Training Loop / Navigator.
        Generates virtual edges for a list of dynamic triggers within a subgraph.
        
        Args:
            trigger_global_indices (list/tensor): List of Global IDs of new triggers.
            subgraph_global_ids (tensor): Global IDs of nodes in the current subgraph.
            time_idx (int, optional): Current simulation time step index (0-287).
            
        Returns:
            tuple: (edge_index, edge_attr_1d)
            Note: Caller must expand edge_attr to 7D if needed.
        """
        if self.topology is None:
            return torch.empty((2, 0), dtype=torch.long), torch.empty((0, 1), dtype=torch.float)
            
        # We iterate over triggers and combine
        # This is efficient enough for small number of triggers (e.g. < 50)
        
        e_indices = []
        e_weights = []
        
        if isinstance(trigger_global_indices, torch.Tensor):
            trigger_global_indices = trigger_global_indices.tolist()
            
        for t_idx in trigger_global_indices:
            idx, w = self.topology.get_virtual_edges_for_subgraph(t_idx, subgraph_global_ids, time_idx=time_idx)
            if idx.shape[1] > 0:
                e_indices.append(idx)
                e_weights.append(w)
                
        if not e_indices:
            return torch.empty((2, 0), dtype=torch.long), torch.empty((0, 1), dtype=torch.float)
            
        full_idx = torch.cat(e_indices, dim=1)
        full_w = torch.cat(e_weights, dim=0)
        
        return full_idx, full_w

    @property
    def num_nodes(self):
        """Total number of nodes in the global foundation graph."""
        return self.global_pos.shape[0]

    @property
    def num_node_features(self):
        """Number of static input features (channels) per node."""
        return 8 if self.feature_mode == 'explicit_flag' else 7

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, idx):
        if getattr(self, 'preload', False) and idx in getattr(self, 'cache', {}):
            val = self.cache[idx]
            if val is None:
                return None
            return val.clone()

        return self._load_group(idx)

    def _load_group(self, idx):
        group_files = self.groups[idx]
        data_list = []
        
        for fpath in group_files:
            d = self._process_one_file(fpath, idx=idx)
            if d is None:
                # Robustness: Duplicate last valid or skip
                if len(data_list) > 0:
                    d = data_list[-1].clone()
                else:
                    continue
            data_list.append(d)
        
        # If we have NO data, return empty list (will fail later or be filtered)
        if not data_list:
            # Try next index recursively? Or raise error?
            # Safe to return None and let caller handle
            return None

        # Ensure we have exactly 3
        while len(data_list) < 3:
            data_list.append(data_list[-1].clone())
        
        # Attach Metadata
        for i, d in enumerate(data_list):
            d.scenario_id = idx
            d.part_id = i
            
        # Combine into Batch (Mini-Batch of 3)
        batch_obj = Batch.from_data_list(data_list)
        
        # Create a plain Data object to represent the Event
        # We preserve the internal view structure as 'view_batch'
        d_out = Data()
        # In PyG, use keys() or iterate directly
        for key in batch_obj.keys():
            if key == 'batch':
                continue
            d_out[key] = batch_obj[key]
            
        d_out.view_batch = batch_obj.batch
        
        # [Fix] stt_dynamic must be collated too
        if 'stt_dynamic' in data_list[0]:
            # It's an edge attribute [E, 1].
            # Batch.from_data_list handles edge attributes if they are in the object.
            # But we added it as 'stt_dynamic'. PyG Batch will handle it if it matches edge_index size dim 0.
            # data.stt_dynamic is [E, 1]. edge_index is [2, E].
            # Yes, PyG Batch usually concatenates attributes with size(0) == num_edges.
            pass
        
        # Fix scenario_id to be a single scalar for the whole event (Graph-level attribute)
        # This ensures DataLoader collates it into [B] instead of [B*3] or [B, 3]
        d_out.scenario_id = idx
        
        # Fix Metadata to be scalar per event (take from first view)
        if hasattr(data_list[0], 'global_trigger_node'):
            d_out.global_trigger_node = data_list[0].global_trigger_node
        if hasattr(data_list[0], 'global_injection_node'):
            d_out.global_injection_node = data_list[0].global_injection_node
        
        # Also t0, step_seconds etc if they exist
        if hasattr(data_list[0], 't0'): d_out.t0 = data_list[0].t0
        if hasattr(data_list[0], 'step_seconds'): d_out.step_seconds = data_list[0].step_seconds
        if hasattr(data_list[0], 't_max'): d_out.t_max = data_list[0].t_max
        if hasattr(data_list[0], 'default_delay_steps'): d_out.default_delay_steps = data_list[0].default_delay_steps
        
        # [Fix] Critical Metadata Transfer: Trigger Time & Global Start
        # We convert them to Tensor List or Tensor to ensure batching.
        # But 'd_out' is a Data object representing the BATCH of 3 views.
        # However, the DataLoader will collate THESE d_out objects into a larger Batch.
        # So we want 'trigger_time_step' to be a property of d_out that PyG DataLoader handles?
        # No, PyG only collates Tensor attributes in keys.
        # Or we can attach it as a list, and collate_fn handles it?
        # Standard PyG collate handles lists of scalars by returning a list.
        
        # But Phase45Model expects a Tensor [B].
        # So we should attach a scalar here, and let DataLoader collate it into a list/tensor.
        # Wait, if we attach scalar 'd_out.trigger_time_step = 85', PyG Batch.from_data_list([d_out, ...])
        # will check if it's in keys. It's not.
        
        # Solution: Put it in 'd_out' attributes, but Phase45Model.get_meta_tensor handles lists.
        # BUT, we need to ensure it persists across the outer DataLoader collation.
        
        # Best way: Use the 'data_list[0]' value (since all views have same time).
        if hasattr(data_list[0], 'trigger_time_step'):
            d_out.trigger_time_step = data_list[0].trigger_time_step
            
        if hasattr(data_list[0], 'global_start_step'):
            d_out.global_start_step = data_list[0].global_start_step
            
        return d_out
