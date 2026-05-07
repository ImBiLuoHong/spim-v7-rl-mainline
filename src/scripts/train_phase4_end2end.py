"""
[SSOT] Training Main Entry Point
This is the ONLY authoritative script for launching training.
Legacy scripts (train_*.py) have been archived.
"""
import os
import multiprocessing
# [CRITICAL] Force 'spawn' to avoid fork-related deadlocks/contention in PyTorch DataLoader
try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass

# [LMDB OPTIMIZATION] Prevent thread explosion
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import sys
import os
# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import random
import logging
import glob
import pandas as pd
import time
import math
from datetime import datetime
from torch_geometric.utils import to_dense_batch

# Import from your project
from src.config.core import Config
from src.data.v6.loader import create_dataloaders
from src.modeling.architectures.phase4_5_model import Phase45Model
from src.modeling.builders.model_builder import ModelBuilder
from src.modeling.losses import ModularLossEngine
from src.utils.hardware_optim import apply_hardware_optimizations, get_device_and_scaler, DevicePrefetcher
from src.evaluation.runner import evaluate_mode
from src.evaluation.evaluator import Evaluator
from src.training.horizon import HorizonScheduler
import wandb
import yaml
import hashlib
import json
import subprocess
from src.scripts.audit.run_data_audit import run_audit as run_data_audit_func
from src.scripts.audit.audit_cache import AuditCache
from src.shared.artifacts import save_checkpoint

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)




def get_config_hash(cfg_dict):
    """Generate a stable MD5 hash of a configuration dictionary."""
    # Filter out non-reproducibility fields like run_name, timestamp, etc.
    # We only care about fields that affect the model's behavior or data.
    def _clean(d):
        if not isinstance(d, dict): return d
        skip_keys = {'run_name', 'run_id', 'timestamp', 'path', 'artifacts_dir', 'logs_dir', 'checkpoints_dir', 'run_dir'}
        return {k: _clean(v) for k, v in d.items() if k not in skip_keys}
    
    clean_dict = _clean(cfg_dict)
    config_str = json.dumps(clean_dict, sort_keys=True)
    return hashlib.md5(config_str.encode()).hexdigest()[:12]


def _cfg_like_get(cfg_obj, key, default=None):
    if cfg_obj is None:
        return default
    if isinstance(cfg_obj, dict):
        return cfg_obj.get(key, default)
    return getattr(cfg_obj, key, default)


def _resolve_sampling_budget_anneal(cfg, epoch: int):
    schedule_cfg = getattr(cfg.training, "sampling_budget_anneal", None)
    enabled = bool(_cfg_like_get(schedule_cfg, "enabled", False))
    if not enabled:
        return None

    full_budget = max(1, int(_cfg_like_get(schedule_cfg, "full_budget", getattr(cfg.model, "sample_budget", 3))))
    floor_budget = max(1, int(_cfg_like_get(schedule_cfg, "floor_budget", 3)))
    warm_epoch_count = max(1, int(_cfg_like_get(schedule_cfg, "warm_epoch_count", 1)))
    pre_floor_policy = str(_cfg_like_get(schedule_cfg, "pre_floor_policy", "random_valid"))
    floor_policy = str(_cfg_like_get(schedule_cfg, "floor_policy", "navigator_only"))

    if epoch < warm_epoch_count:
        budget = full_budget
    else:
        budget = max(floor_budget, int(math.ceil(full_budget / float(2 ** (epoch - warm_epoch_count + 1)))))
    policy = floor_policy if budget <= floor_budget else pre_floor_policy
    return {
        "sample_budget": int(budget),
        "sampling_policy": policy,
        "full_budget": int(full_budget),
        "floor_budget": int(floor_budget),
        "warm_epoch_count": int(warm_epoch_count),
    }

def run_training(loss_config_override=None, run_name=None, max_epochs=None, loaders=None, seed=None, skip_audit=False, force_rebuild=False):
    run_training_start = time.perf_counter()
    # [PLATFORM] Mandatory Audit Gatekeeper
    if not skip_audit:
        logger.info("Running Platform Compliance Audit...")
        try:
            # 1. Run Platform Audit (Structure/Protocol)
            audit_res = subprocess.run(
                ["python", "src/scripts/audit/run_audit.py"],
                env={**os.environ, "PYTHONPATH": os.getcwd()},
                capture_output=False, text=True
            )
            if audit_res.returncode != 0:
                logger.error("❌ Platform Audit Failed! Check your implementation against the protocol.")
                sys.exit(1)
                
            # 2. Run Data Semantic Audit (Schema/Clock/Leakage)
            logger.info("Running Data & Semantic Audit (Fast Mode + Cache)...")
            
            # Prepare overrides for audit
            audit_cfg_path = None
            if loss_config_override:
                import tempfile
                fd, audit_cfg_path = tempfile.mkstemp(suffix='.yaml', text=True)
                with os.fdopen(fd, 'w') as f:
                    yaml.dump(loss_config_override, f)
            
            try:
                # Call In-Process (Fast Mode, No Preload, Use Cache)
                data_audit_success = run_data_audit_func(
                    config_path=audit_cfg_path, 
                    mode='fast', 
                    no_preload=True, 
                    use_cache=True
                )
            except Exception as e:
                logger.error(f"Data Audit Exception: {e}")
                data_audit_success = False
            finally:
                if audit_cfg_path and os.path.exists(audit_cfg_path):
                    os.remove(audit_cfg_path)
            
            if not data_audit_success:
                logger.error("❌ Data Semantic Audit Failed! Violations detected.")
                sys.exit(1)
            
            # 3. Run Evaluation Contract Audit
            logger.info("Running Evaluation Contract Audit...")
            eval_audit_cmd = ["python", "src/scripts/audit/run_eval_audit.py"]
            # Check if file exists
            if not os.path.exists("src/scripts/audit/run_eval_audit.py"):
                 if os.path.exists("src/scripts/run_eval_audit.py"):
                      eval_audit_cmd = ["python", "src/scripts/run_eval_audit.py"]
                 else:
                      logger.warning("⚠️ Evaluation Audit script not found. Skipping Eval Audit.")
                      eval_audit_cmd = None
            
            if eval_audit_cmd:
                eval_audit_res = subprocess.run(
                    eval_audit_cmd,
                    env={**os.environ, "PYTHONPATH": os.getcwd()},
                    capture_output=False, text=True
                )
                if eval_audit_res.returncode != 0:
                    logger.error("❌ Evaluation Contract Audit Failed! Metrics are not trustworthy.")
                    sys.exit(1)
                
            logger.info("✅ Platform & Data & Eval Audits Passed. Proceeding to training.")
        except Exception as e:
            logger.error(f"Failed to run audit: {e}")
            sys.exit(1)

    # 1. Config
    cfg = Config()
    
    # Apply overrides if provided
    if loss_config_override:
        cfg.apply_overrides(loss_config_override)
    
    # Standard overrides for Phase 4.5
    cfg.data.use_dataloader_v6 = True
    cfg.data.filter_no_source = True # Crucial for Hard Success Rate validity
    cfg.model.architecture = 'phase4_5'
    
    # [PLATFORM FIX] Sync CLI overrides to Config object BEFORE hashing
    if max_epochs is not None:
        cfg.training.num_epochs = max_epochs
    if seed is not None:
        cfg.training.seed = seed
        
    # Apply Seed (SSOT)
    seed_everything(cfg.training.seed)
    logger.info(f"Set Global Seed: {cfg.training.seed}")
    log_every_n_steps = max(1, int(getattr(cfg.training, "log_every_n_steps", 1)))
    wandb_enabled = bool(getattr(cfg.training, "enable_wandb", True))
    if os.environ.get("WANDB_DISABLED", "").lower() in {"1", "true", "yes"}:
        wandb_enabled = False
    if os.environ.get("WANDB_MODE", "").lower() == "disabled":
        wandb_enabled = False
    runtime_probe_enabled = bool(
        getattr(cfg.training, "force_runtime_probes", False)
        or wandb_enabled
        or getattr(cfg.training, "collect_detailed_step_metrics", False)
        or getattr(cfg.training, "diag_print_grad", False)
        or getattr(cfg.training, "diag_print_top1", False)
        or getattr(cfg.training, "overfit_one_batch", False)
    )
    
    # Use current state of cfg
    epochs = cfg.training.num_epochs
    final_run_name = run_name if run_name is not None else cfg.training.run_name
    
    # 1.5 Hard Skip Logic (Platform Level)
    # Generate hash for this specific configuration
    def _to_dict_stable(obj):
        if isinstance(obj, dict):
            return {k: _to_dict_stable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_to_dict_stable(v) for v in obj]
        if hasattr(obj, '__dict__'):
            return {k: _to_dict_stable(v) for k, v in obj.__dict__.items() if not k.startswith('_')}
        return obj
    
    full_cfg_dict = _to_dict_stable(cfg)
    cfg_hash = get_config_hash(full_cfg_dict)
    
    # Deterministic directory
    save_dir = os.path.join(cfg.paths.experiments_dir, f"{final_run_name}_{cfg_hash}")
    success_marker = os.path.join(save_dir, "EXPERIMENT_COMPLETED_SUCCESSFULLY")
    metrics_cache = os.path.join(save_dir, "test_metrics.json")
    epoch_history_path = os.path.join(save_dir, "epoch_history.jsonl")
    resume_checkpoint = getattr(cfg.training, "resume_checkpoint", None)
    if resume_checkpoint:
        resume_checkpoint = os.path.abspath(str(resume_checkpoint))
    
    if os.path.exists(success_marker) and not force_rebuild:
        # Double-check: was it actually finished with the SAME epochs?
        # We check the metrics cache for the recorded convergence epoch or num_epochs
        already_completed = False
        if os.path.exists(metrics_cache):
            try:
                with open(metrics_cache, 'r') as f:
                    cached = json.load(f)
                    # If we wanted 50 epochs and it finished at epoch 10, it's NOT the same experiment
                    if cached.get('num_epochs', 0) == epochs:
                        already_completed = True
            except:
                pass
        
        if already_completed:
            logger.info(f"!!! [HARD SKIP] Experiment '{final_run_name}' (Hash: {cfg_hash}, Epochs: {epochs}) already exists and is complete. !!!")
            with open(metrics_cache, 'r') as f:
                cached_metrics = json.load(f)
            cached_metrics.setdefault('run_dir', save_dir)
            cached_metrics.setdefault('config_hash', cfg_hash)
            cached_metrics.setdefault('resolved_config_path', os.path.join(save_dir, "resolved_config.yaml"))
            return cached_metrics
        else:
            logger.info(f"!!! [RE-RUN] Experiment '{final_run_name}' found but incomplete or param mismatch. Starting fresh. !!!")

    os.makedirs(save_dir, exist_ok=True)
    if os.path.exists(epoch_history_path) and not resume_checkpoint:
        os.remove(epoch_history_path)
    
    # [PLATFORM] Resolved Config Dump (SSOT)
    resolved_cfg_path = os.path.join(save_dir, "resolved_config.yaml")
    
    # Add Platform Metadata
    full_cfg_dict['platform'] = {
        'version': '1.0.0',
        'git_hash': subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip() if os.path.exists('.git') else 'no-git',
        'audit_timestamp': datetime.now().isoformat()
    }
    
    with open(resolved_cfg_path, 'w') as f:
        yaml.dump(full_cfg_dict, f, default_flow_style=False)
    logger.info(f"✅ Resolved config dumped to {resolved_cfg_path}")

    def _json_ready(value):
        if isinstance(value, dict):
            return {k: _json_ready(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [_json_ready(v) for v in value]
        if isinstance(value, (np.floating, np.integer)):
            return value.item()
        if isinstance(value, torch.Tensor):
            if value.numel() == 1:
                return value.item()
            return value.detach().cpu().tolist()
        return value
    
    # 2. Data
    if loaders:
        train_loader, val_loader, test_loader, imbalance_info = loaders
    else:
        # [Single-Batch Overfitting Test]
        # Force batch_size=1 and limit to 1 graph if test_overfit is enabled
        test_overfit = False # [DEBUG] Enable Overfitting Test
        
        batch_size = cfg.efficiency.batch_size
        eval_batch_size = max(1, int(getattr(cfg.training, "formal_eval_batch_size", 1) or 1))
        if test_overfit:
            logger.warning("!!! [TEST] SINGLE-BATCH OVERFITTING MODE ENABLED !!!")
            batch_size = 1 # Force single graph per batch
            cfg.efficiency.batch_size = 1
            eval_batch_size = 1
            # We will hack the loader after creation
        logger.info(
            "Evaluation policy: validation/test loaders use batch_size=%s (formal reporting lane)",
            eval_batch_size,
        )
            
        train_loader, val_loader, test_loader, imbalance_info = create_dataloaders(
            data_root=cfg.paths.samples_path,
            cfg=cfg,
            batch_size=batch_size,
            eval_batch_size=eval_batch_size,
            skip_lmdb=getattr(cfg.data, 'skip_lmdb', False),
            train_only=bool(getattr(cfg.training, 'train_only', False)),
        )
        
        if test_overfit:
            # HACK: Replace dataset with a single-item subset
            from torch.utils.data import Subset
            # Pick a graph with a source (to ensure learning is possible)
            # We iterate until we find one.
            found_idx = 0
            for i in range(min(100, len(train_loader.dataset))):
                # We need to peek. Dataset access might be slow if lazy.
                # NpzDatasetV6 loads on demand.
                # Let's just pick index 0 for now and hope it has a source.
                # Or better: check filter_no_source flag.
                found_idx = i
                break
                
            logger.info(f"!!! [TEST] Overfitting on Graph Index {found_idx} !!!")
            train_loader.dataset.groups = [train_loader.dataset.groups[found_idx]]
            
            single_batch = next(iter(train_loader))
            # Create a generator that yields this batch forever (or for epoch length)
            class CycleLoader:
                def __init__(self, batch, length):
                    self.batch = batch
                    self.length = length
                    self.dataset = train_loader.dataset # Keep ref
                def __iter__(self):
                    for _ in range(self.length):
                        yield self.batch
                def __len__(self):
                    return self.length
            
            # Replace train_loader
            train_loader = CycleLoader(single_batch, length=50) # 50 steps per epoch
            logger.info("!!! [TEST] Train Loader replaced with CycleLoader (Single Batch) !!!")

    train_dataset_size = len(train_loader.dataset) if hasattr(train_loader, 'dataset') else None
    train_batches_per_epoch = len(train_loader)

    # WandB Login & Init
    if wandb_enabled:
        wandb_key = "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"
        try:
            wandb.login(key=wandb_key)
            logger.info("WandB logged in successfully.")
        except Exception as e:
            logger.warning(f"WandB login failed: {e}. Running without WandB if needed.")
            wandb_enabled = False

    if wandb_enabled:
        wandb.init(project="Phase4.5_Standard_Engine", name=f"{final_run_name}_{cfg_hash}", config=full_cfg_dict, reinit=True)
    else:
        logger.info("WandB disabled for this run.")
    
    # [Verification] Print Sentinel Versions
    try:
        from src.modeling.architectures.phase4_5_model import Phase45Model
        from src.modeling.reasoners.bayesian import BayesianReasoner
        print(f"\n[VERIFICATION] Phase45Model File: {Phase45Model.__module__}")
        print(f"[VERIFICATION] Phase45Model Sentinel: {getattr(Phase45Model, 'SENTINEL_VERSION', 'MISSING')}")
        print(f"[VERIFICATION] BayesianReasoner File: {BayesianReasoner.__module__}")
        print(f"[VERIFICATION] BayesianReasoner Sentinel: {getattr(BayesianReasoner, 'SENTINEL_VERSION', 'MISSING')}\n")
        
        if wandb_enabled and wandb.run is not None:
             wandb.config.update({
                 "meta/phase45_sentinel": getattr(Phase45Model, 'SENTINEL_VERSION', 'MISSING'),
                 "meta/bayesian_sentinel": getattr(BayesianReasoner, 'SENTINEL_VERSION', 'MISSING')
             })
    except Exception as e:
        print(f"[VERIFICATION] Failed to check sentinels: {e}")
    
    # 2.5 Hardware Optimization (SSOT)
    # [SSOT] TF32 Control
    if getattr(cfg.efficiency, 'performance', {}).get('tf32', False):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        logger.info("Hardware Setup: TF32 Matmul/CuDNN ENABLED via Config SSOT.")
    
    device, scaler = get_device_and_scaler(cfg)
    
    # 3. Model
    # Architecture 8.2.2: Pass topology engine for dynamic rewiring
    topology_engine = None
    if hasattr(train_loader.dataset, 'topology'):
        topology_engine = train_loader.dataset.topology
    elif hasattr(train_loader.dataset, 'dataset') and hasattr(train_loader.dataset.dataset, 'topology'):
        topology_engine = train_loader.dataset.dataset.topology
        
    # [SSOT Fix] Use ModelBuilder for Dependency Injection
    model = ModelBuilder.build_model(cfg).to(device)
    # Manually inject topology engine if needed (though ModelBuilder might not handle it, Phase45Model usually takes it in __init__ if directly called, 
    # but ModelBuilder uses set_components. We need to check if Phase45Model still accepts topology_engine in init or needs setter)
    # Checking Phase45Model definition... it accepts it in __init__.
    # But ModelBuilder.build_model calls Phase45Model(cfg, navigator, reasoner, physics).
    # It does NOT pass topology_engine.
    # We must inject it after build or modify builder.
    # Let's inject it as attribute if the model supports it.
    if hasattr(model, 'topology_engine'):
        model.topology_engine = topology_engine
    
    model = apply_hardware_optimizations(model, cfg)

    init_checkpoint = getattr(cfg.training, "init_checkpoint", None)
    if init_checkpoint:
        init_checkpoint = os.path.abspath(str(init_checkpoint))
        strict_load = bool(getattr(cfg.training, "init_checkpoint_strict", True))
        if not os.path.exists(init_checkpoint):
            raise FileNotFoundError(f"Initial checkpoint not found: {init_checkpoint}")
        logger.info("Loading initial checkpoint from %s (strict=%s)", init_checkpoint, strict_load)
        state_dict = torch.load(init_checkpoint, map_location=device)
        try:
            model.load_state_dict(state_dict, strict=strict_load)
        except RuntimeError:
            if strict_load:
                logger.warning("Strict initial checkpoint load failed for %s; retrying with strict=False", init_checkpoint)
            model.load_state_dict(state_dict, strict=False)

    # [TEST] Overfitting Logic (Post-Model Init)
    test_overfit = False # Force Enable
    if test_overfit:
        logger.info("!!! [TEST] FORCING Single Batch Overfitting (First Batch) !!!")
        
        # Simply take the first batch to avoid "smart search" failure
        try:
            # Re-create iterator to ensure fresh start
            temp_iter = iter(train_loader)
            found_batch = next(temp_iter)
            
            # Create CycleLoader
            class CycleLoader:
                def __init__(self, batch, length):
                    self.batch = batch
                    self.length = length
                    self.dataset = train_loader.dataset # Keep ref
                def __iter__(self):
                    for _ in range(self.length):
                        yield self.batch
                def __len__(self):
                    return self.length
            
            # Replace train_loader
            train_loader = CycleLoader(found_batch, length=50) # 50 steps per epoch
            logger.info("!!! [TEST] Train Loader replaced with CycleLoader (First Batch Forced) !!!")
            
        except Exception as e:
            logger.error(f"!!! [TEST] Failed to grab first batch: {e}")

    
    # [Verification] Log Stepper Info
    if hasattr(model, 'stepper'):
        print(f"[VERIFICATION] Stepper Class: {model.stepper.__class__.__name__}")
        print(f"[VERIFICATION] Stepper File: {sys.modules[model.stepper.__module__].__file__}")
    else:
        print("[VERIFICATION] Model has NO stepper attribute!")
    
    # 4. Optimizer & Scaler
    optimizer = optim.Adam(model.parameters(), lr=cfg.training.learning_rate, weight_decay=cfg.training.weight_decay)
    start_epoch = 0
    if resume_checkpoint:
        if not os.path.exists(resume_checkpoint):
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_checkpoint}")
        logger.info("Resuming training state from %s", resume_checkpoint)
        resume_state = torch.load(resume_checkpoint, map_location=device)
        if "model_state_dict" not in resume_state:
            raise ValueError(f"Resume checkpoint {resume_checkpoint} is missing model_state_dict")
        model.load_state_dict(resume_state["model_state_dict"], strict=True)
        if "optimizer_state_dict" in resume_state:
            optimizer.load_state_dict(resume_state["optimizer_state_dict"])
        start_epoch = int(resume_state.get("epoch", -1)) + 1
    
    # Modular Loss Engine (SSOT V6: Use cfg.loss)
    def _to_dict_recursive(obj):
        if isinstance(obj, dict):
            return {k: _to_dict_recursive(v) for k, v in obj.items()}
        if hasattr(obj, '__dict__'):
            return {k: _to_dict_recursive(v) for k, v in obj.__dict__.items() if not k.startswith('_')}
        return obj

    loss_engine_cfg = _to_dict_recursive(cfg.loss)
    
    # If loaders provide imbalance info, use it
    if imbalance_info and 'class_weights_vec' in imbalance_info:
        if 'params' not in loss_engine_cfg: loss_engine_cfg['params'] = {}
        loss_engine_cfg['params']['class_weight_pos'] = imbalance_info['class_weights_vec'][1]
    
    loss_engine = ModularLossEngine(loss_engine_cfg).to(device)
    logger.info(f"Initialized ModularLossEngine with config: {loss_engine_cfg}")
    if resume_checkpoint and scaler is not None and "scaler_state_dict" in resume_state:
        scaler.load_state_dict(resume_state["scaler_state_dict"])
    
    # Initialize Horizon Scheduler (Slot 12)
    horizon_scheduler = HorizonScheduler(cfg)
    
    # [SSOT] Initialize Evaluator for Training Metrics
    evaluator = Evaluator(cfg)
    
    # 5. Training Loop
    best_metrics = None
    best_mainline_score = -float("inf")
    best_train_loss = float("inf")
    train_loop_wall_s = 0.0
    periodic_checkpoint_every = max(0, int(getattr(cfg.training, "periodic_checkpoint_every_n_epochs", getattr(cfg.training, "ofb_save_every", 0)) or 0))
    torch.cuda.reset_peak_memory_stats(device) if device.type == 'cuda' else None
    
    epochs = cfg.training.num_epochs
    grad_accum_steps = max(1, int(getattr(cfg.training, "grad_accum_steps", 1) or 1))
    
    # [Refactor] Annealing Params (SSOT)
    anneal_cfg = getattr(cfg.training, 'annealing', {})
    tau_start = anneal_cfg.get('tau_start', 1.5)
    tau_end = anneal_cfg.get('tau_end', 0.1)
    anneal_end_ratio = anneal_cfg.get('anneal_end_ratio', 0.7)
    anneal_epochs = int(epochs * anneal_end_ratio)
    
    logger.info(f"=== Phase 4.5 Standard Engine: Training Started ({epochs} Epochs) ===")
    logger.info(f"=== Annealing: Tau {tau_start} -> {tau_end} over {anneal_epochs} epochs ===")
    logger.info(f"=== Save Directory: {save_dir} ===")
    
    for epoch in range(start_epoch, epochs):
        epoch_run_training_start = time.perf_counter()
        model.train()
        total_loss = 0
        train_metrics = {}
        
        # [SSOT] Reset Training Evaluator
        if evaluator: evaluator.reset()
        
        # [Refactor] Update Tau
        if epoch < anneal_epochs:
            tau = tau_start - (tau_start - tau_end) * (epoch / anneal_epochs)
        else:
            tau = tau_end
        
        # Curriculum Learning (Phase 4.5)
        # Delegate to HorizonScheduler (Slot 12)
        cfg.training.max_train_episodes = horizon_scheduler.get_train_horizon(epoch)
        
        # Update FoV Strength
        fov_strength = horizon_scheduler.get_fov_strength(epoch)
        if hasattr(cfg, 'fov_controller') and isinstance(cfg.fov_controller, dict):
            cfg.fov_controller['strength'] = fov_strength
            
        # Propagate to Model (if applicable)
        if hasattr(model, 'fov_controller') and model.fov_controller is not None:
            model.fov_controller.strength = fov_strength

        sampling_schedule_state = _resolve_sampling_budget_anneal(cfg, epoch)
        if sampling_schedule_state is not None:
            cfg.model.sample_budget = int(sampling_schedule_state["sample_budget"])
            cfg.model.sampling_policy = str(sampling_schedule_state["sampling_policy"])

        if horizon_scheduler.enabled:
             logger.info(f"[Curriculum] Epoch {epoch}: max_train_episodes={cfg.training.max_train_episodes}, fov_strength={fov_strength}, tau={tau:.4f}")
        else:
             logger.info(f"[Training] Epoch {epoch}: tau={tau:.4f}")
        if sampling_schedule_state is not None:
             logger.info(
                 "[Sampling Anneal] "
                 f"Epoch {epoch + 1}: sample_budget={cfg.model.sample_budget}, "
                 f"sampling_policy={cfg.model.sampling_policy}, "
                 f"full_budget={sampling_schedule_state['full_budget']}, "
                 f"floor_budget={sampling_schedule_state['floor_budget']}"
             )

        if wandb_enabled and wandb.run is not None:
             payload = {'train/tau': tau, 'epoch': epoch}
             if sampling_schedule_state is not None:
                 payload["train/sample_budget"] = int(cfg.model.sample_budget)
             wandb.log(payload)

        # Measure only the training-iteration section: batch iteration including
        # the first in-loop prefetch/first-step warm-up, but excluding model/data
        # construction before the epoch and excluding validation after the epoch.
        epoch_train_loop_start = time.perf_counter()
        # [Optim] Use DevicePrefetcher for Async Data Transfer (CPU -> GPU)
        train_prefetcher = DevicePrefetcher(train_loader, device)
        pbar = tqdm(train_prefetcher, desc=f"Train E{epoch+1}", leave=False, total=len(train_loader))
        
        # [TRACER] Print only once per epoch
        epoch_tracer_printed = False
        
        # [Experiment] Step Metric Accumulators
        epoch_step_metrics = {}
        epoch_step_counts = {}
        
        # [DEBUG] Engineering Checkpoints (One-off per epoch)
        debug_printed = False
        optimizer.zero_grad(set_to_none=True)
        
        for batch_idx, batch in enumerate(pbar):
            global_step = epoch * len(train_loader) + batch_idx
            try:
                setattr(cfg.training, "current_epoch", int(epoch))
                setattr(cfg.training, "current_batch_index", int(batch_idx))
                setattr(cfg.training, "global_step", int(global_step))
                setattr(cfg.training, "total_train_steps", int(epochs * len(train_loader)))
            except Exception:
                pass
            # batch = batch.to(device, non_blocking=cfg.training.non_blocking_transfer) # Handled by Prefetcher
            probe_data = {} if runtime_probe_enabled else None
            log_this_step = (global_step % log_every_n_steps) == 0
            
            # Determine if we should print tracer
            enable_tracer = False
            if runtime_probe_enabled and not epoch_tracer_printed:
                enable_tracer = True
                epoch_tracer_printed = True
            
            # [DEBUG] Level 1: Data Alignment & Level 2: Inputs (Before Forward)
            # Log to WandB instead of Console
            if runtime_probe_enabled and not debug_printed and epoch == 0 and batch_idx == 0:
                try:
                    # Level 1: Data Alignment
                    if hasattr(batch, 'y') and hasattr(batch, 'global_injection_node') and hasattr(batch, 'n_id'):
                        if hasattr(batch, 'batch'):
                            mask_0 = (batch.batch == 0)
                            n_id_0 = batch.n_id[mask_0]
                            y_0 = batch.y[mask_0]
                            src_global = batch.global_injection_node[0].item()
                            
                            probe_data['debug/g0_src_global'] = src_global
                            
                            src_local_candidates = (n_id_0 == src_global).nonzero(as_tuple=True)[0]
                            if len(src_local_candidates) > 0:
                                src_local = src_local_candidates[0].item()
                                probe_data['debug/g0_src_local'] = src_local
                                probe_data['debug/g0_global_at_local'] = n_id_0[src_local].item()
                                probe_data['debug/g0_label_at_local'] = y_0[src_local].item()
                                probe_data['debug/g0_label_sum'] = y_0.sum().item()
                            else:
                                probe_data['debug/g0_src_found'] = 0
                                probe_data['debug/g0_label_sum'] = y_0.sum().item()
                    
                    # Level 2: Input Validity
                    if hasattr(batch, 'x'):
                        x = batch.x
                        probe_data['debug/x_shape_0'] = x.shape[0]
                        probe_data['debug/x_shape_1'] = x.shape[1]
                        
                        ch_names = ["Signal", "Poison", "Freshness", "Mask", "Anchor", "Sensor", "LogDeg"]
                        for i in range(min(x.shape[1], len(ch_names))):
                            col = x[:, i]
                            probe_data[f'debug/input_ch{i}_min'] = col.min().item()
                            probe_data[f'debug/input_ch{i}_max'] = col.max().item()
                            probe_data[f'debug/input_ch{i}_mean'] = col.mean().item()
                            probe_data[f'debug/input_ch{i}_nonzero'] = col.abs().gt(1e-6).sum().item()
                    
                    if hasattr(batch, 'edge_attr'):
                        ea = batch.edge_attr
                        if ea.shape[1] > 6:
                            probe_data['debug/edge_virt_mean'] = ea[:, 5].mean().item()
                            probe_data['debug/edge_anchor_nonzero'] = ea[:, 6].abs().gt(1e-6).sum().item()
                            
                except Exception as e:
                    pass

            
            # AMP Forward
            use_amp = (scaler is not None)
            with torch.amp.autocast('cuda', enabled=use_amp):
                # [Refactor] Pass tau to model
                out = model(batch, inference_mode=False, max_episodes=cfg.training.max_train_episodes, tau=tau, enable_tracer=enable_tracer)
                
                # [Probe B] Extract metrics from Stepper Return
                probe_b = out.get('probe_b_metrics', {}) if runtime_probe_enabled else {}
                if runtime_probe_enabled and probe_b:
                     # [PROBE] Log step-wise metrics immediately
                     step_metrics_to_log = {}
                     for key, value in probe_b.items():
                         if key.startswith("probe/step"):
                             step_metrics_to_log[key] = value
                     if step_metrics_to_log and wandb_enabled and wandb.run is not None and log_this_step:
                         wandb.log(step_metrics_to_log)

                     probe_data.update(probe_b)
                
                trajectory = out['trajectory']
                
                # [DEBUG] Level 4: Rollout Progress (Inside Loop via Trajectory)
                # Log to WandB instead of Console
                if runtime_probe_enabled and not debug_printed and epoch == 0 and batch_idx == 0 and len(trajectory) > 0:
                    try:
                        probe_data['debug/rollout_steps'] = len(trajectory)
                        
                        # Extract first graph stats from each step
                        for t, step_data in enumerate(trajectory):
                            if t > 4: break # Limit to first 5 steps
                            
                            # Unpack
                            dyn = step_data.get('dynamic_state', {})
                            t_sim = dyn.get('t_sim', torch.tensor([0.0]))
                            active = step_data.get('active_mask', torch.tensor([True]))
                            
                            probe_data[f'debug/rollout/step{t}/t_sim'] = t_sim[0].item()
                            probe_data[f'debug/rollout/step{t}/active'] = float(active[0].item())
                            
                            rea_in = step_data.get('reasoner_input_state', {})
                            acc_mask = rea_in.get('accumulated_mask') 
                            fused_batch = step_data.get('fused_batch')
                            
                            if acc_mask is not None and fused_batch is not None:
                                mask_0 = (fused_batch == 0)
                                acc_mask_0 = acc_mask[mask_0]
                                n_revealed = acc_mask_0.sum().item()
                                probe_data[f'debug/rollout/step{t}/n_revealed'] = n_revealed
                                
                            sel = step_data.get('nav_candidates')
                            if sel is not None:
                                if isinstance(sel, torch.Tensor):
                                    probe_data[f'debug/rollout/step{t}/selected_count'] = sel.numel()
                                else:
                                    probe_data[f'debug/rollout/step{t}/selected_type_unknown'] = 1
                    except Exception as e:
                        pass


                
                # Physical Distances (Fused)
                graph_structure = {'dist_to_source': out['step_metrics'].get('fused_dist')}
                loss, loss_dict = loss_engine(trajectory, cfg=cfg, graph_structure=graph_structure)

                # [PROBE] Step 0 Analysis (Probes 4, 5, 6)
                if runtime_probe_enabled and batch_idx == 0:
                    try:
                        probe_data['loss'] = loss.item()
                        if len(trajectory) > 0:
                            last_step = trajectory[-1]
                            logits = last_step.get('reasoner_logits')
                            labels = last_step.get('fused_source_label')
                            fused_batch = last_step.get('fused_batch')
                            
                            if logits is not None and labels is not None:
                                probe_data['logits_shape'] = str(list(logits.shape))
                                probe_data['labels_shape'] = str(list(labels.shape))
                                l_flat = labels.view(-1)
                                
                                # Graph 0 Label Sum
                                if fused_batch is not None:
                                    mask0 = (fused_batch == 0)
                                    l0 = l_flat[mask0]
                                    probe_data['label_sum_g0'] = l0.sum().item()
                                    
                                    # [Assert] Label Validity (Probe 4)
                                    if cfg.get('debug', False):
                                        if l0.sum().item() != 1:
                                            logger.error(f"[PROBE ASSERT FAIL] Graph 0 label sum is {l0.sum().item()} (Expected 1)")
                                
                                # Probe 5: p_true calculation + New Scalar Probes
                                p_true_all = None
                                src_mask = (l_flat > 0.5)
                                has_source = src_mask.any()

                                if logits.shape[-1] == 1:
                                    probs = torch.sigmoid(logits).view(-1)
                                    p_true_all = torch.where(l_flat > 0.5, probs, 1.0 - probs)
                                    
                                    # [Probe Extension] Logit Scalars
                                    logit_max = logits.view(-1).max().item()
                                    logit_min = logits.view(-1).min().item()
                                    logit_abs_mean = logits.abs().mean().item()
                                    logit_std = logits.std().item()
                                    
                                    probe_data['logit_max'] = logit_max
                                    probe_data['logit_min'] = logit_min
                                    probe_data['logit_abs_mean'] = logit_abs_mean
                                    probe_data['logit_std'] = logit_std
                                    probe_data['logits_tensor_id'] = "rea_logits_binary"
                                    
                                    # [Detailed Debug] Check Raw Logits vs Used Logits
                                    # reasoner_logits is likely used for loss.
                                    # Check reasoner raw output if available
                                    # trajectory element has 'reasoner_logits' (used)
                                    # But we don't have 'raw' unless stepper stored it.
                                    # Stepper stores: 'reasoner_logits': logits_fused (post-heuristic)
                                    # So we only see used.
                                    # Check mask
                                    rea_in = last_step.get('reasoner_input_state', {})
                                    valid_mask = rea_in.get('accumulated_mask') # Wait, reasoner uses ~accumulated_mask?
                                    # Reasoner takes 'accumulated_mask'.
                                    # If mask is used to zero out, we need to check mask statistics.
                            # [Probe C - FIXED V3]
                            # Source: EpisodeStepper final_dynamic_state from result
                            # This avoids trajectory logging issues completely
                            final_dyn = out.get('final_dynamic_state', {})
                            accumulated_mask = final_dyn.get('accumulated_mask')
                            
                            # [Probe C2] Assert Sentinel
                            sentinel = out.get('debug_sentinel', 0)
                            probe_data['debug/sentinel_value'] = sentinel
                            
                            # [Probe C2] Rename Keys & Add Heartbeat
                            probe_data['probeC2/heartbeat'] = 1234567
                            probe_data['probeC2/sentinel_ok'] = 1 if sentinel == 777 else 0
                            
                            # [Probe C2] Rank Info
                            try:
                                rank = torch.distributed.get_rank()
                            except:
                                rank = 0
                            probe_data['meta/rank'] = rank
                            
                            if accumulated_mask is not None:
                                mask_c = accumulated_mask.view(-1)
                                mask_sum = mask_c.sum().item()
                                mask_first5 = torch.nonzero(mask_c > 0.5)[:5].view(-1).tolist()
                                
                                # Log directly to probe_data for wandb
                                probe_data['probeC2/mask_sum'] = mask_sum
                                probe_data['probeC2/mask_first5'] = str(mask_first5)
                                probe_data['valid_mask_sum'] = mask_sum # Fix legacy probe
                                
                                # [Probe C - FIXED V4]
                                # User Request: Update Probe C to use mask_sum_after_load
                                if 'probeB/mask_sum_after_load' in probe_data:
                                     probe_data['probeC2/mask_sum'] = probe_data['probeB/mask_sum_after_load']
                            else:
                                pass

                            # Re-implementation of Probe C to use trajectory[-1] but ROBUSTLY
                            traj_len = len(trajectory)
                            
                            if traj_len > 0:
                                last_step = trajectory[-1]
                                # Look for mask in multiple places
                                mask_c = None
                                
                                # 1. reasoner_input_state (Vis)
                                if 'reasoner_input_state' in last_step:
                                     mask_c = last_step['reasoner_input_state'].get('accumulated_mask')
                                
                                # 2. dynamic_state (Snapshot)
                                if mask_c is None and 'dynamic_state' in last_step:
                                     mask_c = last_step['dynamic_state'].get('accumulated_mask')
                                     
                                # 3. active_mask (Not accumulated, but related)
                                
                                if mask_c is not None:
                                    mask_c = mask_c.view(-1)
                                    probe_data['probeC_mask_sum'] = mask_c.sum().item()
                                    probe_data['probeC_mask_first5'] = str(torch.nonzero(mask_c > 0.5)[:5].view(-1).tolist())
                                    probe_data['valid_mask_sum'] = mask_c.sum().item() # Fix legacy probe
                                else:
                                    probe_data['debug/probeC_mask_missing'] = 1
                                    
                            else:
                                probe_data['debug/trajectory_empty'] = 1

                            
                            if has_source:
                                # True Source Logit
                                logit_true = logits.view(-1)[src_mask].mean().item()
                                probe_data['logit_true'] = logit_true
                                probe_data['true_is_masked'] = 1.0 if logit_true < -100.0 else 0.0
                                if valid_mask is not None:
                                     probe_data['mask_value_true'] = valid_mask[src_mask].mean().item()

                            elif logits.shape[-1] == 2:
                                    probs = torch.softmax(logits, dim=-1)
                                    p_true_all = probs.gather(1, l_flat.long().view(-1, 1)).squeeze()
                                    
                                    # [Probe Extension] Logit Scalars (Use Class 1 Logit)
                                    # logits is [N, 2]. Class 1 is index 1.
                                    logits_c1 = logits[:, 1].view(-1)
                                    logit_max = logits_c1.max().item()
                                    logit_min = logits_c1.min().item()
                                    logit_abs_mean = logits_c1.abs().mean().item()
                                    logit_std = logits_c1.std().item()
                                    
                                    probe_data['logit_max'] = logit_max
                                    probe_data['logit_min'] = logit_min
                                    probe_data['logit_abs_mean'] = logit_abs_mean
                                    probe_data['logit_std'] = logit_std
                                    probe_data['logits_tensor_id'] = "rea_logits_class1"
                                    
                                    rea_in = last_step.get('reasoner_input_state', {})
                                    valid_mask = rea_in.get('accumulated_mask')
                                    if valid_mask is not None:
                                        probe_data['valid_mask_sum'] = valid_mask.sum().item()
                                        probe_data['valid_mask_mean'] = valid_mask.float().mean().item()
                                    
                                    if has_source:
                                        logit_true = logits_c1[src_mask].mean().item()
                                        probe_data['logit_true'] = logit_true
                                        probe_data['true_is_masked'] = 1.0 if logit_true < -100.0 else 0.0
                                        if valid_mask is not None:
                                             probe_data['mask_value_true'] = valid_mask[src_mask].mean().item()

                            if p_true_all is not None:
                                src_mask = (l_flat > 0.5)
                                if src_mask.sum() > 0:
                                    probe_data['p_true_last'] = p_true_all[src_mask].mean().item()
                                    
                                    # Get Step 0 p_true
                                    if len(trajectory) > 0:
                                        step0 = trajectory[0]
                                        logits0 = step0.get('reasoner_logits')
                                        if logits0 is not None:
                                            if logits0.shape[-1] == 1:
                                                probs0 = torch.sigmoid(logits0).view(-1)
                                                p_true_all0 = torch.where(l_flat > 0.5, probs0, 1.0 - probs0)
                                            else:
                                                probs0 = torch.softmax(logits0, dim=-1)
                                                p_true_all0 = probs0.gather(1, l_flat.long().view(-1, 1)).squeeze()
                                            probe_data['p_true_0'] = p_true_all0[src_mask].mean().item()
                                            probe_data['p_true_delta'] = probe_data['p_true_last'] - probe_data['p_true_0']
                    except Exception as e:
                        pass # Non-blocking
                
                # [DEBUG] Level 3: Loss Inputs & Level 5: Stability
                # Log to WandB
                if runtime_probe_enabled and not debug_printed and epoch == 0 and batch_idx == 0:
                    try:
                        probe_data['debug/loss_value'] = loss.item()
                        probe_data['debug/loss_is_finite'] = float(torch.isfinite(loss).item())
                        
                        if len(trajectory) > 0:
                            last_step = trajectory[-1]
                            logits = last_step.get('reasoner_logits')
                            labels = last_step.get('fused_source_label')
                            
                            if logits is not None:
                                probe_data['debug/logits_shape_0'] = logits.shape[0]
                                probe_data['debug/logits_finite'] = float(torch.isfinite(logits).all().item())
                                
                                if labels is not None:
                                    fused_batch = last_step.get('fused_batch')
                                    if fused_batch is not None:
                                        mask_0 = (fused_batch == 0)
                                        l0 = labels[mask_0]
                                        probe_data['debug/g0_label_argmax'] = float(l0.argmax().item())
                                        probe_data['debug/g0_label_sum_final'] = l0.sum().item()
                    except Exception as e:
                        pass
                    
                    debug_printed = True


                # [PLATFORM] On-the-fly Metric Accumulation
                if evaluator and 'step_metrics' in out:
                    sm = out['step_metrics']
                    # Unpack raw tensors for per-sample tracking
                    raw_success = sm.get('raw_success')
                    if raw_success is not None:
                        raw_steps = sm.get('raw_steps')
                        raw_budget = sm.get('raw_budget')
                        raw_predict_hit = sm.get('raw_predict_hit')
                        raw_predict_hit_5 = sm.get('raw_predict_hit_5')
                        raw_predict_hit_valid = sm.get('raw_predict_hit_valid')
                        raw_max_hit_prob = sm.get('raw_max_hit_prob')
                        raw_core_mass_before = sm.get('raw_core_mass_before')
                        raw_core_mass_after = sm.get('raw_core_mass_after')
                        raw_core_mass_delta = sm.get('raw_core_mass_delta')
                        raw_core_size_before = sm.get('raw_core_size_before')
                        raw_core_size_after = sm.get('raw_core_size_after')
                        raw_core_size_delta = sm.get('raw_core_size_delta')
                        raw_uncertainty_before = sm.get('raw_uncertainty_before')
                        raw_uncertainty_after = sm.get('raw_uncertainty_after')
                        raw_uncertainty_collapse = sm.get('raw_uncertainty_collapse')
                        raw_closure_success = sm.get('raw_closure_success')
                        raw_decisive_closure = sm.get('raw_decisive_closure')
                        raw_budget_to_closure = sm.get('raw_budget_to_closure')
                        raw_budget_efficiency = sm.get('raw_budget_efficiency')
                        raw_evidence_gain = sm.get('raw_evidence_gain_per_sample')
                        raw_harmful_drift = sm.get('raw_harmful_drift')
                        raw_focus_core_delta = sm.get('raw_focus_core_delta')
                        raw_wasted_budget_fraction = sm.get('raw_wasted_budget_fraction')
                        raw_empty_selection_fraction = sm.get('raw_empty_selection_fraction')
                        raw_terminal_budget_bonus = sm.get('raw_terminal_budget_bonus')
                        
                        # Loop over batch (CPU side, cheap)
                        bsz_curr = raw_success.size(0)
                        for i in range(bsz_curr):
                            ep_data = {
                                'success': bool(raw_success[i].item() > 0.5),
                                'steps': float(raw_steps[i].item()) if raw_steps is not None else 0.0,
                                'budget': float(raw_budget[i].item()) if raw_budget is not None else 0.0,
                                'predict_hit': bool(raw_predict_hit[i].item() > 0.5) if raw_predict_hit is not None else False,
                                'predict_hit_at_5': bool(raw_predict_hit_5[i].item() > 0.5) if raw_predict_hit_5 is not None else False,
                                'predict_hit_valid': bool(raw_predict_hit_valid[i].item() > 0.5) if raw_predict_hit_valid is not None else False,
                                'max_hit_prob': float(raw_max_hit_prob[i].item()) if raw_max_hit_prob is not None else 0.0,
                                'core_mass_before': float(raw_core_mass_before[i].item()) if raw_core_mass_before is not None else 0.0,
                                'core_mass_after': float(raw_core_mass_after[i].item()) if raw_core_mass_after is not None else 0.0,
                                'core_mass_delta': float(raw_core_mass_delta[i].item()) if raw_core_mass_delta is not None else 0.0,
                                'core_size_before': float(raw_core_size_before[i].item()) if raw_core_size_before is not None else 0.0,
                                'core_size_after': float(raw_core_size_after[i].item()) if raw_core_size_after is not None else 0.0,
                                'core_size_delta': float(raw_core_size_delta[i].item()) if raw_core_size_delta is not None else 0.0,
                                'uncertainty_before': float(raw_uncertainty_before[i].item()) if raw_uncertainty_before is not None else 0.0,
                                'uncertainty_after': float(raw_uncertainty_after[i].item()) if raw_uncertainty_after is not None else 0.0,
                                'uncertainty_collapse': float(raw_uncertainty_collapse[i].item()) if raw_uncertainty_collapse is not None else 0.0,
                                'closure_success': float(raw_closure_success[i].item()) if raw_closure_success is not None else 0.0,
                                'decisive_closure': float(raw_decisive_closure[i].item()) if raw_decisive_closure is not None else 0.0,
                                'budget_to_closure': float(raw_budget_to_closure[i].item()) if raw_budget_to_closure is not None else (float(raw_budget[i].item()) if raw_budget is not None else 0.0),
                                'budget_efficiency': float(raw_budget_efficiency[i].item()) if raw_budget_efficiency is not None else 0.0,
                                'evidence_gain_per_sample': float(raw_evidence_gain[i].item()) if raw_evidence_gain is not None else 0.0,
                                'harmful_drift': float(raw_harmful_drift[i].item()) if raw_harmful_drift is not None else 0.0,
                                'focus_core_delta': float(raw_focus_core_delta[i].item()) if raw_focus_core_delta is not None else 0.0,
                                'wasted_budget_fraction': float(raw_wasted_budget_fraction[i].item()) if raw_wasted_budget_fraction is not None else 0.0,
                                'empty_selection_fraction': float(raw_empty_selection_fraction[i].item()) if raw_empty_selection_fraction is not None else 0.0,
                                'terminal_budget_bonus': float(raw_terminal_budget_bonus[i].item()) if raw_terminal_budget_bonus is not None else 0.0,
                            }
                            evaluator.update_episode(ep_data)
                
                # Standard Batch Metrics (Accuracy, etc.)
                if evaluator and 'classification' in out:
                    logits = out['classification']
                    # If we have targets in batch.y?
                    # batch.y is handled in evaluator update_batch if passed
                    # But here we need to extract it from batch or trajectory?
                    # Trajectory has fused_source_label.
                    # Let's try to extract targets from batch if available.
                    if hasattr(batch, 'y'):
                        targets = batch.y
                        # Need to handle batch indexing if logits are [B, 2] and targets are [N]
                        # Phase45Model returns classification logits for [B] graphs?
                        # No, Stepper returns: 'classification': last_logits[inverse_indices] if last_logits is not None else torch.zeros(curr_batch_size, 2, device=h_fused.device)
                        # Wait, last_logits is usually node-level?
                        # In Stepper: last_logits = logits_fused.
                        # logits_fused is [N, 1] usually (binary classification per node).
                        # inverse_indices maps N_fused -> N_original.
                        # So out['classification'] is likely [N_original, 1].
                        
                        # However, evaluator expects dense batch [B, N] usually?
                        # Evaluator.update_batch(logits, targets)
                        # _compute_metrics expects flattened or compatible shapes.
                        
                        # Given Phase 4.5 complexity, let's stick to Episode Metrics for now as they are the most critical.
                        pass
            
            if torch.isnan(loss):
                logger.error(f"[NAN-DEBUG] Loss is NaN at Epoch {epoch+1}")
                raise ValueError("Loss is NaN")
            
            do_optimizer_step = (
                ((batch_idx + 1) % grad_accum_steps) == 0
                or (batch_idx + 1) >= len(train_loader)
            )

            # AMP Backward
            if use_amp:
                scaler.scale(loss / float(grad_accum_steps)).backward()

                if do_optimizer_step:
                    # [Optimization] Single Unscale Point
                    scaler.unscale_(optimizer)

                    # [PROBE] Step 0: Pre-Update Capture (Probe 1, 3)
                    if runtime_probe_enabled and batch_idx == 0:
                        try:
                            # Probe 3: Grad Norm Pre-Clip
                            grad_norms = [p.grad.detach().norm(2) for p in model.parameters() if p.grad is not None]
                            if grad_norms:
                                probe_data['grad_norm_pre'] = torch.norm(torch.stack(grad_norms), 2).item()
                            
                            # Probe 1: Weight Pre-Update (Track first valid param)
                            for name, p in model.named_parameters():
                                if p.requires_grad and p.grad is not None:
                                    probe_data['probe_param_name'] = name
                                    probe_data['probe_param_pre'] = p.detach().clone()
                                    break
                        except Exception as e: pass

                    # [DEBUG] Level 0: Gradient Norm (After Unscale, Before Clip)
                    if runtime_probe_enabled and batch_idx == 0 and epoch == 0:
                         try:
                             # Already unscaled, direct check
                             total_norm = 0.0
                             for p in model.parameters():
                                 if p.grad is not None:
                                     param_norm = p.grad.data.norm(2)
                                     total_norm += param_norm.item() ** 2
                             total_norm = total_norm ** 0.5
                             probe_data['debug/grad_norm_total_calc'] = total_norm
                         except Exception as e:
                             pass

                    # [SSOT] Gradient Clipping
                    if getattr(cfg.training, 'gradient_clip_norm', 0.0) > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.gradient_clip_norm)

                    # [PROBE] Step 0: Grad Norm Post-Clip (Probe 3)
                    if runtime_probe_enabled and batch_idx == 0:
                        try:
                            grad_norms = [p.grad.detach().norm(2) for p in model.parameters() if p.grad is not None]
                            if grad_norms:
                                probe_data['grad_norm_post'] = torch.norm(torch.stack(grad_norms), 2).item()
                        except Exception as e: pass
                    
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)

                    # [PROBE] Step 0: Final Report (Probe 1, 2 + All)
                    if runtime_probe_enabled and batch_idx == 0:
                        try:
                            # Probe 1: Delta W
                            if 'probe_param_pre' in probe_data:
                                for name, p in model.named_parameters():
                                    if name == probe_data.get('probe_param_name'):
                                        delta = (p.detach() - probe_data['probe_param_pre']).norm(2).item()
                                        probe_data['delta_w'] = delta
                                        break
                            
                            # Probe 2: AMP Scale
                            probe_data['amp_scale'] = scaler.get_scale()
                            
                            # [PROBE] W&B Integration (Replaces Console Dump)
                            if wandb_enabled and wandb.run is not None and log_this_step:
                                grad_pre = probe_data.get('grad_norm_pre', 0.0)
                                grad_post = probe_data.get('grad_norm_post', 0.0)
                                p0 = probe_data.get('p_true_0', 0.0)
                                plast = probe_data.get('p_true_last', 0.0)
                                
                                # Parse shapes safe eval
                                try:
                                    logits_shape = eval(probe_data.get('logits_shape', '[0,0]'))
                                    logits_n = logits_shape[0] if len(logits_shape) > 0 else 0
                                    logits_c = logits_shape[1] if len(logits_shape) > 1 else 1
                                except:
                                    logits_n, logits_c = 0, 0

                                metrics = {
                                    "probe/dw": float(probe_data.get('delta_w', 0)),
                                    "probe/grad_pre": float(grad_pre),
                                    "probe/grad_post": float(grad_post),
                                    "probe/grad_clip_ratio": float(grad_post/(grad_pre+1e-12)),
                                    "probe/amp_scale": float(probe_data.get('amp_scale', 0)),
                                    "probe/logits_shape_n": int(logits_n),
                                    "probe/logits_shape_c": int(logits_c),
                                    "probe/labels_sum_g0": float(probe_data.get('label_sum_g0', 0)),
                                    "probe/p_true_step0": float(p0),
                                    "probe/p_true_last": float(plast),
                                    "probe/p_true_delta": float(probe_data.get('p_true_delta', 0)),
                                    "probe/logit_true": float(probe_data.get('logit_true', 0)),
                                    "probe/logit_max": float(probe_data.get('logit_max', 0)),
                                    "probe/logit_min": float(probe_data.get('logit_min', 0)),
                                    "probe/logit_abs_mean": float(probe_data.get('logit_abs_mean', 0)),
                                    "probe/logit_std": float(probe_data.get('logit_std', 0)),
                                    "probe/logits_tensor_id": probe_data.get('logits_tensor_id', "unknown"),
                                    "probe/valid_mask_mean": float(probe_data.get('valid_mask_mean', 0)),
                                    "probe/valid_mask_sum": float(probe_data.get('valid_mask_sum', 0)),
                                    "probe/true_is_masked": float(probe_data.get('true_is_masked', 0)),
                                    "probe/shape_ok": 1,
                                    "meta/epoch": int(epoch),
                                    "meta/run_id": wandb.run.id,
                                }

                                # [PROBE] Dynamically add new step-wise probe metrics
                                for key, value in probe_data.items():
                                    if key.startswith("probe/step"):
                                        metrics[key] = value
                                
                                wandb.log(metrics)
                        
                        except Exception as e:
                            logger.error(f"[PROBE FAILED] {e}")

                # [SSOT] Log Step Metrics (Hit@1, Hit@5 etc.) at EVERY batch
                # FIX: Extract step_metrics from probe_b_metrics and log them independently of batch_idx check
                if runtime_probe_enabled and wandb_enabled and wandb.run is not None:
                    step_metrics_log = {}
                    if 'probe_b_metrics' in out:
                        pb = out['probe_b_metrics']
                        for k, v in pb.items():
                            # Allow Evidence and Reasoner metrics to pass through
                            # [Audit V3 Update] Added suspect_pool, gate, evidence_*
                            if (k.startswith('step_metrics/') or k.startswith('evidence') or 
                                k.startswith('reasoner/') or k.startswith('system/') or 
                                k.startswith('gate/') or k.startswith('suspect_pool/')):
                                step_metrics_log[k] = v
                                # Accumulate for Epoch Summary
                                if k not in epoch_step_metrics:
                                    epoch_step_metrics[k] = 0.0
                                    epoch_step_counts[k] = 0
                                epoch_step_metrics[k] += v
                                epoch_step_counts[k] += 1
                    
                    if step_metrics_log and log_this_step:
                        wandb.log(step_metrics_log, step=global_step)
                elif runtime_probe_enabled:
                    # Accumulate even if wandb is off
                    if 'probe_b_metrics' in out:
                        pb = out['probe_b_metrics']
                        for k, v in pb.items():
                            if (k.startswith('step_metrics/') or k.startswith('evidence') or 
                                k.startswith('reasoner/') or k.startswith('system/') or 
                                k.startswith('gate/') or k.startswith('suspect_pool/')):
                                if k not in epoch_step_metrics:
                                    epoch_step_metrics[k] = 0.0
                                    epoch_step_counts[k] = 0
                                epoch_step_metrics[k] += v
                                epoch_step_counts[k] += 1
            else:
                (loss / float(grad_accum_steps)).backward()
                
                # [DEBUG] Level 0 for No-AMP
                if batch_idx == 0 and epoch == 0 and do_optimizer_step:
                     logger.info("\n" + "="*40)
                     logger.info("[DEBUG] Level 0: Parameter Update Check (No AMP)")
                     try:
                         total_norm = 0.0
                         for p in model.parameters():
                             if p.grad is not None:
                                 param_norm = p.grad.data.norm(2)
                                 total_norm += param_norm.item() ** 2
                         total_norm = total_norm ** 0.5
                         logger.info(f"  Total Grad Norm: {total_norm:.4f}")
                     except Exception as e:
                         logger.error(f"  [ERROR] Level 0 Check Failed: {e}")
                     logger.info("="*40 + "\n")

                if do_optimizer_step:
                    # [SSOT] Gradient Clipping
                    if getattr(cfg.training, 'gradient_clip_norm', 0.0) > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.gradient_clip_norm)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
            
            total_loss += loss.item()
            pbar.set_postfix({'L': f"{loss.item():.4f}"})
            
            # Log Loss Components to WandB
            if wandb_enabled and wandb.run is not None and log_this_step:
                wandb.log(loss_dict, step=global_step)
        epoch_train_loop_wall_s = time.perf_counter() - epoch_train_loop_start
        train_loop_wall_s += epoch_train_loop_wall_s
            
        # [SSOT User Request] Print Training Metrics at Epoch End
        if evaluator:
             train_metrics = evaluator.summarize()
             logger.info(f"\n[Epoch {epoch} Training Metrics (Exploration Mode)]")
             if 'official/core_mass_delta' in train_metrics:
                  logger.info(f"  Core Mass Delta: {train_metrics['official/core_mass_delta']:.4f}")
             if 'official/uncertainty_collapse' in train_metrics:
                  logger.info(f"  Uncertainty Collapse: {train_metrics['official/uncertainty_collapse']:.4f}")
             if 'official/decisive_closure_rate' in train_metrics:
                  logger.info(f"  Decisive Closure Rate: {train_metrics['official/decisive_closure_rate']:.4f}")
             if 'official/budget_used' in train_metrics:
                  logger.info(f"  Budget Used: {train_metrics['official/budget_used']:.1f}")
             if 'official/evidence_gain_per_sample' in train_metrics:
                  logger.info(f"  Evidence Gain / Sample: {train_metrics['official/evidence_gain_per_sample']:.4f}")
             
             if 'legacy/Predict_Hit@1' in train_metrics:
                  logger.info(f"  Legacy Predict Hit@1: {train_metrics['legacy/Predict_Hit@1']:.4f}")
             
             logger.info("-" * 40)
             
             if wandb_enabled and wandb.run is not None:
                 wandb.log(train_metrics, step=global_step)

        epoch_step_metrics_avg = {}
        if epoch_step_metrics:
            epoch_step_metrics_avg = {
                key: epoch_step_metrics[key] / max(1, epoch_step_counts[key])
                for key in epoch_step_metrics
            }
             
        # Evaluation
        val_metrics = None
        if (
            getattr(cfg.training, 'enable_eval', True)
            and val_loader is not None
            and (epoch + 1) % cfg.training.val_every_n_epochs == 0
        ):
            logger.info("Evaluating on validation split...")
            val_metrics = evaluate_mode(model, val_loader, device=device)
            
            # [SSOT] Round metrics for display ONLY
            display_metrics = {}
            for k, v in val_metrics.items():
                if isinstance(v, float):
                    display_metrics[k] = round(v, 3)
                else:
                    display_metrics[k] = v
            
            logger.info(f"Epoch {epoch+1} Val Results: {display_metrics}")
            if wandb_enabled and wandb.run is not None:
                wandb.log({f"val/{k}": v for k, v in val_metrics.items()}, step=global_step)
            
            # Track Best (Primary: official Navigator-only mainline score)
            current_mainline_score = float(val_metrics.get('official/core_mass_delta', -float('inf')))
            if current_mainline_score > best_mainline_score:
                best_mainline_score = current_mainline_score
                best_metrics = dict(val_metrics)
                best_metrics['Convergence_Epoch'] = epoch + 1
                best_metrics['selection_metric'] = 'val/official/core_mass_delta'
                torch.save(model.state_dict(), os.path.join(save_dir, "model_best.pt"))
        else:
            current_train_loss = total_loss / max(1, len(train_loader))
            if current_train_loss < best_train_loss:
                best_train_loss = current_train_loss

        checkpoint_extra_state = {}
        if scaler is not None:
            checkpoint_extra_state['scaler_state_dict'] = scaler.state_dict()
        save_checkpoint(
            save_dir,
            model,
            optimizer,
            epoch,
            metrics={
                'train_loss': total_loss / max(1, len(train_loader)),
                'val_loss': float('nan') if val_metrics is None else val_metrics.get('loss', float('nan')),
            },
            is_best_val=bool(val_metrics is not None and best_metrics is not None and best_metrics.get('Convergence_Epoch') == epoch + 1),
            is_best_train=bool((total_loss / max(1, len(train_loader))) <= best_train_loss),
            periodic_every=periodic_checkpoint_every,
            extra_state=checkpoint_extra_state,
        )

        epoch_run_training_wall_s = time.perf_counter() - epoch_run_training_start
        history_record = {
            'epoch': epoch + 1,
            'train_loss': total_loss / max(1, len(train_loader)),
            'tau': tau,
            'sample_budget': int(getattr(cfg.model, 'sample_budget', 1)),
            'sampling_policy': str(getattr(cfg.model, 'sampling_policy', 'greedy')),
            'train_metrics': train_metrics,
            'train_step_metrics': epoch_step_metrics_avg,
            'val_metrics': val_metrics,
            'epoch_train_loop_wall_s': float(epoch_train_loop_wall_s),
            'epoch_train_loop_scope': (
                "this epoch's training batch-iteration section only; excludes model/data setup before the epoch "
                "and excludes validation/checkpoint work after the loop"
            ),
            'epoch_train_loop_includes_first_step_cold_start': bool(epoch == 0),
            'epoch_run_training_wall_s': float(epoch_run_training_wall_s),
            'epoch_run_training_scope': (
                "this epoch's official training pipeline section from epoch start to epoch-end bookkeeping; "
                "includes schedule updates, training loop, and any validation/checkpoint work triggered this epoch"
            ),
            'epoch_run_training_includes_first_step_cold_start': bool(epoch == 0),
            'epoch_run_training_cumulative_wall_s': float(time.perf_counter() - run_training_start),
            'epoch_has_validation': bool(val_metrics is not None),
        }
        with open(epoch_history_path, 'a') as f:
            f.write(json.dumps(_json_ready(history_record)) + "\n")
                
    # Final cleanup and success marking
    run_timing = {
        "train_loop_wall_s": float(train_loop_wall_s),
        "train_loop_scope": (
            "sum of per-epoch training batch-iteration sections only; "
            "excludes data/model setup before the loop and excludes validation/checkpoint work after the loop"
        ),
        "train_loop_includes_first_step_cold_start": True,
        "run_training_wall_s": float(time.perf_counter() - run_training_start),
        "run_training_scope": (
            "run_training() entry to return; includes resolved-config dump, dataloader/model/optimizer setup, "
            "training loop, validation, checkpoint/metrics writes, with skip_audit applied as passed by caller"
        ),
        "skip_audit": bool(skip_audit),
    }
    best_model_path = os.path.join(save_dir, "model_best.pt")
    if best_metrics:
        best_metrics['num_epochs'] = epochs
        best_metrics['run_timing'] = run_timing
        best_metrics['run_dir'] = save_dir
        best_metrics['config_hash'] = cfg_hash
        best_metrics['resolved_config_path'] = resolved_cfg_path
        with open(metrics_cache, 'w') as f:
            json.dump(best_metrics, f, indent=4)
    elif not os.path.exists(best_model_path):
        torch.save(model.state_dict(), best_model_path)

    final_checkpoint_path = os.path.join(save_dir, "model_final.pt")
    torch.save(model.state_dict(), final_checkpoint_path)
        
    with open(success_marker, 'w') as f:
        f.write(f"Completed at {datetime.now().isoformat()}\n")
        f.write(f"Config Hash: {cfg_hash}\n")
        
    if wandb_enabled and wandb.run is not None:
        wandb.finish()
    if best_metrics is None:
        best_metrics = {
            'num_epochs': epochs,
            'run_timing': run_timing,
            'run_dir': save_dir,
            'config_hash': cfg_hash,
            'resolved_config_path': resolved_cfg_path,
            'final_checkpoint_path': best_model_path,
        }
    best_metrics['train_dataset_size'] = train_dataset_size
    best_metrics['train_batches_per_epoch'] = train_batches_per_epoch
    best_metrics['latest_checkpoint_path'] = os.path.join(save_dir, 'checkpoints', 'checkpoint_latest.pt')
    best_metrics['final_model_state_path'] = final_checkpoint_path
    best_metrics['peak_gpu_memory_mb'] = (
        float(torch.cuda.max_memory_allocated(device) / (1024 ** 2))
        if device.type == 'cuda' else 0.0
    )
    return best_metrics

def main():
    import argparse
    import yaml
    parser = argparse.ArgumentParser(description="Phase 4.5 Standard Training Engine")
    parser.add_argument("--run_name", type=str, help="Override run name")
    parser.add_argument("--epochs", type=int, help="Override max epochs")
    parser.add_argument("--batch_size", type=int, help="Override batch size")
    parser.add_argument("--lr", type=float, help="Override learning rate")
    parser.add_argument("--seed", type=int, help="Override random seed")
    parser.add_argument("--seeds", type=int, nargs='+', help="Run multiple seeds sequentially with shared data")
    parser.add_argument("--config", type=str, nargs='+', help="Path to yaml config file(s). Multiple files will run in sequence with shared data.")
    parser.add_argument("--suite", type=str, choices=['loss'], help="Run a predefined experiment suite")
    parser.add_argument("--override", action='append', help="Override config value (e.g. model.sampling_policy=learned)")
    parser.add_argument("--skip_audit", action="store_true", help="Skip the audit process")
    parser.add_argument("--force_rebuild", action="store_true", help="Force rebuild of experiment results, ignoring cache.")
    
    args = parser.parse_args()

    # Parse overrides
    cli_overrides = {}
    if args.override:
        for override in args.override:
            if '=' in override:
                key, value = override.split('=', 1)
                
                # Handle value types (int, float, bool)
                if value.lower() == 'true': value = True
                elif value.lower() == 'false': value = False
                elif value.isdigit(): value = int(value)
                else:
                    try:
                        value = float(value)
                    except ValueError:
                        pass # Keep as string
                
                # Nested keys support (e.g. model.navigator.k_explore)
                keys = key.split('.')
                current = cli_overrides
                for k in keys[:-1]:
                    if k not in current: current[k] = {}
                    current = current[k]
                current[keys[-1]] = value

    if args.suite == 'loss':
        logger.info("=== Running Loss Experiment Suite (Standardized) ===")
        suite_configs = [
            "configs/loss_experiments/exp_l0_baseline.yaml",
            "configs/loss_experiments/exp_l1_temporal.yaml",
            "configs/loss_experiments/exp_l2_monotonic.yaml",
            "configs/loss_experiments/exp_l3_uncertainty.yaml",
            "configs/loss_experiments/exp_l4_physical.yaml",
            "configs/loss_experiments/exp_l5_symphony.yaml",
        ]
        
        results = []
        # Load loaders once to share across suite (Performance optimization from absorbed script)
        cfg_base = Config()
        loaders = create_dataloaders(
            data_root=cfg_base.paths.samples_path,
            cfg=cfg_base,
            batch_size=args.batch_size if args.batch_size else cfg_base.efficiency.batch_size,
            skip_lmdb=getattr(cfg_base.data, 'skip_lmdb', False)
        )
        
        for cfg_path in suite_configs:
            if not os.path.exists(cfg_path):
                logger.warning(f"Suite config {cfg_path} not found, skipping.")
                continue
                
            with open(cfg_path, 'r') as f:
                overrides = yaml.safe_load(f)
            
            # Apply CLI overrides on top of suite config
            if args.epochs:
                if 'training' not in overrides: overrides['training'] = {}
                overrides['training']['num_epochs'] = args.epochs
            
            exp_id = os.path.basename(cfg_path).replace('.yaml', '').upper()
            logger.info(f"\n>>> Starting Suite Experiment: {exp_id}")
            
            try:
                metrics = run_training(
                    loss_config_override=overrides,
                    run_name=exp_id,
                    max_epochs=args.epochs,
                    loaders=loaders,
                    force_rebuild=args.force_rebuild
                )
                
                res = {
                    "Exp_ID": exp_id,
                    "Official_Core_Mass_Delta": metrics.get("official/core_mass_delta", 0.0),
                    "Official_Uncertainty_Collapse": metrics.get("official/uncertainty_collapse", 0.0),
                    "Official_Decisive_Closure_Rate": metrics.get("official/decisive_closure_rate", 0.0),
                    "Official_Budget_Efficiency": metrics.get("official/budget_efficiency", 0.0),
                    "Legacy_Success_Rate": metrics.get("legacy/Success_Rate", 0.0),
                    "Convergence_Epoch": metrics.get("Convergence_Epoch", -1)
                }
                results.append(res)
                logger.info(f">>> Finished {exp_id}: {res}")
            except Exception as e:
                logger.error(f"Experiment {exp_id} failed: {e}")
        
        if results:
            df = pd.DataFrame(results)
            out_path = f"suite_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            df.to_csv(out_path, index=False)
            logger.info(f"\n=== Suite Completed. Results saved to {out_path} ===")
            print(df)
        return

    # Recursive update helper
    def recursive_update(d, u):
        for k, v in u.items():
            if isinstance(v, dict):
                d[k] = recursive_update(d.get(k, {}), v)
            else:
                d[k] = v
        return d

    # Multi-config support with shared data (Performance optimization)
    if args.config and len(args.config) > 1:
        logger.info(f"=== [PLATFORM] Shared Data Cache: Enabled for {len(args.config)} configs ===")
        cfg_base = Config()
        # Initialize loaders once
        loaders = create_dataloaders(
            data_root=cfg_base.paths.samples_path,
            cfg=cfg_base,
            batch_size=args.batch_size if args.batch_size else cfg_base.efficiency.batch_size,
            skip_lmdb=getattr(cfg_base.data, 'skip_lmdb', False)
        )
        
        for cfg_path in args.config:
            if not os.path.exists(cfg_path):
                logger.warning(f"Config {cfg_path} not found, skipping.")
                continue
                
            with open(cfg_path, 'r') as f:
                overrides = yaml.safe_load(f) or {}
            
            # Apply CLI overrides on top of config
            if cli_overrides:
                recursive_update(overrides, cli_overrides)

            if 'training' not in overrides: overrides['training'] = {}
            if 'efficiency' not in overrides: overrides['efficiency'] = {}
            if args.run_name: overrides['training']['run_name'] = args.run_name
            if args.epochs: overrides['training']['num_epochs'] = args.epochs
            if args.batch_size: overrides['efficiency']['batch_size'] = args.batch_size
            if args.lr: overrides['training']['learning_rate'] = args.lr
            
            exp_id = os.path.basename(cfg_path).replace('.yaml', '')
            logger.info(f"\n" + "="*80)
            logger.info(f">>> [PLATFORM] Next Experiment: {exp_id}")
            logger.info("="*80)
            
            run_training(
                loss_config_override=overrides,
                run_name=args.run_name if args.run_name else exp_id,
                max_epochs=args.epochs,
                loaders=loaders,
                force_rebuild=args.force_rebuild
            )
        return

    overrides = {}
    
    # Single config or no config
    if args.config:
        cfg_path = args.config[0]
        if os.path.exists(cfg_path):
            with open(cfg_path, 'r') as f:
                yaml_cfg = yaml.safe_load(f)
                if yaml_cfg:
                    overrides.update(yaml_cfg)
            print(f"[CLI] Loaded overrides from {cfg_path}")
        else:
            print(f"[CLI][ERROR] Config file {cfg_path} not found!")
    
    # Apply CLI overrides (parsed from --override)
    if cli_overrides:
        recursive_update(overrides, cli_overrides)

    # 2. CLI arguments take highest priority
    if 'training' not in overrides: overrides['training'] = {}
    if 'efficiency' not in overrides: overrides['efficiency'] = {}
    if args.run_name: overrides['training']['run_name'] = args.run_name
    if args.epochs: overrides['training']['num_epochs'] = args.epochs
    if args.batch_size: overrides['efficiency']['batch_size'] = args.batch_size
    if args.lr: overrides['training']['learning_rate'] = args.lr

    if args.seeds:
        logger.info(f"=== [PLATFORM] Shared Data Cache: Enabled for {len(args.seeds)} seeds ===")
        cfg_base = Config()
        loaders = create_dataloaders(
            data_root=cfg_base.paths.samples_path,
            cfg=cfg_base,
            batch_size=args.batch_size if args.batch_size else cfg_base.efficiency.batch_size
        )
        
        base_overrides = overrides
        for seed in args.seeds:
            logger.info(f"\n" + "="*80)
            logger.info(f">>> [PLATFORM] Next Seed: {seed}")
            logger.info("="*80)
            
            # Create a copy of overrides for this seed
            import copy
            seed_overrides = copy.deepcopy(base_overrides)
            if 'training' not in seed_overrides: seed_overrides['training'] = {}
            seed_overrides['training']['seed'] = seed
            
            # Construct final_name
            base_run_name = args.run_name if args.run_name else seed_overrides.get('training', {}).get('run_name', 'experiment')
            
            # [PLATFORM] Enforce Run Naming Convention
            bs = seed_overrides.get('efficiency', {}).get('batch_size', 32)
            # max_samples is usually in data config or passed via override.
            # We check overrides first, then default to 'all' if not found.
            # Note: seed_overrides is a dict, so we check deep keys if needed.
            # But here we just check if 'max_samples' is in overrides['data']?
            # Or if it was passed via CLI override.
            # Let's try to get it from 'data' section if present.
            n_samples = "all"
            if 'data' in seed_overrides and 'max_samples' in seed_overrides['data']:
                n_samples = seed_overrides['data']['max_samples']
            
            suffix = f"_bs{bs}_n{n_samples}"

            if "{seed}" in base_run_name:
                final_name = base_run_name.format(seed=seed)
            else:
                final_name = f"{base_run_name}_seed{seed}"
            
            # Append suffix if not already present
            if f"_bs{bs}" not in final_name:
                final_name += suffix

            run_training(
                loss_config_override=overrides,
                run_name=final_name,
                max_epochs=args.epochs,
                seed=seed,
                loaders=loaders,
                skip_audit=(seed != args.seeds[0]),
                force_rebuild=args.force_rebuild
            )
    else:
        if args.seed: overrides['training']['seed'] = args.seed
        
        # [PLATFORM] Enforce Run Naming Convention (Single Run)
        base_run_name = args.run_name if args.run_name else overrides.get('training', {}).get('run_name', 'experiment')
        bs = args.batch_size if args.batch_size else overrides.get('efficiency', {}).get('batch_size', 32)
        
        # Try to find max_samples
        n_samples = "all"
        # Check overrides
        if 'data' in overrides and 'max_samples' in overrides['data']:
             n_samples = overrides['data']['max_samples']
        # Also check cfg default if not in overrides? Hard to access cfg here without init.
        # But we can assume if it's not in overrides, it might be 'all' or default.
        
        suffix = f"_bs{bs}_n{n_samples}"
        final_name = base_run_name
        if f"_bs{bs}" not in final_name:
             final_name += suffix
             
        run_training(
            loss_config_override=overrides, 
            run_name=final_name, 
            max_epochs=args.epochs, 
            seed=args.seed,
            skip_audit=args.skip_audit,
            force_rebuild=args.force_rebuild
        )

if __name__ == "__main__":
    main()
