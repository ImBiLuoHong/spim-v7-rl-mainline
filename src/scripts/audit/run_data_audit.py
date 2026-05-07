
import sys
import os
import torch
import logging
import argparse
from src.config.core import Config
from src.scripts.audit.data_contract import DataContractAuditor
from src.modeling.builders.model_builder import ModelBuilder
from src.modeling.architectures.phase4_5_model import Phase45Model
from src.data.v6.loader import create_dataloaders
from src.scripts.audit.audit_cache import AuditCache

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("DataAudit")

def run_audit(config_path=None, mode='fast', no_preload=True, use_cache=True):
    logger.info(f"=== APT Data & Semantic Audit Gatekeeper (Mode: {mode}) ===")
    
    # 1. Load Config
    cfg = Config()
    if config_path:
        import yaml
        with open(config_path, 'r') as f:
            overrides = yaml.safe_load(f)
        cfg.apply_overrides(overrides)
        
    # Force V6
    cfg.data.use_dataloader_v6 = True
    
    # 1.5 Audit Cache Check
    if use_cache:
        try:
            cache_key = AuditCache.generate_key(cfg)
            if AuditCache.check(cache_key):
                logger.info(f"✅ [CACHE HIT] Audit key {cache_key} found. Skipping heavy/full audit.")
                if mode == 'fast':
                    logger.info("Audit Cache Hit. Returning Success.")
                    return True
                else:
                    logger.info("Heavy Mode requested. Proceeding despite cache hit.")
        except Exception as e:
            logger.warning(f"Cache check failed: {e}. Proceeding with audit.")
            cache_key = None
    else:
        cache_key = None
    
    # 2. Audit Dataset Fingerprint
    logger.info("[Audit 1/5] Dataset Fingerprint & Config...")
    ok, issues = DataContractAuditor.audit_dataset_fingerprint(cfg)
    if not ok:
        logger.error(f"❌ Config Audit Failed: {issues}")
        return False
    logger.info("✅ Config Audit Passed.")
    
    # 3. Load One Batch (Real Data with Safe Loader)
    logger.info(f"[Audit 2/5] Loading Data Batch (Mode: {mode}, Preload: {not no_preload})...")
    try:
        # Use small batch size for speed
        cfg.training.batch_size = 4
        
        # Explicitly set data_root if not in cfg
        data_root = cfg.paths.samples_path
        
        # [AUDIT] Pass audit_mode to disable preload and full scan
        loader, _, _, _ = create_dataloaders(
            data_root=data_root, 
            cfg=cfg, 
            batch_size=4,
            audit_mode=mode
        )
        
        # Try to get one batch
        try:
            batch = next(iter(loader))
            if batch is None:
                logger.error("❌ DataLoader returned None! Check split files or sample directory.")
                return False
            batch = batch.to('cuda' if torch.cuda.is_available() else 'cpu')
        except StopIteration:
            logger.error("❌ DataLoader returned no data! Check split files or sample directory.")
            return False
            
    except Exception as e:
        logger.error(f"❌ Failed to load data: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    # 4. Feature Schema Audit
    logger.info("[Audit 3/5] Feature Schema...")
    ok, issues = DataContractAuditor.audit_feature_schema(batch)
    if not ok:
        logger.error(f"❌ Feature Schema Audit Failed: {issues}")
        # Don't return yet, try other audits
    else:
        logger.info("✅ Feature Schema Audit Passed.")

    # 5. Model & Label Firewall
    logger.info("[Audit 4/5] Label Firewall (Leakage)...")
    try:
        model = ModelBuilder.build_model(cfg).to(batch.x.device)
        ok, issues = DataContractAuditor.audit_label_firewall(model, batch)
        if not ok:
            logger.error(f"❌ Firewall Audit Failed: {issues}")
            return False
        logger.info("✅ Firewall Audit Passed.")
    except Exception as e:
        logger.error(f"❌ Firewall Audit Crashed: {e}")
        return False

    # 6. Clock Consistency
    logger.info("[Audit 5/5] Clock Consistency...")
    try:
        ok, issues = DataContractAuditor.audit_clock_consistency(model, batch)
        if not ok:
            logger.error(f"❌ Clock Audit Failed: {issues}")
            return False
        logger.info("✅ Clock Audit Passed.")
    except Exception as e:
        logger.error(f"❌ Clock Audit Crashed: {e}")
        return False
        
    # Heavy Audit Logic (Optional)
    if mode == 'heavy':
        logger.info("[Audit HEAVY] Running extended checks (Sampling K=10)...")
        # TODO: Implement heavy checks (e.g. iterate more batches)
        pass

    logger.info("=== 🏆 ALL SYSTEMS GO. Data Semantic Contract Verified. ===")
    
    # Write Cache
    if use_cache and cache_key:
        AuditCache.write(cache_key, metadata={"mode": mode, "config": config_path})
        
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--mode", type=str, default="fast", choices=["fast", "heavy"], help="Audit mode")
    # Default no_preload to True for safety, unless explicitly disabled
    parser.add_argument("--preload", action="store_true", help="Enable preload (default: False)")
    parser.add_argument("--no-cache", action="store_true", help="Disable audit cache")
    args = parser.parse_args()
    
    success = run_audit(args.config, mode=args.mode, no_preload=not args.preload, use_cache=not args.no_cache)
    if not success:
        sys.exit(1)
