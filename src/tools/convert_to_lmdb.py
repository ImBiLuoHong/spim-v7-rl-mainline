import os
import lmdb
import logging
import shutil
from tqdm import tqdm
from src.data.v6.lmdb_codec import encode_lmdb_payload
from src.utils.hash_utils import generate_dir_fingerprint

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _resolve_lmdb_map_size_bytes(output_dir):
    """Choose a generous logical map size without tightly coupling it to current free space."""
    gib = 1024 ** 3
    preferred_map = 64 * gib
    max_map = 1024 * gib
    return min(max_map, preferred_map)

def convert_to_lmdb(dataset, mode='train', output_dir=None, cache_version='v1'):
    """
    将给定的 dataset (Raw NpzDatasetV6) 转换为 LMDB 格式。
    单进程、极致瘦身、分批提交。
    """
    if output_dir is None:
        output_dir = dataset.samples_dir
    os.makedirs(output_dir, exist_ok=True)
        
    lmdb_path = os.path.join(output_dir, f"v6_dataset_{mode}_{cache_version}.lmdb")
    hash_path = os.path.join(output_dir, f"v6_dataset_{mode}_{cache_version}.hash")
    
    logger.info(f"Starting LMDB conversion for {mode}...")
    logger.info(f"Target: {lmdb_path}")
    
    # Pick a map size that fits the actual target filesystem instead of assuming 1TB.
    map_size = _resolve_lmdb_map_size_bytes(output_dir)
    logger.info("Using LMDB map_size=%d bytes (%.2f GiB)", map_size, map_size / (1024 ** 3))
    
    # 清理旧文件
    if os.path.exists(lmdb_path):
        if os.path.isdir(lmdb_path):
            shutil.rmtree(lmdb_path)
        else:
            os.remove(lmdb_path)
            
    # 打开 LMDB 环境
    env = lmdb.open(lmdb_path, map_size=map_size)
    
    count = 0
    batch_size = 200 # Reduced batch size for smaller transactions
    
    try:
        txn = env.begin(write=True)
        for i in tqdm(range(len(dataset)), desc=f"Converting {mode} to LMDB"):
            try:
                data = dataset[i]
                if data is None: continue
                
                # === 极致瘦身 (CRITICAL) ===
                # [FIX] Phase 4.5 requires x_raw for dynamic simulation (EpisodeStepper).
                # We must PRESERVE x_raw.
                # if hasattr(data, 'x_raw'):
                #    del data.x_raw
                # if hasattr(data, 'x_raw_signal'):
                #    del data.x_raw_signal
                
                # Use a light compression codec so the full-train cache fits repo-local disk.
                key = f"{i}".encode('ascii')
                val = encode_lmdb_payload(data)
                txn.put(key, val)
                
                count += 1
                
                # 分批提交，防止内存积压
                if count % batch_size == 0:
                    txn.commit()
                    txn = env.begin(write=True)
                    
            except Exception as e:
                logger.error(f"Failed to convert sample {i}: {e}")
                txn.abort()
                raise
        
        # 提交剩余数据
        txn.commit()
        
    except Exception as e:
        logger.error(f"LMDB conversion failed: {e}")
        env.close()
        # 清理可能损坏的文件
        if os.path.exists(lmdb_path):
            shutil.rmtree(lmdb_path)
        return False
        
    env.close()
    
    # 生成 Hash 并写入
    try:
        dir_hash = generate_dir_fingerprint(dataset.samples_dir)
        with open(hash_path, 'w') as f:
            f.write(dir_hash)
        logger.info(f"LMDB conversion complete: {count} samples written.")
        logger.info(f"Hash stored in {hash_path}: {dir_hash}")
        return True
    except Exception as e:
        logger.error(f"Hash generation failed: {e}")
        return False
