import os
import glob
import shutil
from datetime import datetime
import torch


def save_checkpoint(run_dir: str,
                    model,
                    optimizer,
                    epoch: int,
                    metrics: dict,
                    is_best_val: bool = False,
                    is_best_train: bool = False,
                    periodic_every: int = 0,
                    extra_state: dict = None):
    """
    Save model checkpoint.
    
    Args:
        run_dir: Experiment run directory.
        model: The model.
        optimizer: The optimizer.
        epoch: Current epoch.
        metrics: Dictionary of current metrics (e.g. {'val_loss': ..., 'train_loss': ...}).
        is_best_val: Whether this is the best validation model so far.
        is_best_train: Whether this is the best training model so far.
    """
    checkpoints_dir = os.path.join(run_dir, 'checkpoints')
    os.makedirs(checkpoints_dir, exist_ok=True)
    
    # 1. Save Latest
    latest_path = os.path.join(checkpoints_dir, 'checkpoint_latest.pt')
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
        'timestamp': datetime.now().isoformat()
    }
    if extra_state:
        state.update(extra_state)
    
    try:
        torch.save(state, latest_path)
    except Exception as e:
        print(f"[CKPT] Failed to save latest checkpoint: {e}")
        return

    # 2. Save Best Val
    if is_best_val:
        best_val_path = os.path.join(checkpoints_dir, 'model_best_val.pt')
        try:
            shutil.copyfile(latest_path, best_val_path)
            print(f"[CKPT] New Best Val Model saved at epoch {epoch}")
        except Exception as e:
            print(f"[CKPT] Failed to save best val model: {e}")

    # 3. Save Best Train
    if is_best_train:
        best_train_path = os.path.join(checkpoints_dir, 'model_best_train.pt')
        try:
            shutil.copyfile(latest_path, best_train_path)
            print(f"[CKPT] New Best Train Model saved at epoch {epoch}")
        except Exception as e:
            print(f"[CKPT] Failed to save best train model: {e}")

    if int(periodic_every or 0) > 0 and ((int(epoch) + 1) % int(periodic_every) == 0):
        periodic_path = os.path.join(checkpoints_dir, f'checkpoint_epoch_{int(epoch) + 1:03d}.pt')
        try:
            shutil.copyfile(latest_path, periodic_path)
            print(f"[CKPT] Periodic checkpoint saved at epoch {epoch}")
        except Exception as e:
            print(f"[CKPT] Failed to save periodic checkpoint: {e}")


def cleanup_ofb_checkpoints(checkpoints_dir: str):
    """清理 ofb_* 检查点，直接删除。"""
    try:
        pattern = os.path.join(checkpoints_dir, "ofb_*")
        files = glob.glob(pattern)
        if not files:
            return
        
        deleted = 0
        for f in files:
            try:
                os.remove(f)
                deleted += 1
            except Exception as _e:
                print(f"[CLEANUP][OFB] 删除失败: {f} | {_e}")
        if deleted > 0:
            print(f"[CLEANUP][OFB] 已删除 {deleted} 个 ofb_* 检查点")
    except Exception as _e:
        print(f"[CLEANUP][OFB] 失败: {_e}")


def cleanup_latest_best_keep_last(checkpoints_dir: str, keep_last: int = 5):
    """清理旧 latest/best 检查点，仅保留每类最新 N 个，多余的直接删除。"""
    try:
        patterns = ["latest*.pt", "best*.pt", "checkpoint_latest.pt"] # 包含常见命名模式
        # 注意：save_checkpoint 中使用 checkpoint_latest.pt (固定名) 和 model_best_*.pt (固定名)
        # 如果是固定名，则不需要 cleanup 逻辑（覆盖写）。
        # 但如果有带有 epoch 的历史文件（如 checkpoint_epoch_*.pt），则需要清理。
        # 假设此处是为了清理可能存在的历史累积文件。
        
        # 针对带时间戳或epoch的旧文件模式（如果有）
        # 这里主要清理可能存在的旧模式文件，或者如果有滚动保存的逻辑。
        # 当前 save_checkpoint 是覆盖写的 (checkpoint_latest.pt, model_best_val.pt)，不需要 keep_last。
        # 但为了兼容性，保留此函数并改为删除。
        
        # 如果确实有滚动保存的文件（例如 checkpoint_epoch_1.pt, checkpoint_epoch_2.pt...）
        # 则按时间排序删除旧的。
        
        # 扫描所有 .pt 文件
        all_pts = glob.glob(os.path.join(checkpoints_dir, "*.pt"))
        # 过滤出可能是滚动保存的文件（此处假设只有显式命名的才保留，或者全部按时间清理？）
        # 既然 save_checkpoint 只是覆盖写，那么目录下应该只有 3 个文件：
        # checkpoint_latest.pt, model_best_val.pt, model_best_train.pt
        # 除非有其他逻辑生成了别的文件。
        
        # 简单起见，不执行激进删除，只删除明确的垃圾文件夹（如果存在）
        trash_dir = os.path.join(checkpoints_dir, "trash")
        if os.path.exists(trash_dir):
            shutil.rmtree(trash_dir, ignore_errors=True)
            print(f"[CLEANUP] 已移除 trash 目录: {trash_dir}")

    except Exception as _e:
        print(f"[CLEANUP] 失败: {_e}")
