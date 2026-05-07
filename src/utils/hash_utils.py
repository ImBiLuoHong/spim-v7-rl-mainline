import os
import hashlib
import logging

logger = logging.getLogger(__name__)

def generate_dir_fingerprint(data_dir: str) -> str:
    """
    生成目录的轻量级指纹哈希。
    只读取文件名、大小和修改时间，不读取文件内容。
    用于快速检测原始数据是否变更。
    """
    if not os.path.exists(data_dir):
        return "0" * 64

    meta_strings = []
    try:
        # 使用 os.walk 遍历所有文件
        # 为了保证顺序一致性，必须排序
        for root, _, files in os.walk(data_dir):
            for file in sorted(files):
                # 只关注数据文件，忽略隐藏文件和其他
                if not file.endswith('.npz'):
                    continue
                
                path = os.path.join(root, file)
                try:
                    stat = os.stat(path)
                    # 拼接：相对路径 + 大小 + mtime (转为整数以避免浮点精度问题)
                    rel_path = os.path.relpath(path, data_dir)
                    meta_strings.append(f"{rel_path}|{stat.st_size}|{int(stat.st_mtime)}")
                except OSError:
                    continue
                    
    except Exception as e:
        logger.warning(f"Hash generation failed: {e}")
        return "error"

    full_string = "".join(meta_strings)
    return hashlib.sha256(full_string.encode()).hexdigest()
