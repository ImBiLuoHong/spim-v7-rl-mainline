import lmdb
import torch
from torch.utils.data import Dataset
import logging
from src.data.v6.lmdb_codec import decode_lmdb_payload

logger = logging.getLogger(__name__)

class LmdbDatasetV6(Dataset):
    """
    终极懒加载 Dataset (LMDB)
    彻底解决多进程 Worker 共享 LMDB 句柄导致的死锁和 I/O 阻塞。
    """
    def __init__(self, lmdb_path, transform=None):
        self.lmdb_path = lmdb_path
        self.transform = transform
        self.env = None
        self.txn = None
        self._length = 0
        
        # 预先获取长度
        try:
            # 这里的 open 只是为了读一下 stat，读完立即关掉
            # 必须设置 lock=False 以避免产生锁文件残留？
            # 只要是 readonly，lock=False 是安全的且推荐的
            env = lmdb.open(
                lmdb_path, 
                readonly=True, 
                lock=False, 
                readahead=False, 
                meminit=False
            )
            with env.begin() as txn:
                self._length = txn.stat()['entries']
            env.close()
            logger.info(f"[LmdbDataset] Initialized {lmdb_path} with {self._length} entries.")
        except Exception as e:
            logger.error(f"[LmdbDataset] Failed to initialize LMDB {lmdb_path}: {e}")
            self._length = 0

    def __len__(self):
        return self._length

    def __getitem__(self, idx):
        # 懒加载（Lazy Init）：在 __getitem__ 中进行判断
        # 确保每个 Worker 进程有自己独立的 env 句柄
        if self.env is None:
            self.env = lmdb.open(
                self.lmdb_path, 
                readonly=True, 
                lock=False, 
                readahead=False, 
                meminit=False
            )
            self.txn = self.env.begin(buffers=True)
        
        # 键是 ascii 编码的索引
        key = f"{idx}".encode('ascii')
        val_bytes = self.txn.get(key)
        
        if val_bytes is None:
            # 可能是索引越界或数据缺失
            # 尝试做一点容错？或者直接报错
            # 如果是随机访问，idx 应该在范围内。
            # 如果 LMDB 数据不完整（比如写入中断），这里会 None。
            raise IndexError(f"LMDB key {idx} not found in {self.lmdb_path}")
            
        # 反序列化
        try:
            data = decode_lmdb_payload(val_bytes)
        except Exception as e:
            raise RuntimeError(f"Failed to unpickle data at index {idx}: {e}")
        
        if self.transform:
            data = self.transform(data)
            
        return data

    def __getstate__(self):
        """
        Pickling support for multi-processing.
        When passing dataset to workers, we must NOT pickle the env/txn.
        """
        state = self.__dict__.copy()
        state['env'] = None
        state['txn'] = None
        return state
