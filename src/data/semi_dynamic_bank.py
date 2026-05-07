from __future__ import annotations

import json
import logging
import os
import pickle
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import lmdb
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch, Data


logger = logging.getLogger(__name__)


def semi_dynamic_bank_collate_fn(data_list: List[Data]) -> Optional[Batch]:
    data_list = [d for d in data_list if d is not None]
    if not data_list:
        return None
    return Batch.from_data_list(data_list)


class SemiDynamicTrajectoryBankDataset(Dataset):
    def __init__(self, lmdb_path: str, transform=None):
        self.lmdb_path = lmdb_path
        self.transform = transform
        self.env = None
        self.txn = None
        self._length = 0
        env = lmdb.open(
            lmdb_path,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        with env.begin() as txn:
            self._length = txn.stat()["entries"]
        env.close()
        logger.info("[SemiDynamicBank] Initialized %s with %d entries.", lmdb_path, self._length)

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int):
        if self.env is None:
            self.env = lmdb.open(
                self.lmdb_path,
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
            )
            self.txn = self.env.begin(buffers=True)
        key = f"{idx}".encode("ascii")
        payload = self.txn.get(key)
        if payload is None:
            raise IndexError(f"LMDB key {idx} missing in {self.lmdb_path}")
        sample = pickle.loads(payload)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample

    def __getstate__(self):
        state = self.__dict__.copy()
        state["env"] = None
        state["txn"] = None
        return state


@dataclass
class SemiDynamicBankStats:
    case_count: int = 0
    trajectory_count: int = 0
    sample_count: int = 0
    total_steps: int = 0
    total_selected: int = 0
    unique_signature_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        avg_traj_per_case = self.trajectory_count / max(1, self.case_count)
        avg_steps_per_traj = self.total_steps / max(1, self.trajectory_count)
        avg_selected_per_step = self.total_selected / max(1, self.sample_count)
        avg_unique_signatures_per_case = self.unique_signature_count / max(1, self.case_count)
        return {
            "case_count": int(self.case_count),
            "trajectory_count": int(self.trajectory_count),
            "sample_count": int(self.sample_count),
            "avg_trajectories_per_case": float(avg_traj_per_case),
            "avg_steps_per_trajectory": float(avg_steps_per_traj),
            "avg_selected_per_step": float(avg_selected_per_step),
            "avg_unique_trajectory_signatures_per_case": float(avg_unique_signatures_per_case),
        }


class SemiDynamicTrajectoryBankWriter:
    def __init__(self, lmdb_path: Path, *, map_size: int = 256 * 1024 * 1024 * 1024, batch_commit: int = 200):
        self.lmdb_path = Path(lmdb_path)
        self.batch_commit = int(batch_commit)
        if self.lmdb_path.exists():
            if self.lmdb_path.is_dir():
                shutil.rmtree(self.lmdb_path)
            else:
                self.lmdb_path.unlink()
        self.lmdb_path.parent.mkdir(parents=True, exist_ok=True)
        self.env = lmdb.open(str(self.lmdb_path), map_size=int(map_size))
        self.txn = self.env.begin(write=True)
        self.count = 0

    def add(self, sample: Data) -> None:
        key = f"{self.count}".encode("ascii")
        self.txn.put(key, pickle.dumps(sample, protocol=pickle.HIGHEST_PROTOCOL))
        self.count += 1
        if self.count % self.batch_commit == 0:
            self.txn.commit()
            self.txn = self.env.begin(write=True)

    def close(self) -> int:
        if self.txn is not None:
            self.txn.commit()
            self.txn = None
        if self.env is not None:
            self.env.close()
            self.env = None
        return int(self.count)


def build_bank_sample(
    *,
    node_features: torch.Tensor,
    edge_index: torch.Tensor,
    graph_features: torch.Tensor,
    valid_mask: torch.Tensor,
    source_mask: torch.Tensor,
    case_id: int,
    trajectory_id: int,
    step_id: int,
    budget_used: float,
    t_sim_minutes: float,
) -> Data:
    data = Data(
        x=node_features.detach().cpu().float(),
        edge_index=edge_index.detach().cpu().long(),
        clean_aligned_graph_features=graph_features.detach().cpu().view(1, -1).float(),
        valid_mask=valid_mask.detach().cpu().bool(),
        source_mask=source_mask.detach().cpu().float(),
    )
    data.case_id = int(case_id)
    data.trajectory_id = int(trajectory_id)
    data.step_id = int(step_id)
    data.budget_used = float(budget_used)
    data.t_sim_minutes = float(t_sim_minutes)
    return data


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def path_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return int(path.stat().st_size)
    total = 0
    for child in path.rglob("*"):
        if child.is_file():
            total += int(child.stat().st_size)
    return total
