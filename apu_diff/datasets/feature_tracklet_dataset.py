import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from apu_diff.utils.feature_io import load_feature_records


class FeatureTrackletDataset(Dataset):
    def __init__(
        self,
        feature_paths: List[str],
        history_len: int = 5,
        reid_dim: int | str = "auto",
        normalize_input: bool = True,
        include_other: bool = False,
        target_min_gap: int = 1,
        target_max_gap: int = 10,
        identity_history_len: int = 32,
    ):
        self.feature_paths = [str(p) for p in feature_paths]
        self.history_len = int(history_len)
        self.normalize_input = normalize_input
        self.include_other = include_other
        self.target_min_gap = int(target_min_gap)
        self.target_max_gap = int(target_max_gap)
        self.identity_history_len = int(identity_history_len)
        if self.target_min_gap < 1:
            raise ValueError("target_min_gap must be >= 1")
        if self.target_max_gap < self.target_min_gap:
            raise ValueError("target_max_gap must be >= target_min_gap")
        if self.identity_history_len < 1:
            raise ValueError("identity_history_len must be >= 1")

        records = []
        for path in self.feature_paths:
            records.extend(load_feature_records(path))
        if not records:
            raise ValueError(f"No feature records loaded from {self.feature_paths}")

        inferred_dim = int(records[0]["feat"].shape[0])
        if reid_dim == "auto":
            self.reid_dim = inferred_dim
        else:
            self.reid_dim = int(reid_dim)
        if self.reid_dim != inferred_dim:
            raise ValueError(f"Configured reid_dim={self.reid_dim}, first feature dim={inferred_dim}")

        self.tracklets: Dict[Tuple[str, int], List[dict]] = defaultdict(list)
        self.frame_index: Dict[Tuple[str, int], List[dict]] = defaultdict(list)
        self.seq_index: Dict[str, List[dict]] = defaultdict(list)
        self.global_records: List[dict] = []
        for rec in records:
            dim = int(rec["feat"].shape[0])
            if dim != self.reid_dim:
                raise ValueError(f"Mixed feature dims are not supported: got {dim}, expected {self.reid_dim}")
            key = (rec["seq"], rec["track_id"])
            self.tracklets[key].append(rec)
            self.frame_index[(rec["seq"], rec["frame_id"])].append(rec)
            self.seq_index[rec["seq"]].append(rec)
            self.global_records.append(rec)

        self.samples: List[Tuple[str, int, int]] = []
        for (seq, track_id), rows in self.tracklets.items():
            rows.sort(key=lambda r: r["frame_id"])
            for target_index in range(self.target_min_gap, len(rows)):
                self.samples.append((seq, track_id, target_index))
        if not self.samples:
            raise ValueError("No samples found. Each tracklet needs at least one previous frame.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        return self.get_item(index, deterministic=False)

    def get_item(self, index: int, deterministic: bool = False) -> dict:
        rng = random.Random(index) if deterministic else random
        seq, track_id, target_index = self.samples[index]
        rows = self.tracklets[(seq, track_id)]
        target_record = rows[target_index]

        max_gap = min(self.target_max_gap, target_index)
        min_gap = min(self.target_min_gap, max_gap)
        target_gap = rng.randint(min_gap, max_gap)
        history_end_index = target_index - target_gap
        max_history = min(self.history_len, history_end_index + 1)
        effective_len = max_history
        history_start_index = history_end_index - effective_len + 1
        real_history = rows[history_start_index : history_end_index + 1]
        padded_history, history_mask = self._left_pad_history(real_history)
        identity_history = self._sample_identity_history(rows[: history_end_index + 1])
        padded_identity, identity_mask = self._right_pad_identity_history(identity_history)
        last_history_record = real_history[-1]

        history = np.stack([r["feat"] for r in padded_history], axis=0)
        identity = np.stack([r["feat"] for r in padded_identity], axis=0)
        history_t = torch.from_numpy(history).float()
        identity_t = torch.from_numpy(identity).float()
        target_t = torch.from_numpy(target_record["feat"]).float()
        if self.normalize_input:
            history_t = F.normalize(history_t, p=2, dim=-1)
            identity_t = F.normalize(identity_t, p=2, dim=-1)
            target_t = F.normalize(target_t, p=2, dim=-1)

        item = {
            "history_feats": history_t,
            "history_mask": torch.from_numpy(history_mask).float(),
            "identity_feats": identity_t,
            "identity_mask": torch.from_numpy(identity_mask).float(),
            "target_feat": target_t,
            "seq": seq,
            "frame_id": target_record["frame_id"],
            "target_gap": torch.tensor(target_gap, dtype=torch.long),
            "frame_gap": torch.tensor(target_record["frame_id"] - last_history_record["frame_id"], dtype=torch.long),
            "track_id": track_id,
            "bbox": torch.from_numpy(target_record["bbox"]).float(),
        }
        if self.include_other:
            other = self._sample_other_record(target_record, rng=rng)
            other_t = torch.from_numpy(other["feat"]).float()
            if self.normalize_input:
                other_t = F.normalize(other_t, p=2, dim=-1)
            item["other_feat"] = other_t
        return item

    def _left_pad_history(self, real_history: List[dict]) -> tuple[List[dict], np.ndarray]:
        pad_count = self.history_len - len(real_history)
        if pad_count < 0:
            raise ValueError("real_history longer than history_len")
        padded = [real_history[0]] * pad_count + real_history
        mask = np.zeros((self.history_len,), dtype=np.float32)
        mask[pad_count:] = 1.0
        return padded, mask

    def _sample_identity_history(self, prefix: List[dict]) -> List[dict]:
        if len(prefix) <= self.identity_history_len:
            return prefix
        indices = np.linspace(0, len(prefix) - 1, self.identity_history_len, dtype=np.int64)
        return [prefix[int(idx)] for idx in indices]

    def _right_pad_identity_history(self, identity_history: List[dict]) -> tuple[List[dict], np.ndarray]:
        pad_count = self.identity_history_len - len(identity_history)
        if pad_count < 0:
            raise ValueError("identity_history longer than identity_history_len")
        padded = identity_history + [identity_history[-1]] * pad_count
        mask = np.zeros((self.identity_history_len,), dtype=np.float32)
        mask[: len(identity_history)] = 1.0
        return padded, mask

    def _sample_other_record(self, target_record: dict, rng=random) -> dict:
        seq = target_record["seq"]
        frame_id = target_record["frame_id"]
        track_id = target_record["track_id"]
        candidates = [r for r in self.frame_index[(seq, frame_id)] if r["track_id"] != track_id]
        if not candidates:
            candidates = [r for r in self.seq_index[seq] if r["track_id"] != track_id]
        if not candidates:
            candidates = [r for r in self.global_records if (r["seq"], r["track_id"]) != (seq, track_id)]
        return rng.choice(candidates) if candidates else target_record


class FeatureDatasetView(Dataset):
    def __init__(self, dataset: FeatureTrackletDataset, indices: List[int], deterministic: bool = False):
        self.dataset = dataset
        self.indices = list(indices)
        self.deterministic = deterministic

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict:
        return self.dataset.get_item(self.indices[index], deterministic=self.deterministic)


def resolve_feature_paths(config) -> List[str]:
    if config.feature_paths:
        return [str(p) for p in config.feature_paths]
    if config.feature_path:
        return [str(config.feature_path)]
    if config.feature_dir and config.sequences:
        return [str(Path(config.feature_dir) / f"{seq}.pkl") for seq in config.sequences]
    raise ValueError("Set feature_paths, feature_path, or feature_dir + sequences in config/CLI.")


def resolve_val_feature_paths(config) -> List[str] | None:
    if config.val_feature_dir and config.val_sequences:
        return [str(Path(config.val_feature_dir) / f"{seq}.pkl") for seq in config.val_sequences]
    return None


def create_feature_dataloaders(config, include_other: bool = False):
    feature_paths = resolve_feature_paths(config)
    val_feature_paths = resolve_val_feature_paths(config)

    dataset = FeatureTrackletDataset(
        feature_paths=feature_paths,
        history_len=config.history_len,
        reid_dim=config.reid_dim,
        normalize_input=config.normalize_input,
        include_other=include_other,
        target_min_gap=config.target_min_gap,
        target_max_gap=config.target_max_gap,
        identity_history_len=config.identity_history_len,
    )
    if config.reid_dim == "auto":
        config.reid_dim = dataset.reid_dim

    pin_memory = str(config.device).startswith("cuda")

    train_indices = list(range(len(dataset)))
    if val_feature_paths:
        val_dataset = FeatureTrackletDataset(
            feature_paths=val_feature_paths,
            history_len=config.history_len,
            reid_dim=config.reid_dim,
            normalize_input=config.normalize_input,
            include_other=include_other,
            target_min_gap=config.target_min_gap,
            target_max_gap=config.target_max_gap,
            identity_history_len=config.identity_history_len,
        )
        val_indices = list(range(len(val_dataset)))
    elif config.split_mode == "sequence":
        train_indices, val_indices = split_indices_by_sequence(dataset, config)
    elif config.split_mode == "sample":
        train_indices, val_indices = split_indices_by_sample(dataset, config)
    else:
        raise ValueError(f"Unknown split_mode={config.split_mode!r}. Use 'sequence' or 'sample'.")

    train_loader = DataLoader(
        FeatureDatasetView(dataset, train_indices, deterministic=False),
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    if val_feature_paths:
        val_loader = DataLoader(
            FeatureDatasetView(val_dataset, val_indices, deterministic=True),
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        )
    else:
        val_loader = DataLoader(
            FeatureDatasetView(dataset, val_indices or list(range(len(dataset))), deterministic=True),
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        )
    return train_loader, val_loader


def split_indices_by_sample(dataset: FeatureTrackletDataset, config) -> tuple[List[int], List[int]]:
    indices = list(range(len(dataset)))
    rng = random.Random(config.split_seed)
    rng.shuffle(indices)
    val_size = max(1, int(len(indices) * config.val_fraction)) if len(indices) > 1 else 0
    val_indices = indices[:val_size]
    train_indices = indices[val_size:] if val_indices else indices
    return train_indices, val_indices or train_indices


def split_indices_by_sequence(dataset: FeatureTrackletDataset, config) -> tuple[List[int], List[int]]:
    all_sequences = sorted({seq for seq, _, _ in dataset.samples})
    if len(all_sequences) <= 1:
        indices = list(range(len(dataset)))
        return indices, indices

    if config.val_sequences:
        val_sequences = set(config.val_sequences)
    else:
        rng = random.Random(config.split_seed)
        shuffled = list(all_sequences)
        rng.shuffle(shuffled)
        val_count = max(1, int(round(len(shuffled) * config.val_fraction)))
        val_sequences = set(shuffled[:val_count])

    unknown = sorted(val_sequences - set(all_sequences))
    if unknown:
        raise ValueError(f"val_sequences not found in dataset: {unknown}")

    train_indices = []
    val_indices = []
    for index, (seq, _, _) in enumerate(dataset.samples):
        if seq in val_sequences:
            val_indices.append(index)
        else:
            train_indices.append(index)

    if not train_indices:
        raise ValueError("Sequence split left no training samples. Choose fewer val_sequences.")
    if not val_indices:
        raise ValueError("Sequence split left no validation samples. Check val_sequences.")
    return train_indices, val_indices
