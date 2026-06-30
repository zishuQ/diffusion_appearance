import json
import pickle
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np


REQUIRED_KEYS = ("seq", "frame_id", "track_id", "bbox", "feat")


def load_feature_records(path: str) -> List[Dict[str, Any]]:
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix in {".pkl", ".pickle"}:
        with p.open("rb") as f:
            obj = pickle.load(f)
        records = _records_from_object(obj)
    elif suffix == ".json":
        with p.open("r", encoding="utf-8") as f:
            records = _records_from_object(json.load(f))
    elif suffix == ".jsonl":
        records = []
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    elif suffix == ".npz":
        records = _records_from_npz(p)
    elif suffix in {".pt", ".pth"}:
        import torch

        obj = torch.load(p, map_location="cpu")
        records = _records_from_object(obj)
    else:
        raise ValueError(f"Unsupported feature file extension: {p.suffix}")
    return [_normalize_record(r) for r in records]


def _records_from_object(obj: Any) -> List[Dict[str, Any]]:
    if isinstance(obj, list):
        return obj
    if isinstance(obj, tuple):
        return list(obj)
    if isinstance(obj, dict):
        if all(k in obj for k in ("seq", "frame_id", "track_id")) and ("feat" in obj or "features" in obj):
            return _records_from_column_dict(obj)
        if "records" in obj:
            return _records_from_object(obj["records"])
        if all(isinstance(v, (list, tuple)) for v in obj.values()):
            rows = []
            for seq, seq_rows in obj.items():
                for row in seq_rows:
                    rec = dict(row)
                    rec.setdefault("seq", seq)
                    rows.append(rec)
            return rows
    raise ValueError(
        "Unsupported feature object. Expected a list of records, a column dict, "
        "or {'records': [...]}. TrackTrack detection-feature pickles do not contain "
        "GT track_id by default and must be converted to records first."
    )


def _records_from_column_dict(obj: Dict[str, Any]) -> List[Dict[str, Any]]:
    feats = obj.get("feat", obj.get("features"))
    n = len(feats)
    records = []
    for i in range(n):
        records.append(
            {
                "seq": obj["seq"][i],
                "frame_id": obj["frame_id"][i],
                "track_id": obj["track_id"][i],
                "bbox": obj["bbox"][i],
                "feat": feats[i],
            }
        )
    return records


def _records_from_npz(path: Path) -> List[Dict[str, Any]]:
    data = np.load(path, allow_pickle=True)
    keys = set(data.files)
    feat_key = "feat" if "feat" in keys else "features"
    required = {"seq", "frame_id", "track_id", "bbox", feat_key}
    missing = sorted(required - keys)
    if missing:
        raise ValueError(f"NPZ feature file missing keys: {missing}")
    return _records_from_column_dict({k: data[k] for k in required})


def _normalize_record(record: Dict[str, Any]) -> Dict[str, Any]:
    if "features" in record and "feat" not in record:
        record = dict(record)
        record["feat"] = record.pop("features")
    missing = [k for k in REQUIRED_KEYS if k not in record]
    if missing:
        raise ValueError(f"Feature record missing keys {missing}: {record.keys()}")
    feat = np.asarray(record["feat"], dtype=np.float32)
    if feat.ndim != 1:
        raise ValueError(f"Expected feature shape [D], got {feat.shape}")
    return {
        "seq": str(record["seq"]),
        "frame_id": int(record["frame_id"]),
        "track_id": int(record["track_id"]),
        "bbox": np.asarray(record["bbox"], dtype=np.float32),
        "feat": feat,
    }
