from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml


@dataclass
class APUDiffConfig:
    reid_dim: Union[int, str] = 2048
    latent_dim: int = 2048
    history_len: int = 5
    identity_history_len: int = 32
    denoiser_hidden_dim: int = 512
    batch_size: int = 64
    num_workers: int = 0
    num_epochs_stage1: int = 30
    lr_stage1: float = 1e-4
    weight_decay: float = 1e-4
    stage1_diff_weight: float = 0.2
    stage1_improve_weight: float = 2.0
    ema_alpha: float = 0.9
    num_diffusion_steps: int = 1
    time_dim: int = 64
    val_fraction: float = 0.1
    split_mode: str = "sequence"
    val_sequences: Optional[List[str]] = None
    target_min_gap: int = 1
    target_max_gap: int = 1
    split_seed: int = 42
    seed: int = 42
    grad_clip: float = 1.0
    device: str = "cuda"
    log_interval: int = 20
    save_interval: int = 5
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    feature_path: Optional[str] = None
    feature_paths: Optional[List[str]] = None
    feature_dir: Optional[str] = None
    val_feature_dir: Optional[str] = None
    sequences: Optional[List[str]] = None
    normalize_input: bool = True

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "APUDiffConfig":
        data = _flatten_config(data)
        legacy_keys = {
            "projection_hidden_dim",
            "update_hidden_dim",
            "num_epochs_stage2",
            "lr_stage2",
            "stage1_mu_weight",
            "stage1_metric_weight",
            "stage1_improve_margin",
            "stage1_margin",
            "stage1_projection_warmup_epochs",
            "stage1_supcon_weight",
            "stage1_distill_weight",
            "stage1_supcon_temperature",
            "freeze_projection_after_warmup",
            "lambda_clean_prob",
            "lambda_dirty_min",
            "lambda_dirty_max",
            "obs_clean_prob",
            "obs_dirty_lambda_min",
            "obs_dirty_lambda_max",
            "identity_loss_weight",
            "match_loss_weight",
            "gate_loss_weight",
            "match_temperature",
            "match_margin",
            "app_alpha",
            "lost_decay",
            "max_strong_app_lost_age",
        }
        for key in legacy_keys:
            data.pop(key, None)
        valid = {f.name for f in fields(cls)}
        unknown = sorted(set(data) - valid)
        if unknown:
            raise ValueError(f"Unknown config keys: {unknown}")
        if data.get("feature_path") and not data.get("feature_paths"):
            data["feature_paths"] = [data["feature_path"]]
        return cls(**data)

    def to_dict(self) -> Dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}


def load_config(path: Optional[str] = None, overrides: Optional[Dict[str, Any]] = None) -> APUDiffConfig:
    data: Dict[str, Any] = {}
    if path:
        data.update(_load_yaml_with_bases(Path(path)))
    data = _flatten_config(data)
    if overrides:
        data.update({k: v for k, v in overrides.items() if v is not None})
    return APUDiffConfig.from_dict(data)


def _load_yaml_with_bases(path: Path, seen: Optional[set[Path]] = None) -> Dict[str, Any]:
    path = path.expanduser().resolve()
    seen = set() if seen is None else set(seen)
    if path in seen:
        chain = " -> ".join(str(p) for p in [*seen, path])
        raise ValueError(f"Recursive config base reference detected: {chain}")
    seen.add(path)

    with path.open("r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Config file must contain a mapping: {path}")

    base_refs = loaded.pop("base", None)
    if base_refs is None:
        base_refs = loaded.pop("bases", None)
    if base_refs is None:
        return loaded
    if isinstance(base_refs, (str, Path)):
        base_refs = [base_refs]
    if not isinstance(base_refs, list):
        raise ValueError(f"Config base must be a string or list in {path}")

    merged: Dict[str, Any] = {}
    for base_ref in base_refs:
        base_path = Path(base_ref)
        if not base_path.is_absolute():
            base_path = path.parent / base_path
        merged = _deep_update(merged, _load_yaml_with_bases(base_path, seen))
    return _deep_update(merged, loaded)


def _deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged


def _flatten_config(data: Dict[str, Any]) -> Dict[str, Any]:
    data = dict(data)
    data.pop("base", None)
    data.pop("bases", None)
    predictor = data.pop("predictor", {}) or {}
    training = data.pop("training", {}) or {}
    data.pop("projection", None)
    data.pop("update_block", None)
    data.pop("obs_synthesis", None)
    data.pop("tracker", None)

    mapping = {
        "num_diffusion_steps": predictor.get("num_diffusion_steps"),
        "time_dim": predictor.get("time_dim"),
        "denoiser_hidden_dim": predictor.get("denoiser_hidden_dim"),
        "batch_size": training.get("batch_size"),
        "num_epochs_stage1": training.get("stage1_epochs"),
        "lr_stage1": training.get("lr_stage1"),
        "weight_decay": training.get("weight_decay"),
        "stage1_diff_weight": training.get("stage1_diff_weight"),
        "stage1_improve_weight": training.get("stage1_improve_weight"),
        "ema_alpha": training.get("ema_alpha"),
    }
    for key, value in mapping.items():
        if value is not None:
            data[key] = value
    if "input_dim" in data and "reid_dim" not in data:
        data["reid_dim"] = data.pop("input_dim")
    return data
