from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml


@dataclass
class APUDiffConfig:
    reid_dim: Union[int, str] = 2048
    latent_dim: int = 256
    history_len: int = 5
    projection_hidden_dim: int = 512
    denoiser_hidden_dim: int = 512
    update_hidden_dim: int = 512
    batch_size: int = 256
    num_workers: int = 0
    num_epochs_stage1: int = 30
    num_epochs_stage2: int = 20
    lr_stage1: float = 1e-4
    lr_stage2: float = 1e-4
    weight_decay: float = 1e-4
    stage1_mu_weight: float = 0.5
    stage1_diff_weight: float = 0.2
    stage1_metric_weight: float = 0.5
    stage1_improve_weight: float = 2.0
    stage1_improve_margin: float = 0.0
    stage1_margin: float = 0.2
    stage1_projection_warmup_epochs: int = 5
    stage1_supcon_weight: float = 1.0
    stage1_distill_weight: float = 1.0
    stage1_supcon_temperature: float = 0.1
    freeze_projection_after_warmup: bool = True
    num_diffusion_steps: int = 8
    time_dim: int = 64
    lambda_clean_prob: float = 0.5
    lambda_dirty_min: float = 0.3
    lambda_dirty_max: float = 0.8
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
    obs_clean_prob: float = 0.5
    obs_dirty_lambda_min: float = 0.3
    obs_dirty_lambda_max: float = 0.8
    identity_loss_weight: float = 0.5
    match_loss_weight: float = 0.2
    gate_loss_weight: float = 0.5
    match_temperature: float = 0.1
    match_margin: float = 0.05
    app_alpha: float = 0.5
    lost_decay: float = 0.8
    max_strong_app_lost_age: int = 5

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "APUDiffConfig":
        data = _flatten_config(data)
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
        with Path(path).open("r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"Config file must contain a mapping: {path}")
        data.update(loaded)
    data = _flatten_config(data)
    if overrides:
        data.update({k: v for k, v in overrides.items() if v is not None})
    return APUDiffConfig.from_dict(data)


def _flatten_config(data: Dict[str, Any]) -> Dict[str, Any]:
    data = dict(data)
    projection = data.pop("projection", {}) or {}
    predictor = data.pop("predictor", {}) or {}
    update_block = data.pop("update_block", {}) or {}
    training = data.pop("training", {}) or {}
    obs = data.pop("obs_synthesis", {}) or {}
    tracker = data.pop("tracker", {}) or {}

    mapping = {
        "projection_hidden_dim": projection.get("hidden_dim"),
        "num_diffusion_steps": predictor.get("num_diffusion_steps"),
        "time_dim": predictor.get("time_dim"),
        "denoiser_hidden_dim": predictor.get("denoiser_hidden_dim"),
        "update_hidden_dim": update_block.get("hidden_dim"),
        "batch_size": training.get("batch_size"),
        "num_epochs_stage1": training.get("stage1_epochs"),
        "num_epochs_stage2": training.get("stage2_epochs"),
        "lr_stage1": training.get("lr_stage1"),
        "lr_stage2": training.get("lr_stage2"),
        "weight_decay": training.get("weight_decay"),
        "stage1_mu_weight": training.get("stage1_mu_weight"),
        "stage1_diff_weight": training.get("stage1_diff_weight"),
        "stage1_metric_weight": training.get("stage1_metric_weight"),
        "stage1_improve_weight": training.get("stage1_improve_weight"),
        "stage1_improve_margin": training.get("stage1_improve_margin"),
        "stage1_margin": training.get("stage1_margin"),
        "stage1_projection_warmup_epochs": training.get("stage1_projection_warmup_epochs"),
        "stage1_supcon_weight": training.get("stage1_supcon_weight"),
        "stage1_distill_weight": training.get("stage1_distill_weight"),
        "stage1_supcon_temperature": training.get("stage1_supcon_temperature"),
        "freeze_projection_after_warmup": training.get("freeze_projection_after_warmup"),
        "gate_loss_weight": training.get("gate_loss_weight"),
        "match_temperature": training.get("match_temperature"),
        "match_margin": training.get("match_margin"),
        "obs_clean_prob": obs.get("clean_prob"),
        "obs_dirty_lambda_min": obs.get("dirty_lambda_min"),
        "obs_dirty_lambda_max": obs.get("dirty_lambda_max"),
        "app_alpha": tracker.get("app_alpha"),
        "lost_decay": tracker.get("lost_decay"),
        "max_strong_app_lost_age": tracker.get("max_strong_app_lost_age"),
    }
    for key, value in mapping.items():
        if value is not None:
            data[key] = value
    if "input_dim" in data and "reid_dim" not in data:
        data["reid_dim"] = data.pop("input_dim")
    return data
