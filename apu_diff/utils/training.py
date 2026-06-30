import logging
import os
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(device_name: str) -> torch.device:
    if device_name == "cuda" and not torch.cuda.is_available():
        logging.warning("CUDA requested but unavailable; falling back to CPU")
        return torch.device("cpu")
    return torch.device(device_name)


def setup_logging(log_dir: str, name: str) -> str:
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{name}.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
        force=True,
    )
    return log_file


def save_checkpoint(path: str, model, optimizer, epoch: int, val_loss: float, config, extra: Optional[dict] = None) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": epoch,
        "val_loss": val_loss,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "config": config.to_dict() if hasattr(config, "to_dict") else dict(config),
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_model_state(model, checkpoint_path: str, strict: bool = True):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state = checkpoint.get("model_state_dict", checkpoint)
    load_result = model.load_state_dict(state, strict=strict)
    if not strict and (load_result.missing_keys or load_result.unexpected_keys):
        logging.warning(
            "Loaded %s with key mismatch: missing=%s unexpected=%s",
            checkpoint_path,
            list(load_result.missing_keys),
            list(load_result.unexpected_keys),
        )
    return checkpoint
