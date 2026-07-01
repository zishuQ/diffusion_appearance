import argparse
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from apu_diff.config import load_config
from apu_diff.datasets.feature_tracklet_dataset import FeatureDatasetView, FeatureTrackletDataset
from apu_diff.models import APUDiff
from apu_diff.utils.training import get_device, load_model_state, set_seed, setup_logging
from train_stage1_predictor import run_epoch


def parse_args():
    parser = argparse.ArgumentParser("Evaluate APUDiff Stage 1 predictor by sequence")
    parser.add_argument("--config", default="configs/apu_diff_default.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/stage1_predictor_best_improve.pth")
    parser.add_argument("--feature-dir", default=None)
    parser.add_argument("--sequences", nargs="+", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--target-min-gap", type=int, default=None)
    parser.add_argument("--target-max-gap", type=int, default=None)
    parser.add_argument("--log-name", default="eval_stage1_by_sequence")
    return parser.parse_args()


def make_loader(config, sequence: str | None, num_workers: int):
    if sequence is None:
        if not config.sequences:
            raise ValueError("config.sequences is required for all-sequence evaluation")
        feature_paths = [str(Path(config.feature_dir) / f"{seq}.pkl") for seq in config.sequences]
    else:
        feature_paths = [str(Path(config.feature_dir) / f"{sequence}.pkl")]

    dataset = FeatureTrackletDataset(
        feature_paths=feature_paths,
        history_len=config.history_len,
        reid_dim=config.reid_dim,
        normalize_input=config.normalize_input,
        include_other=True,
        target_min_gap=config.target_min_gap,
        target_max_gap=config.target_max_gap,
    )
    if config.reid_dim == "auto":
        config.reid_dim = dataset.reid_dim
    pin_memory = str(config.device).startswith("cuda")
    return DataLoader(
        FeatureDatasetView(dataset, list(range(len(dataset))), deterministic=True),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )


def build_model(config, device):
    model = APUDiff(
        reid_dim=config.reid_dim,
        latent_dim=config.latent_dim,
        time_dim=config.time_dim,
        num_diffusion_steps=config.num_diffusion_steps,
        denoiser_hidden_dim=config.denoiser_hidden_dim,
    ).to(device)
    return model


@torch.no_grad()
def evaluate_one(model, loader, device, config):
    return run_epoch(
        model,
        loader,
        optimizer=None,
        device=device,
        config=config,
        train=False,
        show_progress=False,
    )


def print_summary(name: str, metrics: dict):
    keys = [
        "cos_pred",
        "cos_last",
        "cos_ema",
        "cos_pred_minus_last",
        "loss",
        "target_gap",
    ]
    summary = {key: metrics[key] for key in keys if key in metrics}
    print(f"{name}: {summary}")
    logging.info("%s %s", name, metrics)


def main():
    args = parse_args()
    overrides = {
        "feature_dir": args.feature_dir,
        "sequences": args.sequences,
        "device": args.device,
        "batch_size": args.batch_size,
        "target_min_gap": args.target_min_gap,
        "target_max_gap": args.target_max_gap,
    }
    config = load_config(args.config, overrides)
    if not config.feature_dir:
        raise ValueError("feature_dir must be set in config or CLI")
    if not config.sequences:
        raise ValueError("sequences must be set in config or CLI")

    set_seed(config.seed)
    log_file = setup_logging(config.log_dir, args.log_name)
    device = get_device(config.device)
    config.device = str(device)
    logging.info("Log file: %s", log_file)
    logging.info("Checkpoint: %s", args.checkpoint)
    logging.info("Config: %s", config)

    warmup_loader = make_loader(config, config.sequences[0], args.num_workers)
    model = build_model(config, device)
    load_model_state(model, args.checkpoint, strict=False)
    model.eval()
    del warmup_loader

    for sequence in config.sequences:
        loader = make_loader(config, sequence, args.num_workers)
        metrics = evaluate_one(model, loader, device, config)
        print_summary(sequence, metrics)

    all_loader = make_loader(config, None, args.num_workers)
    all_metrics = evaluate_one(model, all_loader, device, config)
    print_summary("ALL", all_metrics)


if __name__ == "__main__":
    main()
