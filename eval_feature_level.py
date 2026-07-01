import argparse
import logging

import torch
import torch.nn.functional as F
from tqdm import tqdm

from apu_diff.config import load_config
from apu_diff.datasets import create_feature_dataloaders
from apu_diff.models import APUDiff
from apu_diff.utils.metrics import compute_ema, last_real_feature
from apu_diff.utils.training import get_device, load_model_state, set_seed, setup_logging


def parse_args():
    parser = argparse.ArgumentParser("Evaluate APUDiff feature-level metrics")
    parser.add_argument("--config", default="configs/apu_diff_default.yaml")
    parser.add_argument("--feature-path", default=None)
    parser.add_argument("--feature-dir", default=None)
    parser.add_argument("--sequences", nargs="+", default=None)
    parser.add_argument("--checkpoint", default="checkpoints/apu_diff_full.pth")
    parser.add_argument("--split", choices=["train", "val"], default="val")
    parser.add_argument("--device", default=None)
    parser.add_argument("--val-feature-dir", default=None)
    parser.add_argument("--val-sequences", nargs="+", default=None)
    parser.add_argument("--split-mode", default=None)
    parser.add_argument("--log-dir", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--target-min-gap", type=int, default=None)
    parser.add_argument("--target-max-gap", type=int, default=None)
    parser.add_argument("--ema-alpha", type=float, default=None)
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--log-name", default="eval_feature_level")
    return parser.parse_args()


def _rank_accuracy(anchor: torch.Tensor, target_z: torch.Tensor, other_z: torch.Tensor) -> torch.Tensor:
    pos_cost = 1.0 - F.cosine_similarity(anchor, target_z, dim=-1)
    neg_cost = 1.0 - F.cosine_similarity(anchor, other_z, dim=-1)
    return (pos_cost < neg_cost).float().mean()


@torch.no_grad()
def main():
    args = parse_args()
    config = load_config(
        args.config,
        {
            "feature_path": args.feature_path,
            "feature_dir": args.feature_dir,
            "sequences": args.sequences,
            "val_feature_dir": args.val_feature_dir,
            "val_sequences": args.val_sequences,
            "split_mode": args.split_mode,
            "device": args.device,
            "batch_size": args.batch_size,
            "target_min_gap": args.target_min_gap,
            "target_max_gap": args.target_max_gap,
            "ema_alpha": args.ema_alpha,
            "log_dir": args.log_dir,
        },
    )
    set_seed(config.seed)
    setup_logging(config.log_dir, args.log_name)
    device = get_device(config.device)
    config.device = str(device)
    train_loader, val_loader = create_feature_dataloaders(config, include_other=True)
    loader = train_loader if args.split == "train" else val_loader

    model = APUDiff(
        reid_dim=config.reid_dim,
        latent_dim=config.latent_dim,
        time_dim=config.time_dim,
        num_diffusion_steps=config.num_diffusion_steps,
        denoiser_hidden_dim=config.denoiser_hidden_dim,
    ).to(device)
    load_model_state(model, args.checkpoint, strict=False)
    model.eval()

    totals = {
        "cos_pred": 0.0,
        "cos_ema": 0.0,
        "cos_last": 0.0,
        "loss_pred": 0.0,
        "loss_ema": 0.0,
        "loss_last": 0.0,
        "cos_pred_minus_ema": 0.0,
        "cos_pred_minus_last": 0.0,
        "rank_acc_pred": 0.0,
        "rank_acc_ema": 0.0,
        "rank_acc_last": 0.0,
    }
    n = 0
    for batch in tqdm(loader, desc=f"eval_{args.split}"):
        history = batch["history_feats"].to(device, non_blocking=True)
        history_mask = batch["history_mask"].to(device, non_blocking=True)
        target = batch["target_feat"].to(device, non_blocking=True)
        other = batch["other_feat"].to(device, non_blocking=True)

        local_queue = model.project(history)
        target_z = model.project(target)
        other_z = model.project(other)
        pred_feat = model.predict(local_queue, history_mask, deterministic=True)

        last_z = last_real_feature(local_queue, history_mask)
        ema_feat = compute_ema(local_queue, history_mask, alpha=config.ema_alpha)

        pred_cos = F.cosine_similarity(pred_feat, target_z, dim=-1)
        ema_cos = F.cosine_similarity(ema_feat, target_z, dim=-1)
        last_cos = F.cosine_similarity(last_z, target_z, dim=-1)

        rank_pred = _rank_accuracy(pred_feat, target_z, other_z)
        rank_ema = _rank_accuracy(ema_feat, target_z, other_z)
        rank_last = _rank_accuracy(last_z, target_z, other_z)

        totals["cos_pred"] += pred_cos.mean().item()
        totals["cos_ema"] += ema_cos.mean().item()
        totals["cos_last"] += last_cos.mean().item()
        totals["loss_pred"] += (1.0 - pred_cos.mean()).item()
        totals["loss_ema"] += (1.0 - ema_cos.mean()).item()
        totals["loss_last"] += (1.0 - last_cos.mean()).item()
        totals["cos_pred_minus_ema"] += (pred_cos - ema_cos).mean().item()
        totals["cos_pred_minus_last"] += (pred_cos - last_cos).mean().item()
        totals["rank_acc_pred"] += rank_pred.item()
        totals["rank_acc_ema"] += rank_ema.item()
        totals["rank_acc_last"] += rank_last.item()
        n += 1
        if args.max_batches and n >= args.max_batches:
            break

    metrics = {k: v / max(n, 1) for k, v in totals.items()}
    logging.info("Feature-level metrics: %s", metrics)
    for key in sorted(metrics):
        print(f"{key}: {metrics[key]:.6f}")


if __name__ == "__main__":
    main()
