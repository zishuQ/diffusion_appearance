import argparse
import logging

import torch
import torch.nn.functional as F
from tqdm import tqdm

from apu_diff.config import load_config
from apu_diff.datasets import create_feature_dataloaders
from apu_diff.models import APUDiff
from apu_diff.utils.metrics import last_real_feature, mean_cosine
from apu_diff.utils.training import get_device, load_model_state, set_seed, setup_logging
from train_stage2_update import synthesize_obs_z


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
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--log-name", default="eval_feature_level")
    return parser.parse_args()


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
        projection_hidden_dim=config.projection_hidden_dim,
        time_dim=config.time_dim,
        num_diffusion_steps=config.num_diffusion_steps,
        denoiser_hidden_dim=config.denoiser_hidden_dim,
        update_hidden_dim=config.update_hidden_dim,
    ).to(device)
    load_model_state(model, args.checkpoint, strict=False)
    model.eval()

    totals = {
        "cos_last": 0.0,
        "cos_pred": 0.0,
        "cos_obs": 0.0,
        "cos_updated": 0.0,
        "cos_identity": 0.0,
        "gate_mean": 0.0,
    }
    n = 0
    sample_offset = 0
    for batch in tqdm(loader, desc=f"eval_{args.split}"):
        history = batch["history_feats"].to(device, non_blocking=True)
        history_mask = batch["history_mask"].to(device, non_blocking=True)
        target = batch["target_feat"].to(device, non_blocking=True)
        other = batch["other_feat"].to(device, non_blocking=True)

        local_queue = model.project(history)
        target_z = model.project(target)
        other_z = model.project(other)
        identity_token = model.init_identity(local_queue, history_mask)
        pred_feat = model.predict(local_queue, history_mask, identity_token, deterministic=True)
        obs_z, _ = synthesize_obs_z(target_z, other_z, config, deterministic=True, offset=sample_offset)
        sample_offset += target_z.shape[0]
        updated_local, updated_identity = model.update(pred_feat, obs_z, identity_token)
        gate = model.match_gate_value(pred_feat, obs_z, identity_token)

        totals["cos_last"] += mean_cosine(last_real_feature(local_queue, history_mask), target_z).item()
        totals["cos_pred"] += mean_cosine(pred_feat, target_z).item()
        totals["cos_obs"] += mean_cosine(obs_z, target_z).item()
        totals["cos_updated"] += mean_cosine(updated_local, target_z).item()
        totals["cos_identity"] += mean_cosine(updated_identity, target_z).item()
        totals["gate_mean"] += gate.mean().item()
        n += 1
        if args.max_batches and n >= args.max_batches:
            break

    metrics = {k: v / max(n, 1) for k, v in totals.items()}
    logging.info("Feature-level metrics: %s", metrics)
    print(metrics)


if __name__ == "__main__":
    main()
