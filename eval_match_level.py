import argparse
import logging

import torch
import torch.nn.functional as F
from tqdm import tqdm

from apu_diff.config import load_config
from apu_diff.datasets import create_feature_dataloaders
from apu_diff.models import APUDiff
from apu_diff.utils.training import get_device, load_model_state, set_seed, setup_logging
from train_stage2_update import identity_match_mask, synthesize_obs_z


def parse_args():
    parser = argparse.ArgumentParser("Evaluate APUDiff candidate ranking at feature level")
    parser.add_argument("--config", default="configs/apu_diff_default.yaml")
    parser.add_argument("--feature-path", default=None)
    parser.add_argument("--feature-dir", default=None)
    parser.add_argument("--sequences", nargs="+", default=None)
    parser.add_argument("--val-feature-dir", default=None)
    parser.add_argument("--val-sequences", nargs="+", default=None)
    parser.add_argument("--split-mode", default=None)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=["train", "val"], default="val")
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--target-min-gap", type=int, default=None)
    parser.add_argument("--target-max-gap", type=int, default=None)
    parser.add_argument("--log-dir", default=None)
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--log-name", default="eval_match_level")
    return parser.parse_args()


def pairwise_costs(model, pred_feat, obs_z, other_z, identity_token):
    batch_size = pred_feat.shape[0]
    candidates = torch.cat([obs_z, other_z], dim=0)
    num_candidates = candidates.shape[0]

    pred_flat = pred_feat[:, None, :].expand(-1, num_candidates, -1).reshape(-1, pred_feat.shape[-1])
    id_flat = identity_token[:, None, :].expand(-1, num_candidates, -1).reshape(-1, identity_token.shape[-1])
    cand_flat = candidates[None, :, :].expand(batch_size, -1, -1).reshape(-1, candidates.shape[-1])

    apu_cost, gate = model.appearance_cost(pred_flat, cand_flat, id_flat)
    identity_cost = 1.0 - F.cosine_similarity(id_flat, cand_flat, dim=-1)
    pred_cost = 1.0 - F.cosine_similarity(pred_flat, cand_flat, dim=-1)
    avg_cost = 0.5 * identity_cost + 0.5 * pred_cost
    return {
        "apu": apu_cost.reshape(batch_size, num_candidates),
        "identity": identity_cost.reshape(batch_size, num_candidates),
        "pred": pred_cost.reshape(batch_size, num_candidates),
        "avg": avg_cost.reshape(batch_size, num_candidates),
        "gate": gate.reshape(batch_size, num_candidates),
    }


def ranking_metrics(cost, positive_candidates):
    best_idx = cost.argmin(dim=1)
    rows = torch.arange(cost.shape[0], device=cost.device)
    rank1 = positive_candidates[rows, best_idx].float().mean()
    pos_cost = cost[positive_candidates].mean()
    neg_cost = cost[~positive_candidates].mean()
    logits = -cost
    pos_logits = logits.masked_fill(~positive_candidates, -torch.inf)
    nll = -(torch.logsumexp(pos_logits, dim=1) - torch.logsumexp(logits, dim=1)).mean()
    return rank1, pos_cost, neg_cost, nll


def main():
    args = parse_args()
    overrides = {
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
    }
    config = load_config(args.config, overrides)
    set_seed(config.seed)
    log_file = setup_logging(config.log_dir, args.log_name)
    device = get_device(config.device)
    config.device = str(device)
    logging.info("Log file: %s", log_file)
    logging.info("Config: %s", config)

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
    checkpoint = load_model_state(model, args.checkpoint, strict=False)
    model.eval()
    logging.info("Loaded checkpoint %s (epoch=%s)", args.checkpoint, checkpoint.get("epoch"))

    totals = {
        "gate": 0.0,
        "cos_pred": 0.0,
        "cos_identity": 0.0,
        "cos_obs": 0.0,
    }
    for prefix in ("apu", "identity", "pred", "avg"):
        totals[f"{prefix}_rank1"] = 0.0
        totals[f"{prefix}_pos_cost"] = 0.0
        totals[f"{prefix}_neg_cost"] = 0.0
        totals[f"{prefix}_nll"] = 0.0

    num_batches = 0
    sample_offset = 0
    with torch.no_grad():
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

            positive_mask = identity_match_mask(batch, device)
            positive_candidates = torch.cat(
                [
                    positive_mask,
                    torch.zeros_like(positive_mask),
                ],
                dim=1,
            )
            costs = pairwise_costs(model, pred_feat, obs_z, other_z, identity_token)

            totals["gate"] += costs["gate"].mean().item()
            totals["cos_pred"] += F.cosine_similarity(pred_feat, target_z, dim=-1).mean().item()
            totals["cos_identity"] += F.cosine_similarity(identity_token, target_z, dim=-1).mean().item()
            totals["cos_obs"] += F.cosine_similarity(obs_z, target_z, dim=-1).mean().item()
            for prefix in ("apu", "identity", "pred", "avg"):
                rank1, pos_cost, neg_cost, nll = ranking_metrics(costs[prefix], positive_candidates)
                totals[f"{prefix}_rank1"] += rank1.item()
                totals[f"{prefix}_pos_cost"] += pos_cost.item()
                totals[f"{prefix}_neg_cost"] += neg_cost.item()
                totals[f"{prefix}_nll"] += nll.item()

            num_batches += 1
            if args.max_batches and num_batches >= args.max_batches:
                break

    metrics = {key: value / max(num_batches, 1) for key, value in totals.items()}
    logging.info("metrics %s", metrics)
    for key in sorted(metrics):
        print(f"{key}: {metrics[key]:.6f}")


if __name__ == "__main__":
    main()
