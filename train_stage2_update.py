import argparse
import logging
import os

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from apu_diff.config import load_config
from apu_diff.datasets import create_feature_dataloaders
from apu_diff.models import APUDiff
from apu_diff.utils.losses import cosine_loss
from apu_diff.utils.training import get_device, load_model_state, save_checkpoint, set_seed, setup_logging


def parse_args():
    parser = argparse.ArgumentParser("Train APUDiff Stage 2 UpdateBlock")
    parser.add_argument("--config", default="configs/apu_diff_default.yaml")
    parser.add_argument("--feature-path", default=None)
    parser.add_argument("--feature-dir", default=None)
    parser.add_argument("--sequences", nargs="+", default=None)
    parser.add_argument("--stage1-checkpoint", default="checkpoints/stage1_predictor_best_improve.pth")
    parser.add_argument("--device", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--target-min-gap", type=int, default=None)
    parser.add_argument("--target-max-gap", type=int, default=None)
    parser.add_argument("--val-feature-dir", default=None)
    parser.add_argument("--val-sequences", nargs="+", default=None)
    parser.add_argument("--split-mode", default=None)
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--log-dir", default=None)
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--log-name", default="stage2_update")
    return parser.parse_args()


def synthesize_obs_z(target_z, other_z, config, deterministic: bool = False, offset: int = 0):
    batch_size = target_z.shape[0]
    device = target_z.device
    if deterministic:
        idx = torch.arange(offset, offset + batch_size, device=device)
        clean_score = idx.mul(37).remainder(1000).float() / 1000.0
        clean = clean_score < float(config.obs_clean_prob)
        span = float(config.obs_dirty_lambda_max - config.obs_dirty_lambda_min)
        frac = idx.mul(997).remainder(1000).float() / 999.0
        dirty_lambda = float(config.obs_dirty_lambda_min) + span * frac
    else:
        clean = torch.rand(batch_size, device=device) < config.obs_clean_prob
        dirty_lambda = torch.empty(batch_size, device=device).uniform_(
            config.obs_dirty_lambda_min,
            config.obs_dirty_lambda_max,
        )
    lam = torch.where(clean, torch.ones_like(dirty_lambda), dirty_lambda).unsqueeze(-1)
    obs_z = F.normalize(lam * target_z + (1.0 - lam) * other_z, p=2, dim=-1)
    return obs_z, clean


def identity_match_mask(batch, device):
    seqs = [str(seq) for seq in batch["seq"]]
    track_ids = batch["track_id"]
    if torch.is_tensor(track_ids):
        track_ids = track_ids.cpu().tolist()
    keys = [(seq, int(track_id)) for seq, track_id in zip(seqs, track_ids)]
    mask = [[left == right for right in keys] for left in keys]
    return torch.tensor(mask, dtype=torch.bool, device=device)


def candidate_ranking_loss(model, pred_feat, obs_z, other_z, identity_token, positive_mask, config):
    """Train APUDiff cost on a tracker-like candidate set.

    Each row is one track state. Candidate columns are current observations from
    the whole batch plus the dataset-sampled hard negative for each row.
    Lower APUDiff cost should rank any same-identity observation above all
    other candidates.
    """
    batch_size = pred_feat.shape[0]
    candidates = torch.cat([obs_z, other_z], dim=0)
    num_candidates = candidates.shape[0]

    pred_flat = pred_feat[:, None, :].expand(-1, num_candidates, -1).reshape(-1, pred_feat.shape[-1])
    id_flat = identity_token[:, None, :].expand(-1, num_candidates, -1).reshape(-1, identity_token.shape[-1])
    cand_flat = candidates[None, :, :].expand(batch_size, -1, -1).reshape(-1, candidates.shape[-1])
    cost_flat, gate_flat = model.appearance_cost(pred_flat, cand_flat, id_flat)
    cost = cost_flat.reshape(batch_size, num_candidates)
    gate = gate_flat.reshape(batch_size, num_candidates)

    positive_candidates = torch.cat(
        [
            positive_mask,
            torch.zeros(batch_size, batch_size, dtype=torch.bool, device=positive_mask.device),
        ],
        dim=1,
    )
    temperature = max(float(config.match_temperature), 1e-6)
    logits = -cost / temperature
    pos_logits = logits.masked_fill(~positive_candidates, -torch.inf)
    loss_nll = -(torch.logsumexp(pos_logits, dim=1) - torch.logsumexp(logits, dim=1)).mean()

    pos_cost_row = cost.masked_fill(~positive_candidates, torch.inf).min(dim=1).values
    neg_cost_row = cost.masked_fill(positive_candidates, torch.inf).min(dim=1).values
    loss_margin = F.relu(pos_cost_row + float(config.match_margin) - neg_cost_row).mean()
    loss = loss_nll + loss_margin

    with torch.no_grad():
        best_idx = cost.argmin(dim=1)
        row_idx = torch.arange(batch_size, device=cost.device)
        rank_acc = positive_candidates[row_idx, best_idx].float().mean()
        pos_cost = pos_cost_row.mean()
        neg_cost = neg_cost_row.mean()
        gate_mean = gate.mean()
    return loss, rank_acc, pos_cost, neg_cost, gate_mean, loss_nll.detach(), loss_margin.detach()


def gate_teacher_loss(model, pred_feat, obs_z, identity_token, target_z):
    """Per-channel cross-attention gate teacher loss.

    The teacher is high on channels where identity_token is closer to target_z
    than pred_feat, and low where pred_feat is closer.
    """
    _, attn = model.cross_attn_gate(pred_feat, obs_z, identity_token)
    mem_error = torch.abs(identity_token - target_z)
    pred_error = torch.abs(pred_feat - target_z)
    teacher = torch.sigmoid((pred_error - mem_error) / 0.05).detach()
    loss_bce = F.binary_cross_entropy(attn.clamp(1e-6, 1.0 - 1e-6), teacher)
    loss_mean = F.mse_loss(attn.mean(dim=-1), teacher.mean(dim=-1))
    loss = loss_bce + 0.1 * loss_mean
    return loss, attn.detach().mean(), teacher.detach().mean()


def run_epoch(model, loader, optimizer, device, config, train: bool, show_progress: bool = True):
    model.train(train)
    model.projection.eval()
    model.predictor.eval()

    totals = {
        "loss_stage2": 0.0,
        "loss_local": 0.0,
        "loss_identity": 0.0,
        "loss_match": 0.0,
        "loss_match_nll": 0.0,
        "loss_match_margin": 0.0,
        "loss_gate": 0.0,
        "loss_select": 0.0,
        "rank_acc": 0.0,
        "pos_cost": 0.0,
        "neg_cost": 0.0,
        "cos_obs": 0.0,
        "cos_updated": 0.0,
        "cos_identity": 0.0,
        "cos_gate": 0.0,
        "gate_pos": 0.0,
        "gate_teacher": 0.0,
        "cos_clean_obs": 0.0,
        "cos_clean_updated": 0.0,
        "cos_dirty_obs": 0.0,
        "cos_dirty_updated": 0.0,
    }
    num_batches = clean_batches = dirty_batches = 0

    pbar = tqdm(loader, desc="train" if train else "val", disable=not show_progress)
    sample_offset = 0
    for batch in pbar:
        history = batch["history_feats"].to(device, non_blocking=True)
        history_mask = batch["history_mask"].to(device, non_blocking=True)
        target = batch["target_feat"].to(device, non_blocking=True)
        other = batch["other_feat"].to(device, non_blocking=True)

        with torch.no_grad():
            local_queue = model.project(history)
            target_z = model.project(target)
            other_z = model.project(other)
            identity_token = model.init_identity(local_queue, history_mask)
            pred_feat = model.predict(local_queue, history_mask, identity_token, deterministic=True)
            obs_z, clean_mask = synthesize_obs_z(
                target_z,
                other_z,
                config,
                deterministic=not train,
                offset=sample_offset,
            )
            sample_offset += target_z.shape[0]

        with torch.set_grad_enabled(train):
            updated_local, updated_identity = model.update(pred_feat.detach(), obs_z.detach(), identity_token.detach())
            loss_local = cosine_loss(updated_local, target_z.detach())
            loss_identity = cosine_loss(updated_identity, target_z.detach())
            positive_mask = identity_match_mask(batch, device)
            loss_match, rank_acc, pos_cost, neg_cost, gate_mean, loss_match_nll, loss_match_margin = candidate_ranking_loss(
                model,
                pred_feat.detach(),
                obs_z.detach(),
                other_z.detach(),
                identity_token.detach(),
                positive_mask,
                config,
            )
            loss_gate, gate_pos, gate_teacher = gate_teacher_loss(
                model,
                pred_feat.detach(),
                obs_z.detach(),
                identity_token.detach(),
                target_z.detach(),
            )
            loss = (
                loss_local
                + config.identity_loss_weight * loss_identity
                + config.match_loss_weight * loss_match
                + config.gate_loss_weight * loss_gate
            )
            loss_select = loss_match.detach() + config.gate_loss_weight * loss_gate.detach()
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if config.grad_clip > 0:
                    params = list(model.update_block.parameters()) + list(model.cross_attn_gate.parameters())
                    torch.nn.utils.clip_grad_norm_(params, config.grad_clip)
                optimizer.step()

        with torch.no_grad():
            cos_obs = F.cosine_similarity(obs_z, target_z, dim=-1)
            cos_updated = F.cosine_similarity(updated_local, target_z, dim=-1)
            cos_identity = F.cosine_similarity(updated_identity, target_z, dim=-1)

        totals["loss_stage2"] += loss.item()
        totals["loss_local"] += loss_local.item()
        totals["loss_identity"] += loss_identity.item()
        totals["loss_match"] += loss_match.item()
        totals["loss_match_nll"] += loss_match_nll.item()
        totals["loss_match_margin"] += loss_match_margin.item()
        totals["loss_gate"] += loss_gate.item()
        totals["loss_select"] += loss_select.item()
        totals["rank_acc"] += rank_acc.item()
        totals["pos_cost"] += pos_cost.item()
        totals["neg_cost"] += neg_cost.item()
        totals["cos_obs"] += cos_obs.mean().item()
        totals["cos_updated"] += cos_updated.mean().item()
        totals["cos_identity"] += cos_identity.mean().item()
        totals["cos_gate"] += gate_mean.item()
        totals["gate_pos"] += gate_pos.item()
        totals["gate_teacher"] += gate_teacher.item()
        if clean_mask.any():
            totals["cos_clean_obs"] += cos_obs[clean_mask].mean().item()
            totals["cos_clean_updated"] += cos_updated[clean_mask].mean().item()
            clean_batches += 1
        dirty_mask = ~clean_mask
        if dirty_mask.any():
            totals["cos_dirty_obs"] += cos_obs[dirty_mask].mean().item()
            totals["cos_dirty_updated"] += cos_updated[dirty_mask].mean().item()
            dirty_batches += 1

        num_batches += 1
        pbar.set_postfix(loss=f"{loss.item():.4f}", upd=f"{totals['cos_updated'] / num_batches:.4f}")

    denom = max(num_batches, 1)
    metrics = {k: v / denom for k, v in totals.items()}
    metrics["cos_clean_obs"] = totals["cos_clean_obs"] / max(clean_batches, 1)
    metrics["cos_clean_updated"] = totals["cos_clean_updated"] / max(clean_batches, 1)
    metrics["cos_dirty_obs"] = totals["cos_dirty_obs"] / max(dirty_batches, 1)
    metrics["cos_dirty_updated"] = totals["cos_dirty_updated"] / max(dirty_batches, 1)
    return metrics


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
        "num_epochs_stage2": args.epochs,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "target_min_gap": args.target_min_gap,
        "target_max_gap": args.target_max_gap,
        "checkpoint_dir": args.checkpoint_dir,
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
    model = APUDiff(
        reid_dim=config.reid_dim,
        latent_dim=config.latent_dim,
        projection_hidden_dim=config.projection_hidden_dim,
        time_dim=config.time_dim,
        num_diffusion_steps=config.num_diffusion_steps,
        denoiser_hidden_dim=config.denoiser_hidden_dim,
        update_hidden_dim=config.update_hidden_dim,
    ).to(device)
    checkpoint = load_model_state(model, args.stage1_checkpoint, strict=False)
    logging.info("Loaded stage1 checkpoint %s (epoch=%s)", args.stage1_checkpoint, checkpoint.get("epoch"))

    model.freeze_predictor()
    for param in model.cross_attn_gate.parameters():
        param.requires_grad = True
    optimizer = torch.optim.AdamW(
        list(model.update_block.parameters()) + list(model.cross_attn_gate.parameters()),
        lr=config.lr_stage2,
        weight_decay=config.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=config.num_epochs_stage2) if config.num_epochs_stage2 > 0 else None

    best_val_select = float("inf")
    for epoch in range(1, config.num_epochs_stage2 + 1):
        logging.info("Epoch %d/%d", epoch, config.num_epochs_stage2)
        train_metrics = run_epoch(
            model,
            train_loader,
            optimizer,
            device,
            config,
            train=True,
            show_progress=not args.no_progress,
        )
        val_metrics = run_epoch(
            model,
            val_loader,
            optimizer,
            device,
            config,
            train=False,
            show_progress=not args.no_progress,
        )
        logging.info("train %s", train_metrics)
        logging.info("val %s", val_metrics)
        last_path = os.path.join(config.checkpoint_dir, "apu_diff_full_last.pth")
        save_checkpoint(last_path, model, optimizer, epoch, val_metrics["loss_stage2"], config, {"metrics": val_metrics})
        if val_metrics["loss_select"] < best_val_select:
            best_val_select = val_metrics["loss_select"]
            best_path = os.path.join(config.checkpoint_dir, "apu_diff_full.pth")
            save_checkpoint(best_path, model, optimizer, epoch, best_val_select, config, {"metrics": val_metrics})
            logging.info("New best full checkpoint by val loss_select: %s", best_path)
        if scheduler is not None:
            scheduler.step()
            logging.info("LR: %.2e", scheduler.get_last_lr()[0])


if __name__ == "__main__":
    main()
