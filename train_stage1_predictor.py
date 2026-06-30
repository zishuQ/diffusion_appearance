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
from apu_diff.utils.losses import (
    cosine_loss,
    cosine_margin_loss,
    mean_cosine,
    pairwise_similarity_distillation,
    supervised_contrastive_loss,
)
from apu_diff.utils.metrics import last_real_feature
from apu_diff.utils.training import get_device, load_model_state, save_checkpoint, set_seed, setup_logging


def parse_args():
    parser = argparse.ArgumentParser("Train APUDiff Stage 1 predictor")
    parser.add_argument("--config", default="configs/apu_diff_default.yaml")
    parser.add_argument("--feature-path", default=None)
    parser.add_argument("--feature-dir", default=None)
    parser.add_argument("--sequences", nargs="+", default=None)
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
    parser.add_argument("--log-name", default="stage1_predictor")
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def make_identity_labels(batch, device):
    seqs = batch["seq"]
    track_ids = batch["track_id"]
    if torch.is_tensor(track_ids):
        track_ids = track_ids.cpu().tolist()
    mapping = {}
    labels = []
    for seq, track_id in zip(seqs, track_ids):
        key = (str(seq), int(track_id))
        if key not in mapping:
            mapping[key] = len(mapping)
        labels.append(mapping[key])
    return torch.tensor(labels, dtype=torch.long, device=device)


def pairwise_identity_cosine(features, labels):
    features = F.normalize(features, p=2, dim=-1)
    sim = features @ features.T
    same = labels[:, None].eq(labels[None, :])
    offdiag = ~torch.eye(features.shape[0], dtype=torch.bool, device=features.device)
    same = same & offdiag
    diff = (~labels[:, None].eq(labels[None, :])) & offdiag
    same_mean = sim[same].mean() if same.any() else features.new_tensor(0.0)
    diff_mean = sim[diff].mean() if diff.any() else features.new_tensor(0.0)
    return same_mean, diff_mean


def set_module_trainable(module, trainable: bool):
    for param in module.parameters():
        param.requires_grad = trainable


def update_gap_stats(stats, gaps, pred_cos, last_cos):
    gaps_cpu = gaps.detach().cpu().long()
    pred_cpu = pred_cos.detach().cpu()
    last_cpu = last_cos.detach().cpu()
    for gap in gaps_cpu.unique().tolist():
        mask = gaps_cpu == gap
        count = int(mask.sum().item())
        name = f"gap_{int(gap):02d}"
        if name not in stats:
            stats[name] = {
                "n": 0,
                "cos_pred": 0.0,
                "cos_last": 0.0,
                "cos_pred_minus_last": 0.0,
            }
        stats[name]["n"] += count
        stats[name]["cos_pred"] += pred_cpu[mask].sum().item()
        stats[name]["cos_last"] += last_cpu[mask].sum().item()
        stats[name]["cos_pred_minus_last"] += (pred_cpu[mask] - last_cpu[mask]).sum().item()


def finalize_gap_stats(stats):
    metrics = {}
    for name in sorted(stats):
        count = max(stats[name]["n"], 1)
        metrics[f"{name}_n"] = stats[name]["n"]
        metrics[f"{name}_cos_pred"] = stats[name]["cos_pred"] / count
        metrics[f"{name}_cos_last"] = stats[name]["cos_last"] / count
        metrics[f"{name}_pred_minus_last"] = stats[name]["cos_pred_minus_last"] / count
    return metrics


def run_projection_epoch(model, loader, optimizer, device, config, train: bool, show_progress: bool = True):
    model.projection.train(train)
    model.predictor.eval()
    totals = {
        "loss": 0.0,
        "loss_supcon": 0.0,
        "loss_distill": 0.0,
        "cos_pos": 0.0,
        "cos_same": 0.0,
        "cos_diff": 0.0,
        "target_gap": 0.0,
        "frame_gap": 0.0,
    }
    num_batches = 0
    pbar = tqdm(loader, desc="projection_train" if train else "projection_val", disable=not show_progress)
    for batch in pbar:
        history = batch["history_feats"].to(device, non_blocking=True)
        history_mask = batch["history_mask"].to(device, non_blocking=True)
        target = batch["target_feat"].to(device, non_blocking=True)
        labels = make_identity_labels(batch, device)

        with torch.set_grad_enabled(train):
            last_raw = last_real_feature(history, history_mask)
            target_raw = F.normalize(target, p=2, dim=-1)
            last_raw = F.normalize(last_raw, p=2, dim=-1)
            target_z = model.project(target_raw)
            last_z = model.project(last_raw)
            projected = torch.cat([target_z, last_z], dim=0)
            reference = torch.cat([target_raw, last_raw], dim=0)
            contrast_labels = torch.cat([labels, labels], dim=0)

            loss_supcon = supervised_contrastive_loss(
                projected,
                contrast_labels,
                temperature=config.stage1_supcon_temperature,
            )
            loss_distill = pairwise_similarity_distillation(projected, reference)
            loss = config.stage1_supcon_weight * loss_supcon + config.stage1_distill_weight * loss_distill
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if config.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.projection.parameters(), config.grad_clip)
                optimizer.step()

        same_cos, diff_cos = pairwise_identity_cosine(projected.detach(), contrast_labels)
        totals["loss"] += loss.item()
        totals["loss_supcon"] += loss_supcon.item()
        totals["loss_distill"] += loss_distill.item()
        totals["cos_pos"] += mean_cosine(target_z.detach(), last_z.detach()).item()
        totals["cos_same"] += same_cos.item()
        totals["cos_diff"] += diff_cos.item()
        totals["target_gap"] += batch["target_gap"].float().mean().item()
        totals["frame_gap"] += batch["frame_gap"].float().mean().item()
        num_batches += 1
        pbar.set_postfix(
            loss=f"{loss.item():.4f}",
            pos=f"{totals['cos_pos'] / num_batches:.4f}",
            diff=f"{totals['cos_diff'] / num_batches:.4f}",
        )
    denom = max(num_batches, 1)
    return {key: value / denom for key, value in totals.items()}


def run_epoch(model, loader, optimizer, device, config, train: bool, show_progress: bool = True):
    model.train(train)
    if not any(param.requires_grad for param in model.projection.parameters()):
        model.projection.eval()
    totals = {
        "loss": 0.0,
        "loss_pred": 0.0,
        "loss_mu": 0.0,
        "loss_diff": 0.0,
        "loss_metric": 0.0,
        "loss_improve": 0.0,
        "cos_pred": 0.0,
        "cos_mu": 0.0,
        "cos_last": 0.0,
        "cos_pos": 0.0,
        "cos_neg": 0.0,
        "cos_pred_neg": 0.0,
        "cos_pred_minus_last": 0.0,
        "base_scale": 0.0,
        "history_mix_scale": 0.0,
        "residual_scale": 0.0,
        "target_gap": 0.0,
        "frame_gap": 0.0,
    }
    gap_stats = {}
    num_batches = 0
    pbar = tqdm(loader, desc="train" if train else "val", disable=not show_progress)
    for batch in pbar:
        history = batch["history_feats"].to(device, non_blocking=True)
        history_mask = batch["history_mask"].to(device, non_blocking=True)
        target = batch["target_feat"].to(device, non_blocking=True)
        other = batch["other_feat"].to(device, non_blocking=True)
        with torch.set_grad_enabled(train):
            outputs = model.predictor_training_forward(history, history_mask, target)
            other_z = model.project(other)
            pred_feat = outputs["pred_feat_det"]
            last_z = last_real_feature(outputs["local_queue"], history_mask)
            loss_pred = cosine_loss(pred_feat, outputs["target_z"])
            loss_mu = cosine_loss(outputs["mu"], outputs["target_z"])
            loss_c = F.smooth_l1_loss(outputs["c_hat"], outputs["c_target"])
            loss_noise = F.smooth_l1_loss(outputs["noise_hat"], outputs["noise"])
            loss_diff = 0.5 * loss_c + 0.5 * loss_noise
            loss_proj_margin, cos_pos, cos_neg = cosine_margin_loss(
                outputs["target_z"],
                last_z,
                other_z,
                config.stage1_margin,
            )
            loss_pred_margin, _, cos_pred_neg = cosine_margin_loss(
                pred_feat,
                outputs["target_z"],
                other_z,
                config.stage1_margin,
            )
            loss_metric = 0.5 * (loss_proj_margin + loss_pred_margin)
            pred_cos_for_loss = F.cosine_similarity(pred_feat, outputs["target_z"], dim=-1)
            last_cos_for_loss = F.cosine_similarity(last_z.detach(), outputs["target_z"].detach(), dim=-1)
            loss_improve = F.relu(
                last_cos_for_loss + float(config.stage1_improve_margin) - pred_cos_for_loss
            ).mean()
            loss = (
                loss_pred
                + config.stage1_mu_weight * loss_mu
                + config.stage1_diff_weight * loss_diff
                + config.stage1_metric_weight * loss_metric
                + config.stage1_improve_weight * loss_improve
            )
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if config.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                optimizer.step()
        totals["loss"] += loss.item()
        totals["loss_pred"] += loss_pred.item()
        totals["loss_mu"] += loss_mu.item()
        totals["loss_diff"] += loss_diff.item()
        totals["loss_metric"] += loss_metric.item()
        totals["loss_improve"] += loss_improve.item()
        pred_cos = F.cosine_similarity(pred_feat.detach(), outputs["target_z"].detach(), dim=-1)
        last_cos = F.cosine_similarity(last_z.detach(), outputs["target_z"].detach(), dim=-1)
        totals["cos_pred"] += pred_cos.mean().item()
        totals["cos_mu"] += mean_cosine(outputs["mu"].detach(), outputs["target_z"].detach()).item()
        totals["cos_last"] += last_cos.mean().item()
        totals["cos_pos"] += cos_pos.detach().item()
        totals["cos_neg"] += cos_neg.detach().item()
        totals["cos_pred_neg"] += cos_pred_neg.detach().item()
        totals["cos_pred_minus_last"] += (pred_cos - last_cos).mean().item()
        totals["base_scale"] += outputs["base_scale"].detach().item()
        totals["history_mix_scale"] += outputs["history_mix_scale"].detach().item()
        totals["residual_scale"] += outputs["residual_scale"].detach().item()
        totals["target_gap"] += batch["target_gap"].float().mean().item()
        totals["frame_gap"] += batch["frame_gap"].float().mean().item()
        update_gap_stats(gap_stats, batch["target_gap"], pred_cos, last_cos)
        num_batches += 1
        pbar.set_postfix(loss=f"{loss.item():.4f}", cos=f"{totals['cos_pred'] / num_batches:.4f}")
    denom = max(num_batches, 1)
    metrics = {key: value / denom for key, value in totals.items()}
    metrics.update(finalize_gap_stats(gap_stats))
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
        "num_epochs_stage1": args.epochs,
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
    projection_optimizer = torch.optim.AdamW(
        model.projection.parameters(),
        lr=config.lr_stage1,
        weight_decay=config.weight_decay,
    )
    predictor_optimizer = None
    scheduler = None
    warmup_epochs = min(config.stage1_projection_warmup_epochs, config.num_epochs_stage1)
    projection_scheduler = CosineAnnealingLR(projection_optimizer, T_max=warmup_epochs) if warmup_epochs > 0 else None
    warmup_best = float("inf")
    warmup_best_path = None
    best_val = float("inf")
    best_improve = float("-inf")
    for epoch in range(1, config.num_epochs_stage1 + 1):
        logging.info("Epoch %d/%d", epoch, config.num_epochs_stage1)
        if epoch <= warmup_epochs:
            logging.info("Phase: projection warmup")
            train_metrics = run_projection_epoch(
                model,
                train_loader,
                projection_optimizer,
                device,
                config,
                train=True,
                show_progress=not args.no_progress,
            )
            val_metrics = run_projection_epoch(
                model,
                val_loader,
                projection_optimizer,
                device,
                config,
                train=False,
                show_progress=not args.no_progress,
            )
            logging.info("projection train %s", train_metrics)
            logging.info("projection val %s", val_metrics)
            last_path = os.path.join(config.checkpoint_dir, "projection_warmup_last.pth")
            save_checkpoint(
                last_path,
                model,
                projection_optimizer,
                epoch,
                val_metrics["loss"],
                config,
                {"metrics": val_metrics, "phase": "projection_warmup"},
            )
            if val_metrics["loss"] < warmup_best:
                warmup_best = val_metrics["loss"]
                best_path = os.path.join(config.checkpoint_dir, "projection_warmup_best.pth")
                warmup_best_path = best_path
                save_checkpoint(
                    best_path,
                    model,
                    projection_optimizer,
                    epoch,
                    warmup_best,
                    config,
                    {"metrics": val_metrics, "phase": "projection_warmup"},
                )
                logging.info("New best projection checkpoint: %s", best_path)
            if projection_scheduler is not None:
                projection_scheduler.step()
            continue

        if predictor_optimizer is None:
            if warmup_best_path:
                checkpoint = load_model_state(model, warmup_best_path, strict=True)
                logging.info(
                    "Restored best ProjectionHead before predictor phase: %s (epoch=%s, val_loss=%.6f)",
                    warmup_best_path,
                    checkpoint.get("epoch"),
                    checkpoint.get("val_loss", float("nan")),
                )
            if config.freeze_projection_after_warmup:
                set_module_trainable(model.projection, False)
                logging.info("ProjectionHead frozen after %d warmup epochs", warmup_epochs)
            predictor_params = [p for p in model.parameters() if p.requires_grad]
            predictor_optimizer = torch.optim.AdamW(
                predictor_params,
                lr=config.lr_stage1,
                weight_decay=config.weight_decay,
            )
            predictor_epochs = config.num_epochs_stage1 - warmup_epochs
            scheduler = CosineAnnealingLR(predictor_optimizer, T_max=predictor_epochs) if predictor_epochs > 0 else None

        logging.info("Phase: predictor")
        train_metrics = run_epoch(
            model,
            train_loader,
            predictor_optimizer,
            device,
            config,
            train=True,
            show_progress=not args.no_progress,
        )
        val_metrics = run_epoch(
            model,
            val_loader,
            predictor_optimizer,
            device,
            config,
            train=False,
            show_progress=not args.no_progress,
        )
        logging.info("train %s", train_metrics)
        logging.info("val %s", val_metrics)
        last_path = os.path.join(config.checkpoint_dir, "APUDiff_stage1_last.pth")
        save_checkpoint(last_path, model, predictor_optimizer, epoch, val_metrics["loss"], config, {"metrics": val_metrics})
        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            best_path = os.path.join(config.checkpoint_dir, "stage1_predictor.pth")
            save_checkpoint(best_path, model, predictor_optimizer, epoch, best_val, config, {"metrics": val_metrics})
            logging.info("New best stage1 checkpoint: %s", best_path)
        val_improve = val_metrics.get("cos_pred_minus_last", float("-inf"))
        if val_improve > best_improve:
            best_improve = val_improve
            best_improve_path = os.path.join(config.checkpoint_dir, "stage1_predictor_best_improve.pth")
            save_checkpoint(
                best_improve_path,
                model,
                predictor_optimizer,
                epoch,
                val_metrics["loss"],
                config,
                {"metrics": val_metrics, "selection_metric": "cos_pred_minus_last"},
            )
            logging.info(
                "New best stage1 improve checkpoint: %s (cos_pred_minus_last=%.6f)",
                best_improve_path,
                best_improve,
            )
        if scheduler is not None:
            scheduler.step()
            logging.info("LR: %.2e", scheduler.get_last_lr()[0])


if __name__ == "__main__":
    main()
