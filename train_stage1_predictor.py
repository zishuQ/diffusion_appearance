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
from apu_diff.utils.metrics import compute_ema, last_real_feature
from apu_diff.utils.training import get_device, save_checkpoint, set_seed, setup_logging


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
    parser.add_argument("--ema-alpha", type=float, default=None)
    parser.add_argument("--log-name", default="stage1_predictor")
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def update_gap_stats(stats, gaps, pred_cos, ema_cos, last_cos):
    gaps_cpu = gaps.detach().cpu().long()
    pred_cpu = pred_cos.detach().cpu()
    ema_cpu = ema_cos.detach().cpu()
    last_cpu = last_cos.detach().cpu()
    for gap in gaps_cpu.unique().tolist():
        mask = gaps_cpu == gap
        count = int(mask.sum().item())
        name = f"gap_{int(gap):02d}"
        if name not in stats:
            stats[name] = {
                "n": 0,
                "cos_pred": 0.0,
                "cos_ema": 0.0,
                "cos_last": 0.0,
                "cos_pred_minus_ema": 0.0,
                "cos_pred_minus_last": 0.0,
            }
        stats[name]["n"] += count
        stats[name]["cos_pred"] += pred_cpu[mask].sum().item()
        stats[name]["cos_ema"] += ema_cpu[mask].sum().item()
        stats[name]["cos_last"] += last_cpu[mask].sum().item()
        stats[name]["cos_pred_minus_ema"] += (pred_cpu[mask] - ema_cpu[mask]).sum().item()
        stats[name]["cos_pred_minus_last"] += (pred_cpu[mask] - last_cpu[mask]).sum().item()


def finalize_gap_stats(stats):
    metrics = {}
    for name in sorted(stats):
        count = max(stats[name]["n"], 1)
        metrics[f"{name}_n"] = stats[name]["n"]
        metrics[f"{name}_cos_pred"] = stats[name]["cos_pred"] / count
        metrics[f"{name}_cos_ema"] = stats[name]["cos_ema"] / count
        metrics[f"{name}_cos_last"] = stats[name]["cos_last"] / count
        metrics[f"{name}_pred_minus_ema"] = stats[name]["cos_pred_minus_ema"] / count
        metrics[f"{name}_pred_minus_last"] = stats[name]["cos_pred_minus_last"] / count
    return metrics


def _rank_accuracy(anchor: torch.Tensor, target_z: torch.Tensor, other_z: torch.Tensor) -> torch.Tensor:
    pos_cost = 1.0 - F.cosine_similarity(anchor, target_z, dim=-1)
    neg_cost = 1.0 - F.cosine_similarity(anchor, other_z, dim=-1)
    return (pos_cost < neg_cost).float().mean()


def run_epoch(model, loader, optimizer, device, config, train: bool, show_progress: bool = True):
    model.train(train)
    if not any(param.requires_grad for param in model.projection.parameters()):
        model.projection.eval()

    totals = {
        "loss": 0.0,
        "loss_pred": 0.0,
        "loss_ema": 0.0,
        "loss_last": 0.0,
        "loss_diff": 0.0,
        "loss_improve": 0.0,
        "cos_pred": 0.0,
        "cos_ema": 0.0,
        "cos_last": 0.0,
        "cos_pred_minus_ema": 0.0,
        "cos_pred_minus_last": 0.0,
        "rank_acc_pred": 0.0,
        "rank_acc_ema": 0.0,
        "rank_acc_last": 0.0,
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
            target_z = outputs["target_z"]
            last_z = last_real_feature(outputs["local_queue"], history_mask)
            ema_feat = compute_ema(outputs["local_queue"], history_mask, alpha=config.ema_alpha)

            loss_pred = 1.0 - F.cosine_similarity(pred_feat, target_z, dim=-1).mean()
            loss_c = F.smooth_l1_loss(outputs["c_hat"], outputs["c_target"])
            loss_noise = F.smooth_l1_loss(outputs["noise_hat"], outputs["noise"])
            loss_diff = 0.5 * loss_c + 0.5 * loss_noise
            cos_ema_target = F.cosine_similarity(ema_feat, target_z, dim=-1)
            cos_pred_target = F.cosine_similarity(pred_feat, target_z, dim=-1)
            loss_improve = F.relu(cos_ema_target - cos_pred_target).mean()

            loss = loss_pred + config.stage1_diff_weight * loss_diff + config.stage1_improve_weight * loss_improve

            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if config.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                optimizer.step()

        with torch.no_grad():
            pred_cos = F.cosine_similarity(pred_feat.detach(), target_z.detach(), dim=-1)
            ema_cos = F.cosine_similarity(ema_feat.detach(), target_z.detach(), dim=-1)
            last_cos = F.cosine_similarity(last_z.detach(), target_z.detach(), dim=-1)

            rank_pred = _rank_accuracy(pred_feat.detach(), target_z.detach(), other_z.detach())
            rank_ema = _rank_accuracy(ema_feat.detach(), target_z.detach(), other_z.detach())
            rank_last = _rank_accuracy(last_z.detach(), target_z.detach(), other_z.detach())

        totals["loss"] += loss.item()
        totals["loss_pred"] += loss_pred.item()
        totals["loss_ema"] += (1.0 - ema_cos.mean()).item()
        totals["loss_last"] += (1.0 - last_cos.mean()).item()
        totals["loss_diff"] += loss_diff.item()
        totals["loss_improve"] += loss_improve.item()
        totals["cos_pred"] += pred_cos.mean().item()
        totals["cos_ema"] += ema_cos.mean().item()
        totals["cos_last"] += last_cos.mean().item()
        totals["cos_pred_minus_ema"] += (pred_cos - ema_cos).mean().item()
        totals["cos_pred_minus_last"] += (pred_cos - last_cos).mean().item()
        totals["rank_acc_pred"] += rank_pred.item()
        totals["rank_acc_ema"] += rank_ema.item()
        totals["rank_acc_last"] += rank_last.item()
        totals["target_gap"] += batch["target_gap"].float().mean().item()
        totals["frame_gap"] += batch["frame_gap"].float().mean().item()

        update_gap_stats(gap_stats, batch["target_gap"], pred_cos, ema_cos, last_cos)
        num_batches += 1

        if train and num_batches == 1 and torch.cuda.is_available():
            allocated_mb = torch.cuda.memory_allocated() / 1024**2
            reserved_mb = torch.cuda.memory_reserved() / 1024**2
            peak_mb = torch.cuda.max_memory_allocated() / 1024**2
            logging.info(
                "CUDA memory: allocated=%.1fMB reserved=%.1fMB peak=%.1fMB",
                allocated_mb,
                reserved_mb,
                peak_mb,
            )
            print(f"cuda allocated MB: {allocated_mb:.1f}")
            print(f"cuda reserved MB: {reserved_mb:.1f}")
            print(f"cuda peak MB: {peak_mb:.1f}")

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
        "ema_alpha": args.ema_alpha,
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
    for module in (model.projection, model.update_block, model.cross_attn_gate):
        for param in module.parameters():
            param.requires_grad = False

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info("Device: %s", device)
    logging.info("Model device: %s", next(model.parameters()).device)
    logging.info("Total params: %d", total_params)
    logging.info("Trainable params: %d", trainable_params)
    print("device:", device)
    print("model device:", next(model.parameters()).device)
    print(f"total params: {total_params / 1e6:.2f}M")
    print(f"trainable params: {trainable_params / 1e6:.2f}M")

    predictor_params = [p for p in model.predictor.parameters() if p.requires_grad]
    predictor_optimizer = torch.optim.AdamW(
        predictor_params,
        lr=config.lr_stage1,
        weight_decay=config.weight_decay,
    )
    scheduler = CosineAnnealingLR(predictor_optimizer, T_max=config.num_epochs_stage1) if config.num_epochs_stage1 > 0 else None

    best_val = float("inf")
    best_improve = float("-inf")

    for epoch in range(1, config.num_epochs_stage1 + 1):
        logging.info("Epoch %d/%d", epoch, config.num_epochs_stage1)
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
