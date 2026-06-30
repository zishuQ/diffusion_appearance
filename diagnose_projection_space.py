import argparse
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from apu_diff.config import load_config
from apu_diff.datasets.feature_tracklet_dataset import resolve_feature_paths
from apu_diff.models import APUDiff
from apu_diff.utils.feature_io import load_feature_records
from apu_diff.utils.training import get_device, load_model_state, set_seed


def parse_args():
    parser = argparse.ArgumentParser("Diagnose APUDiff projection-space identity separation")
    parser.add_argument("--config", default="configs/apu_diff_default.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/stage1_predictor.pth")
    parser.add_argument("--feature-dir", default=None)
    parser.add_argument("--sequences", nargs="+", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--num-pairs", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def summarize(name, values):
    values = values.detach().cpu()
    print(
        f"{name}: mean={values.mean().item():.6f}, std={values.std().item():.6f}, "
        f"p05={values.quantile(0.05).item():.6f}, p95={values.quantile(0.95).item():.6f}"
    )


@torch.no_grad()
def main():
    args = parse_args()
    config = load_config(
        args.config,
        {
            "feature_dir": args.feature_dir,
            "sequences": args.sequences,
            "device": args.device,
        },
    )
    set_seed(args.seed)
    device = get_device(config.device)

    model = APUDiff(
        reid_dim=config.reid_dim,
        latent_dim=config.latent_dim,
        projection_hidden_dim=config.projection_hidden_dim,
        time_dim=config.time_dim,
        num_diffusion_steps=config.num_diffusion_steps,
        denoiser_hidden_dim=config.denoiser_hidden_dim,
        update_hidden_dim=config.update_hidden_dim,
        predictor_anchor_delta_scale=config.predictor_anchor_delta_scale,
    ).to(device)
    load_model_state(model, args.checkpoint, strict=False)
    model.eval()

    records = []
    for path in resolve_feature_paths(config):
        records.extend(load_feature_records(Path(path)))

    by_identity = defaultdict(list)
    for record in records:
        by_identity[(record["seq"], record["track_id"])].append(record)
    identities = [key for key, rows in by_identity.items() if len(rows) >= 2]
    if len(identities) < 2:
        raise ValueError("Need at least two identities with two records each.")

    rng = random.Random(args.seed)
    anchors = []
    positives = []
    negatives = []
    for _ in range(args.num_pairs):
        identity = rng.choice(identities)
        anchor, positive = rng.sample(by_identity[identity], 2)
        negative_identity = rng.choice(identities)
        while negative_identity == identity:
            negative_identity = rng.choice(identities)
        negative = rng.choice(by_identity[negative_identity])
        anchors.append(anchor["feat"])
        positives.append(positive["feat"])
        negatives.append(negative["feat"])

    anchor_raw = F.normalize(torch.as_tensor(np.stack(anchors), dtype=torch.float32, device=device), dim=-1)
    positive_raw = F.normalize(torch.as_tensor(np.stack(positives), dtype=torch.float32, device=device), dim=-1)
    negative_raw = F.normalize(torch.as_tensor(np.stack(negatives), dtype=torch.float32, device=device), dim=-1)
    anchor_z = model.project(anchor_raw)
    positive_z = model.project(positive_raw)
    negative_z = model.project(negative_raw)

    same_raw = F.cosine_similarity(anchor_raw, positive_raw, dim=-1)
    diff_raw = F.cosine_similarity(anchor_raw, negative_raw, dim=-1)
    same_proj = F.cosine_similarity(anchor_z, positive_z, dim=-1)
    diff_proj = F.cosine_similarity(anchor_z, negative_z, dim=-1)

    summarize("same_raw", same_raw)
    summarize("diff_raw", diff_raw)
    summarize("same_proj", same_proj)
    summarize("diff_proj", diff_proj)
    print(f"raw_gap={same_raw.mean().item() - diff_raw.mean().item():.6f}")
    print(f"proj_gap={same_proj.mean().item() - diff_proj.mean().item():.6f}")


if __name__ == "__main__":
    main()
