import torch
import torch.nn.functional as F


def last_real_feature(local_queue: torch.Tensor, history_mask: torch.Tensor) -> torch.Tensor:
    valid = history_mask.to(device=local_queue.device, dtype=torch.bool)
    positions = torch.arange(local_queue.shape[1], device=local_queue.device)
    idx = positions.unsqueeze(0).expand_as(valid).masked_fill(~valid, -1).max(dim=1).values
    idx = idx.clamp_min(0)
    batch = torch.arange(local_queue.shape[0], device=local_queue.device)
    return local_queue[batch, idx]


def mean_cosine(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.cosine_similarity(pred, target, dim=-1).mean()


def compute_ema(
    history_feats: torch.Tensor,
    history_mask: torch.Tensor | None = None,
    alpha: float = 0.9,
) -> torch.Tensor:
    """Compute a mask-aware EMA over normalized history appearance features."""
    batch_size, history_len, _dim = history_feats.shape
    if history_mask is None:
        ema = history_feats[:, 0]
        for idx in range(1, history_len):
            ema = F.normalize(alpha * ema + (1.0 - alpha) * history_feats[:, idx], dim=-1)
        return F.normalize(ema, dim=-1)

    valid = history_mask.to(device=history_feats.device, dtype=torch.bool)
    first_idx = valid.float().argmax(dim=1)
    batch = torch.arange(batch_size, device=history_feats.device)
    ema = history_feats[batch, first_idx]
    for idx in range(history_len):
        feat = history_feats[:, idx]
        is_valid = valid[:, idx]
        updated = F.normalize(alpha * ema + (1.0 - alpha) * feat, dim=-1)
        ema = torch.where(is_valid.unsqueeze(-1), updated, ema)
    return F.normalize(ema, dim=-1)
