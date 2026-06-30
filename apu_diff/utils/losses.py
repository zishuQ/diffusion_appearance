import torch
import torch.nn.functional as F


def cosine_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return 1.0 - F.cosine_similarity(pred, target, dim=-1).mean()


def mean_cosine(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.cosine_similarity(pred, target, dim=-1).mean()


def cosine_margin_loss(
    anchor: torch.Tensor,
    positive: torch.Tensor,
    negative: torch.Tensor,
    margin: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pos_cos = F.cosine_similarity(anchor, positive, dim=-1)
    neg_cos = F.cosine_similarity(anchor, negative, dim=-1)
    loss = F.relu(float(margin) + neg_cos - pos_cos).mean()
    return loss, pos_cos.mean(), neg_cos.mean()


def supervised_contrastive_loss(
    features: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    features = F.normalize(features, p=2, dim=-1)
    labels = labels.view(-1, 1)
    positive_mask = torch.eq(labels, labels.T).float().to(features.device)
    self_mask = torch.eye(features.shape[0], device=features.device)
    positive_mask = positive_mask * (1.0 - self_mask)

    logits = features @ features.T / float(temperature)
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    exp_logits = torch.exp(logits) * (1.0 - self_mask)
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12))

    positive_count = positive_mask.sum(dim=1)
    valid = positive_count > 0
    if not valid.any():
        return features.new_tensor(0.0)
    mean_log_prob_pos = (positive_mask * log_prob).sum(dim=1) / positive_count.clamp_min(1.0)
    return -mean_log_prob_pos[valid].mean()


def pairwise_similarity_distillation(projected: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    projected = F.normalize(projected, p=2, dim=-1)
    reference = F.normalize(reference, p=2, dim=-1)
    projected_sim = projected @ projected.T
    reference_sim = reference @ reference.T
    mask = ~torch.eye(projected.shape[0], device=projected.device, dtype=torch.bool)
    return F.smooth_l1_loss(projected_sim[mask], reference_sim[mask])
