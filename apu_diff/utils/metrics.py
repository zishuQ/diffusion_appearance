import torch


def last_real_feature(local_queue: torch.Tensor, history_mask: torch.Tensor) -> torch.Tensor:
    return local_queue[:, -1]


def mean_cosine(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    import torch.nn.functional as F

    return F.cosine_similarity(pred, target, dim=-1).mean()
