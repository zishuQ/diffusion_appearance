import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def l2_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-12) -> torch.Tensor:
    return F.normalize(x, p=2, dim=dim, eps=eps)


def timestep_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=timesteps.device, dtype=torch.float32) / max(half - 1, 1)
    )
    args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb


class DiffusionSchedule(nn.Module):
    def __init__(self, num_steps: int, beta_start: float = 1e-4, beta_end: float = 2e-2):
        super().__init__()
        if num_steps < 1:
            raise ValueError("num_steps must be >= 1")
        betas = torch.linspace(beta_start, beta_end, num_steps, dtype=torch.float32)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        self.num_steps = num_steps
        self.register_buffer("alpha_bars", alpha_bars)

    def add_noise(self, residual: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        alpha_bar = self.alpha_bars[t].to(device=residual.device, dtype=residual.dtype).unsqueeze(-1)
        return alpha_bar.sqrt() * residual + (1.0 - alpha_bar).sqrt() * noise
