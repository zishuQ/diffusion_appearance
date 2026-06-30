import torch
import torch.nn as nn

from apu_diff.utils.diffusion import l2_normalize


class ProjectionHead(nn.Module):
    def __init__(self, reid_dim: int = 2048, latent_dim: int = 256, hidden_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(reid_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return l2_normalize(self.net(feat), dim=-1)
