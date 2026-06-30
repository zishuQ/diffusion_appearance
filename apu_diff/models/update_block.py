import torch
import torch.nn as nn
import torch.nn.functional as F

from apu_diff.utils.diffusion import l2_normalize


class UpdateBlock(nn.Module):
    def __init__(self, latent_dim: int = 256, hidden_dim: int = 512, local_delta_scale: float = 0.5):
        super().__init__()
        self.local_delta_scale = float(local_delta_scale)
        self.shared = nn.Sequential(
            nn.Linear(latent_dim * 5 + 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.local_head = nn.Linear(hidden_dim, latent_dim)
        self.identity_candidate = nn.Linear(hidden_dim, latent_dim)
        self.identity_cell = nn.GRUCell(latent_dim, latent_dim)
        nn.init.zeros_(self.local_head.weight)
        nn.init.zeros_(self.local_head.bias)

    def forward(self, pred_feat: torch.Tensor, obs_z: torch.Tensor, identity_token: torch.Tensor):
        x = _build_update_input(pred_feat, obs_z, identity_token)
        h = self.shared(x)
        local_delta = torch.tanh(self.local_head(h)) * self.local_delta_scale
        updated_local_feat = l2_normalize(obs_z + local_delta, dim=-1)
        candidate_identity = self.identity_candidate(h)
        updated_identity_token = self.identity_cell(candidate_identity, identity_token)
        updated_identity_token = l2_normalize(updated_identity_token, dim=-1)
        return updated_local_feat, updated_identity_token


class CrossAttentionGate(nn.Module):
    """Channel-wise gate blending identity (memory) and pred_feat (prediction).

    Produces a gate vector [B, D] using per-channel features:
        pred_i, det_i, identity_i, |pred_i - det_i|, |identity_i - det_i|
    plus global cosine similarities cos(pred, det) and cos(identity, det).
    gated_feat = L2(gate * identity + (1 - gate) * pred).
    """

    def __init__(self, latent_dim: int):
        super().__init__()
        hidden_dim = max(int(latent_dim) // 4, 16)
        self.gate_net = nn.Sequential(
            nn.Linear(7, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.cos_weight = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        pred_feat: torch.Tensor,
        det_z: torch.Tensor,
        identity_token: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, D = pred_feat.shape

        cos_pred_det = F.cosine_similarity(pred_feat, det_z, dim=-1)  # [B]
        cos_id_det = F.cosine_similarity(identity_token, det_z, dim=-1)  # [B]

        per_channel = torch.stack(
            [
                pred_feat,
                det_z,
                identity_token,
                torch.abs(pred_feat - det_z),
                torch.abs(identity_token - det_z),
                cos_pred_det.unsqueeze(-1).expand(-1, D),
                cos_id_det.unsqueeze(-1).expand(-1, D),
            ],
            dim=-1,
        )  # [B, D, 7]

        flat = per_channel.reshape(B * D, 7)
        gate_flat = torch.sigmoid(self.gate_net(flat)).squeeze(-1)  # [B*D]
        gate = gate_flat.reshape(B, D)

        cos_bias = self.cos_weight * (cos_id_det - cos_pred_det).unsqueeze(-1)
        gate = torch.sigmoid(torch.logit(gate.clamp(1e-6, 1.0 - 1e-6)) + cos_bias)

        gated_feat = gate * identity_token + (1.0 - gate) * pred_feat
        gated_feat = l2_normalize(gated_feat, dim=-1)
        return gated_feat, gate


def _build_update_input(pred_feat: torch.Tensor, obs_z: torch.Tensor, identity_token: torch.Tensor) -> torch.Tensor:
    cos_pred_obs = F.cosine_similarity(pred_feat, obs_z, dim=-1).unsqueeze(-1)
    cos_id_obs = F.cosine_similarity(identity_token, obs_z, dim=-1).unsqueeze(-1)
    return torch.cat(
        [
            pred_feat,
            obs_z,
            identity_token,
            torch.abs(pred_feat - obs_z),
            torch.abs(identity_token - obs_z),
            cos_pred_obs,
            cos_id_obs,
        ],
        dim=-1,
    )
