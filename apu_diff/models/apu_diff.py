import torch
import torch.nn as nn
import torch.nn.functional as F

from .diffusion_predictor import DiffusionPredictor
from .projection import ProjectionHead
from .update_block import CrossAttentionGate, UpdateBlock


class APUDiff(nn.Module):
    def __init__(
        self,
        reid_dim: int = 2048,
        latent_dim: int = 2048,
        projection_hidden_dim: int = 512,
        time_dim: int = 64,
        num_diffusion_steps: int = 1,
        denoiser_hidden_dim: int = 512,
        update_hidden_dim: int = 512,
    ):
        super().__init__()
        self.reid_dim = int(reid_dim)
        self.latent_dim = int(latent_dim)
        self.projection = ProjectionHead(
            reid_dim=self.reid_dim,
            latent_dim=self.latent_dim,
            hidden_dim=projection_hidden_dim,
        )
        for param in self.projection.parameters():
            param.requires_grad = False
        self.predictor = DiffusionPredictor(
            latent_dim=self.latent_dim,
            time_dim=time_dim,
            num_diffusion_steps=num_diffusion_steps,
            denoiser_hidden_dim=denoiser_hidden_dim,
        )
        self.update_block = UpdateBlock(latent_dim=self.latent_dim, hidden_dim=update_hidden_dim)
        self.cross_attn_gate = CrossAttentionGate(latent_dim=self.latent_dim)

    def project(self, feat_reid: torch.Tensor) -> torch.Tensor:
        squeeze = feat_reid.dim() == 1
        if squeeze:
            feat_reid = feat_reid.unsqueeze(0)
        z = F.normalize(feat_reid, dim=-1)
        return z.squeeze(0) if squeeze else z

    def init_identity(self, local_queue: torch.Tensor, history_mask: torch.Tensor) -> torch.Tensor:
        squeeze = local_queue.dim() == 2
        if squeeze:
            local_queue = local_queue.unsqueeze(0)
            history_mask = history_mask.unsqueeze(0)
        idx = history_mask.float().argmax(dim=1)
        # Masks are left padded, so the last real feature is always the last position.
        idx = torch.where(history_mask[:, -1] > 0, torch.full_like(idx, local_queue.shape[1] - 1), idx)
        batch = torch.arange(local_queue.shape[0], device=local_queue.device)
        identity = local_queue[batch, idx]
        return identity.squeeze(0) if squeeze else identity

    @torch.no_grad()
    def predict(
        self,
        local_queue: torch.Tensor,
        history_mask: torch.Tensor,
        identity_token: torch.Tensor | None = None,
        deterministic: bool = True,
        sample_steps: int | None = None,
    ):
        squeeze = local_queue.dim() == 2
        if squeeze:
            local_queue = local_queue.unsqueeze(0)
            history_mask = history_mask.unsqueeze(0)
        pred = (
            self.predictor.deterministic_predict(
                local_queue,
                history_mask,
                sample_steps=sample_steps,
            )
            if deterministic
            else self.predictor.sample(
                local_queue,
                history_mask,
                sample_steps=sample_steps if sample_steps is not None else 1,
                deterministic=False,
            )
        )
        return pred.squeeze(0) if squeeze else pred

    def update(self, pred_feat: torch.Tensor, obs_feat: torch.Tensor, identity_token: torch.Tensor):
        squeeze = pred_feat.dim() == 1
        if squeeze:
            pred_feat = pred_feat.unsqueeze(0)
            obs_feat = obs_feat.unsqueeze(0)
            identity_token = identity_token.unsqueeze(0)
        obs_z = obs_feat if obs_feat.shape[-1] == self.latent_dim else self.project(obs_feat)
        updated_local, updated_identity = self.update_block(pred_feat, obs_z, identity_token)
        if squeeze:
            return updated_local.squeeze(0), updated_identity.squeeze(0)
        return updated_local, updated_identity

    def match_gate_value(self, pred_feat: torch.Tensor, det_z: torch.Tensor, identity_token: torch.Tensor) -> torch.Tensor:
        _, attn = self.cross_attn_gate(pred_feat, det_z, identity_token)
        return attn.mean(dim=-1)

    def appearance_cost(
        self,
        pred_feat: torch.Tensor,
        det_z: torch.Tensor,
        identity_token: torch.Tensor,
        mode: str = "pred",
    ):
        c_pred = 1.0 - F.cosine_similarity(pred_feat, det_z, dim=-1)
        c_mem = 1.0 - F.cosine_similarity(identity_token, det_z, dim=-1)
        if mode == "pred":
            return c_pred
        if mode == "identity":
            return c_mem
        if mode == "avg":
            return 0.5 * c_pred + 0.5 * c_mem
        if mode == "min":
            return torch.minimum(c_mem, c_pred)
        raise ValueError(f"Unknown APUDiff appearance cost mode: {mode!r}")

    def forward(self, local_queue: torch.Tensor, history_mask: torch.Tensor, identity_token: torch.Tensor, obs_feat: torch.Tensor):
        pred_feat = self.predict(local_queue, history_mask, identity_token, deterministic=True)
        updated_local, updated_identity = self.update(pred_feat, obs_feat, identity_token)
        return pred_feat, updated_local, updated_identity

    def predictor_training_forward(
        self,
        history_feats: torch.Tensor,
        history_mask: torch.Tensor,
        target_feat: torch.Tensor,
    ):
        local_queue = self.project(history_feats)
        target_z = self.project(target_feat)
        outputs = self.predictor.training_forward(local_queue, target_z, history_mask)
        outputs["target_z"] = target_z
        outputs["local_queue"] = local_queue
        outputs["pred_feat_det"] = outputs["pred_feat"]
        return outputs

    def freeze_predictor(self) -> None:
        for module in (self.projection, self.predictor):
            for param in module.parameters():
                param.requires_grad = False

    def unfreeze_all(self) -> None:
        for param in self.parameters():
            param.requires_grad = True
