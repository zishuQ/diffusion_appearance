import torch
import torch.nn as nn
import torch.nn.functional as F

from .diffusion_predictor import DiffusionPredictor


class APUDiff(nn.Module):
    def __init__(
        self,
        reid_dim: int = 2048,
        latent_dim: int = 2048,
        time_dim: int = 64,
        num_diffusion_steps: int = 1,
        denoiser_hidden_dim: int = 512,
    ):
        super().__init__()
        self.reid_dim = int(reid_dim)
        self.latent_dim = int(latent_dim)
        if self.reid_dim != self.latent_dim:
            raise ValueError(
                "Raw-delta APUDiff requires reid_dim == latent_dim. "
                f"Got reid_dim={self.reid_dim}, latent_dim={self.latent_dim}."
            )
        self.predictor = DiffusionPredictor(
            latent_dim=self.latent_dim,
            time_dim=time_dim,
            num_diffusion_steps=num_diffusion_steps,
            denoiser_hidden_dim=denoiser_hidden_dim,
        )

    def project(self, feat_reid: torch.Tensor) -> torch.Tensor:
        squeeze = feat_reid.dim() == 1
        if squeeze:
            feat_reid = feat_reid.unsqueeze(0)
        if feat_reid.shape[-1] != self.latent_dim:
            raise ValueError(f"Expected {self.latent_dim}-d raw ReID features, got {feat_reid.shape[-1]}.")
        z = F.normalize(feat_reid, dim=-1)
        return z.squeeze(0) if squeeze else z

    @torch.no_grad()
    def predict(
        self,
        local_queue: torch.Tensor,
        history_mask: torch.Tensor,
        identity_state: torch.Tensor | None = None,
        deterministic: bool = True,
        sample_steps: int | None = None,
    ):
        squeeze = local_queue.dim() == 2
        if squeeze:
            local_queue = local_queue.unsqueeze(0)
            history_mask = history_mask.unsqueeze(0)
            if identity_state is not None:
                identity_state = identity_state.unsqueeze(0)
        pred = (
            self.predictor.deterministic_predict(
                local_queue,
                history_mask,
                identity_state=identity_state,
                sample_steps=sample_steps,
            )
            if deterministic
            else self.predictor.sample(
                local_queue,
                history_mask,
                identity_state=identity_state,
                sample_steps=sample_steps if sample_steps is not None else 1,
                deterministic=False,
            )
        )
        return pred.squeeze(0) if squeeze else pred

    def init_identity(self, first_feat: torch.Tensor) -> torch.Tensor:
        first_z = first_feat if first_feat.shape[-1] == self.latent_dim else self.project(first_feat)
        return self.predictor.identity_memory.init_identity(first_z)

    def update_identity(
        self,
        identity_state: torch.Tensor,
        obs_feat: torch.Tensor,
        update_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        obs_z = obs_feat if obs_feat.shape[-1] == self.latent_dim else self.project(obs_feat)
        return self.predictor.identity_memory.update_identity(identity_state, obs_z, update_mask)

    def build_identity(
        self,
        identity_feats: torch.Tensor,
        identity_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        identity_z = self.project(identity_feats)
        return self.predictor.identity_memory.build_from_history(identity_z, identity_mask)

    def appearance_cost(
        self,
        pred_feat: torch.Tensor,
        det_z: torch.Tensor,
        mode: str = "pred",
    ):
        if mode != "pred":
            raise ValueError(f"Raw-delta APUDiff only supports appearance_cost mode='pred', got {mode!r}.")
        c_pred = 1.0 - F.cosine_similarity(pred_feat, det_z, dim=-1)
        return c_pred

    def forward(self, local_queue: torch.Tensor, history_mask: torch.Tensor):
        return self.predict(local_queue, history_mask, deterministic=True)

    def predictor_training_forward(
        self,
        history_feats: torch.Tensor,
        history_mask: torch.Tensor,
        target_feat: torch.Tensor,
        identity_feats: torch.Tensor | None = None,
        identity_mask: torch.Tensor | None = None,
    ):
        local_queue = self.project(history_feats)
        target_z = self.project(target_feat)
        identity_state = None
        if identity_feats is not None:
            identity_state = self.build_identity(identity_feats, identity_mask)
        outputs = self.predictor.training_forward(local_queue, target_z, history_mask, identity_state=identity_state)
        outputs["target_z"] = target_z
        outputs["local_queue"] = local_queue
        outputs["pred_feat_det"] = outputs["pred_feat"]
        return outputs

    def freeze_predictor(self) -> None:
        for param in self.predictor.parameters():
            param.requires_grad = False

    def unfreeze_predictor(self) -> None:
        for param in self.predictor.parameters():
            param.requires_grad = True
