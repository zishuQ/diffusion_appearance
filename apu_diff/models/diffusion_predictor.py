import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from apu_diff.utils.diffusion import l2_normalize, timestep_embedding


class MFL(nn.Module):
    """Modulated fully-connected layer used by DiffMOT-style HMINet."""

    def __init__(self, dim_in: int, dim_out: int, dim_ctx: int):
        super().__init__()
        self.linear = nn.Linear(dim_in, dim_out)
        self.hyper_bias = nn.Linear(dim_ctx, dim_out, bias=False)
        self.hyper_gate = nn.Linear(dim_ctx, dim_out)

    def forward(self, ctx: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.hyper_gate(ctx))
        bias = self.hyper_bias(ctx)
        return self.linear(x) * gate + bias


class HistoryTokenEmbedding(nn.Module):
    """Project history tokens to context space with type and position embeddings.

    Outputs context_tokens [B, K+1, D] and context_mask [B, K+1] for
    cross-attention inside the denoising network.  No TransformerEncoder
    pooling is applied here -- the denoiser attends to raw tokens.
    """

    def __init__(self, latent_dim: int, context_dim: int):
        super().__init__()
        self.history_proj = nn.Linear(latent_dim, context_dim)
        self.identity_proj = nn.Linear(latent_dim, context_dim)
        self.history_type = nn.Parameter(torch.zeros(1, 1, context_dim))
        self.identity_type = nn.Parameter(torch.zeros(1, 1, context_dim))
        nn.init.trunc_normal_(self.history_type, std=0.02)
        nn.init.trunc_normal_(self.identity_type, std=0.02)

    def forward(
        self,
        history_z: torch.Tensor,
        identity_token: torch.Tensor,
        history_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, history_len, _ = history_z.shape
        hist = self.history_proj(history_z) + self.history_type
        ident = self.identity_proj(identity_token).unsqueeze(1) + self.identity_type
        tokens = torch.cat([hist, ident], dim=1)

        pos = timestep_embedding(
            torch.arange(history_len + 1, device=history_z.device, dtype=torch.float32),
            tokens.shape[-1],
        ).to(dtype=tokens.dtype)
        tokens = tokens + pos.unsqueeze(0)

        if history_mask is not None:
            ctx_mask = torch.cat(
                [
                    history_mask.to(device=history_z.device, dtype=tokens.dtype),
                    torch.ones(batch_size, 1, device=history_z.device, dtype=tokens.dtype),
                ],
                dim=1,
            )
        else:
            ctx_mask = torch.ones(batch_size, history_len + 1, device=history_z.device, dtype=tokens.dtype)
        return tokens, ctx_mask


class HMINet(nn.Module):
    """DDM denoiser with cross-attention over history context tokens.

    At each denoising step the current residual / noise state together with
    the beta time embedding forms a query that cross-attends to the
    context_tokens (history observations + identity).  The resulting
    step_context vector then drives the MFL-modulated layers.
    """

    def __init__(
        self,
        point_dim: int,
        context_dim: int,
        hidden_dim: int,
        tf_layer: int = 2,
        num_heads: int = 4,
    ):
        super().__init__()
        self.context_dim = int(context_dim)
        self.hidden_dim = int(hidden_dim)
        ctx_dim = self.context_dim + 3
        mid_dim = max(self.context_dim // 2, 128)
        heads_hidden = num_heads if self.hidden_dim % num_heads == 0 else 1
        heads_context = num_heads if self.context_dim % num_heads == 0 else 1
        heads_cross = max(1, num_heads if context_dim % num_heads == 0 else 1)

        self.query_proj = nn.Linear(point_dim + 3, self.context_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.context_dim,
            num_heads=heads_cross,
            batch_first=True,
        )
        self.cross_norm = nn.LayerNorm(self.context_dim)

        self.concat1 = MFL(point_dim, mid_dim, ctx_dim)
        self.concat2 = MFL(mid_dim, self.context_dim, ctx_dim)
        self.concat3 = MFL(self.context_dim, self.hidden_dim, ctx_dim)
        layer1 = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=heads_hidden,
            dim_feedforward=self.hidden_dim * 2,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.transformer1 = nn.TransformerEncoder(layer1, num_layers=tf_layer)
        self.reduce1 = MFL(self.hidden_dim, self.context_dim, ctx_dim)
        layer2 = nn.TransformerEncoderLayer(
            d_model=self.context_dim,
            nhead=heads_context,
            dim_feedforward=self.context_dim * 2,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.transformer2 = nn.TransformerEncoder(layer2, num_layers=tf_layer)
        self.reduce2 = MFL(self.context_dim, mid_dim, ctx_dim)
        self.out = MFL(mid_dim, point_dim, ctx_dim)

    def forward(
        self,
        x: torch.Tensor,
        beta: torch.Tensor,
        context_tokens: torch.Tensor,
        context_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if beta.dim() == 1:
            beta = beta.unsqueeze(-1)
        time_emb = torch.cat([beta, torch.sin(beta), torch.cos(beta)], dim=-1)

        query = self.query_proj(torch.cat([x, time_emb], dim=-1))
        key_pad = None
        if context_mask is not None:
            key_pad = context_mask <= 0
        attn_out, _ = self.cross_attn(
            query=query.unsqueeze(1),
            key=context_tokens,
            value=context_tokens,
            key_padding_mask=key_pad,
        )
        step_context = self.cross_norm(attn_out.squeeze(1))

        ctx = torch.cat([time_emb, step_context], dim=-1)
        h = F.gelu(self.concat1(ctx, x))
        h = F.gelu(self.concat2(ctx, h))
        h = F.gelu(self.concat3(ctx, h))
        h = self.transformer1(h.unsqueeze(1)).squeeze(1)
        h = F.gelu(self.reduce1(ctx, h))
        h = self.transformer2(h.unsqueeze(1)).squeeze(1)
        h = F.gelu(self.reduce2(ctx, h))
        return self.out(ctx, h), step_context


class DiffusionPredictor(nn.Module):
    def __init__(
        self,
        latent_dim: int = 512,
        time_dim: int = 64,
        num_diffusion_steps: int = 1,
        denoiser_hidden_dim: int = 1024,
        num_encoder_layers: int = 6,
        num_attention_heads: int = 8,
        min_train_t: float = 1e-3,
    ):
        super().__init__()
        del time_dim
        del num_encoder_layers
        self.latent_dim = int(latent_dim)
        self.context_dim = int(latent_dim)
        self.hidden_dim = int(denoiser_hidden_dim)
        self.num_steps = max(int(num_diffusion_steps), 1)
        self.min_train_t = float(min_train_t)
        self.context_encoder = HistoryTokenEmbedding(
            latent_dim=self.latent_dim,
            context_dim=self.context_dim,
        )
        self.net = HMINet(
            point_dim=self.latent_dim,
            context_dim=self.context_dim,
            hidden_dim=self.hidden_dim,
            tf_layer=2,
            num_heads=4,
        )
        self.base_logit = nn.Parameter(torch.tensor(-4.5951199))
        self.history_mix_logit = nn.Parameter(torch.tensor(-2.1972246))
        self.residual_logit = nn.Parameter(torch.tensor(-2.1972246))
        self.base_head = nn.Sequential(
            nn.Linear(self.context_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.latent_dim),
        )
        self.base_score = nn.Sequential(
            nn.Linear(self.latent_dim + self.context_dim, max(self.hidden_dim // 2, 128)),
            nn.GELU(),
            nn.Linear(max(self.hidden_dim // 2, 128), 1),
        )
        nn.init.zeros_(self.base_score[-1].weight)
        nn.init.zeros_(self.base_score[-1].bias)

    def encode_condition(
        self,
        history_z: torch.Tensor,
        identity_token: torch.Tensor,
        history_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.context_encoder(history_z, identity_token, history_mask)

    def _compute_pooled_cond(
        self,
        context_tokens: torch.Tensor,
        context_mask: torch.Tensor,
    ) -> torch.Tensor:
        mask = context_mask.unsqueeze(-1)
        return (context_tokens * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)

    def _last_feature(
        self,
        history_z: torch.Tensor,
        history_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if history_mask is None:
            return history_z[:, -1]
        mask = history_mask.to(device=history_z.device, dtype=torch.long)
        idx = mask.sum(dim=1).clamp_min(1) - 1
        idx = idx + history_z.shape[1] - mask.sum(dim=1).clamp_min(1)
        batch = torch.arange(history_z.shape[0], device=history_z.device)
        return history_z[batch, idx]

    def _history_pool(
        self,
        history_z: torch.Tensor,
        cond: torch.Tensor,
        history_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        cond_tokens = cond.unsqueeze(1).expand(-1, history_z.shape[1], -1)
        score_input = torch.cat([history_z, cond_tokens], dim=-1)
        scores = self.base_score(score_input).squeeze(-1)
        recency_bias = torch.linspace(
            -1.0, 0.0, history_z.shape[1], device=history_z.device, dtype=scores.dtype,
        )
        scores = scores + recency_bias.unsqueeze(0)
        if history_mask is not None:
            valid = history_mask.to(device=history_z.device, dtype=torch.bool)
            scores = scores.masked_fill(~valid, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=1)
        return (history_z * weights.unsqueeze(-1)).sum(dim=1)

    def history_mix_scale(self) -> torch.Tensor:
        return torch.sigmoid(self.history_mix_logit)

    def base_scale(self) -> torch.Tensor:
        return torch.sigmoid(self.base_logit)

    def residual_scale(self) -> torch.Tensor:
        return torch.sigmoid(self.residual_logit)

    def make_base(
        self,
        history_z: torch.Tensor,
        cond: torch.Tensor,
        history_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        last = self._last_feature(history_z, history_mask)
        history_pool = self._history_pool(history_z, cond, history_mask)
        history_delta = history_pool - last
        base_delta = torch.tanh(self.base_head(cond))
        return l2_normalize(
            last + self.history_mix_scale() * history_delta + self.base_scale() * base_delta,
            dim=-1,
        )

    def apply_residual(self, base: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        return l2_normalize(base + self.residual_scale() * residual, dim=-1)

    def coarse_predict(
        self,
        history_z: torch.Tensor,
        identity_token: torch.Tensor,
        history_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (base, context_tokens, context_mask, pooled_cond)."""
        context_tokens, context_mask = self.encode_condition(history_z, identity_token, history_mask)
        pooled_cond = self._compute_pooled_cond(context_tokens, context_mask)
        base = self.make_base(history_z, pooled_cond, history_mask)
        return base, context_tokens, context_mask, pooled_cond

    def training_forward(
        self,
        history_z: torch.Tensor,
        identity_token: torch.Tensor,
        target_z: torch.Tensor,
        history_mask: torch.Tensor | None = None,
    ):
        base, context_tokens, context_mask, pooled_cond = self.coarse_predict(
            history_z, identity_token, history_mask,
        )
        batch_size = target_z.shape[0]
        t = torch.rand(batch_size, device=target_z.device, dtype=target_z.dtype)
        t = t.mul(1.0 - self.min_train_t).add(self.min_train_t)
        noise = torch.randn_like(target_z)
        residual_target = target_z - base.detach()
        c_target = -residual_target
        diffusion_state = self.ddm_forward(residual_target, noise, t, c_target)
        beta = self.to_beta(t)
        c_hat, step_context = self.net(diffusion_state, beta=beta, context_tokens=context_tokens, context_mask=context_mask)
        noise_hat = self.predict_noise(diffusion_state, c_hat, t)
        pred_residual = self.sample_residual_from_condition(
            context_tokens, context_mask, noise=torch.zeros_like(target_z), sample_steps=1,
        )
        pred_sample_residual = self.sample_residual_from_condition(
            context_tokens, context_mask, noise=torch.randn_like(target_z), sample_steps=1,
        )
        pred_feat = self.apply_residual(base, pred_residual)
        pred_feat_sample = self.apply_residual(base, pred_sample_residual)
        return {
            "pred_feat": pred_feat,
            "pred_feat_sample": pred_feat_sample,
            "mu": base,
            "cond": step_context,
            "context_tokens": context_tokens,
            "context_mask": context_mask,
            "pooled_cond": pooled_cond,
            "base": base,
            "base_scale": self.base_scale(),
            "history_mix_scale": self.history_mix_scale(),
            "residual_scale": self.residual_scale(),
            "residual_target": residual_target,
            "c_target": c_target,
            "c_hat": c_hat,
            "noise": noise,
            "noise_hat": noise_hat,
            "diffusion_state": diffusion_state,
            "timesteps": t,
        }

    def sample(
        self,
        history_z: torch.Tensor,
        identity_token: torch.Tensor,
        history_mask: torch.Tensor | None = None,
        noise: torch.Tensor | None = None,
        sample_steps: int | None = None,
    ) -> torch.Tensor:
        base, context_tokens, context_mask, _pooled = self.coarse_predict(
            history_z, identity_token, history_mask,
        )
        if noise is None:
            noise = torch.randn(
                identity_token.shape[0], self.latent_dim,
                device=identity_token.device, dtype=identity_token.dtype,
            )
        residual = self.sample_residual_from_condition(
            context_tokens, context_mask, noise=noise, sample_steps=sample_steps,
        )
        return self.apply_residual(base, residual)

    def deterministic_predict(
        self,
        history_z: torch.Tensor,
        identity_token: torch.Tensor,
        history_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        base, context_tokens, context_mask, _pooled = self.coarse_predict(
            history_z, identity_token, history_mask,
        )
        noise = torch.zeros(
            identity_token.shape[0], self.latent_dim,
            device=identity_token.device, dtype=identity_token.dtype,
        )
        residual = self.sample_residual_from_condition(
            context_tokens, context_mask, noise=noise, sample_steps=1,
        )
        return self.apply_residual(base, residual)

    def sample_residual_from_condition(
        self,
        context_tokens: torch.Tensor,
        context_mask: torch.Tensor,
        noise: torch.Tensor,
        sample_steps: int | None = None,
    ) -> torch.Tensor:
        steps = max(int(sample_steps if sample_steps is not None else self.num_steps), 1)
        x_t = noise
        cur_time = torch.ones((noise.shape[0],), device=noise.device, dtype=noise.dtype)
        step_size = 1.0 / steps
        for step in range(steps, 0, -1):
            s = torch.full_like(cur_time, step_size)
            if step == 1:
                s = cur_time
            beta = self.to_beta(cur_time)
            c_hat, _step_ctx = self.net(x_t, beta=beta, context_tokens=context_tokens, context_mask=context_mask)
            noise_hat = self.predict_noise(x_t, c_hat, cur_time)
            x0 = self.pred_x0_from_xt(x_t, noise_hat, c_hat, cur_time).clamp(-1.0, 1.0)
            c_hat = -x0
            x_t = self.pred_xtms_from_xt(x_t, noise_hat, c_hat, cur_time, s)
            cur_time = cur_time - s
        return l2_normalize(x_t, dim=-1)

    def to_beta(self, t: torch.Tensor) -> torch.Tensor:
        return t.clamp_min(self.min_train_t).log() / 4.0

    def ddm_forward(self, clean: torch.Tensor, noise: torch.Tensor, t: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        time = t.clamp_min(self.min_train_t).unsqueeze(-1)
        return clean + c * time + time.sqrt() * noise

    def predict_noise(self, x_t: torch.Tensor, c_hat: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        time = t.clamp_min(self.min_train_t).unsqueeze(-1)
        return (x_t - (time - 1.0) * c_hat) / time.sqrt()

    def pred_x0_from_xt(
        self,
        x_t: torch.Tensor,
        noise_hat: torch.Tensor,
        c_hat: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        time = t.clamp_min(self.min_train_t).unsqueeze(-1)
        return x_t - c_hat * time - time.sqrt() * noise_hat

    def pred_xtms_from_xt(
        self,
        x_t: torch.Tensor,
        noise_hat: torch.Tensor,
        c_hat: torch.Tensor,
        t: torch.Tensor,
        s: torch.Tensor,
    ) -> torch.Tensor:
        time = t.clamp_min(self.min_train_t).unsqueeze(-1)
        s = s.unsqueeze(-1)
        mean = x_t + c_hat * (time - s) - c_hat * time - s / time.sqrt() * noise_hat
        sigma = torch.sqrt((s * (time - s) / time).clamp_min(0.0))
        return mean + sigma * torch.randn_like(mean)
