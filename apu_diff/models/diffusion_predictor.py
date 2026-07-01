import torch
import torch.nn as nn
import torch.nn.functional as F

from apu_diff.utils.diffusion import timestep_embedding


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


class ConditionTokenEmbedding(nn.Module):
    """Project local history and optional identity state to DDM context tokens.

    Local history tokens keep relative position embeddings.  The identity
    state is a learned per-track memory vector and is added as one typed token,
    not as a pile of stored historical frames.
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
        history_mask: torch.Tensor | None = None,
        identity_state: torch.Tensor | None = None,
        identity_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, history_len, _ = history_z.shape
        hist = self.history_proj(history_z) + self.history_type

        pos = timestep_embedding(
            torch.arange(history_len, device=history_z.device, dtype=torch.float32),
            hist.shape[-1],
        ).to(dtype=hist.dtype)
        tokens = hist + pos.unsqueeze(0)

        if history_mask is not None:
            ctx_mask = history_mask.to(device=history_z.device, dtype=tokens.dtype)
        else:
            ctx_mask = torch.ones(batch_size, history_len, device=history_z.device, dtype=tokens.dtype)
        if identity_state is None:
            return tokens, ctx_mask
        if identity_state.dim() == 3:
            if identity_state.shape[1] != 1:
                raise ValueError(f"identity_state must be [B,D] or [B,1,D], got {tuple(identity_state.shape)}")
            identity_state = identity_state[:, 0]
        id_tokens = self.identity_proj(identity_state).unsqueeze(1) + self.identity_type
        if identity_mask is None:
            id_mask = torch.ones(batch_size, 1, device=history_z.device, dtype=tokens.dtype)
        else:
            id_mask = identity_mask.view(batch_size, 1).to(device=history_z.device, dtype=tokens.dtype)
        return torch.cat([tokens, id_tokens], dim=1), torch.cat([ctx_mask, id_mask], dim=1)


class IdentityMemory(nn.Module):
    """Learned online identity state, maintained per track by the caller."""

    def __init__(self, latent_dim: int, hidden_dim: int | None = None):
        super().__init__()
        hidden_dim = int(hidden_dim or max(latent_dim // 2, 256))
        self.init_net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.update_net = nn.Sequential(
            nn.Linear(latent_dim * 4 + 1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.update_gate = nn.Sequential(
            nn.Linear(latent_dim * 4 + 1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def init_identity(self, first_feat: torch.Tensor) -> torch.Tensor:
        return F.normalize(first_feat + self.init_net(first_feat), dim=-1)

    def update_identity(
        self,
        identity_state: torch.Tensor,
        obs_feat: torch.Tensor,
        update_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        cos = F.cosine_similarity(identity_state, obs_feat, dim=-1, eps=1e-8).unsqueeze(-1)
        x = torch.cat(
            [
                identity_state,
                obs_feat,
                torch.abs(identity_state - obs_feat),
                identity_state * obs_feat,
                cos,
            ],
            dim=-1,
        )
        delta = self.update_net(x)
        gate = torch.sigmoid(self.update_gate(x))
        updated = F.normalize(identity_state + gate * delta, dim=-1)
        if update_mask is None:
            return updated
        mask = update_mask.to(device=identity_state.device, dtype=torch.bool).unsqueeze(-1)
        return torch.where(mask, updated, identity_state)

    def build_from_history(
        self,
        history_z: torch.Tensor,
        history_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, history_len, _ = history_z.shape
        if history_mask is None:
            identity = self.init_identity(history_z[:, 0])
            start = 1
            for idx in range(start, history_len):
                identity = self.update_identity(identity, history_z[:, idx])
            return identity

        valid = history_mask.to(device=history_z.device, dtype=torch.bool)
        first_idx = valid.float().argmax(dim=1)
        batch = torch.arange(batch_size, device=history_z.device)
        identity = self.init_identity(history_z[batch, first_idx])
        for idx in range(history_len):
            is_valid = valid[:, idx] & (idx != first_idx)
            identity = self.update_identity(identity, history_z[:, idx], is_valid)
        return identity


class HMINet(nn.Module):
    """DDM denoiser with cross-attention over history context tokens.

    At each denoising step the current residual / noise state together with
    the beta time embedding forms a query that cross-attends to the
    context_tokens (history observations only in the current main path).  The resulting
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
        latent_dim: int = 2048,
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
        self.context_encoder = ConditionTokenEmbedding(
            latent_dim=self.latent_dim,
            context_dim=self.context_dim,
        )
        self.identity_memory = IdentityMemory(latent_dim=self.latent_dim)
        self.net = HMINet(
            point_dim=self.latent_dim,
            context_dim=self.context_dim,
            hidden_dim=self.hidden_dim,
            tf_layer=2,
            num_heads=max(int(num_attention_heads), 1),
        )

    def encode_condition(
        self,
        history_z: torch.Tensor,
        history_mask: torch.Tensor | None = None,
        identity_state: torch.Tensor | None = None,
        identity_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.context_encoder(history_z, history_mask, identity_state, identity_mask)

    def _last_feature(
        self,
        history_z: torch.Tensor,
        history_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if history_mask is None:
            return history_z[:, -1]
        valid = history_mask.to(device=history_z.device, dtype=torch.bool)
        positions = torch.arange(history_z.shape[1], device=history_z.device)
        idx = positions.unsqueeze(0).expand_as(valid).masked_fill(~valid, -1).max(dim=1).values
        idx = idx.clamp_min(0)
        batch = torch.arange(history_z.shape[0], device=history_z.device)
        return history_z[batch, idx]

    def training_forward(
        self,
        history_z: torch.Tensor,
        target_z: torch.Tensor,
        history_mask: torch.Tensor | None = None,
        identity_state: torch.Tensor | None = None,
    ):
        if identity_state is None:
            identity_state = self.identity_memory.build_from_history(history_z, history_mask)
        context_tokens, context_mask = self.encode_condition(history_z, history_mask, identity_state)
        last = self._last_feature(history_z, history_mask)
        delta_target = target_z - last.detach()
        batch_size = target_z.shape[0]
        t = torch.rand(batch_size, device=target_z.device, dtype=target_z.dtype)
        t = t.mul(1.0 - self.min_train_t).add(self.min_train_t)
        noise = torch.randn_like(delta_target)
        c_target = -delta_target
        diffusion_state = self.ddm_forward(delta_target, noise, t, c_target)
        beta = self.to_beta(t)
        c_hat, step_context = self.net(diffusion_state, beta=beta, context_tokens=context_tokens, context_mask=context_mask)
        noise_hat = self.predict_noise(diffusion_state, c_hat, t)
        pred_delta = self.sample_delta_from_condition(
            context_tokens,
            context_mask,
            noise=torch.zeros_like(delta_target),
            sample_steps=1,
            deterministic=True,
        )
        pred_feat = F.normalize(last + pred_delta, dim=-1)
        return {
            "pred_feat": pred_feat,
            "last_feat": last,
            "pred_delta": pred_delta,
            "delta_target": delta_target,
            "cond": step_context,
            "context_tokens": context_tokens,
            "context_mask": context_mask,
            "identity_state": identity_state,
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
        history_mask: torch.Tensor | None = None,
        identity_state: torch.Tensor | None = None,
        noise: torch.Tensor | None = None,
        sample_steps: int | None = None,
        deterministic: bool = False,
    ) -> torch.Tensor:
        if identity_state is None:
            identity_state = self.identity_memory.build_from_history(history_z, history_mask)
        context_tokens, context_mask = self.encode_condition(history_z, history_mask, identity_state)
        last = self._last_feature(history_z, history_mask)
        if noise is None:
            noise = torch.randn(
                history_z.shape[0], self.latent_dim,
                device=history_z.device, dtype=history_z.dtype,
            )
        delta = self.sample_delta_from_condition(
            context_tokens,
            context_mask,
            noise=noise,
            sample_steps=sample_steps,
            deterministic=deterministic,
        )
        return F.normalize(last + delta, dim=-1)

    def deterministic_predict(
        self,
        history_z: torch.Tensor,
        history_mask: torch.Tensor | None = None,
        identity_state: torch.Tensor | None = None,
        sample_steps: int | None = None,
    ) -> torch.Tensor:
        if identity_state is None:
            identity_state = self.identity_memory.build_from_history(history_z, history_mask)
        context_tokens, context_mask = self.encode_condition(history_z, history_mask, identity_state)
        last = self._last_feature(history_z, history_mask)
        noise = torch.zeros(
            history_z.shape[0], self.latent_dim,
            device=history_z.device, dtype=history_z.dtype,
        )
        delta = self.sample_delta_from_condition(
            context_tokens,
            context_mask,
            noise=noise,
            sample_steps=sample_steps if sample_steps is not None else 1,
            deterministic=True,
        )
        return F.normalize(last + delta, dim=-1)

    def sample_delta_from_condition(
        self,
        context_tokens: torch.Tensor,
        context_mask: torch.Tensor,
        noise: torch.Tensor,
        sample_steps: int | None = None,
        deterministic: bool = True,
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
            x_t = self.pred_xtms_from_xt(x_t, noise_hat, c_hat, cur_time, s, deterministic=deterministic)
            cur_time = cur_time - s
        return x_t.clamp(-1.0, 1.0)

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
        deterministic: bool = False,
    ) -> torch.Tensor:
        time = t.clamp_min(self.min_train_t).unsqueeze(-1)
        s = s.unsqueeze(-1)
        mean = x_t + c_hat * (time - s) - c_hat * time - s / time.sqrt() * noise_hat
        if deterministic:
            return mean
        sigma = torch.sqrt((s * (time - s) / time).clamp_min(0.0))
        return mean + sigma * torch.randn_like(mean)
