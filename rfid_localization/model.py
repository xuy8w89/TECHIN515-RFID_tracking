"""
Time-Aware Confidence-Gated Dual-Branch Tracker (2D).

- Per-channel features (built in dataset): RSSI_z, sin/cos phase (2048 wrap),
  masks, imputation flags, tslv, time-based confidence, r_dot, p_dot, unwrap_vel, …
- Per-timestep global: dt, log_dt, inv_dt
- Channel Transformer + pooled vector + GRU
- Absolute branch: p_abs_t from pooled RF + global + h_t
- Motion branch: v_t from h_t (bounded)
- p_dyn: recurrent integration from detached p with v (physics-style)
- alpha_t = sigmoid(MLP([h_t, pooled_conf, pooled_obs_rate])) -> fused p_hat
"""
from __future__ import annotations

import torch
import torch.nn as nn


class DualBranchTracker2D(nn.Module):
    IDX_CEXP = 8
    IDX_MR = 3
    IDX_MP = 4

    def __init__(
        self,
        num_channels: int,
        ch_in_dim: int,
        global_dim: int,
        d_ch: int = 32,
        gru_hidden: int = 128,
        n_heads: int = 4,
        dropout: float = 0.1,
        vel_scale: float = 0.08,
    ) -> None:
        super().__init__()
        self.num_channels = num_channels
        self.ch_in_dim = ch_in_dim
        self.global_dim = global_dim
        self.d_ch = d_ch
        self.gru_hidden = gru_hidden
        self.vel_scale = vel_scale

        self.ch_proj = nn.Linear(ch_in_dim + global_dim, d_ch)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_ch,
            nhead=n_heads,
            dim_feedforward=d_ch * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.channel_tx = nn.TransformerEncoder(enc_layer, num_layers=1)

        self.global_enc = nn.Linear(global_dim, d_ch)
        gru_in = d_ch * 2
        self.gru = nn.GRU(gru_in, gru_hidden, num_layers=1, batch_first=True)

        ctx_dim = gru_hidden + d_ch + global_dim
        self.abs_mlp = nn.Sequential(
            nn.Linear(ctx_dim, gru_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(gru_hidden, 2),
        )
        self.vel_mlp = nn.Sequential(
            nn.Linear(gru_hidden, gru_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(gru_hidden, 2),
        )
        self.alpha_mlp = nn.Sequential(
            nn.Linear(gru_hidden + 2, gru_hidden // 2),
            nn.ReLU(inplace=True),
            nn.Linear(gru_hidden // 2, 2),
        )

    def forward(
        self,
        ch_feats: torch.Tensor,
        attn_invalid: torch.Tensor,
        global_feats: torch.Tensor,
        xy0: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        ch_feats: [B, T, C, F]
        attn_invalid: [B, T, C] bool True=dead channel
        global_feats: [B, T, G]
        xy0: [B, 2] ground-truth start position (detached by caller) for p_dyn rollout
        returns p_hat, p_abs, p_dyn, v, alpha  each [B,T,2] except alpha last dim 2 for x,y gates
        """
        b, t, c, f_dim = ch_feats.shape
        g = global_feats
        g_expand = g.unsqueeze(2).expand(-1, -1, c, -1)
        x_in = torch.cat([ch_feats, g_expand], dim=-1)
        u = self.ch_proj(x_in)

        pad = attn_invalid.reshape(b * t, c)
        z = u.reshape(b * t, c, self.d_ch)
        z = self.channel_tx(z, src_key_padding_mask=pad)
        pool = z.mean(dim=1).view(b, t, self.d_ch)

        g_emb = self.global_enc(g)
        gru_in = torch.cat([pool, g_emb], dim=-1)
        h_seq, _ = self.gru(gru_in)

        mean_c = ch_feats[..., self.IDX_CEXP].mean(dim=2)
        mean_obs = torch.maximum(ch_feats[..., self.IDX_MR], ch_feats[..., self.IDX_MP]).mean(dim=2)
        alpha_in = torch.cat([h_seq, mean_c.unsqueeze(-1), mean_obs.unsqueeze(-1)], dim=-1)
        alpha = torch.sigmoid(self.alpha_mlp(alpha_in))

        ctx = torch.cat([h_seq, pool, g], dim=-1)
        p_abs = self.abs_mlp(ctx)
        v = self.vel_scale * torch.tanh(self.vel_mlp(h_seq))

        p_dyn = torch.zeros(b, t, 2, device=ch_feats.device, dtype=ch_feats.dtype)
        p_dyn[:, 0, :] = xy0
        for k in range(1, t):
            dtk = global_feats[:, k, 0].clamp_min(1e-4).unsqueeze(-1)
            step = v[:, k - 1, :] * dtk
            p_dyn[:, k, :] = p_dyn[:, k - 1, :].detach() + step

        p_hat = alpha * p_abs + (1.0 - alpha) * p_dyn
        return p_hat, p_abs, p_dyn, v, alpha
