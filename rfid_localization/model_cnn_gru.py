"""CNN(2x4 grid) + GRU 2D tracker.

Per-cell (antenna x tag) feature vector (11 dims):
  [RSSI_z, sin_phase, cos_phase, phase_dot, rssi_dot,
   phase_mask, rssi_mask, confidence, is_imputed,
   x_local, y_local]

`is_imputed` = max(imp_r, imp_p) from the current dataset pipeline.
`x_local, y_local` are broadcast constants from the grid layout (e.g. A4 corners).

Architecture:
  per-frame: Conv2d(F->32, k=2) -> ReLU -> Conv2d(32->64, k=1) -> ReLU
             -> Flatten -> [optional concat global_feats (dt, log_dt, inv_dt)]
             -> Linear -> frame_embed (+ Dropout)
  per-seq  : GRU(frame_embed -> hidden)
             -> MLP_pos (hidden -> 2)
             -> MLP_vel (hidden -> 2, tanh scaled)
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class CNNGRUTracker(nn.Module):
    # Column indices into ch_feats (14-dim, as built in dataset.build_enriched_table).
    # When use_pdoa=True, the dataset appends 2 extra columns (pdoa_sin, pdoa_cos)
    # at indices 14, 15 and we pull those in too.
    BASE_FEAT_INDICES = (0, 1, 2, 11, 9, 4, 3, 8)
    PDOA_FEAT_INDICES = (14, 15)
    IDX_IMP_R = 5
    IDX_IMP_P = 6

    def __init__(
        self,
        cell_to_channel: np.ndarray,
        local_xy: np.ndarray,
        frame_embed_dim: int = 128,
        gru_hidden: int = 128,
        dropout: float = 0.1,
        vel_scale: float = 0.08,
        global_feat_dim: int = 0,
        anchor_decoder: bool = False,
        anchor_init_scale: float = 0.1,
        residual_scale: float = 0.05,
        use_pdoa: bool = False,
    ) -> None:
        super().__init__()
        if cell_to_channel.ndim != 2:
            raise ValueError("cell_to_channel must be [H, W]")
        H, W = int(cell_to_channel.shape[0]), int(cell_to_channel.shape[1])
        if H < 2 or W < 2:
            raise ValueError("CNN path requires at least a 2x2 grid (kernel_size=2)")
        if int(cell_to_channel.min()) < 0:
            raise ValueError("cell_to_channel has -1 (incomplete grid) not supported")
        self.H = H
        self.W = W
        self.frame_embed_dim = int(frame_embed_dim)
        self.gru_hidden = int(gru_hidden)
        self.vel_scale = float(vel_scale)
        self.global_feat_dim = int(global_feat_dim)
        self.anchor_decoder = bool(anchor_decoder)
        self.residual_scale = float(residual_scale)
        self.n_tags = W
        self.use_pdoa = bool(use_pdoa)
        self.feat_indices = tuple(self.BASE_FEAT_INDICES) + (
            tuple(self.PDOA_FEAT_INDICES) if self.use_pdoa else tuple()
        )

        self.register_buffer(
            "cell_to_channel",
            torch.from_numpy(np.ascontiguousarray(cell_to_channel).astype(np.int64)),
        )
        self.register_buffer(
            "local_xy",
            torch.from_numpy(np.ascontiguousarray(local_xy).astype(np.float32)),
        )

        f_grid = len(self.feat_indices) + 1 + 2  # base (+ pdoa) + is_imputed + (x_local, y_local)
        self.f_grid = f_grid

        self.conv = nn.Sequential(
            nn.Conv2d(f_grid, 32, kernel_size=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=1),
            nn.ReLU(inplace=True),
        )

        flat_dim = 64 * (H - 1) * (W - 1)
        self.frame_embed = nn.Linear(flat_dim + self.global_feat_dim, self.frame_embed_dim)
        self.dropout = nn.Dropout(p=float(dropout))

        self.gru = nn.GRU(
            self.frame_embed_dim, self.gru_hidden, num_layers=1, batch_first=True
        )

        if self.anchor_decoder:
            # Per-tag learnable anchor positions in the (xy_offset-centered) target frame.
            # Init from local_xy (column 0 of each tag), scaled so the initial anchors
            # are roughly in the right spatial range.
            per_tag_xy = np.asarray(local_xy, dtype=np.float32)[0, :, :] * float(anchor_init_scale)
            self.anchor_xy = nn.Parameter(torch.from_numpy(per_tag_xy).clone())
            self.tag_logit_head = nn.Sequential(
                nn.Linear(self.gru_hidden, self.gru_hidden),
                nn.ReLU(inplace=True),
                nn.Linear(self.gru_hidden, self.n_tags),
            )
            self.residual_head = nn.Sequential(
                nn.Linear(self.gru_hidden, self.gru_hidden),
                nn.ReLU(inplace=True),
                nn.Linear(self.gru_hidden, 2),
            )
            self.pos_mlp = None
        else:
            self.pos_mlp = nn.Sequential(
                nn.Linear(self.gru_hidden, self.gru_hidden),
                nn.ReLU(inplace=True),
                nn.Linear(self.gru_hidden, 2),
            )
        self.vel_mlp = nn.Sequential(
            nn.Linear(self.gru_hidden, self.gru_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(self.gru_hidden, 2),
        )

    def build_grid(self, ch_feats: torch.Tensor) -> torch.Tensor:
        """[B, T, C, F_in] -> [B, T, H, W, f_grid].

        F_in is 14 by default, or 16 when use_pdoa=True (last 2 dims = pdoa_sin/cos).
        """
        base_idx = list(self.feat_indices)
        base = ch_feats[..., base_idx]
        imp = torch.maximum(
            ch_feats[..., self.IDX_IMP_R], ch_feats[..., self.IDX_IMP_P]
        ).unsqueeze(-1)
        feat = torch.cat([base, imp], dim=-1)  # [B, T, C, len(feat_indices)+1]
        f_per_cell = feat.shape[-1]

        h_w = self.H * self.W
        flat_idx = self.cell_to_channel.reshape(h_w)
        gathered = feat.index_select(dim=-2, index=flat_idx)  # [B, T, H*W, f_per_cell]
        b, t = gathered.shape[0], gathered.shape[1]
        grid = gathered.view(b, t, self.H, self.W, f_per_cell)

        local = self.local_xy.view(1, 1, self.H, self.W, 2).expand(b, t, self.H, self.W, 2)
        grid = torch.cat([grid, local], dim=-1)  # [B, T, H, W, f_grid]
        return grid

    def forward(
        self,
        ch_feats: torch.Tensor,
        global_feats: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """ch_feats [B, T, C, 14] (+ optional global_feats [B, T, G]) -> (p [B, T, 2], v [B, T, 2])."""
        grid = self.build_grid(ch_feats)
        b, t, h, w, f = grid.shape
        x = grid.permute(0, 1, 4, 2, 3).contiguous().view(b * t, f, h, w)
        x = self.conv(x)
        x = x.reshape(b * t, -1)
        if self.global_feat_dim > 0:
            if global_feats is None:
                raise ValueError(
                    f"model was built with global_feat_dim={self.global_feat_dim} "
                    "but forward() got global_feats=None"
                )
            g = global_feats.reshape(b * t, -1)
            if g.shape[-1] != self.global_feat_dim:
                raise ValueError(
                    f"global_feats last dim {g.shape[-1]} != global_feat_dim {self.global_feat_dim}"
                )
            x = torch.cat([x, g], dim=-1)
        x = self.frame_embed(x)
        x = self.dropout(x)
        x = x.view(b, t, -1)

        h_seq, _ = self.gru(x)
        if self.anchor_decoder:
            logits = self.tag_logit_head(h_seq)                          # [B, T, n_tags]
            w = torch.softmax(logits, dim=-1)                            # [B, T, n_tags]
            anchor_pred = torch.einsum("btn,nk->btk", w, self.anchor_xy) # [B, T, 2]
            residual = self.residual_scale * torch.tanh(self.residual_head(h_seq))
            p = anchor_pred + residual
        else:
            p = self.pos_mlp(h_seq)
        v = self.vel_scale * torch.tanh(self.vel_mlp(h_seq))
        return p, v


def build_cnn_gru_from_ckpt(ckpt: dict) -> CNNGRUTracker:
    """Reconstruct CNNGRUTracker from a saved checkpoint dict."""
    layout = ckpt["grid_layout"]
    return CNNGRUTracker(
        cell_to_channel=np.asarray(layout["cell_to_channel"], dtype=np.int64),
        local_xy=np.asarray(layout["local_xy"], dtype=np.float32),
        frame_embed_dim=int(ckpt.get("frame_embed_dim", 128)),
        gru_hidden=int(ckpt.get("hidden", 128)),
        dropout=float(ckpt.get("dropout", 0.1)),
        vel_scale=float(ckpt.get("vel_scale", 0.08)),
        global_feat_dim=int(ckpt.get("global_feat_dim", 0)),
        anchor_decoder=bool(ckpt.get("anchor_decoder", False)),
        anchor_init_scale=float(ckpt.get("anchor_init_scale", 0.1)),
        residual_scale=float(ckpt.get("residual_scale", 0.05)),
        use_pdoa=bool(ckpt.get("use_pdoa", False)),
    )
