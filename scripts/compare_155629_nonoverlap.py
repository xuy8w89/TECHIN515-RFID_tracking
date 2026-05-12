"""Compare 4 models on the same 30-frame slice of CSV 155629
using SLIDING-WINDOW inference (each frame's prediction comes from the
window ending at that frame; same as eval_cnn_gru's per-CSV plots).

The slice covers frames 10..39, i.e. the same time range as the original
non-overlapping windows 11/21/31, but every output frame now sees a full
10-frame causal window so the trajectory is naturally continuous.

- algo  : DualBranchTracker2D with unwrap=algo, w=10  (checkpoints/rfid_0421_w10_algo.pt)
- v1    : CNN+GRU (baseline, old local_xy)            (checkpoints/rfid_cnn_gru_w10_algo.pt)
- v2f   : CNN+GRU tuned, flipped local_xy             (checkpoints/rfid_cnn_gru_w10_algo_v2f.pt)
- ens   : CNN+GRU v5 3-seed ensemble, flipped         (v5 + s123 + s456)

Saves a 2x2 grid of trajectory plots (GT vs Pred per model) and a
2x2 grid of x(t), y(t) plots.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from rfid_localization.dataset import (
    RFIDTrackingWindowDataset,
    build_enriched_table,
)
from rfid_localization.model import DualBranchTracker2D
from rfid_localization.model_cnn_gru import build_cnn_gru_from_ckpt
from rfid_localization.train import collate


CSV_PATH = Path("data/0421/test/aruco_rfid_20260421_155629.csv")
# Frame range we want predictions for (matches the time span covered by
# the original non-overlapping windows 11/21/31).
FRAME_START = 10
FRAME_END = 40  # exclusive
OUT_DIR = Path("checkpoints/plots_comparison_155629")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _load_dual_branch(path: Path, dev):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = DualBranchTracker2D(
        num_channels=len(ckpt["pairs"]),
        ch_in_dim=int(ckpt.get("ch_feat_dim", 14)),
        global_dim=int(ckpt.get("global_dim", 3)),
        d_ch=int(ckpt.get("d_ch", 32)),
        gru_hidden=int(ckpt.get("hidden", 128)),
        vel_scale=float(ckpt.get("vel_scale", 0.08)),
    ).to(dev)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt


def _load_cnn_gru(path: Path, dev):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = build_cnn_gru_from_ckpt(ckpt).to(dev)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt


def _build_ds_for(ckpt, csv_path: Path) -> RFIDTrackingWindowDataset:
    pairs = ckpt["pairs"]
    r_mean = np.asarray(ckpt["rssi_mean"], dtype=np.float32)
    r_std = np.asarray(ckpt["rssi_std"], dtype=np.float32)
    offset = np.asarray(ckpt["xy_offset"], dtype=np.float32)
    window = int(ckpt["window"])
    df = pd.read_csv(csv_path)
    table = build_enriched_table(
        df, pairs, r_mean, r_std,
        phase_period=float(ckpt["phase_period"]),
        conf_lambda=float(ckpt["conf_lambda"]),
        unwrap_method=str(ckpt.get("unwrap_method", "arctan2")),
        recover=bool(ckpt.get("recover", False)),
    )
    return RFIDTrackingWindowDataset(table, window=window, xy_offset=offset), offset


def _frame_to_window_indices(frame_lo: int, frame_hi: int, window: int, n_windows: int) -> list[int]:
    """Window index j has its last frame at j + window - 1.
    For each frame f in [frame_lo, frame_hi), pick j = f - (window - 1).
    """
    out = []
    for f in range(frame_lo, frame_hi):
        j = f - (window - 1)
        if j < 0 or j >= n_windows:
            raise SystemExit(
                f"Frame {f} cannot be predicted with window={window}: "
                f"need ds[{j}] but n_windows={n_windows}"
            )
        out.append(j)
    return out


@torch.no_grad()
def _pred_cnn_gru_sliding(model, ds, offset, win_idxs, dev):
    """For each window index j in win_idxs, take the last-frame prediction p[:, -1, :]."""
    preds, gts, ts = [], [], []
    for j in win_idxs:
        batch = collate([ds[j]])
        ch = batch["ch_feats"].to(dev)
        xy = batch["xy"].to(dev)
        gf = batch["global_feats"].to(dev) if getattr(model, "global_feat_dim", 0) > 0 else None
        p, _v = model(ch, gf)
        preds.append(p[0, -1, :].cpu().numpy() + offset)
        gts.append(xy[0, -1, :].cpu().numpy() + offset)
        ts.append(float(batch["ts"][0, -1]))
    return np.asarray(preds), np.asarray(gts), np.asarray(ts)


@torch.no_grad()
def _pred_cnn_gru_sliding_ensemble(models, ds, offset, win_idxs, dev):
    preds, gts, ts = [], [], []
    for j in win_idxs:
        batch = collate([ds[j]])
        ch = batch["ch_feats"].to(dev)
        xy = batch["xy"].to(dev)
        ps = []
        for m in models:
            gf = batch["global_feats"].to(dev) if getattr(m, "global_feat_dim", 0) > 0 else None
            p, _v = m(ch, gf)
            ps.append(p)
        p_mean = torch.stack(ps, 0).mean(0)
        preds.append(p_mean[0, -1, :].cpu().numpy() + offset)
        gts.append(xy[0, -1, :].cpu().numpy() + offset)
        ts.append(float(batch["ts"][0, -1]))
    return np.asarray(preds), np.asarray(gts), np.asarray(ts)


@torch.no_grad()
def _pred_dual_branch_sliding(model, ds, offset, win_idxs, dev):
    preds, gts, ts = [], [], []
    for j in win_idxs:
        batch = collate([ds[j]])
        ch = batch["ch_feats"].to(dev)
        inv = batch["attn_invalid"].to(dev)
        gf = batch["global_feats"].to(dev)
        xy = batch["xy"].to(dev)
        xy0 = xy[:, 0, :]
        p_hat, _pa, _pd, _v, _alpha = model(ch, inv, gf, xy0)
        preds.append(p_hat[0, -1, :].cpu().numpy() + offset)
        gts.append(xy[0, -1, :].cpu().numpy() + offset)
        ts.append(float(batch["ts"][0, -1]))
    return np.asarray(preds), np.asarray(gts), np.asarray(ts)


def _metrics(pred, gt):
    err = np.linalg.norm(pred - gt, axis=1)
    return float(np.sqrt(np.mean(err ** 2))), float(np.mean(err)), float(np.max(err))


def main() -> None:
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    algo_model, algo_ckpt = _load_dual_branch(Path("checkpoints/rfid_0421_w10_algo.pt"), dev)
    v1_model, v1_ckpt = _load_cnn_gru(Path("checkpoints/rfid_cnn_gru_w10_algo.pt"), dev)
    v2_model, v2_ckpt = _load_cnn_gru(Path("checkpoints/rfid_cnn_gru_w10_algo_v2f.pt"), dev)
    ens_pairs = [
        Path("checkpoints/rfid_cnn_gru_w10_algo_v5.pt"),
        Path("checkpoints/rfid_cnn_gru_w10_algo_v5_s123.pt"),
        Path("checkpoints/rfid_cnn_gru_w10_algo_v5_s456.pt"),
    ]
    ens_loaded = [_load_cnn_gru(p, dev) for p in ens_pairs]
    ens_models = [m for m, _c in ens_loaded]
    ens_ckpt = ens_loaded[0][1]

    ds_algo, off_algo = _build_ds_for(algo_ckpt, CSV_PATH)
    ds_v1, off_v1 = _build_ds_for(v1_ckpt, CSV_PATH)
    ds_v2, off_v2 = _build_ds_for(v2_ckpt, CSV_PATH)
    ds_ens, off_ens = _build_ds_for(ens_ckpt, CSV_PATH)
    print(f"windows: algo={len(ds_algo)}  v1={len(ds_v1)}  v2={len(ds_v2)}  ens={len(ds_ens)}")

    window = int(ens_ckpt["window"])  # all 4 ckpts share window=10
    win_idxs = _frame_to_window_indices(FRAME_START, FRAME_END, window, len(ds_ens))

    p_algo, g_algo, t_algo = _pred_dual_branch_sliding(algo_model, ds_algo, off_algo, win_idxs, dev)
    p_v1, g_v1, _t = _pred_cnn_gru_sliding(v1_model, ds_v1, off_v1, win_idxs, dev)
    p_v2, g_v2, _t = _pred_cnn_gru_sliding(v2_model, ds_v2, off_v2, win_idxs, dev)
    p_ens, g_ens, t_ens = _pred_cnn_gru_sliding_ensemble(ens_models, ds_ens, off_ens, win_idxs, dev)

    gt = g_ens
    ts_rel = t_ens - t_ens[0]
    rmse_algo, mae_algo, max_algo = _metrics(p_algo, gt)
    rmse_v1, mae_v1, max_v1 = _metrics(p_v1, gt)
    rmse_v2, mae_v2, max_v2 = _metrics(p_v2, gt)
    rmse_ens, mae_ens, max_ens = _metrics(p_ens, gt)

    span = f"frames [{FRAME_START}, {FRAME_END}) -> {len(gt)} pts (sliding-window inference)"
    print(span)
    print(f"  algo (DualBranch) : RMSE={rmse_algo*100:.2f} cm  MAE={mae_algo*100:.2f} cm  max={max_algo*100:.2f} cm")
    print(f"  v1   (CNN+GRU)    : RMSE={rmse_v1*100:.2f} cm  MAE={mae_v1*100:.2f} cm  max={max_v1*100:.2f} cm")
    print(f"  v2f  (CNN+GRU tun, flipped): RMSE={rmse_v2*100:.2f} cm  MAE={mae_v2*100:.2f} cm  max={max_v2*100:.2f} cm")
    print(f"  ens  (v5 x 3-seed): RMSE={rmse_ens*100:.2f} cm  MAE={mae_ens*100:.2f} cm  max={max_ens*100:.2f} cm")

    panels = [
        ("algo  (DualBranch)", p_algo, rmse_algo, mae_algo, max_algo, "#1f77b4"),
        ("v1    (CNN+GRU)", p_v1, rmse_v1, mae_v1, max_v1, "#ff7f0e"),
        ("v2f   (CNN+GRU tuned, flipped)", p_v2, rmse_v2, mae_v2, max_v2, "#2ca02c"),
        ("ensemble (v5 x 3 seeds)", p_ens, rmse_ens, mae_ens, max_ens, "#d62728"),
    ]

    # Common axis limits so the 4 panels are directly comparable.
    all_x = np.concatenate([gt[:, 0], p_algo[:, 0], p_v1[:, 0], p_v2[:, 0], p_ens[:, 0]])
    all_y = np.concatenate([gt[:, 1], p_algo[:, 1], p_v1[:, 1], p_v2[:, 1], p_ens[:, 1]])
    pad = 0.01
    xlim = (all_x.min() - pad, all_x.max() + pad)
    ylim = (all_y.min() - pad, all_y.max() + pad)

    fig, axes = plt.subplots(2, 2, figsize=(11, 10))
    for ax, (name, pred, rmse, mae, mx, color) in zip(axes.ravel(), panels):
        ax.plot(gt[:, 0], gt[:, 1], "k-", lw=1.6, label="GT")
        ax.plot(pred[:, 0], pred[:, 1], "--", color=color, lw=1.4, label="Pred")
        ax.scatter(gt[0, 0], gt[0, 1], c="k", s=45, marker="o", zorder=5)
        ax.scatter(gt[-1, 0], gt[-1, 1], c="k", s=70, marker="X", zorder=5)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_title(f"{name}\nRMSE={rmse*100:.2f} cm  MAE={mae*100:.2f} cm  max={mx*100:.2f} cm",
                     fontsize=10)
        ax.legend(fontsize=8, loc="best")
    fig.suptitle(
        f"155629  sliding-window inference  frames [{FRAME_START}, {FRAME_END})  ({len(gt)} pts)",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_traj = OUT_DIR / "155629_sliding_4panels_traj.png"
    fig.savefig(out_traj, dpi=150)
    plt.close(fig)
    print(f"saved: {out_traj}")

    # x(t), y(t) - 2x2 with each model getting one panel showing both x and y
    fig, axes = plt.subplots(2, 2, figsize=(13, 7), sharex=True)
    for ax, (name, pred, rmse, mae, mx, color) in zip(axes.ravel(), panels):
        ax.plot(ts_rel, gt[:, 0], "k-", lw=1.6, label="x GT")
        ax.plot(ts_rel, pred[:, 0], "--", color=color, lw=1.3, label="x Pred")
        ax.plot(ts_rel, gt[:, 1], "k:", lw=1.6, label="y GT")
        ax.plot(ts_rel, pred[:, 1], ":", color=color, lw=1.3, label="y Pred")
        ax.set_ylabel("position (m)")
        ax.grid(True, alpha=0.3)
        ax.set_title(f"{name}  RMSE={rmse*100:.2f} cm", fontsize=10)
        ax.legend(fontsize=7, ncol=2, loc="best")
    for ax in axes[-1]:
        ax.set_xlabel("time (s, relative)")
    fig.suptitle(
        f"155629  sliding-window  x(t), y(t)  frames [{FRAME_START}, {FRAME_END})",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_xyt = OUT_DIR / "155629_sliding_4panels_xyt.png"
    fig.savefig(out_xyt, dpi=150)
    plt.close(fig)
    print(f"saved: {out_xyt}")


if __name__ == "__main__":
    main()
