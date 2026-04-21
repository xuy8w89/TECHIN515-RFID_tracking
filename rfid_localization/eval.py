#!/usr/bin/env python3
"""
Evaluation for the 2D RFID tracker (metrics + optional plots + spatial bin confusion).

This is regression (XY), not classification: "accuracy" is replaced by RMSE/MAE and hit-rate@eps.

Split policy (must match training):
  temporal       — windows sorted by end time; first (1-val_ratio) train, last val_ratio val
  merged_random  — same as old random window split

Test / data usage:
  --eval_set val|train|all — which subset to score (default val).
  For a strict *held-out test file*, keep one CSV out of `data/` and point --data_dir only to train CSVs
  during training, then evaluate with a separate directory or `last_file`-style split.

Plots (optional --plots_dir):
  fig_trajectories.png — GT vs pred in 2D for several val windows
  fig_xy_vs_time.png   — x,y vs step in one window
  fig_velocity.png     — predicted vs GT velocity in one window

Usage:
  python -m rfid_localization.eval --checkpoint checkpoints/rfid_dual_branch_2d.pt --data_dir data
  python -m rfid_localization.eval --checkpoint ... --plots_dir checkpoints/plots --eval_set val
  python -m rfid_localization.eval ... --traj_pick best --traj_n 9   # top-9 lowest-error val windows
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

try:
    from sklearn.metrics import confusion_matrix
except ImportError as e:  # pragma: no cover
    raise SystemExit("pip install scikit-learn") from e

from rfid_localization.dataset import (
    RFIDTrackingWindowDataset,
    compute_rssi_norm,
    discover_channel_pairs,
    load_csv_paths,
    stack_tables,
)
from rfid_localization.model import DualBranchTracker2D
from rfid_localization.splits import train_val_indices
from rfid_localization.train import collate


def _build_full_ds(paths, pairs, r_mean, r_std, build_kw, window, xy_offset):
    tbl_all = stack_tables(paths, pairs, r_mean, r_std, **build_kw)
    return RFIDTrackingWindowDataset(tbl_all, window=window, xy_offset=xy_offset)


def xy_to_cell(xy: np.ndarray, gx_min: float, gx_max: float, gy_min: float, gy_max: float, n_bins: int) -> np.ndarray:
    rx = gx_max - gx_min + 1e-9
    ry = gy_max - gy_min + 1e-9
    xb = np.floor((xy[:, 0] - gx_min) / rx * (n_bins - 1e-9)).astype(np.int64)
    yb = np.floor((xy[:, 1] - gy_min) / ry * (n_bins - 1e-9)).astype(np.int64)
    xb = np.clip(xb, 0, n_bins - 1)
    yb = np.clip(yb, 0, n_bins - 1)
    return (xb * n_bins + yb).astype(np.int64)


def _val_window_mean_l2_errors(
    model: torch.nn.Module,
    val_ds: Subset,
    dev: torch.device,
) -> list[tuple[float, int]]:
    """Return (mean per-step L2 error, val_ds index) for each val window."""
    model.eval()
    out: list[tuple[float, int]] = []
    with torch.no_grad():
        for j in range(len(val_ds)):
            batch = collate([val_ds[j]])
            ch = batch["ch_feats"].to(dev)
            inv = batch["attn_invalid"].to(dev)
            dt = batch["dt"].to(dev)
            gf = batch["global_feats"].to(dev)
            xy = batch["xy"].to(dev)
            xy0 = xy[:, 0, :].detach()
            p_hat, _, _, _, _ = model(ch, inv, gf, xy0)
            err = float((p_hat - xy).norm(dim=-1).mean().cpu())
            out.append((err, j))
    return out


def plot_val_windows(
    model: torch.nn.Module,
    val_ds: Subset,
    dev: torch.device,
    plots_dir: Path,
    n_show: int = 6,
    traj_pick: str = "best",
) -> None:
    """
    traj_pick:
      best — smallest mean L2 error over the trajectory in the window (recommended).
      even — evenly spaced val_ds indices (legacy behavior).
    """
    plots_dir.mkdir(parents=True, exist_ok=True)
    n = len(val_ds)
    if n == 0:
        return
    n_show = min(int(n_show), n)
    if traj_pick == "even":
        picks = np.linspace(0, n - 1, num=n_show, dtype=int).tolist()
    else:
        scored = _val_window_mean_l2_errors(model, val_ds, dev)
        scored.sort(key=lambda t: t[0])
        picks = [j for _, j in scored[:n_show]]
    n_show = len(picks)

    ncols = min(3, n_show)
    nrows = int(np.ceil(n_show / ncols))
    fig, _ = plt.subplots(nrows, ncols, figsize=(3.5 * ncols + 1, 3.2 * nrows + 0.5))
    axes_list = fig.axes
    for k in range(len(picks), len(axes_list)):
        axes_list[k].axis("off")
    model.eval()
    with torch.no_grad():
        for ax_i, j in enumerate(picks):
            if ax_i >= len(axes_list):
                break
            batch = collate([val_ds[int(j)]])
            ch = batch["ch_feats"].to(dev)
            inv = batch["attn_invalid"].to(dev)
            dt = batch["dt"].to(dev)
            gf = batch["global_feats"].to(dev)
            xy = batch["xy"].to(dev)
            xy0 = xy[:, 0, :].detach()
            p_hat, _, _, _, _ = model(ch, inv, gf, xy0)
            g = xy[0].cpu().numpy()
            pr = p_hat[0].cpu().numpy()
            ax = axes_list[ax_i]
            ax.plot(g[:, 0], g[:, 1], "k-", lw=1.2, label="GT")
            ax.plot(pr[:, 0], pr[:, 1], "r--", lw=1.0, label="Pred")
            ax.set_aspect("equal", adjustable="box")
            ax.grid(True, alpha=0.3)
            e = float((p_hat - xy).norm(dim=-1).mean().cpu())
            ax.set_title(f"val #{j}\nmean L2={e:.4f}", fontsize=8)
            if ax_i == 0:
                ax.legend(fontsize=7)
    pick_label = "best (lowest mean L2)" if traj_pick != "even" else "evenly spaced"
    fig.suptitle(f"Trajectory overlays (val, {pick_label}, n={len(picks)})", fontsize=12)
    fig.tight_layout()
    fig.savefig(plots_dir / "fig_trajectories.png", dpi=150)
    plt.close(fig)

    j0 = int(picks[0])
    batch = collate([val_ds[j0]])
    with torch.no_grad():
        ch = batch["ch_feats"].to(dev)
        inv = batch["attn_invalid"].to(dev)
        dt = batch["dt"].to(dev)
        gf = batch["global_feats"].to(dev)
        xy = batch["xy"].to(dev)
        xy0 = xy[:, 0, :].detach()
        p_hat, _, _, v, _ = model(ch, inv, gf, xy0)
        g = xy[0].cpu().numpy()
        pr = p_hat[0].cpu().numpy()
        dtn = dt[0, 1:].unsqueeze(-1).clamp_min(1e-4)
        v_gt = (xy[0, 1:, :] - xy[0, :-1, :]) / dtn
        t = np.arange(g.shape[0])
    fig, ax = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
    ax[0].plot(t, g[:, 0], "k-", label="x GT")
    ax[0].plot(t, pr[:, 0], "r--", label="x Pred")
    ax[0].set_ylabel("x")
    ax[0].grid(True, alpha=0.3)
    ax[0].legend(fontsize=8)
    ax[1].plot(t, g[:, 1], "k-", label="y GT")
    ax[1].plot(t, pr[:, 1], "r--", label="y Pred")
    ax[1].set_ylabel("y")
    ax[1].set_xlabel("step in window")
    ax[1].grid(True, alpha=0.3)
    fig.suptitle(f"x(t), y(t) — val window #{j0}", fontsize=12)
    fig.tight_layout()
    fig.savefig(plots_dir / "fig_xy_vs_time.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
    tv = np.arange(1, g.shape[0])
    ax[0].plot(tv, v_gt[:, 0].cpu().numpy(), "k-", label="vx GT")
    ax[0].plot(tv, v[0, 1:, 0].cpu().numpy(), "r--", label="vx Pred")
    ax[0].set_ylabel("vx")
    ax[0].grid(True, alpha=0.3)
    ax[0].legend(fontsize=8)
    ax[1].plot(tv, v_gt[:, 1].cpu().numpy(), "k-", label="vy GT")
    ax[1].plot(tv, v[0, 1:, 1].cpu().numpy(), "r--", label="vy Pred")
    ax[1].set_ylabel("vy")
    ax[1].set_xlabel("step (velocity aligned to t>=1)")
    ax[1].grid(True, alpha=0.3)
    fig.suptitle(f"Velocity — val window #{j0}", fontsize=12)
    fig.tight_layout()
    fig.savefig(plots_dir / "fig_velocity.png", dpi=150)
    plt.close(fig)


def main() -> None:
    warnings.filterwarnings("ignore", message=".*nested tensors.*", category=UserWarning)

    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, default="checkpoints/rfid_dual_branch_2d.pt")
    ap.add_argument("--data_dir", type=str, default="data")
    ap.add_argument("--eval_set", type=str, default="val", choices=["val", "train", "all"])
    ap.add_argument("--split", type=str, default="", help="temporal|merged_random; empty = read from checkpoint")
    ap.add_argument("--val_ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--eps_pos", type=float, default=0.02)
    ap.add_argument("--n_bins", type=int, default=8)
    ap.add_argument("--save_cm", type=str, default="", help="optional PNG path for spatial bin confusion")
    ap.add_argument("--plots_dir", type=str, default="", help="if set, save trajectory / x-y / velocity figures")
    ap.add_argument("--report_cm", action="store_true", help="also print RMSE in cm (x100) for wrist-scale coords")
    ap.add_argument(
        "--traj_pick",
        type=str,
        default="best",
        choices=["best", "even"],
        help="trajectory panels: best = lowest mean L2 error in window (default); even = evenly spaced val indices",
    )
    ap.add_argument("--traj_n", type=int, default=6, help="number of trajectory subplots (grid expands automatically)")
    args = ap.parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_file():
        raise SystemExit(f"Missing checkpoint: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    pairs = ckpt["pairs"]
    window = int(ckpt.get("window", 16))
    phase_period = float(ckpt.get("phase_period", 2048.0))
    conf_lambda = float(ckpt.get("conf_lambda", 0.35))
    hidden = int(ckpt.get("hidden", 128))
    d_ch = int(ckpt.get("d_ch", 32))
    vel_scale = float(ckpt.get("vel_scale", 0.08))
    ch_feat_dim = int(ckpt.get("ch_feat_dim", RFIDTrackingWindowDataset.FEAT_DIM))
    global_dim = int(ckpt.get("global_dim", 3))
    r_mean = np.asarray(ckpt["rssi_mean"], dtype=np.float32)
    r_std = np.asarray(ckpt["rssi_std"], dtype=np.float32)
    offset = np.asarray(ckpt["xy_offset"], dtype=np.float32)
    split = args.split.strip() or str(ckpt.get("split", "temporal"))
    if split == "last_file":
        split = "temporal"
    if split not in ("temporal", "merged_random"):
        split = "temporal"
    val_ratio = float(ckpt.get("val_ratio", args.val_ratio))
    seed = int(ckpt.get("seed", args.seed))

    data_dir = Path(args.data_dir)
    paths = load_csv_paths(data_dir)
    if not paths:
        raise SystemExit(f"No CSV under {data_dir}")

    df0 = pd.read_csv(paths[0])
    live = discover_channel_pairs(df0.columns)
    if tuple(live) != tuple(pairs):
        print("Warning: CSV columns differ from checkpoint pairs; using checkpoint list.")

    build_kw = dict(phase_period=phase_period, conf_lambda=conf_lambda)
    full_ds = _build_full_ds(paths, pairs, r_mean, r_std, build_kw, window, offset)
    tr_idx, va_idx = train_val_indices(len(full_ds), val_ratio, split, seed)
    train_ds = Subset(full_ds, tr_idx)
    val_ds = Subset(full_ds, va_idx)

    if args.eval_set == "train":
        ds = train_ds
    elif args.eval_set == "val":
        ds = val_ds
    else:
        ds = full_ds

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, drop_last=False, collate_fn=collate)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DualBranchTracker2D(
        num_channels=len(pairs),
        ch_in_dim=ch_feat_dim,
        global_dim=global_dim,
        d_ch=d_ch,
        gru_hidden=hidden,
        vel_scale=vel_scale,
    ).to(dev)
    model.load_state_dict(ckpt["model"])
    model.eval()

    all_p: list[torch.Tensor] = []
    all_g: list[torch.Tensor] = []
    all_ve: list[torch.Tensor] = []
    win_mae: list[float] = []
    win_final: list[float] = []
    all_v: list[torch.Tensor] = []
    smooth_terms: list[torch.Tensor] = []

    with torch.no_grad():
        for batch in loader:
            ch = batch["ch_feats"].to(dev)
            inv = batch["attn_invalid"].to(dev)
            dt = batch["dt"].to(dev)
            gf = batch["global_feats"].to(dev)
            xy = batch["xy"].to(dev)
            xy0 = xy[:, 0, :].detach()
            p_hat, _, _, v, _ = model(ch, inv, gf, xy0)
            dtn = dt[:, 1:].unsqueeze(-1).clamp_min(1e-4)
            v_gt = (xy[:, 1:, :] - xy[:, :-1, :]) / dtn
            ve = v[:, 1:, :] - v_gt
            err = (p_hat - xy).norm(dim=-1)
            win_mae.extend(err.mean(dim=1).cpu().tolist())
            win_final.extend(err[:, -1].cpu().tolist())
            dv = (v[:, 1:, :] - v[:, :-1, :]).norm(dim=-1)
            smooth_terms.append(dv.mean(dim=1))
            all_p.append(p_hat.cpu())
            all_g.append(xy.cpu())
            all_ve.append(ve.cpu())
            all_v.append(v.cpu())

    p = torch.cat(all_p, dim=0).reshape(-1, 2).numpy()
    g = torch.cat(all_g, dim=0).reshape(-1, 2).numpy()
    ve = torch.cat(all_ve, dim=0).reshape(-1, 2).numpy()
    v_all = torch.cat(all_v, dim=0)

    err = p - g
    mae_xy = np.mean(np.abs(err), axis=0)
    euc = np.linalg.norm(err, axis=1)
    mae_l1 = float(np.mean(euc))
    rmse_l2 = float(np.sqrt(np.mean(euc**2)))
    rmse_xy = np.sqrt(np.mean(err**2, axis=0))
    hit = float(np.mean(euc < args.eps_pos))
    mae_v = np.mean(np.abs(ve), axis=0)
    rmse_v = np.sqrt(np.mean(ve**2, axis=0))
    rmse_v_l2 = float(np.sqrt(np.mean((ve**2).sum(axis=-1))))
    smooth = float(torch.cat(smooth_terms, dim=0).mean()) if smooth_terms else 0.0

    print(f"checkpoint: {ckpt_path}")
    print(f"split={split}  val_ratio={val_ratio}  seed={seed}  eval_set={args.eval_set}")
    print(f"windows={len(ds)}  points={len(p)}  window_T={window}")
    print(f"position RMSE (L2, same units as WRIST): {rmse_l2:.6f}")
    print(f"position MAE (mean L2): {mae_l1:.6f}")
    print(f"position MAE (x,y component): {mae_xy[0]:.6f}, {mae_xy[1]:.6f}")
    print(f"position RMSE (x,y component): {rmse_xy[0]:.6f}, {rmse_xy[1]:.6f}")
    if args.report_cm:
        print(f"position RMSE (L2, cm): {rmse_l2 * 100:.4f} cm")
    print(f"hit_rate@eps (L2<{args.eps_pos}): {hit:.4f}")
    print(f"velocity RMSE (vx,vy): {rmse_v[0]:.6f}, {rmse_v[1]:.6f}  |  L2: {rmse_v_l2:.6f}")
    print(f"per-window mean L2 error: mean={np.mean(win_mae):.6f} std={np.std(win_mae):.6f}")
    print(f"per-window final-step L2 error: mean={np.mean(win_final):.6f} std={np.std(win_final):.6f}")
    print(f"mean temporal smoothness E||v_t - v_(t-1)|| (pred): {smooth:.6f}")

    n_bins = max(2, int(args.n_bins))
    gx_min, gx_max = float(np.min(g[:, 0])), float(np.max(g[:, 0]))
    gy_min, gy_max = float(np.min(g[:, 1])), float(np.max(g[:, 1]))
    ct = xy_to_cell(g, gx_min, gx_max, gy_min, gy_max, n_bins)
    cp = xy_to_cell(p, gx_min, gx_max, gy_min, gy_max, n_bins)
    labels = np.arange(n_bins * n_bins)
    cm = confusion_matrix(ct, cp, labels=labels)
    cell_acc = float(np.trace(cm) / max(np.sum(cm), 1))
    print(f"spatial_cell_diag_rate ({n_bins}x{n_bins} bins on GT range): {cell_acc:.4f}")

    if args.save_cm:
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        ax.set_xlabel("predicted cell")
        ax.set_ylabel("true cell")
        ax.set_title(f"Spatial bin confusion ({n_bins}x{n_bins})")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(args.save_cm, dpi=150)
        plt.close(fig)
        print(f"saved: {args.save_cm}")

    if args.plots_dir:
        plot_dir = Path(args.plots_dir)
        if args.eval_set == "val":
            plot_val_windows(
                model, val_ds, dev, plot_dir, n_show=args.traj_n, traj_pick=args.traj_pick
            )
            print(f"saved figures under: {plot_dir}")
        else:
            print("--plots_dir is implemented for eval_set=val; skip or use --eval_set val")


if __name__ == "__main__":
    main()
