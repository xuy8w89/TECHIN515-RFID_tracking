#!/usr/bin/env python3
"""
Plot one trajectory figure per CSV (GT vs model prediction).

For each CSV under --data_dir (or --val_dir):
  1) build the enriched table (same pipeline used at train/eval)
  2) build a sliding window dataset (stride=1, same window size as checkpoint)
  3) for every valid window, take p_hat[-1] as the prediction at that end-frame
  4) plot GT (x,y) vs Pred (x,y) in 2D; also x(t), y(t) over time

Each CSV produces:
  <out_dir>/<csv_stem>_traj.png   2D overlay
  <out_dir>/<csv_stem>_xyt.png    x(t), y(t) overlay

Usage:
  python -m rfid_localization.plot_per_csv \
    --checkpoint checkpoints/rfid_0421_algo.pt \
    --data_dir data/0421/test \
    --out_dir checkpoints/plots_0421_algo/per_csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from rfid_localization.dataset import (
    RFIDTrackingWindowDataset,
    build_enriched_table,
    discover_channel_pairs,
    load_csv_paths,
)
from rfid_localization.model import DualBranchTracker2D


def _predict_per_frame(
    model: torch.nn.Module,
    table: dict[str, np.ndarray],
    window: int,
    offset: np.ndarray,
    device: torch.device,
    batch_size: int = 64,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run model over all valid windows, return per-end-frame
    (pred_xy, gt_xy, ts) for the last step of each window."""
    ds = RFIDTrackingWindowDataset(table, window=window, xy_offset=offset)
    if len(ds) == 0:
        return (
            np.zeros((0, 2), dtype=np.float32),
            np.zeros((0, 2), dtype=np.float32),
            np.zeros((0,), dtype=np.float64),
        )

    preds: list[np.ndarray] = []
    gts: list[np.ndarray] = []
    times: list[float] = []

    model.eval()
    with torch.no_grad():
        batch_items: list[dict] = []
        for i in range(len(ds)):
            batch_items.append(ds[i])
            if len(batch_items) == batch_size or i == len(ds) - 1:
                ch = torch.stack([b["ch_feats"] for b in batch_items]).to(device)
                inv = torch.stack([b["attn_invalid"] for b in batch_items]).to(device)
                gf = torch.stack([b["global_feats"] for b in batch_items]).to(device)
                xy = torch.stack([b["xy"] for b in batch_items]).to(device)
                xy0 = xy[:, 0, :].detach()
                p_hat, _, _, _, _ = model(ch, inv, gf, xy0)
                p_last = p_hat[:, -1, :].cpu().numpy() + offset.reshape(1, 2)
                g_last = xy[:, -1, :].cpu().numpy() + offset.reshape(1, 2)
                t_last = [float(b["ts"][-1]) for b in batch_items]
                preds.append(p_last)
                gts.append(g_last)
                times.extend(t_last)
                batch_items.clear()

    return (
        np.concatenate(preds, axis=0).astype(np.float32),
        np.concatenate(gts, axis=0).astype(np.float32),
        np.asarray(times, dtype=np.float64),
    )


def _plot_csv(
    pred: np.ndarray,
    gt: np.ndarray,
    ts: np.ndarray,
    out_traj: Path,
    out_xyt: Path,
    title: str,
) -> tuple[float, float]:
    """Save two figures, return (mean L2 error, max L2 error)."""
    err = np.linalg.norm(pred - gt, axis=1)
    mean_e = float(np.mean(err)) if err.size else float("nan")
    max_e = float(np.max(err)) if err.size else float("nan")

    fig, ax = plt.subplots(figsize=(6, 5.5))
    ax.plot(gt[:, 0], gt[:, 1], "k-", lw=1.2, label="GT")
    ax.plot(pred[:, 0], pred[:, 1], "r--", lw=1.0, alpha=0.85, label="Pred")
    if gt.size:
        ax.scatter(gt[0, 0], gt[0, 1], c="k", s=40, marker="o", zorder=5, label="GT start")
        ax.scatter(gt[-1, 0], gt[-1, 1], c="k", s=60, marker="X", zorder=5, label="GT end")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    ax.set_title(f"{title}\nmean L2={mean_e*100:.2f} cm  max L2={max_e*100:.2f} cm  n={len(pred)}")
    fig.tight_layout()
    fig.savefig(out_traj, dpi=150)
    plt.close(fig)

    t_rel = ts - ts[0] if ts.size else ts
    fig, axes = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
    axes[0].plot(t_rel, gt[:, 0], "k-", label="x GT")
    axes[0].plot(t_rel, pred[:, 0], "r--", label="x Pred")
    axes[0].set_ylabel("x")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)
    axes[1].plot(t_rel, gt[:, 1], "k-", label="y GT")
    axes[1].plot(t_rel, pred[:, 1], "r--", label="y Pred")
    axes[1].set_ylabel("y")
    axes[1].set_xlabel("time (s, relative)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_xyt, dpi=150)
    plt.close(fig)

    return mean_e, max_e


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument(
        "--data_dir",
        type=str,
        default="",
        help="folder with CSVs to plot; if empty, falls back to checkpoint's val_dir",
    )
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--batch_size", type=int, default=64)
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
    unwrap_method = str(ckpt.get("unwrap_method", "arctan2"))
    recover = bool(ckpt.get("recover", False))

    data_dir_str = args.data_dir.strip() or str(ckpt.get("val_dir", ""))
    if not data_dir_str:
        raise SystemExit("--data_dir not given and checkpoint has no val_dir")
    data_dir = Path(data_dir_str)
    paths = load_csv_paths(data_dir)
    if not paths:
        raise SystemExit(f"No CSV under {data_dir}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

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

    print(f"window={window}  channels={len(pairs)}  unwrap={unwrap_method}  recover={recover}")
    print(f"plotting {len(paths)} CSV from {data_dir}")
    print(f"output dir: {out_dir}")

    summary: list[tuple[str, int, float, float]] = []
    for p in paths:
        df = pd.read_csv(p)
        table = build_enriched_table(
            df,
            pairs,
            r_mean,
            r_std,
            phase_period=phase_period,
            conf_lambda=conf_lambda,
            unwrap_method=unwrap_method,
            recover=recover,
        )
        pred, gt, ts = _predict_per_frame(model, table, window, offset, dev, args.batch_size)
        if len(pred) == 0:
            print(f"  {p.name}: no valid window, skip")
            continue
        traj_png = out_dir / f"{p.stem}_traj.png"
        xyt_png = out_dir / f"{p.stem}_xyt.png"
        mean_e, max_e = _plot_csv(pred, gt, ts, traj_png, xyt_png, title=p.stem)
        print(f"  {p.name}: n={len(pred)}  mean L2={mean_e*100:.2f} cm  max L2={max_e*100:.2f} cm")
        summary.append((p.name, len(pred), mean_e, max_e))

    if summary:
        print("\n=== summary ===")
        all_n = sum(s[1] for s in summary)
        all_mean = sum(s[1] * s[2] for s in summary) / max(all_n, 1)
        print(f"pooled mean L2 (point-weighted): {all_mean*100:.2f} cm  over {all_n} points in {len(summary)} CSVs")


if __name__ == "__main__":
    main()
