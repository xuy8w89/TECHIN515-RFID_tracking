#!/usr/bin/env python3
"""
Train the Time-Aware Confidence-Gated Dual-Branch 2D RFID tracker.

Loss (2D):
  L_pos = SmoothL1(p_hat, xy) + pos_l2_w * MSE(p_hat, xy)
  L_vel = MSE(v, v_gt) with v_gt from timestamps
  L_dyn = MSE( p_hat_t - (detach(p_hat_{t-1}) + v_{t-1} * dt_t) )
  L_smooth = MSE( v_t - v_{t-1} )

Usage:
  python -m rfid_localization.train --data_dir data --epochs 60
  python -m rfid_localization.train --split temporal      # default: time-ordered windows, no random leakage
  python -m rfid_localization.train --split merged_random # random window split
  python -m rfid_localization.train --split last_file     # train on all CSVs except last
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from rfid_localization.dataset import (
    RFIDTrackingWindowDataset,
    compute_rssi_norm,
    discover_channel_pairs,
    load_csv_paths,
    stack_tables,
)
from rfid_localization.model import DualBranchTracker2D
from rfid_localization.splits import train_val_indices


def collate(batch: list[dict]) -> dict[str, torch.Tensor]:
    return {
        "ch_feats": torch.stack([b["ch_feats"] for b in batch], dim=0),
        "attn_invalid": torch.stack([b["attn_invalid"] for b in batch], dim=0),
        "dt": torch.stack([b["dt"] for b in batch], dim=0),
        "global_feats": torch.stack([b["global_feats"] for b in batch], dim=0),
        "xy": torch.stack([b["xy"] for b in batch], dim=0),
        "ts": torch.stack([b["ts"] for b in batch], dim=0),
    }


def compute_losses(
    p_hat: torch.Tensor,
    v: torch.Tensor,
    xy: torch.Tensor,
    dt: torch.Tensor,
    w_vel: float,
    w_dyn: float,
    w_smooth: float,
    pos_l2_w: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """p_hat, v, xy: [B,T,2]; dt: [B,T]"""
    l_pos = F.smooth_l1_loss(p_hat, xy) + pos_l2_w * F.mse_loss(p_hat, xy)

    dtn = dt[:, 1:].unsqueeze(-1).clamp_min(1e-4)
    v_gt = (xy[:, 1:, :] - xy[:, :-1, :]) / dtn
    l_vel = F.mse_loss(v[:, 1:, :], v_gt)

    dyn = p_hat[:, 1:, :] - (p_hat[:, :-1, :].detach() + v[:, :-1, :] * dtn)
    l_dyn = (dyn**2).mean()

    dv = v[:, 1:, :] - v[:, :-1, :]
    l_smooth = (dv**2).mean() if dv.numel() > 0 else p_hat.new_tensor(0.0)

    loss = l_pos + w_vel * l_vel + w_dyn * l_dyn + w_smooth * l_smooth
    metrics = {
        "l_pos": float(l_pos.detach()),
        "l_vel": float(l_vel.detach()),
        "l_dyn": float(l_dyn.detach()),
        "l_smooth": float(l_smooth.detach()),
    }
    return loss, metrics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="data")
    ap.add_argument(
        "--val_dir",
        type=str,
        default="",
        help="if set, all CSVs in --data_dir are training and all CSVs here are validation; overrides --split",
    )
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--window", type=int, default=16)
    ap.add_argument("--phase_period", type=float, default=2048.0)
    ap.add_argument("--conf_lambda", type=float, default=0.35, help="exp(-lambda * tslv) for confidence feature")
    ap.add_argument("--w_vel", type=float, default=0.4)
    ap.add_argument("--w_dyn", type=float, default=0.25)
    ap.add_argument("--w_smooth", type=float, default=0.08)
    ap.add_argument("--pos_l2_w", type=float, default=0.15)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--d_ch", type=int, default=32)
    ap.add_argument("--vel_scale", type=float, default=0.08)
    ap.add_argument("--out", type=str, default="checkpoints/rfid_dual_branch_2d.pt")
    ap.add_argument(
        "--unwrap_method",
        type=str,
        default="arctan2",
        choices=["arctan2", "algo"],
        help="arctan2 = current min-angle wrap; algo = algorithm/features/phase.py integer-k unwrap",
    )
    ap.add_argument(
        "--recover",
        action="store_true",
        help="when --unwrap_method algo, also run Ridge cross-tag recover for missing phase",
    )
    ap.add_argument(
        "--split",
        type=str,
        default="temporal",
        choices=["temporal", "merged_random", "last_file"],
        help="temporal: time-ordered windows, first (1-val_ratio) train, last val_ratio val (recommended). "
        "merged_random: shuffle windows then split. last_file: train on all but last CSV, val on last.",
    )
    ap.add_argument(
        "--val_ratio",
        type=float,
        default=0.2,
        help="val fraction for temporal/merged_random on stacked CSVs",
    )
    ap.add_argument("--seed", type=int, default=42, help="only used when split=merged_random")
    args = ap.parse_args()

    warnings.filterwarnings("ignore", message=".*nested tensors.*", category=UserWarning)

    data_dir = Path(args.data_dir)
    paths = load_csv_paths(data_dir)
    if not paths:
        raise SystemExit(f"No CSV under {data_dir}")

    val_dir = Path(args.val_dir) if args.val_dir else None
    val_paths: list[Path] = []
    if val_dir is not None:
        val_paths = load_csv_paths(val_dir)
        if not val_paths:
            raise SystemExit(f"--val_dir set but no CSV under {val_dir}")

    df0 = pd.read_csv(paths[0])
    pairs = discover_channel_pairs(df0.columns)
    if not pairs:
        raise SystemExit("No ant*_phase / *_rssi column pairs found.")

    train_dfs = [pd.read_csv(p) for p in paths]
    r_mean, r_std = compute_rssi_norm(train_dfs, pairs)
    build_kw = dict(
        phase_period=args.phase_period,
        conf_lambda=args.conf_lambda,
        unwrap_method=args.unwrap_method,
        recover=bool(args.recover),
    )

    if val_dir is not None:
        tbl_tr = stack_tables(paths, pairs, r_mean, r_std, **build_kw)
        tbl_va = stack_tables(val_paths, pairs, r_mean, r_std, **build_kw)
        vb = tbl_tr["valid_xy"].astype(bool)
        offset = tbl_tr["xy"][vb].mean(axis=0) if np.any(vb) else np.zeros(2, dtype=np.float32)
        train_ds = RFIDTrackingWindowDataset(tbl_tr, window=args.window, xy_offset=offset)
        val_ds = RFIDTrackingWindowDataset(tbl_va, window=args.window, xy_offset=offset)
    elif len(paths) >= 2 and args.split == "last_file":
        tbl_tr = stack_tables(paths[:-1], pairs, r_mean, r_std, **build_kw)
        tbl_va = stack_tables(paths[-1:], pairs, r_mean, r_std, **build_kw)
        vb = tbl_tr["valid_xy"].astype(bool)
        offset = tbl_tr["xy"][vb].mean(axis=0) if np.any(vb) else np.zeros(2, dtype=np.float32)
        train_ds = RFIDTrackingWindowDataset(tbl_tr, window=args.window, xy_offset=offset)
        val_ds = RFIDTrackingWindowDataset(tbl_va, window=args.window, xy_offset=offset)
    else:
        tbl_all = stack_tables(paths, pairs, r_mean, r_std, **build_kw)
        vb = tbl_all["valid_xy"].astype(bool)
        offset = tbl_all["xy"][vb].mean(axis=0) if np.any(vb) else np.zeros(2, dtype=np.float32)
        full_ds = RFIDTrackingWindowDataset(tbl_all, window=args.window, xy_offset=offset)
        split_key = "merged_random" if args.split == "merged_random" else "temporal"
        tr_idx, va_idx = train_val_indices(len(full_ds), args.val_ratio, split_key, args.seed)
        train_ds = Subset(full_ds, tr_idx)
        val_ds = Subset(full_ds, va_idx)

    split_desc = f"val_dir={val_dir}" if val_dir is not None else f"split={args.split}"
    print(
        f"{split_desc}  channels={len(pairs)}  "
        f"train_windows={len(train_ds)}  val_windows={len(val_ds)}"
    )
    if len(train_ds) == 0 or len(val_ds) == 0:
        raise SystemExit("Not enough valid windows; lower --window or check CSV.")

    ch_dim = RFIDTrackingWindowDataset.FEAT_DIM
    g_dim = 3

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True, collate_fn=collate
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False, collate_fn=collate
    )

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DualBranchTracker2D(
        num_channels=len(pairs),
        ch_in_dim=ch_dim,
        global_dim=g_dim,
        d_ch=args.d_ch,
        gru_hidden=args.hidden,
        vel_scale=args.vel_scale,
    ).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    best = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        tr_sum = 0.0
        n_tr = 0
        for batch in train_loader:
            ch = batch["ch_feats"].to(dev)
            inv = batch["attn_invalid"].to(dev)
            dt = batch["dt"].to(dev)
            gf = batch["global_feats"].to(dev)
            xy = batch["xy"].to(dev)

            xy0 = xy[:, 0, :].detach()
            p_hat, _p_abs, _p_dyn, v, _alpha = model(ch, inv, gf, xy0)

            loss, _m = compute_losses(
                p_hat, v, xy, dt, args.w_vel, args.w_dyn, args.w_smooth, args.pos_l2_w
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()

            tr_sum += float(loss.detach()) * ch.size(0)
            n_tr += ch.size(0)

        model.eval()
        va_sum = 0.0
        n_va = 0
        with torch.no_grad():
            for batch in val_loader:
                ch = batch["ch_feats"].to(dev)
                inv = batch["attn_invalid"].to(dev)
                dt = batch["dt"].to(dev)
                gf = batch["global_feats"].to(dev)
                xy = batch["xy"].to(dev)
                xy0 = xy[:, 0, :].detach()
                p_hat, _, _, v, _ = model(ch, inv, gf, xy0)
                loss, _ = compute_losses(
                    p_hat, v, xy, dt, args.w_vel, args.w_dyn, args.w_smooth, args.pos_l2_w
                )
                va_sum += float(loss) * ch.size(0)
                n_va += ch.size(0)

        tr_m = tr_sum / max(n_tr, 1)
        va_m = va_sum / max(n_va, 1)
        print(f"epoch {epoch:03d}  train {tr_m:.5f}  val {va_m:.5f}")
        if va_m < best:
            best = va_m
            torch.save(
                {
                    "model": model.state_dict(),
                    "pairs": pairs,
                    "rssi_mean": r_mean,
                    "rssi_std": r_std,
                    "xy_offset": offset,
                    "window": args.window,
                    "phase_period": args.phase_period,
                    "conf_lambda": args.conf_lambda,
                    "hidden": args.hidden,
                    "d_ch": args.d_ch,
                    "ch_feat_dim": ch_dim,
                    "global_dim": g_dim,
                    "vel_scale": args.vel_scale,
                    "split": args.split,
                    "seed": args.seed,
                    "val_ratio": args.val_ratio,
                    "unwrap_method": args.unwrap_method,
                    "recover": bool(args.recover),
                    "data_dir": str(data_dir),
                    "val_dir": str(val_dir) if val_dir is not None else "",
                },
                out_path,
            )
            print(f"  saved {out_path} (best val {best:.5f})")


if __name__ == "__main__":
    main()
