#!/usr/bin/env python3
"""Train the CNN(2x4 grid) + GRU 2D RFID tracker.

Reuses the same enriched-table pipeline, same losses, and same split options
as `rfid_localization.train`. Only the model is different.

Usage:
    python -m rfid_localization.train_cnn_gru \\
        --data_dir data/0421/train --val_dir data/0421/test \\
        --window 10 --unwrap_method algo \\
        --out checkpoints/rfid_cnn_gru.pt
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from rfid_localization.dataset import (
    RFIDTrackingWindowDataset,
    compute_rssi_norm,
    discover_channel_pairs,
    discover_grid_layout,
    load_csv_paths,
    stack_tables,
)
from rfid_localization.model_cnn_gru import CNNGRUTracker
from rfid_localization.splits import train_val_indices
from rfid_localization.train import collate, compute_losses


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="data")
    ap.add_argument(
        "--val_dir",
        type=str,
        default="",
        help="if set, all CSVs in --data_dir are training and all in --val_dir are validation",
    )
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--window", type=int, default=10)
    ap.add_argument("--phase_period", type=float, default=2048.0)
    ap.add_argument("--conf_lambda", type=float, default=0.35)
    ap.add_argument("--w_vel", type=float, default=0.4)
    ap.add_argument("--w_dyn", type=float, default=0.25)
    ap.add_argument("--w_smooth", type=float, default=0.08)
    ap.add_argument(
        "--pos_l2_w",
        type=float,
        default=0.0,
        help="extra L2 coef on position; spec uses SmoothL1 only (0.0)",
    )
    ap.add_argument("--frame_embed_dim", type=int, default=128)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--vel_scale", type=float, default=0.08)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument(
        "--cosine_lr",
        action="store_true",
        help="use CosineAnnealingLR from --lr down to --min_lr over all epochs",
    )
    ap.add_argument("--min_lr", type=float, default=1e-5)
    ap.add_argument(
        "--rssi_jitter_std",
        type=float,
        default=0.0,
        help="std (in RSSI_z units) of additive Gaussian noise on RSSI_z during training (0=off)",
    )
    ap.add_argument(
        "--use_global_feats",
        action="store_true",
        help="concat per-frame global feats [dt, log_dt, inv_dt] into frame embedding",
    )
    ap.add_argument(
        "--anchor_decoder",
        action="store_true",
        help="decode xy as softmax(w) @ learnable_tag_anchors + tanh residual",
    )
    ap.add_argument(
        "--anchor_init_scale",
        type=float,
        default=0.1,
        help="init scale for learnable tag anchors (in target xy units, after offset)",
    )
    ap.add_argument(
        "--residual_scale",
        type=float,
        default=0.05,
        help="bound on per-frame residual added to anchor-weighted xy (meters)",
    )
    ap.add_argument(
        "--unwrap_method",
        type=str,
        default="algo",
        choices=["arctan2", "algo"],
    )
    ap.add_argument("--recover", action="store_true")
    ap.add_argument(
        "--use_pdoa",
        action="store_true",
        help="add per-tag PDoA features (sin, cos of antenna phase difference); needs H==2",
    )
    ap.add_argument("--out", type=str, default="checkpoints/rfid_cnn_gru.pt")
    ap.add_argument(
        "--split",
        type=str,
        default="temporal",
        choices=["temporal", "merged_random", "last_file"],
    )
    ap.add_argument("--val_ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    warnings.filterwarnings("ignore", message=".*nested tensors.*", category=UserWarning)

    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

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

    layout = discover_grid_layout(pairs)
    print(
        f"grid {layout['H']}x{layout['W']}  "
        f"ants={layout['ant_order']}  tags={layout['tag_order']}"
    )

    train_dfs = [pd.read_csv(p) for p in paths]
    r_mean, r_std = compute_rssi_norm(train_dfs, pairs)

    pdoa_pair_indices: np.ndarray | None = None
    if args.use_pdoa:
        if int(layout["H"]) != 2:
            raise SystemExit(
                f"--use_pdoa needs exactly 2 antennas (H=2), got H={layout['H']}"
            )
        c2c = layout["cell_to_channel"]
        pdoa_pair_indices = np.stack([c2c[0, :], c2c[1, :]], axis=-1).astype(np.int64)
        keep = (pdoa_pair_indices >= 0).all(axis=-1)
        pdoa_pair_indices = pdoa_pair_indices[keep]
        print(f"PDoA pairs: {pdoa_pair_indices.tolist()}")

    build_kw = dict(
        phase_period=args.phase_period,
        conf_lambda=args.conf_lambda,
        unwrap_method=args.unwrap_method,
        recover=bool(args.recover),
        pdoa_pair_indices=pdoa_pair_indices,
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
        f"train_windows={len(train_ds)}  val_windows={len(val_ds)}  window_T={args.window}"
    )
    if len(train_ds) == 0 or len(val_ds) == 0:
        raise SystemExit("Not enough valid windows")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True, collate_fn=collate
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False, collate_fn=collate
    )

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    global_feat_dim = 3 if args.use_global_feats else 0
    model = CNNGRUTracker(
        cell_to_channel=layout["cell_to_channel"],
        local_xy=layout["local_xy"],
        frame_embed_dim=args.frame_embed_dim,
        gru_hidden=args.hidden,
        dropout=args.dropout,
        vel_scale=args.vel_scale,
        global_feat_dim=global_feat_dim,
        anchor_decoder=bool(args.anchor_decoder),
        anchor_init_scale=args.anchor_init_scale,
        residual_scale=args.residual_scale,
        use_pdoa=bool(args.use_pdoa),
    ).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = None
    if args.cosine_lr:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=args.epochs, eta_min=args.min_lr
        )
    print(
        f"model_params={sum(p.numel() for p in model.parameters())}  "
        f"global_feat_dim={global_feat_dim}  cosine_lr={bool(args.cosine_lr)}  "
        f"rssi_jitter_std={args.rssi_jitter_std}  weight_decay={args.weight_decay}  "
        f"anchor_decoder={bool(args.anchor_decoder)}  residual_scale={args.residual_scale}  "
        f"use_pdoa={bool(args.use_pdoa)}"
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    best = float("inf")
    jitter_std = float(args.rssi_jitter_std)
    for epoch in range(1, args.epochs + 1):
        model.train()
        tr_sum = 0.0
        n_tr = 0
        for batch in train_loader:
            ch = batch["ch_feats"].to(dev)
            dt = batch["dt"].to(dev)
            xy = batch["xy"].to(dev)
            gf = batch["global_feats"].to(dev) if args.use_global_feats else None

            # RSSI jitter augmentation: only on RSSI_z (index 0), and only on frames
            # where RSSI was actually observed (m_r = index 3 == 1, i.e. not imputed).
            if jitter_std > 0.0:
                rssi_mask = ch[..., 3:4]
                noise = torch.randn_like(ch[..., 0:1]) * jitter_std * rssi_mask
                ch = ch.clone()
                ch[..., 0:1] = ch[..., 0:1] + noise

            p_hat, v = model(ch, gf)
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
                dt = batch["dt"].to(dev)
                xy = batch["xy"].to(dev)
                gf = batch["global_feats"].to(dev) if args.use_global_feats else None
                p_hat, v = model(ch, gf)
                loss, _ = compute_losses(
                    p_hat, v, xy, dt, args.w_vel, args.w_dyn, args.w_smooth, args.pos_l2_w
                )
                va_sum += float(loss) * ch.size(0)
                n_va += ch.size(0)

        if scheduler is not None:
            scheduler.step()

        tr_m = tr_sum / max(n_tr, 1)
        va_m = va_sum / max(n_va, 1)
        lr_now = opt.param_groups[0]["lr"]
        print(f"epoch {epoch:03d}  train {tr_m:.5f}  val {va_m:.5f}  lr {lr_now:.2e}")
        if va_m < best:
            best = va_m
            torch.save(
                {
                    "model_type": "cnn_gru",
                    "model": model.state_dict(),
                    "pairs": pairs,
                    "rssi_mean": r_mean,
                    "rssi_std": r_std,
                    "xy_offset": offset,
                    "window": args.window,
                    "phase_period": args.phase_period,
                    "conf_lambda": args.conf_lambda,
                    "frame_embed_dim": args.frame_embed_dim,
                    "hidden": args.hidden,
                    "dropout": args.dropout,
                    "vel_scale": args.vel_scale,
                    "ch_feat_dim": RFIDTrackingWindowDataset.FEAT_DIM,
                    "split": args.split,
                    "seed": args.seed,
                    "val_ratio": args.val_ratio,
                    "unwrap_method": args.unwrap_method,
                    "recover": bool(args.recover),
                    "data_dir": str(data_dir),
                    "val_dir": str(val_dir) if val_dir is not None else "",
                    "global_feat_dim": global_feat_dim,
                    "use_global_feats": bool(args.use_global_feats),
                    "rssi_jitter_std": jitter_std,
                    "cosine_lr": bool(args.cosine_lr),
                    "min_lr": float(args.min_lr),
                    "weight_decay": float(args.weight_decay),
                    "anchor_decoder": bool(args.anchor_decoder),
                    "anchor_init_scale": float(args.anchor_init_scale),
                    "residual_scale": float(args.residual_scale),
                    "use_pdoa": bool(args.use_pdoa),
                    "pdoa_pair_indices": (
                        pdoa_pair_indices.tolist() if pdoa_pair_indices is not None else None
                    ),
                    "grid_layout": {
                        "cell_to_channel": layout["cell_to_channel"],
                        "local_xy": layout["local_xy"],
                        "ant_order": layout["ant_order"],
                        "tag_order": layout["tag_order"],
                    },
                },
                out_path,
            )
            print(f"  saved {out_path} (best val {best:.5f})")


if __name__ == "__main__":
    main()
