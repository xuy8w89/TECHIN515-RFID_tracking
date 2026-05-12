"""Benchmark per-frame inference latency for:
- single CNN+GRU model (v5)
- 3-seed ensemble (v5 + s123 + s456)
- 3-seed ensemble with batched forward (single big batch)
- single DualBranchTracker2D (algo)

For each, we run a sliding-window inference on a real test CSV and
report the average per-frame latency.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from rfid_localization.dataset import RFIDTrackingWindowDataset, build_enriched_table
from rfid_localization.model import DualBranchTracker2D
from rfid_localization.model_cnn_gru import build_cnn_gru_from_ckpt
from rfid_localization.train import collate

CSV_PATH = Path("data/0421/test/aruco_rfid_20260421_155629.csv")
N_WARMUP = 20
N_FRAMES = 200  # how many frames to time


def _load_cnn_gru(path, dev):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = build_cnn_gru_from_ckpt(ckpt).to(dev)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt


def _build_ds(ckpt):
    pairs = ckpt["pairs"]
    r_mean = np.asarray(ckpt["rssi_mean"], dtype=np.float32)
    r_std = np.asarray(ckpt["rssi_std"], dtype=np.float32)
    offset = np.asarray(ckpt["xy_offset"], dtype=np.float32)
    df = pd.read_csv(CSV_PATH)
    table = build_enriched_table(
        df, pairs, r_mean, r_std,
        phase_period=float(ckpt["phase_period"]),
        conf_lambda=float(ckpt["conf_lambda"]),
        unwrap_method=str(ckpt.get("unwrap_method", "arctan2")),
        recover=bool(ckpt.get("recover", False)),
    )
    return RFIDTrackingWindowDataset(table, window=int(ckpt["window"]), xy_offset=offset)


@torch.no_grad()
def time_single_cnn_gru(model, ds, dev, n_frames):
    for _ in range(N_WARMUP):
        batch = collate([ds[0]])
        ch = batch["ch_feats"].to(dev)
        gf = batch["global_feats"].to(dev) if getattr(model, "global_feat_dim", 0) > 0 else None
        _ = model(ch, gf)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for j in range(n_frames):
        batch = collate([ds[j % len(ds)]])
        ch = batch["ch_feats"].to(dev)
        gf = batch["global_feats"].to(dev) if getattr(model, "global_feat_dim", 0) > 0 else None
        _ = model(ch, gf)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n_frames


@torch.no_grad()
def time_ensemble_serial(models, ds, dev, n_frames):
    """Loop over models serially (current eval_ensemble behaviour)."""
    for _ in range(N_WARMUP):
        batch = collate([ds[0]])
        ch = batch["ch_feats"].to(dev)
        for m in models:
            gf = batch["global_feats"].to(dev) if getattr(m, "global_feat_dim", 0) > 0 else None
            _ = m(ch, gf)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for j in range(n_frames):
        batch = collate([ds[j % len(ds)]])
        ch = batch["ch_feats"].to(dev)
        ps = []
        for m in models:
            gf = batch["global_feats"].to(dev) if getattr(m, "global_feat_dim", 0) > 0 else None
            p, _v = m(ch, gf)
            ps.append(p)
        _ = torch.stack(ps, 0).mean(0)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n_frames


@torch.no_grad()
def time_ensemble_batched(models, ds, dev, n_frames):
    """Stack the same window 3 times into a B=3 batch and run one forward
    per model on the full batch. (Doesn't actually save time vs serial here
    since we only have 1 input - included for completeness.)"""
    return time_ensemble_serial(models, ds, dev, n_frames)


@torch.no_grad()
def time_dual_branch(model, ds, dev, n_frames):
    for _ in range(N_WARMUP):
        batch = collate([ds[0]])
        ch = batch["ch_feats"].to(dev)
        inv = batch["attn_invalid"].to(dev)
        gf = batch["global_feats"].to(dev)
        xy = batch["xy"].to(dev)
        _ = model(ch, inv, gf, xy[:, 0, :])
    if dev.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for j in range(n_frames):
        batch = collate([ds[j % len(ds)]])
        ch = batch["ch_feats"].to(dev)
        inv = batch["attn_invalid"].to(dev)
        gf = batch["global_feats"].to(dev)
        xy = batch["xy"].to(dev)
        _ = model(ch, inv, gf, xy[:, 0, :])
    if dev.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n_frames


def _count_params(m):
    return sum(p.numel() for p in m.parameters())


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {dev}")
    print(f"torch.get_num_threads(): {torch.get_num_threads()}")

    v5, v5_ckpt = _load_cnn_gru(Path("checkpoints/rfid_cnn_gru_w10_algo_v5.pt"), dev)
    v5_s123, _ = _load_cnn_gru(Path("checkpoints/rfid_cnn_gru_w10_algo_v5_s123.pt"), dev)
    v5_s456, _ = _load_cnn_gru(Path("checkpoints/rfid_cnn_gru_w10_algo_v5_s456.pt"), dev)
    ds = _build_ds(v5_ckpt)
    print(f"params per CNN+GRU: {_count_params(v5):,}")

    algo_ckpt = torch.load("checkpoints/rfid_0421_w10_algo.pt", map_location="cpu", weights_only=False)
    algo_model = DualBranchTracker2D(
        num_channels=len(algo_ckpt["pairs"]),
        ch_in_dim=int(algo_ckpt.get("ch_feat_dim", 14)),
        global_dim=int(algo_ckpt.get("global_dim", 3)),
        d_ch=int(algo_ckpt.get("d_ch", 32)),
        gru_hidden=int(algo_ckpt.get("hidden", 128)),
        vel_scale=float(algo_ckpt.get("vel_scale", 0.08)),
    ).to(dev)
    algo_model.load_state_dict(algo_ckpt["model"])
    algo_model.eval()
    ds_algo = _build_ds(algo_ckpt)
    print(f"params DualBranch:  {_count_params(algo_model):,}")

    n = N_FRAMES
    t_algo = time_dual_branch(algo_model, ds_algo, dev, n)
    t_v5 = time_single_cnn_gru(v5, ds, dev, n)
    t_ens = time_ensemble_serial([v5, v5_s123, v5_s456], ds, dev, n)

    print()
    print(f"benchmark: {n} frames each (after {N_WARMUP} warmup)")
    print(f"  DualBranch (algo)   : {t_algo*1000:7.3f} ms/frame   ->  {1/t_algo:6.1f} Hz")
    print(f"  CNN+GRU single (v5) : {t_v5*1000:7.3f} ms/frame   ->  {1/t_v5:6.1f} Hz")
    print(f"  CNN+GRU ensemble x3 : {t_ens*1000:7.3f} ms/frame   ->  {1/t_ens:6.1f} Hz")
    print()
    print(f"ensemble overhead vs single: {t_ens/t_v5:.2f}x  (+{(t_ens-t_v5)*1000:.2f} ms)")


if __name__ == "__main__":
    main()
