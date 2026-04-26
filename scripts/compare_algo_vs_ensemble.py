"""Compare Algo (DualBranchTracker2D + algo unwrap, w=10) vs V5 Ensemble
(3 x CNN+GRU with anchor decoder) on the held-out test set, using
sliding-window inference (last-frame prediction per window).

Generates three figures under checkpoints/plots_compare_algo_vs_ens/:

  1. fig_error_cdf.png       - cumulative distribution of per-frame L2 error
                               (one curve per model). Includes median, P95,
                               P99 markers.
  2. fig_perframe_error.png  - per-frame L2 error vs time, one panel per CSV,
                               two lines per panel (algo, ensemble).
  3. fig_velocity.png        - speed magnitude over time (top panel) and
                               velocity vector error over time (bottom
                               panel), one column per CSV.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from rfid_localization.dataset import (
    RFIDTrackingWindowDataset,
    build_enriched_table,
    load_csv_paths,
)
from rfid_localization.model import DualBranchTracker2D
from rfid_localization.model_cnn_gru import build_cnn_gru_from_ckpt
from rfid_localization.train import collate

VAL_DIR = Path("data/0421/test")
ALGO_CKPT = Path("checkpoints/rfid_0421_w10_algo.pt")
ENS_CKPTS = [
    Path("checkpoints/rfid_cnn_gru_w10_algo_v5.pt"),
    Path("checkpoints/rfid_cnn_gru_w10_algo_v5_s123.pt"),
    Path("checkpoints/rfid_cnn_gru_w10_algo_v5_s456.pt"),
]
OUT_DIR = Path("checkpoints/plots_compare_algo_vs_ens")
OUT_DIR.mkdir(parents=True, exist_ok=True)

ALGO_COLOR = "#1f77b4"
ENS_COLOR = "#d62728"


def _load_algo(path: Path, dev):
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


def _build_table(ckpt, df):
    return build_enriched_table(
        df,
        ckpt["pairs"],
        np.asarray(ckpt["rssi_mean"], dtype=np.float32),
        np.asarray(ckpt["rssi_std"], dtype=np.float32),
        phase_period=float(ckpt["phase_period"]),
        conf_lambda=float(ckpt["conf_lambda"]),
        unwrap_method=str(ckpt.get("unwrap_method", "arctan2")),
        recover=bool(ckpt.get("recover", False)),
    )


@torch.no_grad()
def _predict_algo(model, ds, offset, dev):
    """Sliding-window inference with the DualBranch model.

    For each window we use the last-frame prediction and last-frame velocity.
    """
    preds, gts, vs, ts = [], [], [], []
    for j in range(len(ds)):
        batch = collate([ds[j]])
        ch = batch["ch_feats"].to(dev)
        inv = batch["attn_invalid"].to(dev)
        gf = batch["global_feats"].to(dev)
        xy = batch["xy"].to(dev)
        xy0 = xy[:, 0, :]
        p, _pa, _pd, v, _alpha = model(ch, inv, gf, xy0)
        preds.append(p[0, -1, :].cpu().numpy() + offset)
        gts.append(xy[0, -1, :].cpu().numpy() + offset)
        vs.append(v[0, -1, :].cpu().numpy())
        ts.append(float(batch["ts"][0, -1]))
    return (
        np.asarray(preds, dtype=np.float32),
        np.asarray(gts, dtype=np.float32),
        np.asarray(vs, dtype=np.float32),
        np.asarray(ts, dtype=np.float64),
    )


@torch.no_grad()
def _predict_ensemble(models, ds, offset, dev):
    preds, gts, vs, ts = [], [], [], []
    for j in range(len(ds)):
        batch = collate([ds[j]])
        ch = batch["ch_feats"].to(dev)
        xy = batch["xy"].to(dev)
        ps, vss = [], []
        for m in models:
            gf = batch["global_feats"].to(dev) if getattr(m, "global_feat_dim", 0) > 0 else None
            p, v = m(ch, gf)
            ps.append(p)
            vss.append(v)
        p_mean = torch.stack(ps, 0).mean(0)
        v_mean = torch.stack(vss, 0).mean(0)
        preds.append(p_mean[0, -1, :].cpu().numpy() + offset)
        gts.append(xy[0, -1, :].cpu().numpy() + offset)
        vs.append(v_mean[0, -1, :].cpu().numpy())
        ts.append(float(batch["ts"][0, -1]))
    return (
        np.asarray(preds, dtype=np.float32),
        np.asarray(gts, dtype=np.float32),
        np.asarray(vs, dtype=np.float32),
        np.asarray(ts, dtype=np.float64),
    )


def _gt_velocity(gt_xy, ts):
    """Numerical GT velocity from consecutive (xy, t)."""
    if len(gt_xy) < 2:
        return np.zeros_like(gt_xy)
    dt = np.diff(ts)
    dt = np.where(dt < 1e-4, 1e-4, dt)
    v = np.zeros_like(gt_xy)
    v[1:, 0] = np.diff(gt_xy[:, 0]) / dt
    v[1:, 1] = np.diff(gt_xy[:, 1]) / dt
    v[0] = v[1]
    return v


def _plot_error_cdf(per_csv_results, out_path):
    """One panel: empirical CDF of per-frame L2 error for both models."""
    err_algo = np.concatenate([r["err_algo"] for r in per_csv_results])
    err_ens = np.concatenate([r["err_ens"] for r in per_csv_results])

    fig, ax = plt.subplots(figsize=(7.5, 5.5))

    def _plot_one(err, label, color):
        e = np.sort(err) * 100.0
        cdf = np.arange(1, len(e) + 1) / len(e)
        ax.plot(e, cdf, color=color, lw=1.8, label=f"{label} (n={len(e)})")
        for q, ls in zip([0.50, 0.95, 0.99], [":", "--", "-."]):
            v = float(np.quantile(err, q)) * 100.0
            ax.axvline(v, color=color, ls=ls, lw=0.8, alpha=0.55)
            ax.text(v, 0.02 + 0.04 * [0.5, 0.95, 0.99].index(q),
                    f"P{int(q*100)}={v:.1f}", color=color, fontsize=8,
                    rotation=90, va="bottom", ha="right")

    _plot_one(err_algo, "DualBranch", ALGO_COLOR)
    _plot_one(err_ens, "3 x CNN+GRU + anchor", ENS_COLOR)

    ax.set_xlabel("per-frame L2 position error (cm)")
    ax.set_ylabel("CDF  P(error ≤ x)")
    ax.set_xlim(0, max(np.quantile(err_algo, 0.995), np.quantile(err_ens, 0.995)) * 100 * 1.05)
    ax.set_ylim(0, 1.005)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=10)
    ax.set_title("Per-frame L2 error CDF (sliding-window inference, full test set)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_perframe_error(per_csv_results, out_path):
    """One panel per CSV; per-frame L2 error overlay for both models."""
    n = len(per_csv_results)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3 * nrows + 0.5),
                             sharey=True, squeeze=False)

    for ax_i, r in enumerate(per_csv_results):
        ax = axes[ax_i // ncols, ax_i % ncols]
        ts = r["ts"] - r["ts"][0]
        ax.plot(ts, r["err_algo"] * 100, color=ALGO_COLOR, lw=1.0, alpha=0.85,
                label=f"DualBranch  (mean={r['err_algo'].mean()*100:.2f} cm)")
        ax.plot(ts, r["err_ens"] * 100, color=ENS_COLOR, lw=1.0, alpha=0.85,
                label=f"3 x CNN+GRU + anchor  (mean={r['err_ens'].mean()*100:.2f} cm)")
        ax.fill_between(ts, 0, r["err_algo"] * 100, color=ALGO_COLOR, alpha=0.08)
        ax.fill_between(ts, 0, r["err_ens"] * 100, color=ENS_COLOR, alpha=0.08)
        ax.set_title(r["name"], fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("time (s, relative)")
        if ax_i % ncols == 0:
            ax.set_ylabel("L2 error (cm)")
        ax.legend(fontsize=8, loc="upper right")

    for k in range(n, nrows * ncols):
        axes[k // ncols, k % ncols].axis("off")

    fig.suptitle("Per-frame L2 error over time (sliding-window inference)",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_velocity(per_csv_results, out_path):
    """Two-row figure: speed magnitude (top) and velocity error (bottom).

    One column per CSV; GT in black, algo blue, ensemble red (top row).
    Bottom row shows ||v_pred - v_gt||.
    """
    n = len(per_csv_results)
    fig, axes = plt.subplots(2, n, figsize=(5 * n, 6), sharex="col",
                             squeeze=False)

    for j, r in enumerate(per_csv_results):
        ts = r["ts"] - r["ts"][0]
        speed_gt = np.linalg.norm(r["v_gt"], axis=1)
        speed_algo = np.linalg.norm(r["v_algo"], axis=1)
        speed_ens = np.linalg.norm(r["v_ens"], axis=1)
        ax = axes[0, j]
        ax.plot(ts, speed_gt, "k-", lw=1.4, label="GT")
        ax.plot(ts, speed_algo, color=ALGO_COLOR, lw=1.0, alpha=0.85, label="DualBranch")
        ax.plot(ts, speed_ens, color=ENS_COLOR, lw=1.0, alpha=0.85, label="3 x CNN+GRU + anchor")
        ax.set_title(f"{r['name']}  speed |v|", fontsize=10)
        ax.grid(True, alpha=0.3)
        if j == 0:
            ax.set_ylabel("|v| (m/s)")
        ax.legend(fontsize=8, loc="upper right")

        ve_algo = np.linalg.norm(r["v_algo"] - r["v_gt"], axis=1)
        ve_ens = np.linalg.norm(r["v_ens"] - r["v_gt"], axis=1)
        ax = axes[1, j]
        ax.plot(ts, ve_algo, color=ALGO_COLOR, lw=1.0, alpha=0.85,
                label=f"DualBranch  (mean={ve_algo.mean():.3f})")
        ax.plot(ts, ve_ens, color=ENS_COLOR, lw=1.0, alpha=0.85,
                label=f"3 x CNN+GRU + anchor  (mean={ve_ens.mean():.3f})")
        ax.fill_between(ts, 0, ve_algo, color=ALGO_COLOR, alpha=0.08)
        ax.fill_between(ts, 0, ve_ens, color=ENS_COLOR, alpha=0.08)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("time (s, relative)")
        if j == 0:
            ax.set_ylabel("‖v_pred - v_gt‖ (m/s)")
        ax.legend(fontsize=8, loc="upper right")

    fig.suptitle("Velocity comparison: speed magnitude (top) and velocity error (bottom)",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    warnings.filterwarnings("ignore", message=".*nested tensors.*", category=UserWarning)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {dev}")

    algo_model, algo_ckpt = _load_algo(ALGO_CKPT, dev)
    ens_loaded = [_load_cnn_gru(p, dev) for p in ENS_CKPTS]
    ens_models = [m for m, _c in ens_loaded]
    ens_ckpt = ens_loaded[0][1]

    algo_offset = np.asarray(algo_ckpt["xy_offset"], dtype=np.float32)
    ens_offset = np.asarray(ens_ckpt["xy_offset"], dtype=np.float32)
    algo_window = int(algo_ckpt["window"])
    ens_window = int(ens_ckpt["window"])
    print(f"algo: window={algo_window}  unwrap={algo_ckpt.get('unwrap_method')}  offset={algo_offset.tolist()}")
    print(f"ens : window={ens_window}  unwrap={ens_ckpt.get('unwrap_method')}  offset={ens_offset.tolist()}  (3 seeds)")

    per_csv_results = []
    csv_paths = sorted(load_csv_paths(VAL_DIR))
    print(f"test CSVs: {len(csv_paths)}")

    for p in csv_paths:
        df = pd.read_csv(p)
        tbl_algo = _build_table(algo_ckpt, df)
        tbl_ens = _build_table(ens_ckpt, df)
        ds_algo = RFIDTrackingWindowDataset(tbl_algo, window=algo_window, xy_offset=algo_offset)
        ds_ens = RFIDTrackingWindowDataset(tbl_ens, window=ens_window, xy_offset=ens_offset)
        if len(ds_algo) == 0 or len(ds_ens) == 0:
            print(f"  {p.name}: empty windows, skip")
            continue

        pa, ga, va, ta = _predict_algo(algo_model, ds_algo, algo_offset, dev)
        pe, ge, ve, te = _predict_ensemble(ens_models, ds_ens, ens_offset, dev)

        n = min(len(pa), len(pe))
        pa, ga, va, ta = pa[:n], ga[:n], va[:n], ta[:n]
        pe, ge, ve, te = pe[:n], ge[:n], ve[:n], te[:n]
        gt = ge
        ts = te
        v_gt = _gt_velocity(gt, ts)

        err_algo = np.linalg.norm(pa - gt, axis=1)
        err_ens = np.linalg.norm(pe - gt, axis=1)

        per_csv_results.append({
            "name": p.stem.replace("aruco_rfid_", ""),
            "ts": ts,
            "err_algo": err_algo,
            "err_ens": err_ens,
            "v_algo": va,
            "v_ens": ve,
            "v_gt": v_gt,
        })
        print(f"  {p.name}: n={n}  algo mean={err_algo.mean()*100:.2f} cm  ens mean={err_ens.mean()*100:.2f} cm")

    if not per_csv_results:
        raise SystemExit("no usable CSV produced predictions")

    err_algo_all = np.concatenate([r["err_algo"] for r in per_csv_results])
    err_ens_all = np.concatenate([r["err_ens"] for r in per_csv_results])
    print()
    print(f"pooled algo: mean={err_algo_all.mean()*100:.2f} cm  median={np.median(err_algo_all)*100:.2f} cm  P95={np.quantile(err_algo_all,0.95)*100:.2f} cm")
    print(f"pooled ens : mean={err_ens_all.mean()*100:.2f} cm  median={np.median(err_ens_all)*100:.2f} cm  P95={np.quantile(err_ens_all,0.95)*100:.2f} cm")

    _plot_error_cdf(per_csv_results, OUT_DIR / "fig_error_cdf.png")
    print(f"saved: {OUT_DIR / 'fig_error_cdf.png'}")
    _plot_perframe_error(per_csv_results, OUT_DIR / "fig_perframe_error.png")
    print(f"saved: {OUT_DIR / 'fig_perframe_error.png'}")
    _plot_velocity(per_csv_results, OUT_DIR / "fig_velocity.png")
    print(f"saved: {OUT_DIR / 'fig_velocity.png'}")


if __name__ == "__main__":
    main()
