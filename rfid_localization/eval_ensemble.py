#!/usr/bin/env python3
"""Evaluate an ensemble of CNN+GRU checkpoints.

Loads N checkpoints, averages their per-frame xy and v predictions, and reports
the same metrics + per-CSV trajectory plots as ``eval_cnn_gru``.

All checkpoints must share the same:
    - channel pairs / rssi_mean / rssi_std
    - xy_offset
    - window, phase_period, conf_lambda
    - unwrap_method, recover
    - grid_layout (cell_to_channel + tag_order)
    - val_dir (and data_dir, if --val_dir not set)

Usage:
    python -m rfid_localization.eval_ensemble \\
        --checkpoints checkpoints/rfid_cnn_gru_w10_algo_v5.pt \\
                      checkpoints/rfid_cnn_gru_w10_algo_v5_s123.pt \\
                      checkpoints/rfid_cnn_gru_w10_algo_v5_s456.pt \\
        --plots_dir checkpoints/plots_cnn_gru_w10_algo_v5_ens
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from rfid_localization.dataset import (
    RFIDTrackingWindowDataset,
    build_enriched_table,
    load_csv_paths,
    stack_tables,
)
from rfid_localization.model_cnn_gru import build_cnn_gru_from_ckpt
from rfid_localization.train import collate


def _maybe_gf(batch, dev, model):
    if getattr(model, "global_feat_dim", 0) > 0:
        return batch["global_feats"].to(dev)
    return None


def _maybe_gf_stack(buf, dev, model):
    if getattr(model, "global_feat_dim", 0) > 0:
        return torch.stack([b["global_feats"] for b in buf]).to(dev)
    return None


def _load_models(ckpt_paths: list[Path], dev) -> list[dict]:
    out = []
    for p in ckpt_paths:
        if not p.is_file():
            raise SystemExit(f"Missing checkpoint: {p}")
        ckpt = torch.load(p, map_location="cpu", weights_only=False)
        if str(ckpt.get("model_type", "")) != "cnn_gru":
            raise SystemExit(f"{p}: model_type != 'cnn_gru'")
        model = build_cnn_gru_from_ckpt(ckpt).to(dev)
        model.load_state_dict(ckpt["model"])
        model.eval()
        out.append({"path": p, "ckpt": ckpt, "model": model})
    return out


def _assert_compatible(models: list[dict]) -> None:
    ref = models[0]["ckpt"]
    keys = ["window", "phase_period", "conf_lambda", "unwrap_method", "recover", "use_pdoa"]
    for m in models[1:]:
        for k in keys:
            if str(m["ckpt"].get(k)) != str(ref.get(k)):
                raise SystemExit(f"{m['path']} differs on '{k}': {m['ckpt'].get(k)} vs {ref.get(k)}")
        if len(m["ckpt"]["pairs"]) != len(ref["pairs"]):
            raise SystemExit(f"{m['path']} differs on channel pairs length")
        if not np.allclose(np.asarray(m["ckpt"]["rssi_mean"]), np.asarray(ref["rssi_mean"]), atol=1e-4):
            raise SystemExit(f"{m['path']} differs on rssi_mean")
        if not np.allclose(np.asarray(m["ckpt"]["xy_offset"]), np.asarray(ref["xy_offset"]), atol=1e-4):
            raise SystemExit(f"{m['path']} differs on xy_offset")


@torch.no_grad()
def _mean_forward(models: list[dict], ch, gfs_per_model, out_p: bool = True):
    """Average predictions across the ensemble."""
    preds_p = []
    preds_v = []
    for i, m in enumerate(models):
        gf = gfs_per_model[i]
        p, v = m["model"](ch, gf)
        preds_p.append(p)
        preds_v.append(v)
    p_mean = torch.stack(preds_p, dim=0).mean(dim=0)
    v_mean = torch.stack(preds_v, dim=0).mean(dim=0)
    return p_mean, v_mean


def _window_l2_ensemble(models, ds, dev):
    out: list[tuple[float, int]] = []
    with torch.no_grad():
        for j in range(len(ds)):
            batch = collate([ds[j]])
            ch = batch["ch_feats"].to(dev)
            xy = batch["xy"].to(dev)
            gfs = [_maybe_gf(batch, dev, m["model"]) for m in models]
            p, _v = _mean_forward(models, ch, gfs)
            err = float((p - xy).norm(dim=-1).mean().cpu())
            out.append((err, j))
    return out


def _plot_window_panels(models, ds, dev, plots_dir: Path, n_show: int, traj_pick: str) -> None:
    plots_dir.mkdir(parents=True, exist_ok=True)
    n = len(ds)
    if n == 0:
        return
    n_show = min(int(n_show), n)
    if traj_pick == "even":
        picks = np.linspace(0, n - 1, num=n_show, dtype=int).tolist()
    else:
        scored = _window_l2_ensemble(models, ds, dev)
        scored.sort(key=lambda t: t[0])
        picks = [j for _e, j in scored[:n_show]]
    n_show = len(picks)

    ncols = min(3, n_show)
    nrows = int(np.ceil(n_show / ncols))
    fig, _ = plt.subplots(nrows, ncols, figsize=(3.5 * ncols + 1, 3.2 * nrows + 0.5))
    axes_list = fig.axes
    for k in range(len(picks), len(axes_list)):
        axes_list[k].axis("off")
    with torch.no_grad():
        for ax_i, j in enumerate(picks):
            if ax_i >= len(axes_list):
                break
            batch = collate([ds[int(j)]])
            ch = batch["ch_feats"].to(dev)
            xy = batch["xy"].to(dev)
            gfs = [_maybe_gf(batch, dev, m["model"]) for m in models]
            p, _v = _mean_forward(models, ch, gfs)
            g = xy[0].cpu().numpy()
            pr = p[0].cpu().numpy()
            ax = axes_list[ax_i]
            ax.plot(g[:, 0], g[:, 1], "k-", lw=1.2, label="GT")
            ax.plot(pr[:, 0], pr[:, 1], "r--", lw=1.0, label="Ensemble pred")
            ax.set_aspect("equal", adjustable="box")
            ax.grid(True, alpha=0.3)
            e = float((p - xy).norm(dim=-1).mean().cpu())
            ax.set_title(f"val #{j}\nmean L2={e:.4f}", fontsize=8)
            if ax_i == 0:
                ax.legend(fontsize=7)
    pick_label = "best (lowest mean L2)" if traj_pick != "even" else "evenly spaced"
    fig.suptitle(f"Ensemble trajectory overlays (val, {pick_label}, n={len(picks)})", fontsize=12)
    fig.tight_layout()
    fig.savefig(plots_dir / "fig_trajectories.png", dpi=150)
    plt.close(fig)

    j0 = int(picks[0])
    batch = collate([ds[j0]])
    with torch.no_grad():
        ch = batch["ch_feats"].to(dev)
        dt = batch["dt"].to(dev)
        xy = batch["xy"].to(dev)
        gfs = [_maybe_gf(batch, dev, m["model"]) for m in models]
        p, v = _mean_forward(models, ch, gfs)
        g = xy[0].cpu().numpy()
        pr = p[0].cpu().numpy()
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
    fig.suptitle(f"x(t), y(t) - val window #{j0}", fontsize=12)
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
    fig.suptitle(f"Velocity - val window #{j0}", fontsize=12)
    fig.tight_layout()
    fig.savefig(plots_dir / "fig_velocity.png", dpi=150)
    plt.close(fig)


def _plot_csv(pred, gt, ts, out_traj: Path, out_xyt: Path, title: str):
    err = np.linalg.norm(pred - gt, axis=1)
    mean_e = float(np.mean(err)) if err.size else float("nan")
    max_e = float(np.max(err)) if err.size else float("nan")

    fig, ax = plt.subplots(figsize=(6, 5.5))
    ax.plot(gt[:, 0], gt[:, 1], "k-", lw=1.2, label="GT")
    ax.plot(pred[:, 0], pred[:, 1], "r--", lw=1.0, alpha=0.85, label="Ensemble pred")
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


def _per_csv_predict(models, table, window, offset, dev, batch_size=64):
    ds = RFIDTrackingWindowDataset(table, window=window, xy_offset=offset)
    if len(ds) == 0:
        return (
            np.zeros((0, 2), dtype=np.float32),
            np.zeros((0, 2), dtype=np.float32),
            np.zeros((0,), dtype=np.float64),
        )
    preds, gts, times = [], [], []
    buf: list[dict] = []
    for i in range(len(ds)):
        buf.append(ds[i])
        if len(buf) == batch_size or i == len(ds) - 1:
            ch = torch.stack([b["ch_feats"] for b in buf]).to(dev)
            xy = torch.stack([b["xy"] for b in buf]).to(dev)
            gfs = [_maybe_gf_stack(buf, dev, m["model"]) for m in models]
            p_mean, _v_mean = _mean_forward(models, ch, gfs)
            p_last = p_mean[:, -1, :].cpu().numpy() + offset.reshape(1, 2)
            g_last = xy[:, -1, :].cpu().numpy() + offset.reshape(1, 2)
            t_last = [float(b["ts"][-1]) for b in buf]
            preds.append(p_last)
            gts.append(g_last)
            times.extend(t_last)
            buf.clear()
    return (
        np.concatenate(preds, axis=0).astype(np.float32),
        np.concatenate(gts, axis=0).astype(np.float32),
        np.asarray(times, dtype=np.float64),
    )


def main() -> None:
    warnings.filterwarnings("ignore", message=".*nested tensors.*", category=UserWarning)
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", type=str, nargs="+", required=True)
    ap.add_argument("--data_dir", type=str, default="")
    ap.add_argument("--val_dir", type=str, default="")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--eps_pos", type=float, default=0.02)
    ap.add_argument("--plots_dir", type=str, default="")
    ap.add_argument("--traj_pick", type=str, default="best", choices=["best", "even"])
    ap.add_argument("--traj_n", type=int, default=9)
    ap.add_argument("--report_cm", action="store_true")
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_paths = [Path(p) for p in args.checkpoints]
    models = _load_models(ckpt_paths, dev)
    _assert_compatible(models)
    print(f"ensemble of {len(models)} checkpoints:")
    for m in models:
        print(f"  {m['path']}")

    ref = models[0]["ckpt"]
    pairs = ref["pairs"]
    window = int(ref.get("window", 10))
    phase_period = float(ref.get("phase_period", 2048.0))
    conf_lambda = float(ref.get("conf_lambda", 0.35))
    r_mean = np.asarray(ref["rssi_mean"], dtype=np.float32)
    r_std = np.asarray(ref["rssi_std"], dtype=np.float32)
    offset = np.asarray(ref["xy_offset"], dtype=np.float32)
    unwrap_method = str(ref.get("unwrap_method", "arctan2"))
    recover = bool(ref.get("recover", False))

    data_dir_str = args.data_dir.strip() or str(ref.get("data_dir", "data"))
    val_dir_str = args.val_dir.strip() or str(ref.get("val_dir", ""))
    if not val_dir_str:
        raise SystemExit("Need a val_dir for evaluation (either via --val_dir or embedded in the checkpoint).")
    data_dir = Path(data_dir_str)
    val_dir = Path(val_dir_str)
    paths = load_csv_paths(data_dir)
    val_paths = load_csv_paths(val_dir)
    if not val_paths:
        raise SystemExit(f"No CSV under {val_dir}")

    pdoa_pair_indices = ref.get("pdoa_pair_indices", None)
    if pdoa_pair_indices is not None:
        pdoa_pair_indices = np.asarray(pdoa_pair_indices, dtype=np.int64)
    build_kw = dict(
        phase_period=phase_period,
        conf_lambda=conf_lambda,
        unwrap_method=unwrap_method,
        recover=recover,
        pdoa_pair_indices=pdoa_pair_indices,
    )
    tbl_va = stack_tables(val_paths, pairs, r_mean, r_std, **build_kw)
    val_ds = RFIDTrackingWindowDataset(tbl_va, window=window, xy_offset=offset)
    loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False, collate_fn=collate)

    all_p: list[torch.Tensor] = []
    all_g: list[torch.Tensor] = []
    all_v: list[torch.Tensor] = []
    all_ve: list[torch.Tensor] = []
    win_mae: list[float] = []
    win_final: list[float] = []
    smooth_terms: list[torch.Tensor] = []

    with torch.no_grad():
        for batch in loader:
            ch = batch["ch_feats"].to(dev)
            dt = batch["dt"].to(dev)
            xy = batch["xy"].to(dev)
            gfs = [_maybe_gf(batch, dev, m["model"]) for m in models]
            p, v = _mean_forward(models, ch, gfs)

            dtn = dt[:, 1:].unsqueeze(-1).clamp_min(1e-4)
            v_gt = (xy[:, 1:, :] - xy[:, :-1, :]) / dtn
            ve = v[:, 1:, :] - v_gt
            err = (p - xy).norm(dim=-1)
            win_mae.extend(err.mean(dim=1).cpu().tolist())
            win_final.extend(err[:, -1].cpu().tolist())
            dv = (v[:, 1:, :] - v[:, :-1, :]).norm(dim=-1)
            smooth_terms.append(dv.mean(dim=1))
            all_p.append(p.cpu())
            all_g.append(xy.cpu())
            all_v.append(v.cpu())
            all_ve.append(ve.cpu())

    p_all = torch.cat(all_p, dim=0).reshape(-1, 2).numpy()
    g_all = torch.cat(all_g, dim=0).reshape(-1, 2).numpy()
    ve_all = torch.cat(all_ve, dim=0).reshape(-1, 2).numpy()

    err = p_all - g_all
    mae_xy = np.mean(np.abs(err), axis=0)
    euc = np.linalg.norm(err, axis=1)
    mae_l1 = float(np.mean(euc))
    rmse_l2 = float(np.sqrt(np.mean(euc**2)))
    rmse_xy = np.sqrt(np.mean(err**2, axis=0))
    hit = float(np.mean(euc < args.eps_pos))
    rmse_v = np.sqrt(np.mean(ve_all**2, axis=0))
    rmse_v_l2 = float(np.sqrt(np.mean((ve_all**2).sum(axis=-1))))
    smooth = float(torch.cat(smooth_terms, dim=0).mean()) if smooth_terms else 0.0

    print(f"val_dir={val_dir}  windows={len(val_ds)}  points={len(p_all)}  window_T={window}  unwrap={unwrap_method}")
    print(f"position RMSE (L2): {rmse_l2:.6f}")
    print(f"position MAE  (L2): {mae_l1:.6f}")
    print(f"position MAE  (x,y): {mae_xy[0]:.6f}, {mae_xy[1]:.6f}")
    print(f"position RMSE (x,y): {rmse_xy[0]:.6f}, {rmse_xy[1]:.6f}")
    if args.report_cm:
        print(f"position RMSE (cm): {rmse_l2 * 100:.4f} cm")
    print(f"hit_rate@eps (L2<{args.eps_pos}): {hit:.4f}")
    print(f"velocity RMSE (vx,vy): {rmse_v[0]:.6f}, {rmse_v[1]:.6f}  |  L2: {rmse_v_l2:.6f}")
    print(f"per-window mean L2: mean={np.mean(win_mae):.6f} std={np.std(win_mae):.6f}")
    print(f"per-window final  L2: mean={np.mean(win_final):.6f} std={np.std(win_final):.6f}")
    print(f"temporal smoothness E||v_t - v_(t-1)||: {smooth:.6f}")

    if args.plots_dir:
        plot_dir = Path(args.plots_dir)
        _plot_window_panels(models, val_ds, dev, plot_dir, n_show=args.traj_n, traj_pick=args.traj_pick)
        print(f"saved window figures under: {plot_dir}")
        per_csv_dir = plot_dir / "per_csv"
        per_csv_dir.mkdir(parents=True, exist_ok=True)
        print(f"plotting {len(val_paths)} CSV from {val_dir}")
        summary: list[tuple[str, int, float, float]] = []
        for p in val_paths:
            df = pd.read_csv(p)
            table = build_enriched_table(
                df, pairs, r_mean, r_std,
                phase_period=phase_period, conf_lambda=conf_lambda,
                unwrap_method=unwrap_method, recover=recover,
                pdoa_pair_indices=pdoa_pair_indices,
            )
            pred, gt, ts = _per_csv_predict(models, table, window, offset, dev, args.batch_size)
            if len(pred) == 0:
                print(f"  {p.name}: no valid window, skip")
                continue
            mean_e, max_e = _plot_csv(
                pred, gt, ts,
                per_csv_dir / f"{p.stem}_traj.png",
                per_csv_dir / f"{p.stem}_xyt.png",
                title=p.stem,
            )
            print(f"  {p.name}: n={len(pred)}  mean L2={mean_e*100:.2f} cm  max L2={max_e*100:.2f} cm")
            summary.append((p.name, len(pred), mean_e, max_e))
        if summary:
            all_n = sum(s[1] for s in summary)
            all_mean = sum(s[1] * s[2] for s in summary) / max(all_n, 1)
            print(f"per-csv pooled mean L2: {all_mean*100:.2f} cm over {all_n} points in {len(summary)} CSVs")


if __name__ == "__main__":
    main()
