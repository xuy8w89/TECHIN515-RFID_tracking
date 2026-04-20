"""
Timestamp-aware + missing-aware + phase-aware (period 2048) feature construction for 2D RFID tracking.

Per channel and timestep:
  - RSSI/phase with forward-fill; is_imputed_r/p flags
  - time_since_last_real_obs (seconds) -> confidence c = exp(-lambda * tslv)
  - r_dot = dr_fill/dt, phase_dot = wrap_angle(dtheta)/dt, validity masks for derivatives
Global per timestep:
  - dt to previous sample (irregular sampling)
Windows require valid WRIST_xy on all steps and at least one real antenna read per step.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# Phase/RSSI pairs to drop entirely (no features, no model inputs).
EXCLUDED_ANTENNA_PAIRS_PHASE = frozenset(
    {
        "ant1_003f0102030405000000ffff_phase",
    }
)


def discover_channel_pairs(columns: Sequence[str]) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for c in columns:
        if c.endswith("_phase") and "ant" in c:
            if c in EXCLUDED_ANTENNA_PAIRS_PHASE:
                continue
            r = c.replace("_phase", "_rssi")
            if r in columns:
                pairs.append((c, r))
    return pairs


def compute_rssi_norm(
    dfs: list[pd.DataFrame],
    pairs: list[tuple[str, str]],
) -> tuple[np.ndarray, np.ndarray]:
    c = len(pairs)
    vals: list[list[float]] = [[] for _ in range(c)]
    for df in dfs:
        for j, (_pc, rc) in enumerate(pairs):
            rv = pd.to_numeric(df[rc], errors="coerce").to_numpy()
            ok_r = np.isfinite(rv) & (rv >= 0)
            if np.any(ok_r):
                vals[j].extend(rv[ok_r].astype(np.float64).tolist())
    mean = np.zeros(c, dtype=np.float32)
    std = np.ones(c, dtype=np.float32)
    for j in range(c):
        if not vals[j]:
            continue
        a = np.asarray(vals[j], dtype=np.float64)
        mean[j] = float(np.mean(a))
        s = float(np.std(a))
        std[j] = s if s > 1e-3 else 1.0
    return mean, std


def _wrap_angle(delta: np.ndarray) -> np.ndarray:
    """Wrap to (-pi, pi]."""
    return np.arctan2(np.sin(delta), np.cos(delta))


def build_enriched_table(
    df: pd.DataFrame,
    pairs: list[tuple[str, str]],
    rssi_mean: np.ndarray,
    rssi_std: np.ndarray,
    phase_period: float = 2048.0,
    conf_lambda: float = 0.35,
    max_tslv_clip_s: float = 30.0,
) -> dict[str, np.ndarray]:
    """
    Returns dict with:
      ch_feats [N, C, F]  channel features
      attn_invalid [N, C] bool True = no real read and no forward-fill (dead)
      dt [N] seconds since previous row (row 0 = row 1 dt or 0.1)
      xy [N, 2], ts [N], valid_xy [N]
    """
    n = len(df)
    c = len(pairs)
    ts = pd.to_numeric(df["timestamp"], errors="coerce").to_numpy(dtype=np.float64)

    phase_obs = np.full((n, c), np.nan, dtype=np.float64)
    rssi_obs = np.full((n, c), np.nan, dtype=np.float64)
    m_p = np.zeros((n, c), dtype=np.float32)
    m_r = np.zeros((n, c), dtype=np.float32)

    for j, (pc, rc) in enumerate(pairs):
        pv = pd.to_numeric(df[pc], errors="coerce").to_numpy(dtype=np.float64)
        rv = pd.to_numeric(df[rc], errors="coerce").to_numpy(dtype=np.float64)
        ok_p = np.isfinite(pv) & (pv >= 0)
        ok_r = np.isfinite(rv) & (rv >= 0)
        phase_obs[ok_p, j] = pv[ok_p]
        rssi_obs[ok_r, j] = rv[ok_r]
        m_p[:, j] = ok_p.astype(np.float32)
        m_r[:, j] = ok_r.astype(np.float32)

    phase_fill = np.zeros((n, c), dtype=np.float32)
    rssi_fill = np.zeros((n, c), dtype=np.float32)
    imp_p = np.zeros((n, c), dtype=np.float32)
    imp_r = np.zeros((n, c), dtype=np.float32)
    tslv = np.zeros((n, c), dtype=np.float32)

    last_phase = np.full(c, np.nan, dtype=np.float64)
    last_rssi = np.full(c, np.nan, dtype=np.float64)
    last_real_ts = np.full(c, np.nan, dtype=np.float64)

    for i in range(n):
        for j in range(c):
            if m_p[i, j] > 0.5:
                last_phase[j] = float(phase_obs[i, j])
                phase_fill[i, j] = last_phase[j]
                imp_p[i, j] = 0.0
            elif np.isfinite(last_phase[j]):
                phase_fill[i, j] = float(last_phase[j])
                imp_p[i, j] = 1.0
            else:
                phase_fill[i, j] = 0.0
                imp_p[i, j] = 0.0

            if m_r[i, j] > 0.5:
                last_rssi[j] = float(rssi_obs[i, j])
                rssi_fill[i, j] = last_rssi[j]
                imp_r[i, j] = 0.0
            elif np.isfinite(last_rssi[j]):
                rssi_fill[i, j] = float(last_rssi[j])
                imp_r[i, j] = 1.0
            else:
                rssi_fill[i, j] = 0.0
                imp_r[i, j] = 0.0

            had_real = (m_p[i, j] > 0.5) or (m_r[i, j] > 0.5)
            if had_real:
                last_real_ts[j] = ts[i]
                tslv[i, j] = 0.0
            elif np.isfinite(last_real_ts[j]):
                tslv[i, j] = float(min(max_tslv_clip_s, ts[i] - last_real_ts[j]))
            else:
                tslv[i, j] = float(max_tslv_clip_s)

    p_wrapped = np.mod(phase_fill.astype(np.float64), float(phase_period))
    ang = (2.0 * math.pi) * (p_wrapped / float(phase_period))
    sin_p = np.sin(ang).astype(np.float32)
    cos_p = np.cos(ang).astype(np.float32)

    r_z = (rssi_fill.astype(np.float64) - rssi_mean.reshape(1, -1)) / rssi_std.reshape(1, -1)
    r_z = r_z.astype(np.float32)

    c_exp = np.exp(-conf_lambda * tslv).astype(np.float32)
    tslv_n = (tslv / max_tslv_clip_s).astype(np.float32)

    dt = np.zeros(n, dtype=np.float32)
    for i in range(1, n):
        d = float(ts[i] - ts[i - 1])
        dt[i] = d if np.isfinite(d) and d > 1e-6 else 0.05
    if n > 1:
        dt[0] = dt[1]
    else:
        dt[0] = 0.05

    r_dot = np.zeros((n, c), dtype=np.float32)
    m_rdot = np.zeros((n, c), dtype=np.float32)
    p_dot = np.zeros((n, c), dtype=np.float32)
    m_pdot = np.zeros((n, c), dtype=np.float32)
    unwrap_vel = np.zeros((n, c), dtype=np.float32)
    unw = np.zeros((n, c), dtype=np.float64)

    for i in range(1, n):
        dti = float(dt[i])
        if dti <= 0:
            continue
        th_prev = (2.0 * math.pi) * (np.mod(phase_fill[i - 1], float(phase_period)) / float(phase_period))
        th_cur = (2.0 * math.pi) * (np.mod(phase_fill[i], float(phase_period)) / float(phase_period))
        dth = _wrap_angle(th_cur - th_prev)
        p_dot[i, :] = (dth / dti).astype(np.float32)
        both_p = (m_p[i, :] > 0.5) & (m_p[i - 1, :] > 0.5)
        m_pdot[i, :] = both_p.astype(np.float32)

        unw[i] = unw[i - 1]
        inc = both_p.astype(np.float64) * dth
        unw[i] = unw[i - 1] + inc
        unwrap_vel[i, :] = ((unw[i] - unw[i - 1]) / dti).astype(np.float32)

        dr = (rssi_fill[i, :] - rssi_fill[i - 1, :]).astype(np.float32)
        r_dot[i, :] = dr / float(dti)
        both_r = (m_r[i, :] > 0.5) & (m_r[i - 1, :] > 0.5)
        m_rdot[i, :] = both_r.astype(np.float32)

    # channel feature stack
    ch_feats = np.stack(
        [
            r_z,
            sin_p,
            cos_p,
            m_r,
            m_p,
            imp_r,
            imp_p,
            tslv_n,
            c_exp,
            r_dot,
            m_rdot,
            p_dot,
            m_pdot,
            unwrap_vel,
        ],
        axis=-1,
    ).astype(np.float32)

    dead = (m_r < 0.5) & (m_p < 0.5) & (imp_r < 0.5) & (imp_p < 0.5)
    attn_invalid = dead

    x = pd.to_numeric(df["WRIST_x"], errors="coerce").to_numpy()
    y = pd.to_numeric(df["WRIST_y"], errors="coerce").to_numpy()
    valid_xy = np.isfinite(x) & np.isfinite(y)
    xy = np.stack([np.where(valid_xy, x, 0.0), np.where(valid_xy, y, 0.0)], axis=-1).astype(np.float32)

    log_dt = np.log(np.maximum(dt.astype(np.float64), 1e-4)).astype(np.float32)
    inv_dt = (1.0 / np.maximum(dt.astype(np.float64), 1e-4)).astype(np.float32)
    global_feats = np.stack([dt.astype(np.float32), log_dt, inv_dt], axis=-1)

    return {
        "ch_feats": ch_feats,
        "attn_invalid": attn_invalid,
        "dt": dt.astype(np.float32),
        "global_feats": global_feats.astype(np.float32),
        "xy": xy,
        "ts": ts,
        "valid_xy": valid_xy.astype(bool),
    }


def stack_tables(paths: list[Path], pairs: list[tuple[str, str]], r_mean, r_std, **kw) -> dict[str, np.ndarray]:
    parts: list[dict[str, np.ndarray]] = []
    for p in paths:
        df = pd.read_csv(p)
        parts.append(build_enriched_table(df, pairs, r_mean, r_std, **kw))
    return {k: np.concatenate([p[k] for p in parts], axis=0) for k in parts[0]}


class RFIDTrackingWindowDataset(Dataset):
    """Returns full-window tensors for sequence losses."""

    FEAT_DIM = 14

    def __init__(
        self,
        table: dict[str, np.ndarray],
        window: int = 16,
        gap_s: float = 2.0,
        xy_offset: np.ndarray | None = None,
    ) -> None:
        super().__init__()
        self.window = window
        ch = table["ch_feats"]
        self.ch_feats = ch
        self.attn_invalid = table["attn_invalid"]
        self.dt = table["dt"]
        self.global_feats = table["global_feats"]
        self.xy = table["xy"]
        self.ts = table["ts"]
        self.valid_xy = table["valid_xy"].astype(bool)
        self.offset = xy_offset if xy_offset is not None else np.zeros(2, dtype=np.float32)

        self.indices: list[int] = []
        n = ch.shape[0]
        seg_start = 0
        for i in range(1, n):
            if (self.ts[i] - self.ts[i - 1]) > gap_s:
                self._add(seg_start, i)
                seg_start = i
        self._add(seg_start, n)

    def _add(self, a: int, b: int) -> None:
        if b - a < self.window:
            return
        inv = self.attn_invalid
        vxy = self.valid_xy
        for end in range(a + self.window - 1, b):
            if not vxy[end]:
                continue
            start = end - self.window + 1
            if not np.all(vxy[start : end + 1]):
                continue
            if not np.all(np.any(~inv[start : end + 1], axis=1)):
                continue
            dte = self.ts[end] - self.ts[end - 1]
            if not np.isfinite(dte) or dte <= 0:
                continue
            self.indices.append(end)

        if self.indices:
            self.indices.sort(key=lambda end: float(self.ts[end]))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        end = self.indices[idx]
        start = end - self.window + 1
        sl = slice(start, end + 1)
        ch = torch.from_numpy(self.ch_feats[sl])
        inv = torch.from_numpy(self.attn_invalid[sl])
        dt = torch.from_numpy(self.dt[sl])
        gf = torch.from_numpy(self.global_feats[sl])
        xy = torch.from_numpy(self.xy[sl] - self.offset)
        ts = torch.from_numpy(self.ts[sl].astype(np.float32))
        return {"ch_feats": ch, "attn_invalid": inv, "dt": dt, "global_feats": gf, "xy": xy, "ts": ts}


def load_csv_paths(data_dir: Path) -> list[Path]:
    return sorted(data_dir.glob("*.csv"))
