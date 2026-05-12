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


def discover_grid_layout(pairs: list[tuple[str, str]]) -> dict:
    """Map ``pairs`` into a (n_antennas x n_tags) grid for CNN input.

    Rows     : sorted antenna names (e.g. ant1 -> row 0, ant4 -> row 1)
    Columns  : sorted tag ids
    Returns a dict with::
        cell_to_channel : [H, W] int, channel index in ``pairs`` (-1 if absent)
        local_xy        : [H, W, 2] float, A4 corner coords for 4 tags.
                          Camera frame is left-right flipped w.r.t. the
                          physical tag layout, so sorted tag ids map to
                          (TR, TL, BR, BL) for cols 0..3.
        ant_order       : list[str]
        tag_order       : list[str]
    """
    parsed: list[tuple[str, str]] = []
    ants: list[str] = []
    tags: list[str] = []
    for ph, _rs in pairs:
        parts = ph.split("_")
        if len(parts) < 3:
            raise ValueError(f"Unexpected phase column: {ph!r}")
        ant = parts[0]
        tag = parts[1]
        parsed.append((ant, tag))
        ants.append(ant)
        tags.append(tag)
    ant_order = sorted(set(ants))
    tag_order = sorted(set(tags))
    H, W = len(ant_order), len(tag_order)
    cell_to_channel = np.full((H, W), -1, dtype=np.int64)
    for idx, (ant, tag) in enumerate(parsed):
        r = ant_order.index(ant)
        c = tag_order.index(tag)
        if cell_to_channel[r, c] != -1:
            raise ValueError(f"Duplicate cell ({ant},{tag}) in pairs")
        cell_to_channel[r, c] = idx

    if W == 4:
        # A4 paper ~ 297x210 (sqrt(2):1). Camera is left-right flipped vs the
        # physical layout, so sorted tag ids map to TR, TL, BR, BL.
        corners = np.array(
            [
                [+1.414, +1.0],
                [-1.414, +1.0],
                [+1.414, -1.0],
                [-1.414, -1.0],
            ],
            dtype=np.float32,
        )
        local_xy_w = corners
    else:
        local_xy_w = np.zeros((W, 2), dtype=np.float32)
        for c in range(W):
            local_xy_w[c, 0] = (c / max(W - 1, 1)) * 2.0 - 1.0

    local_xy = np.broadcast_to(local_xy_w[None, :, :], (H, W, 2)).copy()

    return {
        "cell_to_channel": cell_to_channel,
        "local_xy": local_xy,
        "H": int(H),
        "W": int(W),
        "ant_order": ant_order,
        "tag_order": tag_order,
    }


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


def unwrap_phase_algo(
    raw: np.ndarray,
    mask: np.ndarray,
    phase_period: float = 2048.0,
) -> np.ndarray:
    """Algorithm-style integer-k unwrap (mirrors algorithm/features/phase.py).

    raw  : [N] phase samples in raw period units; NaN allowed.
    mask : [N] bool, True where the sample is real (not imputed/missing).

    Returns absolute cumulative unwrapped phase in raw units (can grow unbounded).
    Reference for choosing the integer wrap count:
      - if previous sample was real, use unwrapped[i-1]
      - else, use the unwrapped value at the most recent real index (no drift across gaps)
    """
    n = int(raw.shape[0])
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    s = pd.Series(raw)
    filled = s.ffill().bfill().to_numpy(dtype=np.float64)
    if not np.any(np.isfinite(filled)):
        return np.zeros(n, dtype=np.float64)
    filled = np.where(np.isfinite(filled), filled, 0.0)

    unwrapped = np.zeros(n, dtype=np.float64)
    unwrapped[0] = float(filled[0])
    last_valid_idx = 0
    period = float(phase_period)
    for i in range(1, n):
        p = float(filled[i])
        if bool(mask[i - 1]):
            ref = unwrapped[i - 1]
            last_valid_idx = i - 1
        else:
            ref = unwrapped[last_valid_idx]
        k = np.round((ref - p) / period)
        unwrapped[i] = p + k * period
    return unwrapped


def recover_phase_ridge(
    unwrapped: np.ndarray,
    masks: np.ndarray,
    rssi_fill: np.ndarray,
) -> np.ndarray:
    """Per-channel Ridge regression to overwrite missing phase positions.

    unwrapped : [N, C] absolute unwrapped phase (no NaN; ffilled internally).
    masks     : [N, C] in {0,1}; 1 = real observation.
    rssi_fill : [N, C] forward-filled RSSI (no NaN expected).

    For each target channel t, regress unwrapped[:,t] on
      (other channels' unwrapped) + (other channels' rssi)
    using only positions where masks[:,t]==1, then overwrite predictions at
    positions where masks[:,t]==0.
    """
    try:
        from sklearn.linear_model import Ridge
    except ImportError as e:  # pragma: no cover
        raise SystemExit("recover=True requires scikit-learn") from e

    n, c = unwrapped.shape
    refined = unwrapped.copy()
    for tgt in range(c):
        others = [j for j in range(c) if j != tgt]
        if not others:
            continue
        X = np.concatenate(
            [unwrapped[:, others], rssi_fill[:, others]], axis=1
        ).astype(np.float64)
        y = unwrapped[:, tgt].astype(np.float64).copy()
        valid = masks[:, tgt] > 0.5
        finite_X = np.all(np.isfinite(X), axis=1)
        train_mask = valid & finite_X
        if int(train_mask.sum()) < 10:
            continue
        model = Ridge()
        model.fit(X[train_mask], y[train_mask])
        missing = (~valid) & finite_X
        if int(missing.sum()) > 0:
            y[missing] = model.predict(X[missing])
        refined[:, tgt] = y
    return refined


def build_enriched_table(
    df: pd.DataFrame,
    pairs: list[tuple[str, str]],
    rssi_mean: np.ndarray,
    rssi_std: np.ndarray,
    phase_period: float = 2048.0,
    conf_lambda: float = 0.35,
    max_tslv_clip_s: float = 30.0,
    unwrap_method: str = "arctan2",
    recover: bool = False,
    pdoa_pair_indices: np.ndarray | None = None,
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

    method = (unwrap_method or "arctan2").lower()
    if method not in ("arctan2", "algo"):
        raise ValueError(f"unwrap_method must be 'arctan2' or 'algo', got {unwrap_method!r}")

    if method == "algo":
        unwrapped_abs = np.zeros((n, c), dtype=np.float64)
        for j in range(c):
            unwrapped_abs[:, j] = unwrap_phase_algo(
                phase_obs[:, j], m_p[:, j] > 0.5, phase_period=float(phase_period)
            )
        if recover:
            unwrapped_abs = recover_phase_ridge(
                unwrapped_abs, m_p.astype(np.float32), rssi_fill.astype(np.float64)
            )
        phase_for_sincos = np.mod(unwrapped_abs, float(phase_period))
    else:
        phase_for_sincos = np.mod(phase_fill.astype(np.float64), float(phase_period))

    ang = (2.0 * math.pi) * (phase_for_sincos / float(phase_period))
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

    rad_per_unit = (2.0 * math.pi) / float(phase_period)

    for i in range(1, n):
        dti = float(dt[i])
        if dti <= 0:
            continue
        if method == "algo":
            # Algorithm-style unwrap is already continuous: differentiate directly.
            dth = (unwrapped_abs[i] - unwrapped_abs[i - 1]) * rad_per_unit
        else:
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

    # Optional PDoA features (sin, cos of antenna-pair phase difference per tag).
    # `pdoa_pair_indices`: int array of shape [n_pairs, 2], each row = (ch_a, ch_b).
    # For each pair, we compute d = wrap(ang_a - ang_b), then store
    #   ch_a slot: ( +sin(d),  cos(d) )
    #   ch_b slot: ( -sin(d),  cos(d) )   (sin antisymmetric, cos symmetric)
    # Channels not listed in any pair stay zero in both slots.
    if pdoa_pair_indices is not None:
        idx = np.asarray(pdoa_pair_indices, dtype=np.int64)
        if idx.ndim != 2 or idx.shape[-1] != 2:
            raise ValueError(
                f"pdoa_pair_indices must have shape [n_pairs, 2], got {idx.shape}"
            )
        pdoa_sin = np.zeros((n, c), dtype=np.float32)
        pdoa_cos = np.zeros((n, c), dtype=np.float32)
        ang_all = (2.0 * math.pi) * (phase_for_sincos / float(phase_period))
        for ch_a, ch_b in idx:
            d = _wrap_angle(ang_all[:, int(ch_a)] - ang_all[:, int(ch_b)])
            sd = np.sin(d).astype(np.float32)
            cd = np.cos(d).astype(np.float32)
            pdoa_sin[:, int(ch_a)] = +sd
            pdoa_sin[:, int(ch_b)] = -sd
            pdoa_cos[:, int(ch_a)] = cd
            pdoa_cos[:, int(ch_b)] = cd
        ch_feats = np.concatenate(
            [ch_feats, pdoa_sin[..., None], pdoa_cos[..., None]],
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
