"""BSISO index and per-year active-season mask, computed from this
project's own OLR and u850 fields (Phase 1 spec SS2: "define the season by
activity ... with a margin of a few weeks, not by a fixed calendar date").

Index: Lee et al. (2013) BSISO1 recipe adapted to our grid --
  - domain 10S-30N, 40E-160E (Lee uses 10S-40N; our grid stops at 30N),
  - daily OLR and u850 anomalies with the trailing 120-day mean removed
    (Lee's step for isolating intraseasonal variability without a
    non-causal band-pass filter),
  - sqrt(cos(lat)) area weighting, each field divided by its domain-mean
    std so OLR and u850 contribute equally,
  - multivariate EOF fitted on May-October days of the DISCOVERY years only
    (<= 2012) so held-out years never shape the index,
  - BSISO1 = (PC1, PC2), each normalized to unit std over the fit days;
    amplitude = sqrt(PC1^2 + PC2^2).

Season (summer BSISO, run first per SS2): per year, inside an Apr 1 - Nov 30
search window, the 31-day centred running mean amplitude must exceed
`threshold` for a run of at least `min_run` consecutive days. Onset is the
start of the first such run, end is the end of the last; both are padded by
`margin_days`. Years with no qualifying run fall back to May 1 - Oct 31.

The mask marks TARGET days only. Source history is never removed (SS2: a
lag-45 parent of a 1 June target reads mid-April).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from causal_moe.data.raw import FIELDS, BSISORawData

DOMAIN_LAT = (-10.0, 30.0)
DOMAIN_LON = (40.0, 160.0)
DISCOVERY_END = "2012-12-31"


@dataclass
class BSISOSeason:
    time: np.ndarray  # (T,) datetime64[D]
    pc: np.ndarray  # (T, 2) normalized BSISO1 PCs
    amplitude: np.ndarray  # (T,)
    in_season: np.ndarray  # (T,) bool -- target-day mask
    onset: dict  # year -> datetime64[D] (padded)
    end: dict  # year -> datetime64[D] (padded)
    explained_variance: np.ndarray  # (2,)
    fallback_years: list


def _remove_trailing_mean(x: np.ndarray, window: int = 120) -> np.ndarray:
    """x - mean(x[t-window:t]) along axis 0 (first day left as is)."""
    csum = np.concatenate([np.zeros((1,) + x.shape[1:], dtype=np.float64), np.cumsum(x, axis=0, dtype=np.float64)])
    t = np.arange(x.shape[0])
    lo = np.maximum(t - window, 0)
    n = np.maximum(t - lo, 1)
    mean = (csum[t] - csum[lo]) / n.reshape((-1,) + (1,) * (x.ndim - 1))
    mean[0] = 0.0
    return (x - mean).astype(np.float32)


def compute_bsiso_index(raw: BSISORawData, fit_end: str = DISCOVERY_END) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (time[D], pcs (T, 2), explained_variance (2,))."""
    li = (raw.lat >= DOMAIN_LAT[0]) & (raw.lat <= DOMAIN_LAT[1])
    lj = (raw.lon >= DOMAIN_LON[0]) & (raw.lon <= DOMAIN_LON[1])
    weight = np.sqrt(np.cos(np.deg2rad(raw.lat[li])))[None, :, None]
    time = raw.time.astype("datetime64[D]")
    dates = pd.DatetimeIndex(time)
    fit = (time <= np.datetime64(fit_end)) & (dates.month >= 5) & (dates.month <= 10) & (np.arange(raw.n_time) >= 120)

    blocks = []
    for name in ("olr", "u850"):
        field = raw.fields[FIELDS.index(name)][:, li][:, :, lj]
        anom = _remove_trailing_mean(field) * weight
        flat = anom.reshape(raw.n_time, -1)
        blocks.append(flat / flat[fit].std())
    X = np.concatenate(blocks, axis=1)

    mu = X[fit].mean(axis=0)
    _, s, vt = np.linalg.svd(X[fit] - mu, full_matrices=False)
    pcs = (X - mu) @ vt[:2].T
    pcs = pcs / pcs[fit].std(axis=0)
    explained = (s[:2] ** 2) / (s**2).sum()
    return time, pcs.astype(np.float32), explained


def season_from_amplitude(
    time: np.ndarray,
    amplitude: np.ndarray,
    threshold: float = 1.0,
    smooth_days: int = 31,
    min_run: int = 20,
    margin_days: int = 21,
) -> tuple[np.ndarray, dict, dict, list]:
    dates = pd.DatetimeIndex(time)
    smoothed = pd.Series(amplitude).rolling(smooth_days, center=True, min_periods=1).mean().to_numpy()
    in_season = np.zeros(time.shape[0], dtype=bool)
    onset, end, fallback = {}, {}, []
    for year in np.unique(dates.year):
        win = (time >= np.datetime64(f"{year}-04-01")) & (time <= np.datetime64(f"{year}-11-30"))
        idx = np.nonzero(win)[0]
        active = smoothed[idx] > threshold
        runs, start = [], None
        for k, a in enumerate(np.append(active, False)):
            if a and start is None:
                start = k
            elif not a and start is not None:
                if k - start >= min_run:
                    runs.append((start, k - 1))
                start = None
        if runs:
            on = time[idx[runs[0][0]]] - np.timedelta64(margin_days, "D")
            off = time[idx[runs[-1][1]]] + np.timedelta64(margin_days, "D")
        else:
            on, off = np.datetime64(f"{year}-05-01"), np.datetime64(f"{year}-10-31")
            fallback.append(int(year))
        onset[int(year)], end[int(year)] = on, off
        in_season |= (time >= on) & (time <= off)
    return in_season, onset, end, fallback


def compute_bsiso_season(raw: BSISORawData, **season_kwargs) -> BSISOSeason:
    time, pcs, explained = compute_bsiso_index(raw)
    amplitude = np.sqrt((pcs**2).sum(axis=1))
    in_season, onset, end, fallback = season_from_amplitude(time, amplitude, **season_kwargs)
    return BSISOSeason(time, pcs, amplitude, in_season, onset, end, explained, fallback)


def target_day_mask(sample_dates: np.ndarray, lead_time_days: int, season_time: np.ndarray, in_season: np.ndarray) -> np.ndarray:
    """True for samples whose TARGET day (sample date + lead) is in season."""
    target = sample_dates.astype("datetime64[D]") + np.timedelta64(lead_time_days, "D")
    lookup = dict(zip(season_time.astype("datetime64[D]").tolist(), in_season.tolist()))
    return np.array([lookup.get(d, False) for d in target.tolist()], dtype=bool)
