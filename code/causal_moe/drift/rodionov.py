"""STARS / Rodionov (2004) sequential regime-shift detection -- Channel 2's
underlying test (SS4.4).

Reference: Rodionov, S.N. (2004), "A sequential algorithm for testing climate
regime shifts", Geophysical Research Letters 31, L09204. Verified working
Python reference identified in SS7: github.com/TrevorJA/Rodionov_regime_shifts
(rodionov.py). Reimplemented here from the algorithm description rather than
vendored, so the project has no extra runtime dependency for a ~100-line
method (river is already a dependency for Channel 1; a single-file
reimplementation of a well-specified published algorithm is cheaper than
another package).

Algorithm, as published:
  1. Fix a cut-off length L (expected minimum regime length) and a
     significance level p. Compute the "diff" threshold: the difference
     between two regime means that would be statistically significant at
     level p for samples of size L, using a two-tailed t-test with
     2L-2 degrees of freedom and a pooled variance estimated from the
     series' own L-length running variance.
  2. Initialize regime 1's mean from the first L points.
  3. Step through the series. For each new point, test whether it lies
     beyond mean_current +/- diff. If not, it belongs to the current
     regime: update the running mean and continue.
  4. If it does, mark it a CANDIDATE shift point and compute the Regime
     Shift Index (RSI): a running sum of normalized anomalies of the next
     L points relative to the hypothesized new regime level. If RSI ever
     goes negative within those L points, the candidate is rejected (the
     series came back), and the scan resumes from the point after the
     candidate. If RSI stays positive across all L points, the shift is
     CONFIRMED at the candidate index.

This is deliberately the plain published version -- no smoothing, no
Huber weighting, no prewhitening (Rodionov's later papers add optional
red-noise prewhitening; BSISO anomaly fields are already climatology- and
harmonic-removed per SS3, so the strongest seasonal autocorrelation source
is already gone).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import stats


@dataclass
class RegimeShift:
    index: int
    rsi: float
    mean_before: float
    mean_after: float


@dataclass
class RodionovResult:
    shifts: list[RegimeShift] = field(default_factory=list)

    @property
    def shift_indices(self) -> list[int]:
        return [s.index for s in self.shifts]


def _diff_threshold(series: np.ndarray, cut_off_length: int, p: float) -> float:
    """The minimum difference between two regime means that counts as
    significant at level `p`, per Rodionov (2004) eq. 1: based on a
    two-tailed t-test with 2L-2 dof and the average L-window variance."""
    L = cut_off_length
    if series.shape[0] < L:
        raise ValueError(f"series shorter ({series.shape[0]}) than cut_off_length ({L})")

    t_crit = stats.t.ppf(1.0 - p / 2.0, df=2 * L - 2)
    # Average variance across all L-length windows (Rodionov uses the mean
    # of running variances as the pooled variance estimate).
    n_windows = series.shape[0] - L + 1
    windows = np.lib.stride_tricks.sliding_window_view(series, L)  # (n_windows, L)
    avg_var = float(windows.var(axis=1, ddof=1).mean()) if n_windows > 0 else float(series.var(ddof=1))
    return float(t_crit * np.sqrt(2.0 * avg_var / L))


def detect_regime_shifts(
    series: np.ndarray,
    cut_off_length: int = 120,
    p: float = 0.05,
) -> RodionovResult:
    """series: (T,) float, the signal to test for regime shifts.
    cut_off_length: L, the expected minimum regime length in samples. For
        daily BSISO data, the default 120 (~4 months) is a deliberate
        choice: shorter than the multi-year El Nino window SS8 targets, but
        long enough not to fire on individual BSISO cycles (30-60 days,
        SS8 "Lead time N").
    p: significance level for the shift test.
    """
    series = np.asarray(series, dtype=np.float64).reshape(-1)
    T = series.shape[0]
    L = cut_off_length
    if T < 2 * L:
        return RodionovResult()

    diff = _diff_threshold(series, L, p)
    sigma = float(np.sqrt(np.mean(np.var(np.lib.stride_tricks.sliding_window_view(series, L), axis=1, ddof=1))))
    if sigma <= 0:
        return RodionovResult()

    result = RodionovResult()

    regime_start = 0
    current_mean = float(series[:L].mean())
    n_in_regime = L

    i = L
    while i < T:
        value = series[i]
        if abs(value - current_mean) <= diff:
            # same regime -- update running mean
            n_in_regime += 1
            current_mean += (value - current_mean) / n_in_regime
            i += 1
            continue

        # candidate shift at i: test with the RSI over the next L points
        direction = 1.0 if value > current_mean else -1.0
        level = current_mean + direction * diff
        rsi = 0.0
        confirmed = True
        horizon = min(i + L, T)
        for j in range(i, horizon):
            rsi += direction * (series[j] - level) / (L * sigma)
            if rsi < 0:
                confirmed = False
                break

        if confirmed:
            # B-audit fix, 2026-09-23: the per-point loop above already
            # rejects (confirmed=False) the instant rsi goes negative, so
            # by the time the loop exits with confirmed=True, rsi is
            # guaranteed >= 0 -- the algorithm's actual accept condition
            # (Rodionov 2004: reject iff RSI ever goes negative). The old
            # extra `and rsi > 0` here silently dropped the boundary case
            # rsi == 0.0 exactly (e.g. a flat post-shift window), which
            # `confirmed` already says should be accepted.
            mean_before = float(series[regime_start:i].mean())
            mean_after = float(series[i:horizon].mean())
            result.shifts.append(
                RegimeShift(index=i, rsi=float(rsi), mean_before=mean_before, mean_after=mean_after)
            )
            regime_start = i
            n_in_regime = horizon - i
            current_mean = mean_after
            i = horizon
        else:
            # rejected candidate: it belongs to the current regime after all
            n_in_regime += 1
            current_mean += (value - current_mean) / n_in_regime
            i += 1

    return result
