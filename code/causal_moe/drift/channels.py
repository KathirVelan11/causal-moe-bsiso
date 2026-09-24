"""Two-channel drift detection + fusion gate (SS4.4).

Channel 1 -- ERROR drift: ADWIN (river.drift.ADWIN, SS7 "don't implement
  from scratch") over the expert's rolling forecast error. Catches "this
  expert has stopped being accurate."

Channel 2 -- CAUSAL-SET drift: STARS/Rodionov (rodionov.py) over a scalar
  summary of the splitter's causal signature through time. Catches "which
  places drive this place has changed" even while error stays flat -- the
  whole reason SS4.4 has two channels ("error alone misses slow, real
  drift; causal-set change alone can wobble from ordinary noise").

Fusion gate -- OR vs AND, left by SS4.4 as a REPORTABLE EXPERIMENT ("run
  both, report detection latency and false-alarm rate for each"). Both are
  implemented here and both are reported; neither is hardcoded as "the"
  rule.

Causal-signature representation: the per-edge score vector the rationale
generator produces for a given timestep, which SS4.5 already fixes as the
"simple importance-vector representation" used for archive matching with
cosine similarity. Channel 2 needs a SCALAR series to run a change-point
test on, so we reduce the signature series to scalar via cosine distance
against a fixed reference signature (the mean signature over a baseline
window). That keeps one consistent similarity mechanism across SS4.4 and
SS4.5, exactly as SS4.5 requires ("same similarity technique as SS4.4's
Channel 2 ... not two separate formulas").
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from river.drift import ADWIN

from causal_moe.drift.rodionov import detect_regime_shifts


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """SS4.5's default similarity: cosine over importance vectors."""
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def center_signatures(signatures: np.ndarray, baseline_window: int = 365) -> np.ndarray:
    """Subtract the per-edge baseline mean from every signature.

    WHY THIS IS REQUIRED (bug found 2026-09-19, step 6): raw generator
    edge scores on real data all sit near the SAME value (measured: all 8
    of place 22's edges had mean ~0.463, per-edge std over time ~0.11).
    Raw signature vectors are therefore nearly parallel -- their shared DC
    offset dominates, and cosine similarity, which measures ANGLE, saturates
    at ~1.0 for every pair (measured cos(1979, 2022) = 0.99997). That made
    both Channel 2's distance series and SS4.5's archive matching
    degenerate: every archived expert matched every query at similarity
    1.000, so the lifecycle "reactivated" 48/48 times purely as an artifact.

    Centering removes the uninformative common offset so similarity reflects
    the DEVIATION PATTERN -- i.e. which edges are scoring above/below their
    own baseline, which is what "the causal set has changed" actually means.
    """
    T = signatures.shape[0]
    w = min(baseline_window, T)
    baseline_mean = signatures[:w].mean(axis=0, keepdims=True)
    return signatures - baseline_mean


def causal_signature_distance_series(
    signatures: np.ndarray,
    baseline_window: int = 365,
    center: bool = True,
) -> np.ndarray:
    """signatures: (T, n_edges) -- the generator's edge-score vector per
        timestep (the causal signature, SS4.5).
    baseline_window: number of leading timesteps whose MEAN signature is
        the fixed reference all later signatures are compared against.
    center: subtract the per-edge baseline mean first (see
        center_signatures -- required on real data, where raw scores share
        a large common offset that otherwise saturates cosine similarity).
    Returns: (T,) cosine DISTANCE (1 - cosine similarity) to the baseline
        reference -- rises when the causal structure moves away from its
        original configuration, which is what Channel 2 tests for shifts in.
    """
    T = signatures.shape[0]
    w = min(baseline_window, T)
    work = center_signatures(signatures, baseline_window) if center else signatures
    reference = work[:w].mean(axis=0)
    # vectorized cosine against a single reference (CPU-only constraint:
    # no per-timestep Python loop)
    ref_norm = float(np.linalg.norm(reference))
    if ref_norm < 1e-12:
        # Centered baseline reference is ~0 by construction when the
        # baseline window itself defined the centering; fall back to
        # per-timestep deviation MAGNITUDE, which still carries "how far
        # from baseline is the causal set now" without needing an angle.
        return np.linalg.norm(work, axis=1).astype(np.float64)
    sig_norms = np.linalg.norm(work, axis=1)
    sig_norms = np.where(sig_norms < 1e-12, np.nan, sig_norms)
    cos = (work @ reference) / (sig_norms * ref_norm)
    cos = np.nan_to_num(cos, nan=0.0)
    return (1.0 - cos).astype(np.float64)


@dataclass
class ChannelResult:
    name: str
    alarm_indices: list[int] = field(default_factory=list)

    def alarm_mask(self, T: int) -> np.ndarray:
        mask = np.zeros(T, dtype=bool)
        for i in self.alarm_indices:
            if 0 <= i < T:
                mask[i] = True
        return mask


def run_channel1_error_adwin(errors: np.ndarray, delta: float = 0.002) -> ChannelResult:
    """errors: (T,) per-timestep forecast error (absolute or squared).
    delta: ADWIN's confidence parameter (river default 0.002); smaller =
        more conservative, fewer false alarms.
    """
    detector = ADWIN(delta=delta)
    alarms: list[int] = []
    for i, e in enumerate(errors):
        detector.update(float(e))
        if detector.drift_detected:
            alarms.append(i)
    return ChannelResult(name="channel1_error_adwin", alarm_indices=alarms)


def run_channel2_causal_stars(
    signature_distance: np.ndarray,
    cut_off_length: int = 120,
    p: float = 0.05,
) -> ChannelResult:
    """signature_distance: (T,) output of causal_signature_distance_series."""
    result = detect_regime_shifts(signature_distance, cut_off_length=cut_off_length, p=p)
    return ChannelResult(name="channel2_causal_stars", alarm_indices=result.shift_indices)


@dataclass
class FusionResult:
    rule: str  # "OR" or "AND"
    alarm_indices: list[int]


def fuse_channels(
    ch1: ChannelResult,
    ch2: ChannelResult,
    T: int,
    rule: str,
    and_tolerance_days: int = 90,
) -> FusionResult:
    """SS4.4's fusion gate. Both rules implemented; SS4.4 explicitly wants
    BOTH reported rather than one picked.

    OR  -- either channel alone triggers (faster, more false alarms).
    AND -- both must agree. Exact same-day agreement is far too strict for
        two detectors with different latencies (ADWIN reacts within its own
        adaptive window; STARS confirms only after L points), so AND means
        "both fired within `and_tolerance_days` of each other," with the
        alarm logged at the LATER of the two (the point at which both have
        actually agreed).
    """
    rule = rule.upper()
    if rule == "OR":
        alarms = sorted(set(ch1.alarm_indices) | set(ch2.alarm_indices))
        return FusionResult(rule="OR", alarm_indices=alarms)
    if rule == "AND":
        alarms = []
        a2 = np.array(ch2.alarm_indices, dtype=np.int64)
        for i1 in ch1.alarm_indices:
            if a2.size == 0:
                break
            near = a2[np.abs(a2 - i1) <= and_tolerance_days]
            if near.size > 0:
                alarms.append(int(max(i1, near.min())))
        return FusionResult(rule="AND", alarm_indices=sorted(set(alarms)))
    raise ValueError(f"unknown fusion rule {rule!r}, expected 'OR' or 'AND'")


@dataclass
class DetectorEvaluation:
    rule: str
    n_alarms: int
    detected: bool
    latency_days: int | None  # days from window start to first in-window alarm
    false_alarms_outside_window: int
    false_alarm_rate_per_year: float


def evaluate_detector(
    alarm_indices: list[int],
    window_mask: np.ndarray,
    rule: str,
    samples_per_year: float = 365.25,
    known_event_mask: np.ndarray | None = None,
) -> DetectorEvaluation:
    """SS6's "Detector reliability -- latency + false-alarm rate, per channel
    AND for the fused rule (both OR and AND, reported)".

    window_mask: (T,) bool, the deliberate-shift window the detector is
        SUPPOSED to fire inside (SS8: the 1997-98 El Nino span, or the
        2013-2022 gradual-drift span).
    known_event_mask: (T,) bool, optional (B13 audit fix, 2026-09-23). Union
        of ALL known real climate events (every major El Nino, plus the
        project's own gradual-drift window) -- alarms inside this mask but
        outside `window_mask` are NOT counted as false alarms, since they
        may be genuine detections of a DIFFERENT real event than the one
        `window_mask` currently tests. Without this, evaluating one window
        at a time (as run_step5_drift.py does) double-penalizes a detector
        that correctly fires during e.g. 1982-83 while being scored against
        the 1997-98 window alone. Defaults to `window_mask` itself
        (original behaviour: only this one window's events are excluded).
    """
    T = window_mask.shape[0]
    if known_event_mask is None:
        known_event_mask = window_mask
    in_window = [i for i in alarm_indices if 0 <= i < T and window_mask[i]]
    outside = [i for i in alarm_indices if 0 <= i < T and not known_event_mask[i]]

    window_start = int(np.argmax(window_mask)) if window_mask.any() else 0
    latency = (min(in_window) - window_start) if in_window else None

    n_outside_years = float((~known_event_mask).sum()) / samples_per_year
    far = (len(outside) / n_outside_years) if n_outside_years > 0 else 0.0

    return DetectorEvaluation(
        rule=rule,
        n_alarms=len(alarm_indices),
        detected=bool(in_window),
        latency_days=int(latency) if latency is not None else None,
        false_alarms_outside_window=len(outside),
        false_alarm_rate_per_year=float(far),
    )
