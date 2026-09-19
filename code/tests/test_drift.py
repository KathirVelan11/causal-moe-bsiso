import numpy as np
import pytest

from causal_moe.drift.channels import (
    causal_signature_distance_series,
    center_signatures,
    cosine_similarity,
    evaluate_detector,
    fuse_channels,
    run_channel1_error_adwin,
    run_channel2_causal_stars,
    ChannelResult,
)
from causal_moe.drift.rodionov import detect_regime_shifts


# --- Rodionov / STARS -------------------------------------------------------


def test_rodionov_detects_obvious_step_change():
    """A clean mean shift well past the significance threshold must be
    found, and found near where it was injected."""
    rng = np.random.default_rng(0)
    T = 800
    shift_at = 400
    series = np.concatenate([
        rng.normal(0.0, 1.0, shift_at),
        rng.normal(5.0, 1.0, T - shift_at),
    ])

    result = detect_regime_shifts(series, cut_off_length=100, p=0.05)

    assert len(result.shifts) >= 1
    nearest = min(result.shift_indices, key=lambda i: abs(i - shift_at))
    assert abs(nearest - shift_at) <= 50


def test_rodionov_quiet_on_stationary_noise():
    """Pure stationary noise must not produce a torrent of shifts (the
    false-alarm side of SS6's detector-reliability metric)."""
    rng = np.random.default_rng(1)
    series = rng.normal(0.0, 1.0, 1500)

    result = detect_regime_shifts(series, cut_off_length=120, p=0.01)

    assert len(result.shifts) <= 2


def test_rodionov_returns_empty_for_too_short_series():
    series = np.arange(50, dtype=float)
    result = detect_regime_shifts(series, cut_off_length=120)
    assert result.shifts == []


def test_rodionov_records_regime_means_around_shift():
    rng = np.random.default_rng(2)
    series = np.concatenate([rng.normal(0.0, 0.5, 300), rng.normal(4.0, 0.5, 300)])

    result = detect_regime_shifts(series, cut_off_length=100, p=0.05)

    assert result.shifts
    first = result.shifts[0]
    assert first.mean_after > first.mean_before
    assert first.rsi > 0


# --- causal signature -> scalar distance ------------------------------------


def test_cosine_similarity_basics():
    a = np.array([1.0, 0.0])
    assert cosine_similarity(a, a) == pytest.approx(1.0)
    assert cosine_similarity(a, np.array([0.0, 1.0])) == pytest.approx(0.0)
    assert cosine_similarity(a, np.zeros(2)) == 0.0  # degenerate guard


def test_signature_distance_zero_when_signature_constant():
    signatures = np.tile(np.array([0.2, 0.8, 0.5]), (500, 1))
    dist = causal_signature_distance_series(signatures, baseline_window=100)
    assert np.allclose(dist, 0.0, atol=1e-9)


def test_centering_rescues_similarity_when_scores_share_a_large_offset():
    """Regression test for the step-6 degenerate-archive bug (2026-09-19).

    Real generator scores all sit near the same value (measured: 8 edges
    all at mean ~0.463). Raw cosine similarity between ANY two such
    signatures saturates near 1.0 because the shared DC offset dominates
    the angle -- which made SS4.5's archive match everything to everything
    at similarity 1.000. Centering must restore real discrimination."""
    baseline = np.full(8, 0.463)
    sig_a = baseline + np.array([0.10, -0.10, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    sig_b = baseline + np.array([-0.10, 0.10, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    # raw: the two opposite deviation patterns look almost identical
    assert cosine_similarity(sig_a, sig_b) > 0.9

    stacked = np.vstack([np.tile(baseline, (10, 1)), sig_a, sig_b])
    centered = center_signatures(stacked, baseline_window=10)
    # centered: the SAME pair is now clearly distinguished (opposite signs)
    assert cosine_similarity(centered[-2], centered[-1]) < -0.9


def test_center_signatures_subtracts_baseline_mean():
    signatures = np.vstack([np.full((100, 3), 0.5), np.full((50, 3), 0.8)])
    centered = center_signatures(signatures, baseline_window=100)
    assert np.allclose(centered[:100], 0.0)
    assert np.allclose(centered[100:], 0.3)


def test_signature_distance_rises_when_causal_set_rotates():
    """The point of Channel 2: if WHICH edges score high changes, the
    distance series must move, even though nothing about magnitude changed."""
    T = 600
    signatures = np.zeros((T, 4))
    signatures[:300] = np.array([0.9, 0.1, 0.1, 0.1])
    signatures[300:] = np.array([0.1, 0.9, 0.1, 0.1])  # driver moved

    dist = causal_signature_distance_series(signatures, baseline_window=200)

    assert dist[:300].mean() < 0.05
    assert dist[300:].mean() > 0.3


# --- channels ---------------------------------------------------------------


def test_channel1_adwin_fires_on_error_jump():
    rng = np.random.default_rng(3)
    errors = np.concatenate([rng.normal(0.1, 0.01, 1000), rng.normal(0.9, 0.01, 1000)])

    result = run_channel1_error_adwin(errors)

    assert result.alarm_indices
    assert any(1000 <= i <= 1200 for i in result.alarm_indices)


def test_channel2_stars_fires_on_signature_shift():
    T = 900
    signatures = np.zeros((T, 3))
    signatures[:450] = np.array([0.9, 0.1, 0.1])
    signatures[450:] = np.array([0.1, 0.9, 0.1])
    dist = causal_signature_distance_series(signatures, baseline_window=200)

    result = run_channel2_causal_stars(dist, cut_off_length=100)

    assert result.alarm_indices


# --- fusion gate ------------------------------------------------------------


def test_fuse_or_is_union():
    ch1 = ChannelResult("c1", [10, 50])
    ch2 = ChannelResult("c2", [50, 90])

    fused = fuse_channels(ch1, ch2, T=100, rule="OR")

    assert fused.alarm_indices == [10, 50, 90]


def test_fuse_and_requires_both_within_tolerance():
    ch1 = ChannelResult("c1", [10, 300])
    ch2 = ChannelResult("c2", [40, 800])  # 40 is within 90d of 10; 800 is not near 300

    fused = fuse_channels(ch1, ch2, T=1000, rule="AND", and_tolerance_days=90)

    assert fused.alarm_indices == [40]  # logged at the LATER of the agreeing pair


def test_fuse_and_empty_when_channels_never_agree():
    ch1 = ChannelResult("c1", [10])
    ch2 = ChannelResult("c2", [900])

    fused = fuse_channels(ch1, ch2, T=1000, rule="AND", and_tolerance_days=90)

    assert fused.alarm_indices == []


def test_fuse_rejects_unknown_rule():
    ch1 = ChannelResult("c1", [1])
    ch2 = ChannelResult("c2", [2])
    with pytest.raises(ValueError):
        fuse_channels(ch1, ch2, T=10, rule="XOR")


# --- evaluation metrics -----------------------------------------------------


def test_evaluate_detector_latency_and_false_alarms():
    T = 1000
    window_mask = np.zeros(T, dtype=bool)
    window_mask[400:600] = True

    # one alarm inside the window (at 450 -> latency 50), two outside
    ev = evaluate_detector([100, 450, 800], window_mask, rule="OR")

    assert ev.detected is True
    assert ev.latency_days == 50
    assert ev.false_alarms_outside_window == 2
    assert ev.false_alarm_rate_per_year > 0


def test_evaluate_detector_reports_miss():
    T = 1000
    window_mask = np.zeros(T, dtype=bool)
    window_mask[400:600] = True

    ev = evaluate_detector([100, 800], window_mask, rule="AND")

    assert ev.detected is False
    assert ev.latency_days is None
