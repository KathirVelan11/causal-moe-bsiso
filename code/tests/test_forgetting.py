import pytest

from causal_moe.drift.forgetting import RegimeObservation, average_forgetting, backward_transfer


def test_forgetting_detects_regression_after_best():
    obs = [
        RegimeObservation("A", 1, 0.10),
        RegimeObservation("A", 2, 0.05),
        RegimeObservation("A", 3, 0.08),
    ]
    result = average_forgetting(obs)
    assert result.n_regimes_revisited == 1
    assert result.per_regime_forgetting["A"] == pytest.approx(0.03)
    assert result.average_forgetting == pytest.approx(0.03)


def test_forgetting_monotonic_improvement_is_nonpositive():
    obs = [RegimeObservation("A", 1, 0.10), RegimeObservation("A", 2, 0.05)]
    result = average_forgetting(obs)
    assert result.average_forgetting <= 0.0


def test_forgetting_ignores_single_visit_regimes():
    obs = [RegimeObservation("A", 1, 0.10), RegimeObservation("B", 1, 0.20)]
    result = average_forgetting(obs)
    assert result.n_regimes_revisited == 0
    assert result.average_forgetting == 0.0
    assert result.per_regime_forgetting == {}


def test_forgetting_empty_input():
    result = average_forgetting([])
    assert result.average_forgetting == 0.0
    assert result.n_regimes_revisited == 0


def test_forgetting_averages_across_multiple_regimes():
    obs = [
        RegimeObservation("A", 1, 0.10),
        RegimeObservation("A", 2, 0.20),  # forgetting = 0.10
        RegimeObservation("B", 1, 0.30),
        RegimeObservation("B", 2, 0.50),  # forgetting = 0.20
    ]
    result = average_forgetting(obs)
    assert result.n_regimes_revisited == 2
    assert result.average_forgetting == pytest.approx(0.15)


def test_forgetting_uses_generation_order_not_list_order():
    # Out-of-order input; generation field determines chronology.
    obs = [
        RegimeObservation("A", 3, 0.08),
        RegimeObservation("A", 1, 0.10),
        RegimeObservation("A", 2, 0.05),
    ]
    result = average_forgetting(obs)
    assert result.per_regime_forgetting["A"] == pytest.approx(0.03)


def test_backward_transfer_none_with_insufficient_regimes():
    obs = [RegimeObservation("A", 1, 0.10), RegimeObservation("A", 2, 0.05)]
    assert backward_transfer(obs) is None


def test_backward_transfer_basic():
    obs = [
        RegimeObservation("A", 1, 0.10),
        RegimeObservation("A", 3, 0.20),
        RegimeObservation("B", 2, 0.30),
        RegimeObservation("B", 3, 0.15),
    ]
    # A: final(gen3)=0.20 - first(gen1)=0.10 = +0.10 (worse)
    # B: final(gen3)=0.15 - first(gen2)=0.30 = -0.15 (better)
    bwt = backward_transfer(obs)
    assert bwt == pytest.approx((0.10 + (-0.15)) / 2)
