"""Backward-transfer / average-forgetting metric (§4.5, §6 -- B12 audit fix,
2026-09-23).

SS6 lists "forgetting / backward transfer" as a metric to report for the
hibernate/reactivate lifecycle (Variant B, §9 Phase 5) vs the no-archive
ablation (Variant A), but no such metric existed anywhere in the codebase
(grep for forget/backward_transfer/bwt returned nothing) -- this module is
the fix.

Formula chosen: AVERAGE FORGETTING (Chaudhry et al. 2018, "Riemannian Walk
for Incremental Learning"; same family as Lopez-Paz & Ranzato 2017 GEM's
Backward Transfer). Average Forgetting is used here rather than the
classic BWT because it only needs, per regime, the BEST error achieved at
any earlier point vs the CURRENT error -- exactly what this project's
existing per-(regime, generation) MSE logs already contain (step 6's
`events` list, `run_step7_all_places.py`'s per-place skill, etc.), with no
need to track a full pairwise task x checkpoint accuracy matrix the way
BWT's original formula does.

Classic (accuracy-based, higher=better) forgetting for task j after
training through task t:
    f_j^t = max_{i in {1,...,t-1}} a(i,j)  -  a(t,j)
    F_t   = mean over all previously-seen tasks j of f_j^t

Adapted here for MSE (lower=better, so the "best" is a MIN not a MAX) and
for REGIMES (recurring climate states) instead of a strictly ordered task
sequence:
    forgetting(regime, current_gen) = mse(regime, current_gen)
                                     - min_{past_gen} mse(regime, past_gen)
    average_forgetting = mean over all regimes with >=2 observations of
                          forgetting(regime, most_recent_gen)

A forgetting value of 0 means the model's most recent visit to a regime is
at least as good as its best-ever visit (no forgetting); positive means
the current generation is WORSE than a past generation was on that same
regime (forgetting occurred); negative means positive backward transfer
(the current generation does even better than any past visit, e.g.
because later training generalized).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RegimeObservation:
    """One (regime, generation) accuracy record -- the unit this module's
    functions consume. `regime` identifies a recurring climate state (e.g.
    an archived expert's matched signature id, or a hand-labeled regime
    name); `generation` is a monotonically increasing index (spawn order,
    or a timestamp/day-index); `mse` is that generation's held-out error
    when active during that regime."""

    regime: str
    generation: int
    mse: float


@dataclass
class ForgettingResult:
    average_forgetting: float
    n_regimes_revisited: int
    per_regime_forgetting: dict[str, float]


def average_forgetting(observations: list[RegimeObservation]) -> ForgettingResult:
    """Computes average forgetting (MSE-delta convention, see module
    docstring) from a flat list of (regime, generation, mse) records.

    Only regimes visited 2+ times contribute (a regime seen once has no
    "past" to forget relative to, by definition -- matches the classic
    formula's requirement of >=2 tasks before forgetting is defined for a
    given task).
    """
    by_regime: dict[str, list[RegimeObservation]] = {}
    for obs in observations:
        by_regime.setdefault(obs.regime, []).append(obs)

    per_regime_forgetting: dict[str, float] = {}
    for regime, obs_list in by_regime.items():
        if len(obs_list) < 2:
            continue
        obs_sorted = sorted(obs_list, key=lambda o: o.generation)
        *past, most_recent = obs_sorted
        best_past_mse = min(o.mse for o in past)
        per_regime_forgetting[regime] = most_recent.mse - best_past_mse

    if not per_regime_forgetting:
        return ForgettingResult(average_forgetting=0.0, n_regimes_revisited=0, per_regime_forgetting={})

    avg = sum(per_regime_forgetting.values()) / len(per_regime_forgetting)
    return ForgettingResult(
        average_forgetting=avg,
        n_regimes_revisited=len(per_regime_forgetting),
        per_regime_forgetting=per_regime_forgetting,
    )


def backward_transfer(observations: list[RegimeObservation]) -> float | None:
    """Classic Backward Transfer (Lopez-Paz & Ranzato 2017 GEM), adapted to
    MSE (lower=better, so the sign convention flips vs the original
    accuracy-based definition): mean over all regimes with >=2 observations
    of (mse right after first training on that regime) - (mse at the FINAL
    generation across the whole record). Requires a well-defined "final"
    generation shared across regimes, unlike average_forgetting's simpler
    per-regime running-min -- included as the standard alternative metric,
    but average_forgetting is the primary one this project reports (see
    module docstring for why).

    Returns None if fewer than 2 regimes have 2+ observations (BWT is
    undefined with less data than that).
    """
    by_regime: dict[str, list[RegimeObservation]] = {}
    for obs in observations:
        by_regime.setdefault(obs.regime, []).append(obs)

    final_gen = max(o.generation for o in observations) if observations else None
    if final_gen is None:
        return None

    deltas = []
    for regime, obs_list in by_regime.items():
        if len(obs_list) < 2:
            continue
        obs_sorted = sorted(obs_list, key=lambda o: o.generation)
        first_obs = obs_sorted[0]
        final_obs_for_regime = next((o for o in reversed(obs_sorted) if o.generation == final_gen), obs_sorted[-1])
        # MSE convention: positive delta = got WORSE by the final generation
        # (forgetting), matching average_forgetting's sign.
        deltas.append(final_obs_for_regime.mse - first_obs.mse)

    if len(deltas) < 2:
        return None
    return sum(deltas) / len(deltas)
