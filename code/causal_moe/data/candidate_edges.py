"""Candidate edge sets for a single target place on the REAL 50-cluster
mesh (SS9 step 4 / SS5.3 stage 2).

Unlike steps 2/3 (CausalDynamics, semi-synthetic patches), where every node
pair was a candidate (fully-connected, no physical pre-filter existed),
the real mesh already carries a physical lattice adjacency
(cache/windowed_clustered50_lead1.npz::edge_index, SS4.1) -- mean degree
~5, max 10 across the 50 clusters. Step 4 asks, for ONE target cluster,
which of the other 49 clusters (at which lag) are its true causal drivers.

Three candidate-set variants, per user decision 2026-09-19 (try a subset,
report which works best) -- built here as plain source-cluster-id lists;
the caller expands each source into 3 (source, lag) pairs per SS4.1's
lag structure, same convention as steps 2/3's edge_index tensors:

  - "direct": target's direct lattice neighbors only (+ self) -- smallest,
    matches SS4.1's physical adjacency directly.
  - "2hop": direct neighbors plus THEIR neighbors (+ self) -- covers
    BSISO propagation one hop further than immediate adjacency.
  - "full": all other 49 clusters (+ self) -- ignores lattice adjacency
    entirely, closest to steps 2/3's methodology, most expensive.

Self (the target cluster's own history) is always included as a candidate
source, since self-persistence is a real, physically meaningful driver
(cf. step 3's injected rule, where every non-rule cluster self-persists).

**B18 fix (2026-09-24)**: `cap_candidate_source_set` below shrinks any
source set (typically "full"'s all-50) down to the top-K sources by raw
lagged |correlation| with the target, plus the target itself. This
directly targets Bug 18's root cause -- the shared DIR-GNN encoder loses
edge-score discrimination as the candidate pool it scores in one forward
pass grows (confirmed both on a synthetic harness and, at the full
50-place mesh-wide level, on the real mesh: `full`'s pool of 50
candidates cleared the discrimination gate at 0/50 places while
`direct`'s ~6-candidate pool cleared it at 27/50, PROJECT_PLAN.md B18 +
Phase 9.4). A capped pool keeps most of `full`'s reach (it can still
pick a source `direct`/`2hop` would never offer) while shrinking the
encoder's simultaneous candidate count back toward the range where
discrimination survives.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

CANDIDATE_SET_NAMES = ("direct", "2hop", "full")


@dataclass
class CandidateSourceSet:
    target: int
    variant: str
    source_clusters: np.ndarray  # (n_sources,) int64, sorted, includes target itself


def _direct_neighbors(edge_index: np.ndarray, node: int) -> set[int]:
    mask = edge_index[0] == node
    return set(edge_index[1, mask].tolist())


def build_candidate_source_set(
    edge_index: np.ndarray,
    target: int,
    variant: str,
    n_clusters: int = 50,
) -> CandidateSourceSet:
    """edge_index: (2, n_edges) int64, the real cluster mesh's directed
    lattice adjacency (both directions already present, SS4.1).
    target: cluster id whose future OLR is being forecast.
    variant: one of CANDIDATE_SET_NAMES.
    """
    if variant not in CANDIDATE_SET_NAMES:
        raise ValueError(f"unknown variant {variant!r}, expected one of {CANDIDATE_SET_NAMES}")

    if variant == "full":
        sources = set(range(n_clusters))
    else:
        direct = _direct_neighbors(edge_index, target)
        if variant == "direct":
            sources = set(direct)
        else:  # "2hop"
            sources = set(direct)
            for n in direct:
                sources |= _direct_neighbors(edge_index, n)
    sources.add(target)  # self-persistence is always a candidate driver

    return CandidateSourceSet(
        target=target,
        variant=variant,
        source_clusters=np.array(sorted(sources), dtype=np.int64),
    )


def rank_sources_by_lagged_correlation(
    olr_lag0_series: np.ndarray,
    target: int,
    source_clusters: np.ndarray,
    lags: tuple[int, ...] = (0, 1, 5, 10),
) -> list[tuple[int, float]]:
    """Ranks candidate sources by their best-lag raw |correlation| with the
    target's own OLR series, descending. Same diagnostic already used
    (post-hoc, for reporting only) as `raw_lagged_correlation_auc` in
    `scripts/train_step4_single_place.py` -- factored out here so
    `cap_candidate_source_set` can use it BEFORE training, not just to
    describe a result after the fact.

    olr_lag0_series: (n_samples, n_clusters) float array, lag-0 (today's)
        OLR for every cluster -- e.g. `features[:, :, olr_lag0_channel]`.
    Returns [(source_id, best_abs_corr), ...] sorted by best_abs_corr desc.
    The target itself is never included (self is handled separately by the
    caller -- it is always kept, never subject to the cap).
    """
    n_t = olr_lag0_series.shape[0]
    scored = []
    for src in source_clusters:
        src = int(src)
        if src == target:
            continue
        best = 0.0
        for lag in lags:
            if lag == 0:
                a, b = olr_lag0_series[:, src], olr_lag0_series[:, target]
            else:
                a, b = olr_lag0_series[: n_t - lag, src], olr_lag0_series[lag:, target]
            if a.std() < 1e-8 or b.std() < 1e-8:
                continue
            corr = float(np.corrcoef(a, b)[0, 1])
            best = max(best, abs(corr))
        scored.append((src, best))
    scored.sort(key=lambda pair: -pair[1])
    return scored


def cap_candidate_source_set(
    candidate_set: CandidateSourceSet,
    olr_lag0_series: np.ndarray,
    max_candidates: int,
) -> CandidateSourceSet:
    """**B18 fix.** Shrinks `candidate_set.source_clusters` to at most
    `max_candidates` non-self sources (plus the target itself, always
    kept) by keeping only the top-`max_candidates` sources ranked by raw
    lagged |correlation| with the target (`rank_sources_by_lagged_correlation`).

    This is a DATA-dependent step (needs the actual OLR series to rank
    sources), so it is kept separate from the pure-topology
    `build_candidate_source_set` above rather than folded into it -- call
    this second, only when `candidate_set.source_clusters` is larger than
    `max_candidates`.

    If the set is already at or below `max_candidates`, returns it
    unchanged (capping `direct`'s ~6 candidates would defeat the point --
    this is meant to shrink `full`'s 50, or `2hop`'s 15-20, not tighten an
    already-small pool further).
    """
    n_non_self = candidate_set.source_clusters.shape[0] - int(
        candidate_set.target in candidate_set.source_clusters
    )
    if n_non_self <= max_candidates:
        return candidate_set

    ranked = rank_sources_by_lagged_correlation(
        olr_lag0_series, candidate_set.target, candidate_set.source_clusters
    )
    kept = {src for src, _ in ranked[:max_candidates]}
    kept.add(candidate_set.target)

    return CandidateSourceSet(
        target=candidate_set.target,
        variant=f"{candidate_set.variant}_capped{max_candidates}",
        source_clusters=np.array(sorted(kept), dtype=np.int64),
    )


def expand_source_clusters_to_edge_index(source_clusters: np.ndarray, target: int) -> np.ndarray:
    """One candidate edge per source cluster, all pointing at `target`
    (lag structure is handled separately, in the feature/window construction
    -- each "edge" here represents "this source cluster's full 21-value
    lagged feature block may be a driver", matching how the splitter's
    encoder already consumes lag-flattened per-node features, SS4.1/SS4.2).

    Returns: (2, n_sources) int64, edge_index[0]=source, edge_index[1]=target
    (repeated).
    """
    n = source_clusters.shape[0]
    dst = np.full(n, target, dtype=np.int64)
    return np.stack([source_clusters, dst], axis=0)
