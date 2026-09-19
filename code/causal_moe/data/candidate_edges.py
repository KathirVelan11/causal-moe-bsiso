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
