import numpy as np
import pytest

from causal_moe.data.candidate_edges import (
    CandidateSourceSet,
    build_candidate_source_set,
    cap_candidate_source_set,
    expand_source_clusters_to_edge_index,
    rank_sources_by_lagged_correlation,
)


def make_toy_edge_index() -> np.ndarray:
    # 0 -- 1 -- 2 -- 3 (undirected chain, both directions stored)
    # plus 4 isolated
    pairs = [(0, 1), (1, 0), (1, 2), (2, 1), (2, 3), (3, 2)]
    src = [p[0] for p in pairs]
    dst = [p[1] for p in pairs]
    return np.array([src, dst], dtype=np.int64)


def test_direct_variant_includes_only_immediate_neighbors_and_self():
    edge_index = make_toy_edge_index()
    result = build_candidate_source_set(edge_index, target=1, variant="direct", n_clusters=5)

    assert set(result.source_clusters.tolist()) == {0, 1, 2}  # neighbors of 1, plus self


def test_2hop_variant_includes_neighbors_of_neighbors():
    edge_index = make_toy_edge_index()
    result = build_candidate_source_set(edge_index, target=1, variant="2hop", n_clusters=5)

    # direct neighbors of 1: {0, 2}; neighbors of those: {1} (from 0), {1, 3} (from 2)
    assert set(result.source_clusters.tolist()) == {0, 1, 2, 3}


def test_full_variant_includes_all_clusters():
    edge_index = make_toy_edge_index()
    result = build_candidate_source_set(edge_index, target=1, variant="full", n_clusters=5)

    assert set(result.source_clusters.tolist()) == {0, 1, 2, 3, 4}


def test_target_always_included_even_if_isolated():
    edge_index = make_toy_edge_index()
    result = build_candidate_source_set(edge_index, target=4, variant="direct", n_clusters=5)

    assert set(result.source_clusters.tolist()) == {4}  # no neighbors, but self always included


def test_unknown_variant_rejected():
    edge_index = make_toy_edge_index()
    with pytest.raises(ValueError):
        build_candidate_source_set(edge_index, target=1, variant="bogus", n_clusters=5)


def test_expand_source_clusters_to_edge_index_shape_and_target_column():
    sources = np.array([0, 2, 4], dtype=np.int64)
    edge_index = expand_source_clusters_to_edge_index(sources, target=7)

    assert edge_index.shape == (2, 3)
    assert (edge_index[1] == 7).all()
    assert edge_index[0].tolist() == [0, 2, 4]


def _make_olr_series_with_known_correlations(n_t: int = 500) -> np.ndarray:
    """5 clusters; target=0. cluster 1 strongly correlated with target
    (lag 0), cluster 2 moderately, clusters 3/4 near-zero (independent
    noise) -- gives rank_sources_by_lagged_correlation an unambiguous
    ground-truth ordering to check against."""
    rng = np.random.default_rng(0)
    target_series = rng.standard_normal(n_t)
    series = np.stack([
        target_series,
        0.9 * target_series + 0.1 * rng.standard_normal(n_t),  # cluster 1: strong
        0.5 * target_series + 0.5 * rng.standard_normal(n_t),  # cluster 2: moderate
        rng.standard_normal(n_t),  # cluster 3: independent
        rng.standard_normal(n_t),  # cluster 4: independent
    ], axis=1)
    return series


def test_rank_sources_by_lagged_correlation_orders_strongest_first():
    series = _make_olr_series_with_known_correlations()
    ranked = rank_sources_by_lagged_correlation(series, target=0, source_clusters=np.array([0, 1, 2, 3, 4]))

    ranked_ids = [src for src, _ in ranked]
    assert ranked_ids[0] == 1  # strongest correlation
    assert ranked_ids[1] == 2  # moderate
    assert set(ranked_ids[2:]) == {3, 4}  # weakest two, order between them not asserted
    assert 0 not in ranked_ids  # target itself never included


def test_cap_candidate_source_set_keeps_strongest_sources_plus_self():
    series = _make_olr_series_with_known_correlations()
    candidate_set = CandidateSourceSet(
        target=0, variant="full", source_clusters=np.array([0, 1, 2, 3, 4], dtype=np.int64)
    )

    capped = cap_candidate_source_set(candidate_set, series, max_candidates=2)

    assert set(capped.source_clusters.tolist()) == {0, 1, 2}  # self + top-2 by correlation
    assert capped.variant == "full_capped2"


def test_cap_candidate_source_set_is_noop_when_already_small_enough():
    series = _make_olr_series_with_known_correlations()
    candidate_set = CandidateSourceSet(
        target=0, variant="direct", source_clusters=np.array([0, 1], dtype=np.int64)
    )

    capped = cap_candidate_source_set(candidate_set, series, max_candidates=10)

    assert capped is candidate_set  # unchanged, same object, not just equal
    assert capped.variant == "direct"
