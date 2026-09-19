import numpy as np
import pytest

from causal_moe.data.candidate_edges import (
    build_candidate_source_set,
    expand_source_clusters_to_edge_index,
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
