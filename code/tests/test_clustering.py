import numpy as np
import pytest

from causal_moe.data.clustering import (
    N_CLUSTERS,
    ClusterAssignment,
    compute_cluster_assignment,
    pool_features_to_clusters,
)
from causal_moe.data.mesh import build_cluster_mesh, build_raw_lattice_mesh, node_id


def make_two_group_olr(n_time: int = 200, n_lat: int = 4, n_lon: int = 4) -> np.ndarray:
    """Synthetic OLR field with two obviously-distinct correlation groups:
    top half (rows 0-1) driven by signal A, bottom half (rows 2-3) by
    signal B, plus small independent noise per cell so cells aren't
    perfectly identical."""
    rng = np.random.default_rng(0)
    t = np.arange(n_time)
    signal_a = np.sin(2 * np.pi * t / 17.0)
    signal_b = np.cos(2 * np.pi * t / 23.0)

    olr = np.empty((n_time, n_lat, n_lon), dtype=np.float32)
    for r in range(n_lat):
        for c in range(n_lon):
            base = signal_a if r < n_lat // 2 else signal_b
            noise = rng.normal(scale=0.01, size=n_time)
            olr[:, r, c] = base + noise
    return olr


def test_kmeans_recovers_two_obvious_groups():
    olr = make_two_group_olr()
    assignment = compute_cluster_assignment(olr, n_clusters=2)

    n_lat, n_lon = olr.shape[1], olr.shape[2]
    top_labels = set()
    bottom_labels = set()
    for r in range(n_lat):
        for c in range(n_lon):
            nid = node_id(r, c, n_lon)
            if r < n_lat // 2:
                top_labels.add(assignment.labels[nid])
            else:
                bottom_labels.add(assignment.labels[nid])

    # all top-half cells should share one label, all bottom-half another,
    # and the two groups should be disjoint labels.
    assert len(top_labels) == 1
    assert len(bottom_labels) == 1
    assert top_labels != bottom_labels


def test_cluster_assignment_covers_all_cells_no_empty_clusters():
    olr = make_two_group_olr(n_lat=6, n_lon=6)
    assignment = compute_cluster_assignment(olr, n_clusters=5)

    assert assignment.labels.shape[0] == 36
    counts = np.bincount(assignment.labels, minlength=5)
    assert (counts > 0).all()
    assert counts.sum() == 36


def test_pool_features_to_clusters_computes_mean():
    # 4 raw places, 2 clusters: {0,1} -> cluster 0, {2,3} -> cluster 1
    assignment = ClusterAssignment(
        n_clusters=2, labels=np.array([0, 0, 1, 1]), n_lat=2, n_lon=2
    )
    features = np.array([[1.0, 2.0, 10.0, 20.0]], dtype=np.float32)  # (1, 4)
    pooled = pool_features_to_clusters(features, assignment)

    assert pooled.shape == (1, 2)
    assert pooled[0, 0] == pytest.approx(1.5)
    assert pooled[0, 1] == pytest.approx(15.0)


def test_pool_features_to_clusters_rejects_empty_cluster():
    assignment = ClusterAssignment(
        n_clusters=2, labels=np.array([0, 0, 0, 0]), n_lat=2, n_lon=2  # cluster 1 empty
    )
    features = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
    with pytest.raises(ValueError):
        pool_features_to_clusters(features, assignment)


def test_build_cluster_mesh_adjacency_from_raw_lattice():
    # 2x4 raw grid, row-major node ids: row0 = {0,1,2,3} (cols 0-3),
    # row1 = {4,5,6,7} (cols 0-3). Cluster by COLUMN PAIRS spanning both
    # rows so cluster adjacency mirrors column adjacency: cluster 0 =
    # {0,4} (col 0), cluster 1 = {1,5} (col 1), cluster 2 = {2,6} (col 2),
    # cluster 3 = {3,7} (col 3). Adjacent clusters should end up connected
    # since their member cells are lattice-adjacent (same-row, adjacent
    # column).
    lat = np.array([10.0, 0.0], dtype=np.float32)
    lon = np.array([0.0, 90.0, 180.0, 270.0], dtype=np.float32)
    raw_mesh = build_raw_lattice_mesh(lat, lon)

    labels = np.array([0, 1, 2, 3, 0, 1, 2, 3], dtype=np.int64)
    assignment = ClusterAssignment(n_clusters=4, labels=labels, n_lat=2, n_lon=4)

    cluster_mesh = build_cluster_mesh(raw_mesh, assignment)

    assert cluster_mesh.n_places == 4
    edges = set(zip(cluster_mesh.edge_index[0].tolist(), cluster_mesh.edge_index[1].tolist()))
    # no self-loops
    for cid in range(4):
        assert (cid, cid) not in edges
    # clusters 0-1, 1-2, 2-3 should be adjacent (columns are neighbors)
    assert (0, 1) in edges and (1, 0) in edges
    assert (1, 2) in edges and (2, 1) in edges
    assert (2, 3) in edges and (3, 2) in edges
    # wraparound: cluster 3 (col 3) and cluster 0 (col 0) should connect
    assert (3, 0) in edges and (0, 3) in edges


def test_n_clusters_constant_is_50():
    assert N_CLUSTERS == 50
