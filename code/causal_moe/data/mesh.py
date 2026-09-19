"""Cluster-level mesh (§4.1, §8 "Place" -- REVISED, fixed at 50 clusters).

nodes = places = clusters of raw grid cells (see clustering.py), not raw
        cells directly
edges = lattice-derived adjacency between clusters: two clusters are
        neighbors if any of their member cells are adjacent in the raw
        25x144 lattice (4-neighbor, longitude wraparound, no latitude
        wraparound)

The raw-cell lattice building block (build_raw_lattice_mesh) is kept
because cluster adjacency is derived from it, and because §5.2's
mid-scale check clusters small real patches using the same method.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from causal_moe.data.clustering import ClusterAssignment


def node_id(row: int, col: int, n_lon: int) -> int:
    """Row-major flatten: raw-cell node = row * n_lon + col."""
    return row * n_lon + col


@dataclass
class LatticeMesh:
    """A lattice mesh -- either raw-cell (n_places = n_lat*n_lon) or, after
    build_cluster_mesh, cluster-level (n_places = n_clusters)."""

    n_places: int
    edge_index: np.ndarray  # (2, n_edges) int64, both directions included
    lat: np.ndarray  # (n_places,) float32 -- per-node latitude (mean for clusters)
    lon: np.ndarray  # (n_places,) float32 -- per-node longitude (mean for clusters)

    @property
    def n_edges(self) -> int:
        return self.edge_index.shape[1]


def build_raw_lattice_mesh(lat: np.ndarray, lon: np.ndarray) -> LatticeMesh:
    """4-neighbor lattice adjacency over raw grid cells, longitude
    wraparound, no latitude wraparound. Edges are undirected, stored as
    both directed pairs (PyG convention).

    lat: (n_lat,), lon: (n_lon,) -- the 1-D grid coordinate arrays from the
    raw .npz files.
    """
    n_lat = lat.shape[0]
    n_lon = lon.shape[0]

    src = []
    dst = []
    for r in range(n_lat):
        for c in range(n_lon):
            u = node_id(r, c, n_lon)

            # East neighbor, with longitude wraparound.
            c_east = (c + 1) % n_lon
            v = node_id(r, c_east, n_lon)
            src += [u, v]
            dst += [v, u]

            # South neighbor, no latitude wraparound.
            if r + 1 < n_lat:
                v = node_id(r + 1, c, n_lon)
                src += [u, v]
                dst += [v, u]

    edge_index = np.array([src, dst], dtype=np.int64)
    edge_index = np.unique(edge_index, axis=1)

    lat_grid, lon_grid = np.meshgrid(lat, lon, indexing="ij")  # (n_lat, n_lon) each
    node_lat = lat_grid.reshape(-1).astype(np.float32)
    node_lon = lon_grid.reshape(-1).astype(np.float32)

    return LatticeMesh(
        n_places=n_lat * n_lon,
        edge_index=edge_index,
        lat=node_lat,
        lon=node_lon,
    )


def build_cluster_mesh(
    raw_mesh: LatticeMesh,
    assignment: ClusterAssignment,
) -> LatticeMesh:
    """Derives cluster-level adjacency from a raw-cell lattice mesh: two
    clusters are neighbors if any of their member raw cells are adjacent
    (§4.1). Self-loops (a cluster "adjacent to itself" via two internally-
    connected member cells) are dropped. Cluster lat/lon = mean of member
    cells' lat/lon.
    """
    labels = assignment.labels
    src_cluster = labels[raw_mesh.edge_index[0]]
    dst_cluster = labels[raw_mesh.edge_index[1]]

    keep = src_cluster != dst_cluster  # drop self-loops
    pairs = np.stack([src_cluster[keep], dst_cluster[keep]], axis=0)
    pairs = np.unique(pairs, axis=1)  # dedup

    n_clusters = assignment.n_clusters
    cluster_lat = np.zeros(n_clusters, dtype=np.float32)
    cluster_lon = np.zeros(n_clusters, dtype=np.float32)
    for cid in range(n_clusters):
        member_idx = assignment.members(cid)
        cluster_lat[cid] = raw_mesh.lat[member_idx].mean()
        # circular mean for longitude would be more correct near the 0/360
        # wrap, but clusters are climatologically-defined (correlation),
        # not geographically contiguous by construction, so a plain mean
        # is already an approximation -- fine for a diagnostic coordinate,
        # not used by the model itself.
        cluster_lon[cid] = raw_mesh.lon[member_idx].mean()

    return LatticeMesh(
        n_places=n_clusters,
        edge_index=pairs,
        lat=cluster_lat,
        lon=cluster_lon,
    )
