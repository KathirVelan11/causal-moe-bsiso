"""Data-driven clustering of the 3,600 raw grid cells into 50 regions
(§4.1, §8 "Place" -- REVISED, fixed at 50 clusters).

Method: k-means on each cell's OLR anomaly correlation with every other
cell (a 3,600 x 3,600 correlation matrix, row i = cell i's correlation
profile against all cells) -- climatologically similar cells (whose
convection rises and falls together) end up in the same cluster,
regardless of exact geographic shape. Ocean/land status is not part of
the clustering signal itself (SST is excluded, and land cells' OLR is
still real and used).

This is a one-time offline step: cluster assignment is computed once from
the full OLR record and reused for the whole pipeline (not re-clustered
per sample).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.cluster import KMeans

from causal_moe.data.raw import FIELDS

N_CLUSTERS = 50  # fixed, §8 "Place" row


@dataclass
class ClusterAssignment:
    n_clusters: int
    labels: np.ndarray  # (n_places_raw,) int64, cluster id per raw grid cell
    n_lat: int
    n_lon: int

    @property
    def n_places_raw(self) -> int:
        return self.n_lat * self.n_lon

    def members(self, cluster_id: int) -> np.ndarray:
        """Raw-cell indices (row-major, matching node_id) belonging to a cluster."""
        return np.nonzero(self.labels == cluster_id)[0]


def compute_cluster_assignment(
    olr_raw: np.ndarray,
    n_clusters: int = N_CLUSTERS,
    random_state: int = 0,
) -> ClusterAssignment:
    """K-means clustering of raw grid cells by OLR anomaly correlation
    structure.

    olr_raw: (T, n_lat, n_lon) float32 -- the raw OLR anomaly field
        (already anomaly-preprocessed per §3, no further detrending here).

    Returns cluster labels for all n_lat*n_lon raw cells (row-major, i.e.
    node_id(r, c, n_lon) = r*n_lon + c indexes into `labels`).
    """
    n_time, n_lat, n_lon = olr_raw.shape
    n_places = n_lat * n_lon

    flat = olr_raw.reshape(n_time, n_places)  # (T, n_places)

    # Correlation matrix between all cell pairs: corrcoef expects variables
    # as rows, so transpose to (n_places, T) first.
    corr = np.corrcoef(flat.T)  # (n_places, n_places)
    corr = np.nan_to_num(corr, nan=0.0)  # a constant-OLR cell (shouldn't
    # happen post-anomaly, but guard anyway) yields NaN correlation rows

    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
    labels = km.fit_predict(corr).astype(np.int64)

    return ClusterAssignment(n_clusters=n_clusters, labels=labels, n_lat=n_lat, n_lon=n_lon)


def pool_features_to_clusters(
    raw_features: np.ndarray,
    assignment: ClusterAssignment,
) -> np.ndarray:
    """Mean-pools raw-cell features into cluster-level features (§4.1:
    "Cluster-level feature value = mean of member cells' values").

    raw_features: (..., n_places_raw) float32 -- any leading dims (e.g.
        samples, channels), last axis = raw grid cells (row-major).

    Returns: (..., n_clusters) float32
    """
    n_clusters = assignment.n_clusters
    out_shape = raw_features.shape[:-1] + (n_clusters,)
    out = np.zeros(out_shape, dtype=np.float32)
    counts = np.zeros(n_clusters, dtype=np.int64)

    for cid in range(n_clusters):
        member_idx = assignment.members(cid)
        counts[cid] = member_idx.shape[0]
        if member_idx.shape[0] == 0:
            continue
        out[..., cid] = raw_features[..., member_idx].mean(axis=-1)

    if (counts == 0).any():
        empty = np.nonzero(counts == 0)[0].tolist()
        raise ValueError(f"clusters with zero members: {empty}")

    return out
