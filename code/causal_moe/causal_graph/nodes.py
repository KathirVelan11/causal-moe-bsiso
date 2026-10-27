"""(region, field) node series for causal discovery (Phase 1 spec SS2).

Each region has 6 fields -> 50 x 6 = 300 nodes; node id = region * 6 +
field index (FIELDS order). Series are region means of the already
anomaly-processed cell fields, re-standardized per node over the discovery
years (pooling shrinks the variance of standardized cells, and SST is
averaged over ocean cells only so coastal regions are not diluted by the
land sentinel).

Also provides the nested 100-region split used by B4 (each 50-region is
cut in two, so every child has exactly one parent) and the region-to-region
distance matrix used by the B2 physical lag mask.
"""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans

from causal_moe.data.raw import FIELDS, BSISORawData

N_FIELDS = len(FIELDS)
EARTH_RADIUS_M = 6.371e6


def node_id(region: int, field: str) -> int:
    return region * N_FIELDS + FIELDS.index(field)


def node_region_field(node: int) -> tuple[int, str]:
    return node // N_FIELDS, FIELDS[node % N_FIELDS]


def build_node_series(raw: BSISORawData, labels: np.ndarray, discovery_mask: np.ndarray) -> np.ndarray:
    """Returns (T, n_regions * 6) float32, standardized over discovery_mask days.

    labels: (n_cells,) region id per raw cell.
    """
    n_regions = int(labels.max()) + 1
    T = raw.n_time
    ocean = raw.ocean_mask.reshape(-1)
    out = np.empty((T, n_regions * N_FIELDS), dtype=np.float32)
    for f, name in enumerate(FIELDS):
        flat = raw.fields[f].reshape(T, -1)
        for r in range(n_regions):
            members = np.nonzero(labels == r)[0]
            if name == "sst":
                members = members[ocean[members]]
            series = flat[:, members].mean(axis=1) if members.size else np.zeros(T, dtype=np.float32)
            out[:, r * N_FIELDS + f] = series
    mu = out[discovery_mask].mean(axis=0)
    sd = out[discovery_mask].std(axis=0)
    valid = sd > 1e-6
    out[:, valid] = (out[:, valid] - mu[valid]) / sd[valid]
    out[:, ~valid] = 0.0
    return out


def valid_nodes(series: np.ndarray) -> np.ndarray:
    """Nodes with non-constant series (an all-land region has no SST node)."""
    return np.nonzero(series.std(axis=0) > 1e-6)[0]


def split_regions_in_two(raw: BSISORawData, labels: np.ndarray, fit_mask: np.ndarray, random_state: int = 0) -> np.ndarray:
    """Nested refinement for B4: child labels in [0, 2 * n_regions), child
    2c and 2c+1 both belong to parent c. Each region's member cells are cut
    with the same rule that built the regions (k-means on OLR correlation
    profiles), fitted on the same years."""
    olr = raw.fields[FIELDS.index("olr")][fit_mask].reshape(int(fit_mask.sum()), -1)
    child = np.empty_like(labels)
    for r in range(int(labels.max()) + 1):
        members = np.nonzero(labels == r)[0]
        corr = np.nan_to_num(np.corrcoef(olr[:, members].T), nan=0.0)
        km = KMeans(n_clusters=2, random_state=random_state, n_init="auto").fit_predict(corr)
        child[members] = 2 * r + km
    return child


def region_min_distance_m(raw: BSISORawData, labels: np.ndarray) -> np.ndarray:
    """(n_regions, n_regions) great-circle distance between the CLOSEST pair
    of member cells -- the distance a signal must at least travel to reach
    the target region (0 for the region itself). Generous by design (SS4 B2)."""
    lat_g, lon_g = np.meshgrid(np.deg2rad(raw.lat), np.deg2rad(raw.lon), indexing="ij")
    lat_c, lon_c = lat_g.reshape(-1), lon_g.reshape(-1)
    n = int(labels.max()) + 1
    out = np.zeros((n, n))
    for a in range(n):
        ma = labels == a
        la, lo = lat_c[ma][:, None], lon_c[ma][:, None]
        for b in range(a + 1, n):
            mb = labels == b
            dlat = lat_c[mb][None, :] - la
            dlon = lon_c[mb][None, :] - lo
            h = np.sin(dlat / 2) ** 2 + np.cos(la) * np.cos(lat_c[mb][None, :]) * np.sin(dlon / 2) ** 2
            d = 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(np.clip(h, 0, 1))).min()
            out[a, b] = out[b, a] = d
    return out
