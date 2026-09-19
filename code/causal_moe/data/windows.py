"""Lagged node-feature windows + forecast targets (§4.1, §8).

Per §8 "Node feature window": each node carries 3 snapshots per day --
today, 5-days-ago, 10-days-ago -- of 7 channels (6 fields + is_ocean flag),
giving 21 values per place per day (3 lags x 7 channels). is_ocean is
static (doesn't vary in time) but is still replicated at each lag so every
lag-block has the same channel count -- simplifies the per-lag reshape
used later by the splitter (§4.2), which treats each lag as a same-shaped
copy of the graph.

Target (§8 "Per-place outcome"): each place's own OLR, N days ahead.

Built at raw-grid-cell resolution first (3,600 places), then pooled to
cluster-level (§4.1/§8: 50 fixed clusters) via
causal_moe.data.clustering.pool_features_to_clusters -- kept as two steps
so the (already-tested) raw windowing logic doesn't need to know about
clustering at all.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from causal_moe.data.clustering import ClusterAssignment, pool_features_to_clusters
from causal_moe.data.raw import FIELDS, TARGET_FIELD, BSISORawData

LAGS_DAYS = (0, 5, 10)  # today, 5-days-ago, 10-days-ago
N_CHANNELS_PER_LAG = len(FIELDS) + 1  # 6 fields + is_ocean flag = 7
N_FEATURES = len(LAGS_DAYS) * N_CHANNELS_PER_LAG  # 21


@dataclass
class WindowedDataset:
    """Model-ready arrays, still indexed by (sample, place, ...) -- not yet
    turned into per-timestep PyG Data objects (that happens per-place /
    per-batch downstream, since 16,071 x 3,600 graphs isn't something we
    want to materialize all at once).

    features: (n_samples, n_places, N_FEATURES) float32
        channel order per lag block: [sst, h850, u200, pw, u850, olr, is_ocean]
        lag block order: [lag0 (today), lag5, lag10]
    targets: (n_samples, n_places) float32 -- OLR, N days ahead
    sample_time_index: (n_samples,) int64 -- index into the original raw
        time axis that each sample's "today" (lag 0) corresponds to
    """

    features: np.ndarray
    targets: np.ndarray
    sample_time_index: np.ndarray
    lead_time_days: int
    lags_days: tuple = LAGS_DAYS
    field_names: tuple = FIELDS


def build_windowed_dataset(
    raw: BSISORawData,
    lead_time_days: int = 1,
) -> WindowedDataset:
    """Builds lagged feature windows + shifted OLR targets.

    A sample at raw time-index t requires:
      - lag10 data at (t - 10)   => t >= 10
      - target OLR at (t + lead_time_days) => t <= n_time - 1 - lead_time_days
    so valid t ranges over [10, n_time - 1 - lead_time_days].
    """
    if lead_time_days < 1:
        raise ValueError("lead_time_days must be >= 1")

    max_lag = max(LAGS_DAYS)
    n_time = raw.n_time
    t_start = max_lag
    t_end = n_time - 1 - lead_time_days  # inclusive
    if t_end < t_start:
        raise ValueError(
            f"not enough timesteps: n_time={n_time}, max_lag={max_lag}, "
            f"lead_time_days={lead_time_days}"
        )

    n_samples = t_end - t_start + 1
    n_places = raw.n_places

    is_ocean_flat = raw.ocean_mask.reshape(-1).astype(np.float32)  # (n_places,)

    fields_flat = raw.fields.reshape(len(FIELDS), n_time, n_places)  # (6, T, n_places)
    # Land cells carry NaN in sst; replace with sentinel 0.0 (§8: fill with
    # sentinel + is_ocean flag -- the flag tells the model to distrust it).
    sst_idx = FIELDS.index("sst")
    fields_flat = fields_flat.copy()
    fields_flat[sst_idx] = np.nan_to_num(fields_flat[sst_idx], nan=0.0)

    target_idx = FIELDS.index(TARGET_FIELD)

    features = np.empty((n_samples, n_places, N_FEATURES), dtype=np.float32)
    targets = np.empty((n_samples, n_places), dtype=np.float32)
    sample_time_index = np.empty((n_samples,), dtype=np.int64)

    for i, t in enumerate(range(t_start, t_end + 1)):
        blocks = []
        for lag in LAGS_DAYS:
            t_lag = t - lag
            channel_block = fields_flat[:, t_lag, :]  # (6, n_places)
            channel_block = np.concatenate(
                [channel_block, is_ocean_flat[None, :]], axis=0
            )  # (7, n_places)
            blocks.append(channel_block)
        # (3, 7, n_places) -> (n_places, 21), lag-major then channel-major
        stacked = np.stack(blocks, axis=0)  # (3 lags, 7 ch, n_places)
        features[i] = stacked.transpose(2, 0, 1).reshape(n_places, N_FEATURES)

        targets[i] = fields_flat[target_idx, t + lead_time_days, :]
        sample_time_index[i] = t

    return WindowedDataset(
        features=features,
        targets=targets,
        sample_time_index=sample_time_index,
        lead_time_days=lead_time_days,
    )


def pool_windowed_dataset_to_clusters(
    ds: WindowedDataset,
    assignment: ClusterAssignment,
) -> WindowedDataset:
    """Mean-pools a raw-cell-resolution WindowedDataset to cluster-level
    (§4.1: "Cluster-level feature value = mean of member cells' values").
    Targets (OLR N days ahead) are pooled the same way -- a cluster's
    target is the mean OLR of its member cells at the target time, matching
    the definition of the cluster's own feature (mean of member cells).
    """
    # features: (n_samples, n_places_raw, N_FEATURES) -> move n_places_raw
    # to the last axis for pool_features_to_clusters, then move back.
    features_raw_last = np.moveaxis(ds.features, 1, -1)  # (n_samples, N_FEATURES, n_places_raw)
    pooled_features = pool_features_to_clusters(features_raw_last, assignment)  # (..., n_clusters)
    pooled_features = np.moveaxis(pooled_features, -1, 1)  # (n_samples, n_clusters, N_FEATURES)

    pooled_targets = pool_features_to_clusters(ds.targets, assignment)  # (n_samples, n_clusters)

    return WindowedDataset(
        features=pooled_features.astype(np.float32),
        targets=pooled_targets.astype(np.float32),
        sample_time_index=ds.sample_time_index,
        lead_time_days=ds.lead_time_days,
    )
