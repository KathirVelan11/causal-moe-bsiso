"""Lagged node-feature windows + forecast targets (§4.1, §8).

Per §8 "Node feature window": each node carries several snapshots per day
of 7 channels (6 fields + is_ocean flag). is_ocean is static (doesn't vary
in time) but is still replicated at each lag so every lag-block has the
same channel count -- simplifies the per-lag reshape used later by the
splitter (§4.2), which treats each lag as a same-shaped copy of the graph.

**Lag set widened 2026-09-22 (audit fix B8).** The original set was
(0, 5, 10) -- today, 5-days-ago, 10-days-ago -- chosen to mirror the BSISO
index's own lag structure. But the PCMCI cross-check on the same data
located place 22's real drivers at lags 3, 5, 6 and 7
(`results/raw/pcmci_place22_direct.json`: cluster 38 @ lag 3, 13 @ lag 5,
3 @ lag 6, 41 @ lag 7). Three of those four lags did not exist as features,
so §1's stated aim -- "which other clusters *and at which lag* are the true
causal drivers" -- was unanswerable by construction. The set is now
(0, 1, 2, 3, 5, 7, 10): 7 lags x 7 channels = 49 values per place per day.

Target (§8 "Per-place outcome"): each place's own OLR, N days ahead.

Two build paths, both producing the same numbers:
  - `build_windowed_dataset` + `pool_windowed_dataset_to_clusters`: the
    original two-step route, raw-cell resolution first then pooled. Kept
    because the raw windowing logic is independently tested, but it
    materializes (n_samples, 3600, 49) float32 = ~11 GB at the widened lag
    set -- too large for this CPU-only dev machine (§8 compute row).
  - `build_clustered_windowed_dataset`: pools the raw fields to cluster
    level FIRST, then windows. Pooling is a per-cell mean and windowing is
    pure time-indexing, so the two commute exactly (verified against the
    existing lead-1 cache). Peak memory ~157 MB instead of ~11 GB. This is
    the path `scripts/build_mesh_cache.py` uses.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from causal_moe.data.clustering import ClusterAssignment, pool_features_to_clusters
from causal_moe.data.raw import FIELDS, TARGET_FIELD, BSISORawData

LAGS_DAYS = (0, 1, 2, 3, 5, 7, 10)
"""Trailing lags each node carries, in days. Widened from (0, 5, 10) on
2026-09-22 so PCMCI's empirically-located driver lags (3, 5, 6, 7) are
representable -- see module docstring."""

LEGACY_LAGS_DAYS = (0, 5, 10)
"""The original lag set, kept so the pre-2026-09-22 lead-1 cache can still
be rebuilt bit-for-bit for regression comparison."""

N_CHANNELS_PER_LAG = len(FIELDS) + 1  # 6 fields + is_ocean flag = 7
N_FEATURES = len(LAGS_DAYS) * N_CHANNELS_PER_LAG  # 49


def channel_index(lag_days: int, field_name: str, lags_days: tuple = LAGS_DAYS) -> int:
    """Absolute column of one (lag, field) pair in a place's feature vector.

    Layout is lag-major then channel-major: [lag0: 7ch][lag1: 7ch]...,
    with channel order [sst, h850, u200, pw, u850, olr, is_ocean].

    Replaces the hardcoded `OLR_LAG0_CHANNEL = 5` constants that were
    scattered across the training scripts -- those were correct only for
    the old 3-lag layout and would have silently pointed at the wrong
    column once the lag set widened.
    """
    if lag_days not in lags_days:
        raise ValueError(f"lag {lag_days} not in lag set {lags_days}")
    lag_block = lags_days.index(lag_days)
    channel = N_CHANNELS_PER_LAG - 1 if field_name == "is_ocean" else FIELDS.index(field_name)
    return lag_block * N_CHANNELS_PER_LAG + channel


@dataclass
class WindowedDataset:
    """Model-ready arrays, still indexed by (sample, place, ...) -- not yet
    turned into per-timestep PyG Data objects (that happens per-place /
    per-batch downstream, since 16,071 x 3,600 graphs isn't something we
    want to materialize all at once).

    features: (n_samples, n_places, n_features) float32
        channel order per lag block: [sst, h850, u200, pw, u850, olr, is_ocean]
        lag block order: `lags_days`, e.g. [lag0 (today), lag1, ..., lag10]
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
    lags_days: tuple = LAGS_DAYS,
) -> WindowedDataset:
    """Builds lagged feature windows + shifted OLR targets, at RAW grid-cell
    resolution.

    A sample at raw time-index t requires:
      - the deepest lag's data at (t - max_lag)   => t >= max_lag
      - target OLR at (t + lead_time_days) => t <= n_time - 1 - lead_time_days
    so valid t ranges over [max_lag, n_time - 1 - lead_time_days].

    Memory warning: the output is (n_samples, 3600, 7*len(lags_days))
    float32 -- ~11 GB at the default 7-lag set. Use
    `build_clustered_windowed_dataset` for anything at full record length;
    this function is kept for the raw-resolution unit tests and small
    patches (§5.2's semi-synthetic check).
    """
    if lead_time_days < 1:
        raise ValueError("lead_time_days must be >= 1")

    max_lag = max(lags_days)
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

    features, targets, sample_time_index = _window_from_flat_fields(
        fields_flat, is_ocean_flat, t_start, t_end, lead_time_days, lags_days
    )
    return WindowedDataset(
        features=features,
        targets=targets,
        sample_time_index=sample_time_index,
        lead_time_days=lead_time_days,
        lags_days=lags_days,
    )


def _window_from_flat_fields(
    fields_flat: np.ndarray,
    is_ocean_flat: np.ndarray,
    t_start: int,
    t_end: int,
    lead_time_days: int,
    lags_days: tuple,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Shared windowing core, resolution-agnostic.

    fields_flat: (n_fields, T, n_places) -- NaNs already filled.
    is_ocean_flat: (n_places,) -- the ocean flag/fraction per place.

    Vectorized across time (no per-sample Python loop, per the project's
    CPU-only constraint): each lag block is one strided slice of the time
    axis rather than a loop over 16,000 days.
    """
    n_fields, _, n_places = fields_flat.shape
    n_features = len(lags_days) * N_CHANNELS_PER_LAG
    target_idx = FIELDS.index(TARGET_FIELD)

    sample_time_index = np.arange(t_start, t_end + 1, dtype=np.int64)
    n_samples = sample_time_index.shape[0]

    features = np.empty((n_samples, n_places, n_features), dtype=np.float32)
    for block, lag in enumerate(lags_days):
        lo = block * N_CHANNELS_PER_LAG
        t_slice = sample_time_index - lag
        # (n_fields, n_samples, n_places) -> (n_samples, n_places, n_fields)
        features[:, :, lo : lo + n_fields] = np.moveaxis(fields_flat[:, t_slice, :], 0, -1)
        features[:, :, lo + n_fields] = is_ocean_flat[None, :]

    targets = fields_flat[target_idx, sample_time_index + lead_time_days, :].astype(np.float32)
    return features, targets, sample_time_index


def build_clustered_windowed_dataset(
    raw: BSISORawData,
    assignment: ClusterAssignment,
    lead_time_days: int = 1,
    lags_days: tuple = LAGS_DAYS,
) -> WindowedDataset:
    """Cluster-resolution windowed dataset, built WITHOUT materializing the
    raw-cell version first.

    Identical output to
    `pool_windowed_dataset_to_clusters(build_windowed_dataset(...))` --
    pooling is a per-cell mean and windowing is pure time-indexing, so they
    commute -- but peak memory is ~157 MB instead of ~11 GB at the 7-lag
    set, which is what makes the multi-lead cache build feasible on this
    CPU-only machine (§8 compute row).
    """
    if lead_time_days < 1:
        raise ValueError("lead_time_days must be >= 1")

    max_lag = max(lags_days)
    n_time = raw.n_time
    t_start = max_lag
    t_end = n_time - 1 - lead_time_days
    if t_end < t_start:
        raise ValueError(
            f"not enough timesteps: n_time={n_time}, max_lag={max_lag}, "
            f"lead_time_days={lead_time_days}"
        )

    fields_flat = raw.fields.reshape(len(FIELDS), n_time, raw.n_places).copy()
    sst_idx = FIELDS.index("sst")
    fields_flat[sst_idx] = np.nan_to_num(fields_flat[sst_idx], nan=0.0)

    # Pool FIRST: (6, T, 3600) -> (6, T, n_clusters), ~19 MB.
    fields_clustered = pool_features_to_clusters(fields_flat, assignment)
    # The is_ocean flag pools to the cluster's OCEAN FRACTION -- exactly what
    # the two-step path produces too (it replicates the 0/1 flag per lag and
    # then means it over member cells), just computed once instead of once
    # per lag block.
    is_ocean_clustered = pool_features_to_clusters(
        raw.ocean_mask.reshape(-1).astype(np.float32), assignment
    )

    features, targets, sample_time_index = _window_from_flat_fields(
        fields_clustered, is_ocean_clustered, t_start, t_end, lead_time_days, lags_days
    )
    return WindowedDataset(
        features=features,
        targets=targets,
        sample_time_index=sample_time_index,
        lead_time_days=lead_time_days,
        lags_days=lags_days,
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
        lags_days=ds.lags_days,
    )
