import numpy as np
import pytest

from causal_moe.data.clustering import ClusterAssignment
from causal_moe.data.raw import FIELDS, BSISORawData
from causal_moe.data.windows import (
    N_CHANNELS_PER_LAG,
    N_FEATURES,
    build_windowed_dataset,
    pool_windowed_dataset_to_clusters,
)


def make_fake_raw(n_time: int = 30, n_lat: int = 2, n_lon: int = 3) -> BSISORawData:
    """Small synthetic dataset with a known, checkable pattern: each field's
    value at (t, r, c) = field_index * 10000 + t * 100 + r * 10 + c, except
    sst on one specific land cell which is NaN everywhere (to exercise the
    is_ocean/sentinel path)."""
    n_places = n_lat * n_lon
    fields = np.empty((len(FIELDS), n_time, n_lat, n_lon), dtype=np.float32)
    for fi in range(len(FIELDS)):
        for t in range(n_time):
            for r in range(n_lat):
                for c in range(n_lon):
                    fields[fi, t, r, c] = fi * 10000 + t * 100 + r * 10 + c

    ocean_mask = np.ones((n_lat, n_lon), dtype=bool)
    ocean_mask[0, 0] = False  # one land cell
    sst_idx = FIELDS.index("sst")
    fields[sst_idx, :, 0, 0] = np.nan

    time = np.array(
        [np.datetime64("2000-01-01") + np.timedelta64(t, "D") for t in range(n_time)]
    )
    lat = np.linspace(10.0, 0.0, n_lat).astype(np.float32)
    lon = np.linspace(0.0, 100.0, n_lon).astype(np.float32)

    return BSISORawData(fields=fields, ocean_mask=ocean_mask, time=time, lat=lat, lon=lon)


def test_shapes_and_valid_range():
    raw = make_fake_raw(n_time=30)
    ds = build_windowed_dataset(raw, lead_time_days=1)

    # valid t in [10, n_time - 1 - 1] = [10, 28] -> 19 samples
    assert ds.features.shape == (19, 6, N_FEATURES)
    assert ds.targets.shape == (19, 6)
    assert N_FEATURES == 21
    assert N_CHANNELS_PER_LAG == 7


def test_lag_values_correct_for_first_sample():
    raw = make_fake_raw(n_time=30)
    ds = build_windowed_dataset(raw, lead_time_days=1)

    # first sample: t_start = 10 (max_lag), so lag0=t10, lag5=t5, lag10=t0
    assert ds.sample_time_index[0] == 10

    place = 4  # r=1, c=1 -> node_id = 1*3+1 = 4
    r, c = 1, 1
    feat = ds.features[0, place]  # (21,) = [lag0: 7ch][lag5: 7ch][lag10: 7ch]

    olr_idx = FIELDS.index("olr")

    def expected_field(fi, t):
        return fi * 10000 + t * 100 + r * 10 + c

    # lag0 block = today = t=10
    assert feat[olr_idx] == pytest.approx(expected_field(olr_idx, 10))
    # lag5 block starts at offset 7, today-5=t5
    assert feat[7 + olr_idx] == pytest.approx(expected_field(olr_idx, 5))
    # lag10 block starts at offset 14, today-10=t0
    assert feat[14 + olr_idx] == pytest.approx(expected_field(olr_idx, 0))

    # is_ocean flag is channel index 6 within each 7-channel block, constant across lags
    assert feat[6] == 1.0
    assert feat[7 + 6] == 1.0
    assert feat[14 + 6] == 1.0


def test_land_cell_sst_sentinel_and_flag():
    raw = make_fake_raw(n_time=30)
    ds = build_windowed_dataset(raw, lead_time_days=1)

    land_place = 0  # r=0, c=0 -> node_id 0, the NaN sst cell
    sst_idx = FIELDS.index("sst")
    feat = ds.features[0, land_place]

    assert feat[sst_idx] == 0.0  # sentinel fill
    assert feat[6] == 0.0  # is_ocean flag = 0 for land


def test_target_is_olr_shifted_by_lead_time():
    raw = make_fake_raw(n_time=30)
    lead = 3
    ds = build_windowed_dataset(raw, lead_time_days=lead)

    olr_idx = FIELDS.index("olr")
    place = 5  # r=1, c=2
    r, c = 1, 2
    t0 = ds.sample_time_index[0]
    expected_target = olr_idx * 10000 + (t0 + lead) * 100 + r * 10 + c
    assert ds.targets[0, place] == pytest.approx(expected_target)


def test_rejects_insufficient_timesteps():
    raw = make_fake_raw(n_time=5)  # smaller than max_lag=10
    with pytest.raises(ValueError):
        build_windowed_dataset(raw, lead_time_days=1)


def test_rejects_bad_lead_time():
    raw = make_fake_raw(n_time=30)
    with pytest.raises(ValueError):
        build_windowed_dataset(raw, lead_time_days=0)


def test_pool_windowed_dataset_to_clusters():
    # 2x3 raw grid = 6 places -> group into 2 clusters: {0,1,2} and {3,4,5}
    raw = make_fake_raw(n_time=30, n_lat=2, n_lon=3)
    ds = build_windowed_dataset(raw, lead_time_days=1)  # (19, 6, 21) features

    labels = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
    assignment = ClusterAssignment(n_clusters=2, labels=labels, n_lat=2, n_lon=3)

    pooled = pool_windowed_dataset_to_clusters(ds, assignment)

    assert pooled.features.shape == (19, 2, N_FEATURES)
    assert pooled.targets.shape == (19, 2)

    # cluster 0 = raw places 0,1,2 (row 0); cluster 1 = raw places 3,4,5 (row 1)
    expected_cluster0 = ds.features[:, 0:3, :].mean(axis=1)
    expected_cluster1 = ds.features[:, 3:6, :].mean(axis=1)
    np.testing.assert_allclose(pooled.features[:, 0, :], expected_cluster0, rtol=1e-5)
    np.testing.assert_allclose(pooled.features[:, 1, :], expected_cluster1, rtol=1e-5)

    expected_target0 = ds.targets[:, 0:3].mean(axis=1)
    np.testing.assert_allclose(pooled.targets[:, 0], expected_target0, rtol=1e-5)

    # metadata preserved
    assert pooled.lead_time_days == ds.lead_time_days
    np.testing.assert_array_equal(pooled.sample_time_index, ds.sample_time_index)
