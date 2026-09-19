import numpy as np
import pytest

from causal_moe.data.clustering import ClusterAssignment, pool_features_to_clusters
from causal_moe.data.raw import FIELDS, BSISORawData
from causal_moe.data.semisynthetic import (
    PATCH_LAT_SLICE,
    PATCH_LON_SLICE,
    build_patch_mesh,
    cluster_patch,
    extract_patch,
    inject_causal_rule,
    make_fully_connected_edge_index,
    pick_abc_avoiding_confounds,
)
from causal_moe.data.windows import build_windowed_dataset, pool_windowed_dataset_to_clusters


def make_fake_full_grid(n_time: int = 60, n_lat: int = 25, n_lon: int = 144) -> BSISORawData:
    """Fake full-size grid so PATCH_LAT_SLICE/PATCH_LON_SLICE (indices into a
    real 25x144 grid) stay in bounds when tests slice it -- values are a
    deterministic, checkable pattern per field/time/cell (mirrors
    tests/test_windows.py's make_fake_raw)."""
    rng = np.random.default_rng(0)
    fields = np.empty((len(FIELDS), n_time, n_lat, n_lon), dtype=np.float32)
    for fi in range(len(FIELDS)):
        fields[fi] = rng.normal(size=(n_time, n_lat, n_lon)).astype(np.float32)

    ocean_mask = np.ones((n_lat, n_lon), dtype=bool)
    time = np.array(
        [np.datetime64("2000-01-01") + np.timedelta64(t, "D") for t in range(n_time)]
    )
    lat = np.linspace(30.0, -30.0, n_lat).astype(np.float32)
    lon = np.linspace(0.0, 357.5, n_lon).astype(np.float32)
    return BSISORawData(fields=fields, ocean_mask=ocean_mask, time=time, lat=lat, lon=lon)


def test_extract_patch_shape_and_slice():
    raw = make_fake_full_grid()
    patch = extract_patch(raw)

    assert patch.fields.shape[2] == 13  # PATCH_LAT_SLICE span
    assert patch.fields.shape[3] == 21  # PATCH_LON_SLICE span
    assert patch.n_places == 273
    np.testing.assert_array_equal(patch.fields, raw.fields[:, :, PATCH_LAT_SLICE, PATCH_LON_SLICE])
    np.testing.assert_array_equal(patch.lat, raw.lat[PATCH_LAT_SLICE])
    np.testing.assert_array_equal(patch.lon, raw.lon[PATCH_LON_SLICE])


def test_cluster_patch_covers_all_cells_no_empty_clusters():
    raw = make_fake_full_grid()
    patch = extract_patch(raw)
    assignment = cluster_patch(patch, n_clusters=5)

    assert assignment.labels.shape[0] == 273
    counts = np.bincount(assignment.labels, minlength=5)
    assert (counts > 0).all()


def test_build_patch_mesh_no_self_loops_and_in_bounds():
    raw = make_fake_full_grid()
    patch = extract_patch(raw)
    assignment = cluster_patch(patch, n_clusters=5)
    mesh = build_patch_mesh(patch, assignment)

    assert mesh.n_places == 5
    for u, v in zip(mesh.edge_index[0].tolist(), mesh.edge_index[1].tolist()):
        assert u != v
        assert 0 <= u < 5 and 0 <= v < 5


# --- pick_abc_avoiding_confounds --------------------------------------------


def test_pick_abc_rejects_when_all_pairs_confounded():
    # 4 clusters, all perfectly correlated with each other -> every pair is
    # a confound, no triple should be returned.
    t = np.linspace(0, 10, 200)
    base = np.sin(t)
    cluster_olr = np.stack([base, base, base, base], axis=1)  # (200, 4), identical

    with pytest.raises(ValueError):
        pick_abc_avoiding_confounds(cluster_olr, max_other_corr=0.8, max_attempts=20)


def test_pick_abc_accepts_independent_clusters():
    rng = np.random.default_rng(0)
    cluster_olr = rng.normal(size=(500, 6)).astype(np.float32)  # independent noise, low cross-corr

    a, b, c = pick_abc_avoiding_confounds(cluster_olr, max_other_corr=0.8, rng=np.random.default_rng(1))
    assert len({a, b, c}) == 3
    assert all(0 <= x < 6 for x in (a, b, c))


# --- inject_causal_rule ------------------------------------------------------


def make_toy_windowed_dataset(n_samples: int = 20, n_clusters: int = 5):
    rng = np.random.default_rng(0)
    n_features = 21  # 3 lags x 7 channels, olr is channel index 5 within each lag block
    features = rng.normal(size=(n_samples, n_clusters, n_features)).astype(np.float32)
    targets = rng.normal(size=(n_samples, n_clusters)).astype(np.float32)  # will be overwritten
    from causal_moe.data.windows import WindowedDataset

    return WindowedDataset(
        features=features,
        targets=targets,
        sample_time_index=np.arange(n_samples, dtype=np.int64),
        lead_time_days=1,
    )


def test_inject_causal_rule_linear_matches_formula():
    ds = make_toy_windowed_dataset(n_samples=10, n_clusters=5)
    a, b, c = 0, 1, 2
    weights = (0.6, 0.4)

    new_ds, true_edge_index = inject_causal_rule(ds, a, b, c, weights=weights)

    olr_channel = FIELDS.index("olr")
    today_olr = ds.features[:, :, olr_channel]
    expected_a = weights[0] * today_olr[:, b] + weights[1] * today_olr[:, c]
    np.testing.assert_allclose(new_ds.targets[:, a], expected_a, rtol=1e-5)

    # other clusters: self-persistence
    for x in range(5):
        if x == a:
            continue
        np.testing.assert_allclose(new_ds.targets[:, x], today_olr[:, x], rtol=1e-5)

    # features untouched (only target changes, §5.2 requirement)
    np.testing.assert_array_equal(new_ds.features, ds.features)


def test_inject_causal_rule_nonlinear_adds_interaction_term():
    ds = make_toy_windowed_dataset(n_samples=10, n_clusters=5)
    a, b, c = 0, 1, 2
    weights = (0.6, 0.4)
    interaction_weight = 0.2

    new_ds, _ = inject_causal_rule(
        ds, a, b, c, weights=weights, nonlinear=True, interaction_weight=interaction_weight
    )

    olr_channel = FIELDS.index("olr")
    today_olr = ds.features[:, :, olr_channel]
    expected_a = (
        weights[0] * today_olr[:, b]
        + weights[1] * today_olr[:, c]
        + interaction_weight * today_olr[:, b] * today_olr[:, c]
    )
    np.testing.assert_allclose(new_ds.targets[:, a], expected_a, rtol=1e-5)


def test_inject_causal_rule_true_edge_index_contents():
    ds = make_toy_windowed_dataset(n_samples=5, n_clusters=5)
    a, b, c = 0, 1, 2

    _, true_edge_index = inject_causal_rule(ds, a, b, c)
    edges = set(zip(true_edge_index[0].tolist(), true_edge_index[1].tolist()))

    # B->A and C->A must be present
    assert (b, a) in edges
    assert (c, a) in edges
    # every other cluster has a self-loop
    for x in range(5):
        if x == a:
            continue
        assert (x, x) in edges
    # A itself has no self-loop (its target doesn't depend on its own past
    # under this rule) and no spurious extra edges
    assert (a, a) not in edges
    assert len(edges) == 2 + 4  # B->A, C->A, plus 4 self-loops (all except A)


def test_inject_causal_rule_rejects_duplicate_ids():
    ds = make_toy_windowed_dataset(n_samples=5, n_clusters=5)
    with pytest.raises(ValueError):
        inject_causal_rule(ds, a=0, b=0, c=1)


def test_inject_causal_rule_rejects_out_of_range_ids():
    ds = make_toy_windowed_dataset(n_samples=5, n_clusters=5)
    with pytest.raises(ValueError):
        inject_causal_rule(ds, a=0, b=1, c=10)


# --- make_fully_connected_edge_index ----------------------------------------


def test_make_fully_connected_edge_index_includes_self_loops():
    edge_index = make_fully_connected_edge_index(4)
    assert edge_index.shape == (2, 16)
    edges = set(zip(edge_index[0].tolist(), edge_index[1].tolist()))
    for i in range(4):
        assert (i, i) in edges
    assert len(edges) == 16


# --- end-to-end on the fake full grid ---------------------------------------


def test_end_to_end_patch_to_injected_targets():
    raw = make_fake_full_grid(n_time=60)
    patch = extract_patch(raw)
    assignment = cluster_patch(patch, n_clusters=5)

    windowed_raw = build_windowed_dataset(patch, lead_time_days=1)
    pooled = pool_windowed_dataset_to_clusters(windowed_raw, assignment)
    assert pooled.features.shape[1] == 5

    cluster_olr = pool_features_to_clusters(
        patch.fields[FIELDS.index("olr")].reshape(patch.n_time, -1), assignment
    )
    a, b, c = pick_abc_avoiding_confounds(cluster_olr, max_other_corr=0.95, rng=np.random.default_rng(0))

    new_ds, true_edge_index = inject_causal_rule(pooled, a, b, c)
    assert new_ds.targets.shape == pooled.targets.shape
    assert true_edge_index.shape[0] == 2

    edge_index = make_fully_connected_edge_index(5)
    assert edge_index.shape == (2, 25)
