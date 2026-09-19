import numpy as np
import pytest

from causal_moe.data.raw import FIELDS, TARGET_FIELD, default_dataset_root, load_raw_dataset


def _dataset_available() -> bool:
    return default_dataset_root().exists()


pytestmark = pytest.mark.skipif(
    not _dataset_available(), reason="real BSISO dataset not present on this machine"
)


def test_load_real_dataset_shapes():
    raw = load_raw_dataset()

    assert raw.fields.shape == (6, 16071, 25, 144)
    assert raw.ocean_mask.shape == (25, 144)
    assert raw.time.shape == (16071,)
    assert raw.lat.shape == (25,)
    assert raw.lon.shape == (144,)

    assert raw.n_places == 3600
    assert raw.n_time == 16071
    assert TARGET_FIELD == "olr"
    assert raw.field_index("olr") == FIELDS.index("olr")


def test_real_dataset_time_range_and_no_gaps():
    raw = load_raw_dataset()
    assert raw.time[0] == np.datetime64("1979-01-01")
    assert raw.time[-1] == np.datetime64("2022-12-31")
    diffs = np.diff(raw.time).astype("timedelta64[D]").astype(int)
    assert (diffs == 1).all()


def test_real_dataset_sst_nan_matches_ocean_mask():
    raw = load_raw_dataset()
    sst = raw.fields[FIELDS.index("sst")]
    nan_cells = np.isnan(sst).any(axis=0)
    assert np.array_equal(nan_cells, ~raw.ocean_mask)
    # matches the count confirmed manually against the dataset (§3): ~942 land cells
    assert int((~raw.ocean_mask).sum()) == 942


def test_real_dataset_no_nan_outside_sst():
    raw = load_raw_dataset()
    for i, name in enumerate(FIELDS):
        if name == "sst":
            continue
        assert not np.isnan(raw.fields[i]).any(), f"{name} has unexpected NaNs"
