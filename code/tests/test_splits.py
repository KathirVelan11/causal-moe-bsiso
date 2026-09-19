from datetime import date

import numpy as np

from causal_moe.data.splits import (
    EL_NINO_1997_98,
    GRADUAL_DRIFT_WINDOW,
    chronological_split,
    deliberate_shift_mask,
)


def test_chronological_split_partitions_all_samples():
    dates = np.array(
        [np.datetime64("1990-01-01") + np.timedelta64(i, "D") for i in range(1000)]
    )
    split = chronological_split(dates, val_start=date(1991, 1, 1), test_start=date(1992, 1, 1))

    assert (split.train_mask | split.val_mask | split.test_mask).all()
    assert not (split.train_mask & split.val_mask).any()
    assert not (split.val_mask & split.test_mask).any()
    assert not (split.train_mask & split.test_mask).any()

    assert dates[split.train_mask].max() < np.datetime64("1991-01-01")
    assert dates[split.val_mask].min() >= np.datetime64("1991-01-01")
    assert dates[split.val_mask].max() < np.datetime64("1992-01-01")
    assert dates[split.test_mask].min() >= np.datetime64("1992-01-01")


def test_el_nino_window_covers_1997_98():
    dates = np.array(
        [np.datetime64("1996-01-01") + np.timedelta64(i, "D") for i in range(1000)]
    )
    mask = deliberate_shift_mask(dates, EL_NINO_1997_98)

    assert mask.any()
    selected = dates[mask]
    assert selected.min() >= np.datetime64("1997-03-01")
    assert selected.max() <= np.datetime64("1998-07-31")
    # peak of the event (mid-1997) should be included
    assert np.datetime64("1997-06-15") in selected


def test_gradual_drift_window_is_last_decade():
    assert GRADUAL_DRIFT_WINDOW == (date(2013, 1, 1), date(2022, 12, 31))

    dates = np.array(
        [np.datetime64("2010-01-01") + np.timedelta64(i, "D") for i in range(5000)]
    )
    mask = deliberate_shift_mask(dates, GRADUAL_DRIFT_WINDOW)
    selected = dates[mask]
    assert selected.min() >= np.datetime64("2013-01-01")
    assert selected.max() <= np.datetime64("2022-12-31")
