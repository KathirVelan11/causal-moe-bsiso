"""Train/val/test split with two deliberate-shift validation windows (§8).

1. Abrupt shift: built around the 1997-98 El Nino (NOAA ONI onset ~Apr/May
   1997, peaked +2.4C, dissipated ~May/Jun 1998). Checks the drift detector
   (§4.4) fires *within* this narrow span.
2. Gradual shift: a later stretch of the record, checked against the
   documented slow MISO/BSISO drift in Arora et al. 2026 (~1 deg/decade
   westward migration of the convective centroid). Checks Channel 2
   (causal-set similarity) shows accumulating drift across those years
   while Channel 1 (error) stays comparatively flat.

These windows are held out as dedicated *test* spans for the drift
detector, separate from an ordinary chronological train/val/test split
used for plain forecast-accuracy evaluation (§6). Both are provided here;
which one a given experiment uses depends on what's being measured.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np

# --- Deliberate-shift windows (§8) -----------------------------------------

EL_NINO_1997_98 = (date(1997, 3, 1), date(1998, 7, 31))
"""Padded slightly around NOAA ONI onset (~Apr/May 1997) to dissipation
(~May/Jun 1998) so the pre-shift baseline and post-shift settling are both
visible to the detector, not just the peak."""

# Arora et al. 2026 describes a slow multi-decade drift, not a single dated
# window the way ONI gives El Nino a start/end. Use the last full decade of
# the record as the "gradual shift" span -- long enough for a ~1deg/decade
# migration to be measurable, recent enough to be furthest from the training
# baseline. (A JJASO convection-centroid-longitude trend test was tried as
# an empirical check -- not significant, p=0.16-0.59 -- but that's a
# limitation of that simple proxy, not evidence against this window; once
# the drift detector exists (§4.4, §9 step 5), Channel 2's causal-set
# similarity signal can locate real drift directly instead.)
GRADUAL_DRIFT_WINDOW = (date(2013, 1, 1), date(2022, 12, 31))


@dataclass
class DateRangeSplit:
    train_mask: np.ndarray  # (n_samples,) bool
    val_mask: np.ndarray
    test_mask: np.ndarray


def _to_datetime64(d: date) -> np.datetime64:
    return np.datetime64(d.isoformat())


def chronological_split(
    sample_dates: np.ndarray,
    val_start: date,
    test_start: date,
) -> DateRangeSplit:
    """Plain chronological split for ordinary forecast-accuracy evaluation
    (§6): train on everything before val_start, validate on
    [val_start, test_start), test on [test_start, end].

    sample_dates: (n_samples,) datetime64[ns] or datetime64[D], the date
        each sample's "today" (lag 0) corresponds to.
    """
    val_start64 = _to_datetime64(val_start)
    test_start64 = _to_datetime64(test_start)

    train_mask = sample_dates < val_start64
    val_mask = (sample_dates >= val_start64) & (sample_dates < test_start64)
    test_mask = sample_dates >= test_start64

    return DateRangeSplit(train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)


def deliberate_shift_mask(
    sample_dates: np.ndarray,
    window: tuple[date, date],
) -> np.ndarray:
    """Boolean mask selecting samples whose date falls inside a deliberate-
    shift window (either EL_NINO_1997_98 or GRADUAL_DRIFT_WINDOW)."""
    start64 = _to_datetime64(window[0])
    end64 = _to_datetime64(window[1])
    return (sample_dates >= start64) & (sample_dates <= end64)
