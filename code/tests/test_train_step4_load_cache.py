"""Regression test for a real bug found 2026-09-24: B22's fix added a
`cache["lead_time_days"]` read to 7 call sites, but `load_cache()` itself
never included that key in its returned dict (even though every real
.npz cache always has it) -- every one of those 7 call sites was broken
(KeyError) until this was caught by actually running a script end to
end, not by `pytest tests -q`, which never exercises `load_cache()`
against a real file. This test closes that gap so it can't regress
silently again.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from scripts.train_step4_single_place import load_cache


def _write_minimal_cache(path: Path, lead_time_days: int | None) -> None:
    n_samples, n_clusters, n_lags = 20, 5, 3
    n_features = n_lags * 7
    kwargs = dict(
        features=np.zeros((n_samples, n_clusters, n_features), dtype=np.float32),
        targets=np.zeros((n_samples, n_clusters), dtype=np.float32),
        sample_time_index=np.arange(n_samples, dtype=np.int64),
        sample_dates=np.array(
            [np.datetime64("2000-01-01") + np.timedelta64(i, "D") for i in range(n_samples)]
        ),
        edge_index=np.zeros((2, 0), dtype=np.int64),
        n_clusters=n_clusters,
        lags_days=np.array([0, 5, 10]),
    )
    if lead_time_days is not None:
        kwargs["lead_time_days"] = lead_time_days
    np.savez(path, **kwargs)


def test_load_cache_includes_lead_time_days(tmp_path: Path) -> None:
    cache_path = tmp_path / "cache.npz"
    _write_minimal_cache(cache_path, lead_time_days=7)

    cache = load_cache(cache_path)

    assert "lead_time_days" in cache
    assert cache["lead_time_days"] == 7
    assert isinstance(cache["lead_time_days"], int)


def test_load_cache_defaults_lead_time_days_for_legacy_cache_without_it(tmp_path: Path) -> None:
    """A cache built before lead_time_days was recorded at all (pre-B22)
    should not crash load_cache -- it should default to 1, the original
    single-lead cache's implicit value."""
    cache_path = tmp_path / "legacy_cache.npz"
    _write_minimal_cache(cache_path, lead_time_days=None)

    cache = load_cache(cache_path)

    assert cache["lead_time_days"] == 1
