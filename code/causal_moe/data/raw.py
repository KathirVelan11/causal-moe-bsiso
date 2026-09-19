"""Loads the 6 raw BSISO fields + SST ocean mask straight off disk.

Dataset layout (see Causal_MoE_Architecture.md §3):
    Dataset/rt_preprocessed_archive/{field}_rt_1979_2022.npz
        data: (16071, 25, 144) float32 -- already anomaly-preprocessed
        time: (16071,) datetime64[ns]
        lat:  (25,) float32   -- 30 .. -30
        lon:  (144,) float32  -- 0 .. 357.5
    Dataset/rt_preprocessed_archive/parameters/sst_ocean_mask.npz
        ocean_mask: (25, 144) bool
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

FIELDS = ("sst", "h850", "u200", "pw", "u850", "olr")

# Index of the field that is the forecast target (§8: OLR, hardest to predict).
TARGET_FIELD = "olr"


def default_dataset_root() -> Path:
    """The dataset root, resolved relative to this repo's known layout.

    code/causal_moe/data/raw.py -> code/ -> "DIR_GNN + DyMoE"/ -> Dataset/
    """
    return Path(__file__).resolve().parents[2].parent / "Dataset"


@dataclass
class BSISORawData:
    """All 6 fields stacked, plus grid metadata and the ocean mask.

    fields: (n_fields, T, 25, 144) float32, order = FIELDS
    ocean_mask: (25, 144) bool, True = ocean (valid SST)
    time: (T,) datetime64[ns]
    lat: (25,) float32
    lon: (144,) float32
    """

    fields: np.ndarray
    ocean_mask: np.ndarray
    time: np.ndarray
    lat: np.ndarray
    lon: np.ndarray
    field_names: tuple = field(default=FIELDS)

    @property
    def n_lat(self) -> int:
        return self.fields.shape[2]

    @property
    def n_lon(self) -> int:
        return self.fields.shape[3]

    @property
    def n_places(self) -> int:
        return self.n_lat * self.n_lon

    @property
    def n_time(self) -> int:
        return self.fields.shape[1]

    def field_index(self, name: str) -> int:
        return self.field_names.index(name)


def load_raw_dataset(root: Path | None = None) -> BSISORawData:
    """Loads all 6 fields + ocean mask from disk. No preprocessing beyond
    what already shipped with the dataset (climatology/harmonics/normalize
    already applied upstream, per §3)."""
    root = Path(root) if root is not None else default_dataset_root()
    archive = root / "rt_preprocessed_archive"

    arrays = []
    time_ref = None
    lat_ref = None
    lon_ref = None
    for name in FIELDS:
        npz = np.load(archive / f"{name}_rt_1979_2022.npz")
        data = npz["data"]
        if data.shape != (16071, 25, 144):
            raise ValueError(f"{name}: unexpected shape {data.shape}")
        if time_ref is None:
            time_ref, lat_ref, lon_ref = npz["time"], npz["lat"], npz["lon"]
        else:
            if not np.array_equal(npz["time"], time_ref):
                raise ValueError(f"{name}: time axis does not match other fields")
            if not np.array_equal(npz["lat"], lat_ref) or not np.array_equal(npz["lon"], lon_ref):
                raise ValueError(f"{name}: lat/lon grid does not match other fields")
        arrays.append(data.astype(np.float32))

    fields = np.stack(arrays, axis=0)  # (6, T, 25, 144)

    mask_npz = np.load(archive / "parameters" / "sst_ocean_mask.npz")
    ocean_mask = mask_npz["ocean_mask"].astype(bool)
    if ocean_mask.shape != (25, 144):
        raise ValueError(f"ocean_mask: unexpected shape {ocean_mask.shape}")

    # SST NaNs should sit exactly over land (§8: fill with sentinel + is_ocean flag).
    sst = fields[FIELDS.index("sst")]
    nan_cells = np.isnan(sst).any(axis=0)
    if not np.array_equal(nan_cells, ~ocean_mask):
        n_mismatch = int((nan_cells != ~ocean_mask).sum())
        raise ValueError(
            f"sst NaN footprint does not match ocean_mask exactly ({n_mismatch} cells differ)"
        )

    return BSISORawData(
        fields=fields,
        ocean_mask=ocean_mask,
        time=time_ref,
        lat=lat_ref,
        lon=lon_ref,
    )
