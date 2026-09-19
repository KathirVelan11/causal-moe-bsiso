"""Loads CausalDynamics Tier-3 (Climate) validation graphs and reshapes them
into the same WindowedDataset/LatticeMesh shapes the real mesh pipeline uses
(§5.1, §9 step 2).

Scope (confirmed with user, 2026-09-18): only the 11 `coupled_enso_modes`
graphs (10 scalar nodes each, dim=1) -- the 12th graph, `coupled_atmos_ocean`,
has 4 nodes but each node is a 374-dim vector (a spatial field, not a
scalar), a structurally different case than our own mesh's per-channel
scalars. Skipped for this step, revisit later if needed.

Source layout (extracted from HuggingFace kausable/CausalDynamics
inputs/climate.tar.gz, verified by direct inspection 2026-09-18):

    climate/coupled_enso_modes/data/{NAME}_N10_T1000.nc
        time_series       (time=1000, system=10, node=10, dim=1) float32
        adjacency_matrix  (node_in=10, node_out=10) float64 -- {0,1}, ground
            truth causal graph. Symmetric in this dataset (undirected), and
            includes nonzero diagonal (self-loops = real autocorrelation,
            kept per user decision 2026-09-18 -- not dropped like the real
            mesh's cluster self-loops).

`system` = 10 independent replicate trajectories of the *same* underlying
graph (same adjacency_matrix, different random initial conditions/noise) --
kept as separate samples, never concatenated across the system axis, since
that would fabricate a fake lag-window continuity across a trajectory
boundary that doesn't physically exist.
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass

import numpy as np
import xarray as xr

from causal_moe.data.windows import LAGS_DAYS, WindowedDataset

ENSO_MODES_GLOB = "climate/coupled_enso_modes/data/*_N10_T1000.nc"


@dataclass
class CausalDynamicsGraph:
    """One graph's raw contents, straight off disk -- pre-windowing.

    time_series: (n_systems, T, n_nodes) float32 -- squeezed dim axis (dim=1
        for all coupled_enso_modes graphs, verified on load)
    adjacency: (n_nodes, n_nodes) float32 -- {0,1} ground truth, self-loops
        included
    name: graph identifier, e.g. "AO", "NONE", "SASD" -- the coupling mode
        tag from the filename
    """

    name: str
    time_series: np.ndarray
    adjacency: np.ndarray

    @property
    def n_nodes(self) -> int:
        return self.adjacency.shape[0]

    @property
    def n_systems(self) -> int:
        return self.time_series.shape[0]

    @property
    def n_time(self) -> int:
        return self.time_series.shape[1]


def load_enso_modes_graphs(root: str) -> list[CausalDynamicsGraph]:
    """Loads all `coupled_enso_modes` graphs from an extracted CausalDynamics
    climate/ directory.

    root: directory containing `climate/coupled_enso_modes/data/*.nc`
        (i.e. the extraction target of inputs/climate.tar.gz).
    """
    pattern = os.path.join(root, ENSO_MODES_GLOB)
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"no coupled_enso_modes .nc files found under {pattern}")

    graphs = []
    for path in paths:
        ds = xr.open_dataset(path)
        try:
            if ds["dim"].size != 1:
                raise ValueError(f"{path}: expected dim=1 (scalar nodes), got {ds['dim'].size}")
            ts = ds["time_series"].values  # (time, system, node, dim=1)
            ts = ts[..., 0]  # (time, system, node)
            ts = np.moveaxis(ts, 1, 0)  # (system, time, node)
            adj = ds["adjacency_matrix"].values.astype(np.float32)  # (node_in, node_out)
            if adj.shape[0] != adj.shape[1]:
                raise ValueError(f"{path}: adjacency_matrix not square: {adj.shape}")
            if not np.array_equal(np.unique(adj), np.array([0.0, 1.0])) and not np.array_equal(
                np.unique(adj), np.array([1.0])
            ):
                uniq = np.unique(adj)
                if not set(uniq.tolist()) <= {0.0, 1.0}:
                    raise ValueError(f"{path}: adjacency_matrix has non-binary values: {uniq}")
        finally:
            ds.close()

        name = os.path.basename(path).split("_N10_T1000.nc")[0]
        graphs.append(
            CausalDynamicsGraph(
                name=name,
                time_series=ts.astype(np.float32),
                adjacency=adj,
            )
        )

    return graphs


def build_windowed_dataset_from_graph(
    graph: CausalDynamicsGraph,
    lead_time_steps: int = 1,
) -> WindowedDataset:
    """Reshapes one CausalDynamics graph into the same WindowedDataset shape
    build_windowed_dataset (§4.1 windows.py) produces for real BSISO data:
    3 lagged snapshots per node (t, t-5, t-10), single scalar channel (no
    is_ocean flag -- not applicable to synthetic single-field data), target
    = each node's own value `lead_time_steps` steps ahead.

    Systems (replicates) are windowed independently, then concatenated along
    the sample axis -- each system contributes its own set of valid
    timesteps, never crossing a system boundary.

    features: (n_samples, n_nodes, 3) float32 -- 3 lags x 1 channel
    targets:  (n_samples, n_nodes) float32
    sample_time_index: (n_samples,) int64 -- the within-system time index of
        the "today" (lag 0) sample; NOT globally unique across systems (use
        alongside a separate system-id array if that distinction matters)
    """
    if lead_time_steps < 1:
        raise ValueError("lead_time_steps must be >= 1")

    max_lag = max(LAGS_DAYS)
    n_time = graph.n_time
    t_start = max_lag
    t_end = n_time - 1 - lead_time_steps
    if t_end < t_start:
        raise ValueError(
            f"not enough timesteps: n_time={n_time}, max_lag={max_lag}, "
            f"lead_time_steps={lead_time_steps}"
        )

    n_nodes = graph.n_nodes
    per_system_samples = t_end - t_start + 1
    n_systems = graph.n_systems
    n_samples = per_system_samples * n_systems
    n_features = len(LAGS_DAYS)  # 3 lags x 1 channel

    features = np.empty((n_samples, n_nodes, n_features), dtype=np.float32)
    targets = np.empty((n_samples, n_nodes), dtype=np.float32)
    sample_time_index = np.empty((n_samples,), dtype=np.int64)
    sample_system_index = np.empty((n_samples,), dtype=np.int64)

    row = 0
    for sys_idx in range(n_systems):
        series = graph.time_series[sys_idx]  # (T, n_nodes)
        for t in range(t_start, t_end + 1):
            for lag_pos, lag in enumerate(LAGS_DAYS):
                features[row, :, lag_pos] = series[t - lag, :]
            targets[row, :] = series[t + lead_time_steps, :]
            sample_time_index[row] = t
            sample_system_index[row] = sys_idx
            row += 1

    ds = WindowedDataset(
        features=features,
        targets=targets,
        sample_time_index=sample_time_index,
        lead_time_days=lead_time_steps,
        lags_days=LAGS_DAYS,
        field_names=("value",),
    )
    ds.sample_system_index = sample_system_index  # type: ignore[attr-defined]
    return ds


def adjacency_to_edge_index(adjacency: np.ndarray) -> np.ndarray:
    """Dense {0,1} adjacency -> PyG-style (2, n_edges) edge_index, directed
    pairs, self-loops included (kept per §5.1 decision -- these are real
    ground-truth causal self-edges, not artifacts)."""
    src, dst = np.nonzero(adjacency)
    return np.stack([src.astype(np.int64), dst.astype(np.int64)], axis=0)
