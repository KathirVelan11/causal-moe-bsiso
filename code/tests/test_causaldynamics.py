import os

import numpy as np
import pytest
import xarray as xr

from causal_moe.data.causaldynamics import (
    CausalDynamicsGraph,
    adjacency_to_edge_index,
    build_windowed_dataset_from_graph,
    load_enso_modes_graphs,
)
from causal_moe.data.windows import LEGACY_LAGS_DAYS

REAL_DATA_ROOT = os.path.join(
    os.path.dirname(__file__), "..", "data_external", "causaldynamics", "extracted_inputs"
)


def make_fake_graph(n_time: int = 30, n_nodes: int = 4, n_systems: int = 2) -> CausalDynamicsGraph:
    """Known, checkable pattern: value(system, t, node) = system*10000 + t*100 + node."""
    ts = np.empty((n_systems, n_time, n_nodes), dtype=np.float32)
    for s in range(n_systems):
        for t in range(n_time):
            for n in range(n_nodes):
                ts[s, t, n] = s * 10000 + t * 100 + n

    adjacency = np.zeros((n_nodes, n_nodes), dtype=np.float32)
    adjacency[0, 0] = 1.0  # self-loop
    adjacency[0, 1] = 1.0
    adjacency[1, 0] = 1.0

    return CausalDynamicsGraph(name="FAKE", time_series=ts, adjacency=adjacency)


def test_build_windowed_dataset_from_graph_shapes():
    graph = make_fake_graph(n_time=30, n_nodes=4, n_systems=2)
    ds = build_windowed_dataset_from_graph(graph, lead_time_steps=1)

    # valid t per system in [10, 30-1-1] = [10, 28] -> 19 samples/system * 2 systems
    assert ds.features.shape == (38, 4, 3)
    assert ds.targets.shape == (38, 4)
    assert ds.lags_days == LEGACY_LAGS_DAYS


def test_build_windowed_dataset_from_graph_lag_values_correct():
    graph = make_fake_graph(n_time=30, n_nodes=4, n_systems=2)
    ds = build_windowed_dataset_from_graph(graph, lead_time_steps=1)

    # first sample of system 0: t_start=10
    assert ds.sample_time_index[0] == 10
    assert ds.sample_system_index[0] == 0  # type: ignore[attr-defined]

    node = 2

    def expected(system, t):
        return system * 10000 + t * 100 + node

    feat = ds.features[0, node]  # [lag0, lag5, lag10]
    assert feat[0] == pytest.approx(expected(0, 10))
    assert feat[1] == pytest.approx(expected(0, 5))
    assert feat[2] == pytest.approx(expected(0, 0))

    assert ds.targets[0, node] == pytest.approx(expected(0, 11))


def test_build_windowed_dataset_from_graph_systems_not_mixed():
    graph = make_fake_graph(n_time=30, n_nodes=4, n_systems=2)
    ds = build_windowed_dataset_from_graph(graph, lead_time_steps=1)

    per_system = 19  # (30 - 10 - 1) + 1
    assert ds.sample_system_index[0] == 0  # type: ignore[attr-defined]
    assert ds.sample_system_index[per_system - 1] == 0  # type: ignore[attr-defined]
    assert ds.sample_system_index[per_system] == 1  # type: ignore[attr-defined]

    # system-1 first sample's lag10 value must come from system 1's own t=0,
    # never system 0's data (no cross-system leakage in the lag window)
    node = 0
    feat = ds.features[per_system, node]
    assert feat[2] == pytest.approx(1 * 10000 + 0 * 100 + node)


def test_build_windowed_dataset_from_graph_rejects_insufficient_timesteps():
    graph = make_fake_graph(n_time=5, n_nodes=4, n_systems=1)
    with pytest.raises(ValueError):
        build_windowed_dataset_from_graph(graph, lead_time_steps=1)


def test_adjacency_to_edge_index_includes_self_loops():
    graph = make_fake_graph()
    edge_index = adjacency_to_edge_index(graph.adjacency)

    assert edge_index.shape[0] == 2
    pairs = set(zip(edge_index[0].tolist(), edge_index[1].tolist()))
    assert (0, 0) in pairs  # self-loop kept
    assert (0, 1) in pairs
    assert (1, 0) in pairs
    assert len(pairs) == 3


def test_load_enso_modes_graphs_rejects_missing_dir(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_enso_modes_graphs(str(tmp_path))


@pytest.mark.skipif(
    not os.path.isdir(REAL_DATA_ROOT), reason="real CausalDynamics download not present"
)
def test_load_enso_modes_graphs_real_data():
    graphs = load_enso_modes_graphs(REAL_DATA_ROOT)

    assert len(graphs) == 11
    names = {g.name for g in graphs}
    assert "NONE" in names
    assert "AO" in names

    for g in graphs:
        assert g.n_nodes == 10
        assert g.n_systems == 10
        assert g.n_time == 1000
        assert g.adjacency.shape == (10, 10)
        assert set(np.unique(g.adjacency).tolist()) <= {0.0, 1.0}
        assert not np.isnan(g.time_series).any()

    none_graph = next(g for g in graphs if g.name == "NONE")
    # "NONE" = the near-empty baseline coupling mode: 2 nodes each with a
    # self-loop plus mutual coupling (4 edges), rest fully isolated.
    assert none_graph.adjacency.sum() == 4


@pytest.mark.skipif(
    not os.path.isdir(REAL_DATA_ROOT), reason="real CausalDynamics download not present"
)
def test_real_graph_reshapes_end_to_end():
    graphs = load_enso_modes_graphs(REAL_DATA_ROOT)
    ao = next(g for g in graphs if g.name == "AO")

    ds = build_windowed_dataset_from_graph(ao, lead_time_steps=1)

    per_system = ao.n_time - max(LEGACY_LAGS_DAYS) - 1  # t in [10, 998]
    assert ds.features.shape == (per_system * ao.n_systems, 10, 3)
    assert ds.targets.shape == (per_system * ao.n_systems, 10)
    assert not np.isnan(ds.features).any()
    assert not np.isnan(ds.targets).any()

    edge_index = adjacency_to_edge_index(ao.adjacency)
    assert edge_index.shape[0] == 2
    assert edge_index.shape[1] == int(ao.adjacency.sum())
