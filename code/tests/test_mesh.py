import numpy as np
import pytest

from causal_moe.data.mesh import build_raw_lattice_mesh, node_id


def test_node_id_row_major():
    assert node_id(0, 0, n_lon=144) == 0
    assert node_id(0, 1, n_lon=144) == 1
    assert node_id(1, 0, n_lon=144) == 144
    assert node_id(24, 143, n_lon=144) == 24 * 144 + 143


def test_small_grid_edge_count_no_wraparound_needed():
    # 2x3 grid, no wraparound triggered since every column already has an
    # east neighbor except the wraparound one -- use this to hand-verify
    # exact edge count.
    lat = np.array([10.0, 0.0], dtype=np.float32)  # n_lat=2
    lon = np.array([0.0, 120.0, 240.0], dtype=np.float32)  # n_lon=3
    mesh = build_raw_lattice_mesh(lat, lon)

    assert mesh.n_places == 6

    # East-West edges (with wraparound): each row has n_lon edges (3 pairs
    # incl. wraparound) -> 2 rows * 3 = 6 undirected pairs.
    # North-South edges (no wraparound): (n_lat-1) * n_lon = 1 * 3 = 3 pairs.
    # Total undirected = 9, stored bidirectionally = 18 directed entries.
    assert mesh.n_edges == 18


def test_longitude_wraparound_present():
    lat = np.array([10.0, 0.0], dtype=np.float32)
    lon = np.array([0.0, 120.0, 240.0], dtype=np.float32)
    mesh = build_raw_lattice_mesh(lat, lon)

    # node (0,0)=0 and node (0,2)=2 should be connected via wraparound.
    edges = set(zip(mesh.edge_index[0].tolist(), mesh.edge_index[1].tolist()))
    assert (0, 2) in edges
    assert (2, 0) in edges


def test_no_latitude_wraparound():
    lat = np.array([10.0, 0.0], dtype=np.float32)
    lon = np.array([0.0, 120.0, 240.0], dtype=np.float32)

    # node (0,0)=0 (top row) and node (1,0)=3 (bottom row) should NOT
    # wrap around to connect directly -- only adjacent rows connect, and
    # there are only 2 rows here so 0-3 is a legitimate adjacent-row edge.
    # Use a 3-row grid to properly test no-wraparound between row 0 and row 2.
    lat3 = np.array([20.0, 10.0, 0.0], dtype=np.float32)
    mesh3 = build_raw_lattice_mesh(lat3, lon)
    edges3 = set(zip(mesh3.edge_index[0].tolist(), mesh3.edge_index[1].tolist()))
    top_row_node = node_id(0, 0, n_lon=3)
    bottom_row_node = node_id(2, 0, n_lon=3)
    assert (top_row_node, bottom_row_node) not in edges3
    assert (bottom_row_node, top_row_node) not in edges3


def test_full_grid_matches_dataset_shape():
    lat = np.linspace(30.0, -30.0, 25).astype(np.float32)
    lon = np.arange(0.0, 360.0, 2.5).astype(np.float32)
    assert lon.shape[0] == 144
    mesh = build_raw_lattice_mesh(lat, lon)

    assert mesh.n_places == 3600
    # EW pairs: 25 * 144 (each row fully wraps) = 3600
    # NS pairs: 24 * 144 = 3456
    # total undirected = 7056, directed = 14112
    assert mesh.n_edges == 14112

    # Every node should have degree 4 (interior rows) or 3 (top/bottom row,
    # no south/north neighbor there).
    degrees = np.zeros(mesh.n_places, dtype=np.int64)
    for u in mesh.edge_index[0]:
        degrees[u] += 1
    assert (degrees[144:3600 - 144] == 4).all()  # interior rows
    assert (degrees[:144] == 3).all()  # top row: no north neighbor
    assert (degrees[3600 - 144:] == 3).all()  # bottom row: no south neighbor
