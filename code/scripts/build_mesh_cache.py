"""Builds the cluster mesh + windowed dataset once and caches to disk
(.npz), so later stages (splitter, experts, ...) don't re-pay the ~11s
raw-load + clustering + windowing rebuild on every run.

Pipeline: load raw 3,600-cell data -> cluster into 50 regions (§4.1, §8) ->
build raw-cell lattice -> derive cluster adjacency -> build raw-resolution
lagged windows -> pool to cluster-level.

Usage:
    python scripts/build_mesh_cache.py [--lead-time-days N] [--n-clusters N] [--out PATH]

Note: uses plain np.savez (uncompressed), not savez_compressed -- the
pooled cluster-level dataset is small (~tens of MB, 72x smaller than the
raw-cell version), so compression isn't needed and isn't worth the extra
memory pressure of compressing a large buffer in one process.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from causal_moe.data.clustering import N_CLUSTERS, compute_cluster_assignment
from causal_moe.data.mesh import build_cluster_mesh, build_raw_lattice_mesh
from causal_moe.data.raw import FIELDS, load_raw_dataset
from causal_moe.data.windows import build_windowed_dataset, pool_windowed_dataset_to_clusters


def default_cache_path(lead_time_days: int, n_clusters: int) -> Path:
    cache_dir = Path(__file__).resolve().parents[1] / "cache"
    cache_dir.mkdir(exist_ok=True)
    return cache_dir / f"windowed_clustered{n_clusters}_lead{lead_time_days}.npz"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lead-time-days", type=int, default=1)
    parser.add_argument("--n-clusters", type=int, default=N_CLUSTERS)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    out_path = args.out or default_cache_path(args.lead_time_days, args.n_clusters)

    t0 = time.time()
    raw = load_raw_dataset()
    print(f"raw loaded: {time.time() - t0:.1f}s")

    t0 = time.time()
    olr = raw.fields[FIELDS.index("olr")]
    assignment = compute_cluster_assignment(olr, n_clusters=args.n_clusters)
    print(f"clustering: {time.time() - t0:.1f}s, {args.n_clusters} clusters")

    raw_mesh = build_raw_lattice_mesh(raw.lat, raw.lon)
    cluster_mesh = build_cluster_mesh(raw_mesh, assignment)
    print(f"cluster mesh: {cluster_mesh.n_places} places, {cluster_mesh.n_edges} directed edges")

    t0 = time.time()
    raw_ds = build_windowed_dataset(raw, lead_time_days=args.lead_time_days)
    print(f"raw windowed dataset built: {time.time() - t0:.1f}s, features {raw_ds.features.shape}")

    t0 = time.time()
    ds = pool_windowed_dataset_to_clusters(raw_ds, assignment)
    print(f"pooled to clusters: {time.time() - t0:.1f}s, features {ds.features.shape}")

    np.savez(
        out_path,
        features=ds.features,
        targets=ds.targets,
        sample_time_index=ds.sample_time_index,
        sample_dates=raw.time[ds.sample_time_index],
        lead_time_days=ds.lead_time_days,
        edge_index=cluster_mesh.edge_index,
        node_lat=cluster_mesh.lat,
        node_lon=cluster_mesh.lon,
        n_clusters=args.n_clusters,
        cluster_labels=assignment.labels,  # (3600,) raw-cell -> cluster id, for later inspection
    )
    print(f"cached to {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
