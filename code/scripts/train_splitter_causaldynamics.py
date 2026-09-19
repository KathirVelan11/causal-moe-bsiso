"""§9 step 2 / §5.1: trains the DIR-GNN splitter (§4.2) on CausalDynamics
Tier-3 climate graphs and checks causal-edge recovery (precision/recall)
against the bundled ground-truth adjacency -- the first real training run
in this project.

Scope (confirmed with user, 2026-09-18): starts with 2 graphs (AO, NONE) --
a mid-complexity case and a near-trivial baseline case -- before expanding
to all 11 coupled_enso_modes graphs.

Memory bank (confirmed with user, 2026-09-18): swaps are drawn from the
SAME system replicate's own history only, both time directions -- not
pooled across the 10 replicate systems.

Usage:
    python scripts/train_splitter_causaldynamics.py [--graphs AO,NONE]
        [--epochs N] [--r 0.5] [--n-swaps 6] [--lr 1e-3]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from causal_moe.data.causaldynamics import (
    adjacency_to_edge_index,
    build_windowed_dataset_from_graph,
    load_enso_modes_graphs,
)
from causal_moe.splitter.dirgnn import DIRGNNSplitter, LambdaWarmupSchedule

DATA_ROOT = Path(__file__).resolve().parents[1] / "data_external" / "causaldynamics" / "extracted_inputs"


def make_fully_connected_edge_index(n_nodes: int) -> torch.Tensor:
    """Candidate edge set for the splitter: every possible node pair incl.
    self-loops (n^2 pairs) -- no physical pre-filter exists for this
    synthetic data, confirmed with user 2026-09-18."""
    src = []
    dst = []
    for i in range(n_nodes):
        for j in range(n_nodes):
            src.append(i)
            dst.append(j)
    return torch.tensor([src, dst], dtype=torch.int64)


def build_memory_bank(ds, n_nodes: int):
    """Groups windowed-dataset rows by system, so swap candidates for a
    given anchor only ever come from that SAME system's other timesteps
    (both time directions) -- confirmed scope 2026-09-18.

    Returns: dict[system_id] -> (features (n_t, n_nodes, 3), n_t)
    """
    system_index = ds.sample_system_index  # type: ignore[attr-defined]
    systems = np.unique(system_index)
    bank = {}
    for sys_id in systems:
        mask = system_index == sys_id
        bank[int(sys_id)] = ds.features[mask]  # (n_t_this_system, n_nodes, 3)
    return bank


def build_generator_windows(
    graph, ds, generator_window_len: int
) -> np.ndarray:
    """Precomputes a trailing history window per windowed-dataset sample,
    for the rationale generator's fixed cross-time blind-spot fix
    (§10 step 2, 2026-09-18) -- vectorized with sliding_window_view, no
    per-sample Python loop (this dev machine is CPU-only, flagged as a hard
    constraint).

    graph.time_series: (n_systems, T, n_nodes) raw values (channel=1,
        scalar per node -- CausalDynamics enso_modes graphs).
    ds.sample_time_index / ds.sample_system_index: which (system, t) each
        windowed-dataset row corresponds to (t = the row's lag-0 "today").

    Returns: (n_samples, n_nodes, generator_window_len, 1) float32 -- each
        sample's trailing window [t-W+1, ..., t] per node, ending at that
        sample's own anchor timestep (matches RationaleGenerator's expected
        (n_nodes, window_len, in_channels) shape once indexed per-sample).
    """
    ts = graph.time_series  # (n_systems, T, n_nodes)
    n_systems, T, n_nodes = ts.shape
    W = generator_window_len

    # left-pad each system's series with W-1 copies of its own first value
    # so every valid windowed-dataset row (t >= max_lag >= 0) still has a
    # full-length window, without needing per-row branching.
    pad = np.repeat(ts[:, :1, :], W - 1, axis=1) if W > 1 else ts[:, :0, :]
    padded = np.concatenate([pad, ts], axis=1)  # (n_systems, T + W - 1, n_nodes)

    # sliding_window_view over the time axis: (n_systems, T, n_nodes, W)
    windows = np.lib.stride_tricks.sliding_window_view(padded, W, axis=1)
    # -> windows[sys, t, node, :] is [t-W+1, ..., t] in padded-index terms,
    # which lines up with original t after the left-pad.

    sys_idx = ds.sample_system_index  # type: ignore[attr-defined]
    t_idx = ds.sample_time_index  # (n_samples,) original-time "today" index

    gen_windows = windows[sys_idx, t_idx]  # (n_samples, n_nodes, W)
    return gen_windows[..., None].astype(np.float32)  # (n_samples, n_nodes, W, 1)


def sample_swaps(bank_entry: np.ndarray, anchor_row: int, n_swaps: int, rng: np.random.Generator) -> np.ndarray:
    """Draws n_swaps other timesteps from the SAME system's history, both
    time directions (§4.2/§8) -- excludes the anchor's own row."""
    n_t = bank_entry.shape[0]
    candidates = np.delete(np.arange(n_t), anchor_row)
    chosen = rng.choice(candidates, size=min(n_swaps, candidates.shape[0]), replace=False)
    return bank_entry[chosen]  # (n_swaps, n_nodes, 3)


def precision_recall(predicted_edges: set, true_edges: set) -> tuple[float, float]:
    if len(predicted_edges) == 0:
        precision = 0.0
    else:
        precision = len(predicted_edges & true_edges) / len(predicted_edges)
    if len(true_edges) == 0:
        recall = 1.0 if len(predicted_edges) == 0 else 0.0
    else:
        recall = len(predicted_edges & true_edges) / len(true_edges)
    return precision, recall


def train_one_graph(
    graph,
    epochs: int,
    r: float,
    n_swaps: int,
    lr: float,
    batch_size: int,
    generator_window_len: int = 10,
    entropy_weight: float = 0.0,
    prior_rate_weight: float = 0.0,
    seed: int = 0,
) -> dict:
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    ds = build_windowed_dataset_from_graph(graph, lead_time_steps=1)
    n_nodes = graph.n_nodes
    n_samples = ds.features.shape[0]

    edge_index = make_fully_connected_edge_index(n_nodes)
    true_edge_index = adjacency_to_edge_index(graph.adjacency)
    true_edges = set(zip(true_edge_index[0].tolist(), true_edge_index[1].tolist()))

    bank = build_memory_bank(ds, n_nodes)
    # Rationale generator's cross-time fix (§10 step 2, 2026-09-18):
    # precomputed once, vectorized -- no per-sample windowing in the loop.
    gen_windows_all = build_generator_windows(graph, ds, generator_window_len)
    gen_windows_t = torch.from_numpy(gen_windows_all)  # (n_samples, n_nodes, W, 1)

    model = DIRGNNSplitter(
        in_channels=3, hidden_channels=16, r=r,
        generator_window_len=generator_window_len, generator_in_channels=1,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # step increments once per SAMPLE (inside the batch loop below), not
    # once per batch -- total_steps must match that same unit or the warm-up
    # schedule finishes far too early.
    total_steps = epochs * n_samples
    schedule = LambdaWarmupSchedule(lambda_max=1.0, total_steps=max(total_steps, 1))

    system_index = ds.sample_system_index  # type: ignore[attr-defined]
    x_all = torch.from_numpy(ds.features)  # (n_samples, n_nodes, 3)
    y_all = torch.from_numpy(ds.targets)  # (n_samples, n_nodes)

    step = 0
    history = []
    t0 = time.time()
    for epoch in range(epochs):
        order = rng.permutation(n_samples)
        epoch_loss = 0.0
        epoch_var = 0.0
        n_batches = 0
        for start in range(0, n_samples - batch_size + 1, batch_size):
            idx_batch = order[start : start + batch_size]
            batch_loss = torch.zeros(())
            batch_var = 0.0
            optimizer.zero_grad()
            for idx in idx_batch:
                sys_id = int(system_index[idx])
                # position of this row within its own system's bank entry
                row_in_system = int((np.nonzero(system_index[: idx + 1] == sys_id)[0]).shape[0] - 1)
                env = sample_swaps(bank[sys_id], row_in_system, n_swaps, rng)
                env_t = torch.from_numpy(env)

                lam = schedule.value(step)
                out = model.compute_loss(
                    x_all[idx], y_all[idx], env_t, edge_index, lam,
                    x_generator_window=gen_windows_t[idx],
                    entropy_weight=entropy_weight,
                    prior_rate_weight=prior_rate_weight,
                )
                batch_loss = batch_loss + out.total_loss
                batch_var += out.variance_risk.item()
                step += 1

            batch_loss = batch_loss / len(idx_batch)
            batch_loss.backward()
            optimizer.step()

            epoch_loss += batch_loss.item()
            epoch_var += batch_var / len(idx_batch)
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        avg_var = epoch_var / max(n_batches, 1)
        history.append((epoch, avg_loss, avg_var))
        print(
            f"  [{graph.name}] epoch {epoch+1}/{epochs}  loss={avg_loss:.4f}  "
            f"var={avg_var:.4f}  lambda={schedule.value(step):.3f}"
        )

    elapsed = time.time() - t0

    # Final edge recovery check: average edge scores over a sample of
    # timesteps (not just one), then apply the same top-r selection used
    # in training.
    model.eval()
    with torch.no_grad():
        sample_idx = rng.choice(n_samples, size=min(200, n_samples), replace=False)
        all_scores = []
        for idx in sample_idx:
            split = model.split(gen_windows_t[idx], edge_index)
            all_scores.append(split.edge_scores.numpy())
        mean_scores = np.mean(all_scores, axis=0)

    n_causal = max(1, round(r * edge_index.shape[1]))
    top_idx = np.argpartition(-mean_scores, n_causal - 1)[:n_causal]
    predicted_edges = set(
        zip(edge_index[0].numpy()[top_idx].tolist(), edge_index[1].numpy()[top_idx].tolist())
    )

    precision, recall = precision_recall(predicted_edges, true_edges)

    return {
        "graph": graph.name,
        "n_nodes": n_nodes,
        "n_true_edges": len(true_edges),
        "n_predicted_edges": len(predicted_edges),
        "precision": precision,
        "recall": recall,
        "final_loss": history[-1][1] if history else None,
        "final_variance": history[-1][2] if history else None,
        "train_time_sec": elapsed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graphs", type=str, default="AO,NONE")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument(
        "--r",
        type=float,
        default=None,
        help="top-r fraction for edge selection; if omitted, set per-graph from the "
        "true adjacency's own sparsity (true_edge_count / n^2) -- confirmed with user "
        "2026-09-18: acceptable for this §5.1 validation step (r is a hyperparameter, "
        "not the causal labels themselves), would need rethinking for the real "
        "ungrounded BSISO run where no such shortcut exists.",
    )
    parser.add_argument("--n-swaps", type=int, default=6)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--generator-window", type=int, default=10,
        help="trailing history window length fed ONLY to the rationale generator's "
        "edge scoring (fix, 2026-09-18: the generator needs cross-time info -- e.g. "
        "lagged cross-correlation -- to separate true from false causal edges; a "
        "single timestep provably cannot carry that signal).",
    )
    parser.add_argument(
        "--entropy-weight", type=float, default=0.0,
        help="sparsity/polarization penalty weight on generator edge scores (candidate "
        "2, §10 step 2 2026-09-18) -- pushes scores toward 0/1. 0.0 = off.",
    )
    parser.add_argument(
        "--prior-rate-weight", type=float, default=0.0,
        help="NRI-style KL-to-sparse-prior weight (§10 step 2 2026-09-18, adapted from "
        "github.com/ethanfetaya/NRI) -- pushes the AVERAGE edge score toward the "
        "target ratio r. 0.0 = off.",
    )
    args = parser.parse_args()

    wanted = set(args.graphs.split(","))
    all_graphs = load_enso_modes_graphs(str(DATA_ROOT))
    graphs = [g for g in all_graphs if g.name in wanted]
    if len(graphs) != len(wanted):
        found = {g.name for g in graphs}
        missing = wanted - found
        raise ValueError(f"graphs not found: {missing}")

    results = []
    for graph in graphs:
        n_true_edges = int(graph.adjacency.sum())
        n_possible = graph.n_nodes * graph.n_nodes
        r = args.r if args.r is not None else n_true_edges / n_possible
        print(f"\n=== training splitter on {graph.name} "
              f"({graph.n_nodes} nodes, {graph.n_systems} systems, "
              f"{n_true_edges} true edges incl self-loops, r={r:.3f}) ===")
        result = train_one_graph(
            graph,
            epochs=args.epochs,
            r=r,
            n_swaps=args.n_swaps,
            lr=args.lr,
            batch_size=args.batch_size,
            generator_window_len=args.generator_window,
            entropy_weight=args.entropy_weight,
            prior_rate_weight=args.prior_rate_weight,
        )
        results.append(result)
        print(
            f"  -> precision={result['precision']:.3f}  recall={result['recall']:.3f}  "
            f"({result['n_predicted_edges']} predicted vs {result['n_true_edges']} true)  "
            f"[{result['train_time_sec']:.1f}s]"
        )

    print("\n=== summary ===")
    for r in results:
        print(
            f"{r['graph']:>6}: precision={r['precision']:.3f}  recall={r['recall']:.3f}  "
            f"loss={r['final_loss']:.4f}  var={r['final_variance']:.4f}"
        )


if __name__ == "__main__":
    main()
