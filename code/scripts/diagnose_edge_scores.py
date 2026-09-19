"""Diagnostic for the AO near/below-chance result (§9 step 2 debugging,
2026-09-18): inspects the trained splitter's RAW edge scores directly,
before the top-r cutoff, split into true-edge vs false-edge groups.

Answers: is there ANY signal (true edges scored higher on average than
false edges), even if the hard top-r selection doesn't land exactly right?
If scores are statistically indistinguishable between true/false edges,
the generator has learned nothing about causal structure at all -- a
design/bug problem, not a tuning problem.

Reuses train_one_graph's training loop unchanged, just adds score
inspection after training instead of only reporting precision/recall.
"""

from __future__ import annotations

import sys
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
from scripts.train_splitter_causaldynamics import (
    build_generator_windows,
    build_memory_bank,
    make_fully_connected_edge_index,
    sample_swaps,
)

DATA_ROOT = Path(__file__).resolve().parents[1] / "data_external" / "causaldynamics" / "extracted_inputs"


def train_and_inspect(
    graph, epochs, r, n_swaps, lr, batch_size,
    generator_window_len=10, entropy_weight=0.0, prior_rate_weight=0.0, seed=0,
):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    ds = build_windowed_dataset_from_graph(graph, lead_time_steps=1)
    n_nodes = graph.n_nodes
    n_samples = ds.features.shape[0]

    edge_index = make_fully_connected_edge_index(n_nodes)
    true_edge_index = adjacency_to_edge_index(graph.adjacency)
    true_edges = set(zip(true_edge_index[0].tolist(), true_edge_index[1].tolist()))

    bank = build_memory_bank(ds, n_nodes)
    gen_windows_t = torch.from_numpy(build_generator_windows(graph, ds, generator_window_len))
    model = DIRGNNSplitter(
        in_channels=3, hidden_channels=16, r=r,
        generator_window_len=generator_window_len, generator_in_channels=1,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    total_steps = epochs * n_samples
    schedule = LambdaWarmupSchedule(lambda_max=1.0, total_steps=max(total_steps, 1))

    system_index = ds.sample_system_index
    x_all = torch.from_numpy(ds.features)
    y_all = torch.from_numpy(ds.targets)

    step = 0
    for epoch in range(epochs):
        order = rng.permutation(n_samples)
        for start in range(0, n_samples - batch_size + 1, batch_size):
            idx_batch = order[start : start + batch_size]
            batch_loss = torch.zeros(())
            optimizer.zero_grad()
            for idx in idx_batch:
                sys_id = int(system_index[idx])
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
                step += 1
            batch_loss = batch_loss / len(idx_batch)
            batch_loss.backward()
            optimizer.step()
        print(f"  [{graph.name}] epoch {epoch+1}/{epochs} done")

    # --- Diagnostic: raw score distribution, true vs false edges ---
    model.eval()
    with torch.no_grad():
        sample_idx = rng.choice(n_samples, size=min(200, n_samples), replace=False)
        all_scores = []
        for idx in sample_idx:
            split = model.split(gen_windows_t[idx], edge_index)
            all_scores.append(split.edge_scores.numpy())
        mean_scores = np.mean(all_scores, axis=0)
        std_scores = np.std(all_scores, axis=0)

    edge_pairs = list(zip(edge_index[0].tolist(), edge_index[1].tolist()))
    is_true = np.array([p in true_edges for p in edge_pairs])

    true_scores = mean_scores[is_true]
    false_scores = mean_scores[~is_true]

    print(f"\n--- {graph.name} raw edge-score diagnostic ---")
    print(f"n_true_edges={is_true.sum()}  n_false_edges={(~is_true).sum()}")
    print(f"true-edge  scores: mean={true_scores.mean():.4f}  std={true_scores.std():.4f}  "
          f"min={true_scores.min():.4f}  max={true_scores.max():.4f}")
    print(f"false-edge scores: mean={false_scores.mean():.4f}  std={false_scores.std():.4f}  "
          f"min={false_scores.min():.4f}  max={false_scores.max():.4f}")

    # AUC-style check: fraction of (true, false) pairs where true > false
    # -- threshold-free measure of whether true edges rank higher at all.
    n_pairs = 2000
    rng2 = np.random.default_rng(1)
    t_idx = rng2.integers(0, len(true_scores), n_pairs)
    f_idx = rng2.integers(0, len(false_scores), n_pairs)
    wins = (true_scores[t_idx] > false_scores[f_idx]).sum()
    auc_estimate = wins / n_pairs
    print(f"P(random true edge scores > random false edge) ~= {auc_estimate:.3f}  "
          f"(0.5 = no signal, 1.0 = perfect separation)")

    # Also: how much did per-timestep scores vary (std_scores) -- high std
    # relative to the true/false score gap would mean scores are mostly
    # noise, not a stable per-edge signal.
    print(f"per-edge score std across the 200 sampled timesteps: "
          f"mean={std_scores.mean():.4f} (higher = less stable/consistent scoring)")

    return {
        "graph": graph.name,
        "true_mean": float(true_scores.mean()),
        "false_mean": float(false_scores.mean()),
        "auc_estimate": float(auc_estimate),
    }


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--graphs", type=str, default="AO,NONE")
    parser.add_argument("--entropy-weight", type=float, default=0.0)
    parser.add_argument("--prior-rate-weight", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()

    graphs = load_enso_modes_graphs(str(DATA_ROOT))
    ao = next(g for g in graphs if g.name == "AO")
    none_g = next(g for g in graphs if g.name == "NONE")
    by_name = {"AO": (ao, 0.49), "NONE": (none_g, 0.04)}
    wanted = args.graphs.split(",")

    results = []
    for name in wanted:
        graph, r = by_name[name]
        print(f"\n=== {graph.name} (r={r}, entropy_weight={args.entropy_weight}, "
              f"prior_rate_weight={args.prior_rate_weight}) ===")
        result = train_and_inspect(
            graph, epochs=args.epochs, r=r, n_swaps=6, lr=1e-3, batch_size=16,
            entropy_weight=args.entropy_weight,
            prior_rate_weight=args.prior_rate_weight,
        )
        results.append(result)

    print("\n=== summary ===")
    for r in results:
        print(f"{r['graph']:>6}: true_mean={r['true_mean']:.4f}  false_mean={r['false_mean']:.4f}  "
              f"auc~={r['auc_estimate']:.3f}")


if __name__ == "__main__":
    main()
