"""SS9 step 3 / SS5.2: mid-scale semi-synthetic check. Real contiguous Bay
of Bengal patch (SS5.2 step 1), clustered with the same method planned for
the full mesh (SS5.2 step 2), real input features, hand-written injected
causal rule replacing the target (SS5.2 steps 3-4) so the true answer is
known by construction. Runs the EXACT SAME splitter pipeline/hyperparameters
as scripts/train_splitter_causaldynamics.py (SS9 step 2) -- same
DIRGNNSplitter, same LambdaWarmupSchedule, same top-r selection, same
memory-bank-swap intervention shape -- adapted only for a single contiguous
real time series (no system replicates: this patch is one place, one long
record, unlike CausalDynamics' 10 independent system replicates) (SS5.2
step 5: "exact same pipeline and hyperparameters intended for the real
full run").

Confound check + rule injection: causal_moe/data/semisynthetic.py
(confirmed with user 2026-09-19 -- see that module's docstring for the
DIR-GNN-precedent research and the confound-avoidance rationale).

Usage:
    python scripts/train_splitter_semisynthetic.py [--n-clusters 5]
        [--epochs 5] [--r 0.5] [--n-swaps 6] [--lr 1e-3]
        [--nonlinear] [--n-samples-cap 2000]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from causal_moe.data.clustering import pool_features_to_clusters
from causal_moe.data.raw import FIELDS, load_raw_dataset
from causal_moe.data.semisynthetic import (
    build_patch_mesh,
    cluster_patch,
    extract_patch,
    inject_causal_rule,
    make_fully_connected_edge_index,
    patch_slices_for_n_clusters,
    pick_abc_avoiding_confounds,
)
from causal_moe.data.windows import build_windowed_dataset, pool_windowed_dataset_to_clusters
from causal_moe.splitter.dirgnn import DIRGNNSplitter, LambdaWarmupSchedule


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


def build_generator_windows(olr_series: np.ndarray, sample_time_index: np.ndarray, generator_window_len: int) -> np.ndarray:
    """Same vectorized approach as
    train_splitter_causaldynamics.py::build_generator_windows (sliding_window_view,
    no per-sample Python loop -- CPU-only dev machine constraint), adapted
    for a single contiguous series instead of (system, time, node).

    olr_series: (T, n_clusters) float32 -- pooled cluster OLR, the same
        channel the generator scores edges from (matches generator_in_channels=1).
    sample_time_index: (n_samples,) int64 -- "today" index into olr_series
        for each windowed-dataset row.
    Returns: (n_samples, n_clusters, generator_window_len, 1) float32.
    """
    T, n_clusters = olr_series.shape
    W = generator_window_len
    pad = np.repeat(olr_series[:1, :], W - 1, axis=0) if W > 1 else olr_series[:0, :]
    padded = np.concatenate([pad, olr_series], axis=0)  # (T + W - 1, n_clusters)
    windows = np.lib.stride_tricks.sliding_window_view(padded, W, axis=0)  # (T, n_clusters, W)
    gen_windows = windows[sample_time_index]  # (n_samples, n_clusters, W)
    return gen_windows[..., None].astype(np.float32)


def sample_swaps(features: np.ndarray, anchor_row: int, n_swaps: int, rng: np.random.Generator) -> np.ndarray:
    """Draws n_swaps other timesteps from this SAME patch's own history,
    both time directions (SS4.2/SS8, same rule as step 2) -- excludes the
    anchor's own row."""
    n_t = features.shape[0]
    candidates = np.delete(np.arange(n_t), anchor_row)
    chosen = rng.choice(candidates, size=min(n_swaps, candidates.shape[0]), replace=False)
    return features[chosen]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-clusters", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--r", type=float, default=None, help="top-r fraction; default = true sparsity of the injected rule's edge set (same convention as step 2)")
    parser.add_argument("--n-swaps", type=int, default=6)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--generator-window", type=int, default=10)
    parser.add_argument("--entropy-weight", type=float, default=0.0)
    parser.add_argument("--prior-rate-weight", type=float, default=0.0)
    parser.add_argument("--nonlinear", action="store_true", help="use the nonlinear (B*C interaction) injected rule variant instead of linear")
    parser.add_argument("--n-samples-cap", type=int, default=0, help="if >0, subsample this many rows (random, fixed seed) for a fast dry run before the full 16k-sample run")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--use-self-features", type=lambda s: s.lower() != "false", default=False,
                         help="Bug 2 audit fix: default False, see PROJECT_PLAN.md")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    t0 = time.time()
    raw = load_raw_dataset()
    print(f"loaded raw dataset in {time.time()-t0:.1f}s")

    lat_slice, lon_slice = patch_slices_for_n_clusters(args.n_clusters)
    patch = extract_patch(raw, lat_slice=lat_slice, lon_slice=lon_slice)
    assignment = cluster_patch(patch, n_clusters=args.n_clusters, random_state=args.seed)
    mesh = build_patch_mesh(patch, assignment)
    print(f"patch: {patch.n_places} raw cells -> {args.n_clusters} clusters "
          f"(sizes {np.bincount(assignment.labels, minlength=args.n_clusters).tolist()}), "
          f"{mesh.n_edges} physical adjacency edges (not used as candidate set, see below)")

    windowed_raw = build_windowed_dataset(patch, lead_time_days=1)
    pooled = pool_windowed_dataset_to_clusters(windowed_raw, assignment)

    cluster_olr = pool_features_to_clusters(
        patch.fields[FIELDS.index("olr")].reshape(patch.n_time, -1), assignment
    )
    a, b, c = pick_abc_avoiding_confounds(cluster_olr, rng=np.random.default_rng(args.seed))
    print(f"injected rule: cluster {a}'s target := f(cluster {b}, cluster {c}) "
          f"[{'nonlinear' if args.nonlinear else 'linear'}], all other clusters self-persist")

    ds, true_edge_index_np = inject_causal_rule(pooled, a, b, c, nonlinear=args.nonlinear)
    true_edges = set(zip(true_edge_index_np[0].tolist(), true_edge_index_np[1].tolist()))

    n_samples_full = ds.features.shape[0]
    if args.n_samples_cap > 0 and args.n_samples_cap < n_samples_full:
        keep_idx = rng.choice(n_samples_full, size=args.n_samples_cap, replace=False)
        keep_idx.sort()
    else:
        keep_idx = np.arange(n_samples_full)
    n_samples = keep_idx.shape[0]
    print(f"using {n_samples}/{n_samples_full} samples")

    n_clusters = args.n_clusters
    edge_index = torch.from_numpy(make_fully_connected_edge_index(n_clusters))
    n_true_edges = len(true_edges)
    n_possible = n_clusters * n_clusters
    r = args.r if args.r is not None else n_true_edges / n_possible
    print(f"n_true_edges={n_true_edges} (incl. self-loops), n_possible={n_possible}, r={r:.3f}")

    # cluster-level OLR (post-pool, whole record) drives the generator's
    # history window -- same channel the injected rule itself reads from.
    cluster_olr_full = cluster_olr  # (T_raw, n_clusters) -- indexed by RAW time, not sample row
    gen_windows_all = build_generator_windows(cluster_olr_full, ds.sample_time_index, args.generator_window)
    gen_windows_t_full = torch.from_numpy(gen_windows_all)  # (n_samples_full, n_clusters, W, 1)

    x_all_full = torch.from_numpy(ds.features)  # (n_samples_full, n_clusters, 21)
    y_all_full = torch.from_numpy(ds.targets)  # (n_samples_full, n_clusters)

    x_all = x_all_full[keep_idx]
    y_all = y_all_full[keep_idx]
    gen_windows_t = gen_windows_t_full[keep_idx]
    features_bank = ds.features[keep_idx]  # numpy, for swap sampling

    n_features = ds.features.shape[-1]
    model = DIRGNNSplitter(
        in_channels=n_features, hidden_channels=16, r=r,
        generator_window_len=args.generator_window, generator_in_channels=1,
        use_self_features=args.use_self_features,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Bug 3 (audit B3) fix: only node A's forecast is edge-dependent by
    # construction (inject_causal_rule); every other cluster self-persists.
    # Without this mask, risk averages over all n_clusters nodes, so at
    # n_clusters=5 only 1/5 of the loss carries edge-dependent signal (4/5
    # at n_clusters=15) -- diluting Var_swaps[risk] and producing the
    # density "cliff" that was misdiagnosed as a VREx non-unique-optima
    # pathology (see PROJECT_PLAN.md Bug 3).
    target_node_mask = torch.zeros(n_clusters, dtype=torch.bool)
    target_node_mask[a] = True

    total_steps = args.epochs * n_samples
    schedule = LambdaWarmupSchedule(lambda_max=1.0, total_steps=max(total_steps, 1))

    step = 0
    t0 = time.time()
    for epoch in range(args.epochs):
        order = rng.permutation(n_samples)
        epoch_loss = 0.0
        epoch_var = 0.0
        n_batches = 0
        for start in range(0, n_samples - args.batch_size + 1, args.batch_size):
            idx_batch = order[start : start + args.batch_size]
            batch_loss = torch.zeros(())
            batch_var = 0.0
            optimizer.zero_grad()
            for idx in idx_batch:
                env = sample_swaps(features_bank, int(idx), args.n_swaps, rng)
                env_t = torch.from_numpy(env)

                lam = schedule.value(step)
                out = model.compute_loss(
                    x_all[idx], y_all[idx], env_t, edge_index, lam,
                    x_generator_window=gen_windows_t[idx],
                    entropy_weight=args.entropy_weight,
                    prior_rate_weight=args.prior_rate_weight,
                    target_node_mask=target_node_mask,
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
        elapsed = time.time() - t0
        print(f"  epoch {epoch+1}/{args.epochs}  loss={avg_loss:.4f}  var={avg_var:.4f}  "
              f"lambda={schedule.value(step):.3f}  elapsed={elapsed:.1f}s")

    elapsed = time.time() - t0
    print(f"training done in {elapsed:.1f}s")

    # Edge recovery check: same convention as step 2 -- average generator
    # scores over a sample of timesteps, then apply the same top-r selection.
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

    # B17 audit fix, 2026-09-23: precision/recall against the FULL
    # true_edges set (node A's 2 real causal parents PLUS every other
    # node's trivial self-loop) mismatches what target_node_mask actually
    # trains for -- the loss only ever rewards getting NODE A's own
    # prediction right, so the model has zero training pressure to score
    # any other node's self-loop correctly. Scoring against all
    # n_clusters "true" edges mechanically gets worse as n_clusters grows
    # (more untrained self-loops dilute the metric), producing a fake
    # "density cliff" independent of whether edge recovery for A itself
    # is working. Fix: report BOTH the full-graph metric (kept for
    # comparability with the OLD, pre-fix numbers) and a
    # node-A-only metric (the one the training objective actually
    # targets) -- the second is the one that should gate Phase 4.
    precision, recall = precision_recall(predicted_edges, true_edges)
    true_edges_a_only = {(src, dst) for src, dst in true_edges if dst == a}
    predicted_edges_a_only = {(src, dst) for src, dst in predicted_edges if dst == a}
    precision_a, recall_a = precision_recall(predicted_edges_a_only, true_edges_a_only)

    # AUC-style separation diagnostic (true-edge vs false-edge mean score),
    # same convention as scripts/diagnose_edge_scores.py in step 2.
    all_idx = set(zip(edge_index[0].tolist(), edge_index[1].tolist()))
    true_mask = np.array([e in true_edges for e in zip(edge_index[0].numpy().tolist(), edge_index[1].numpy().tolist())])
    true_scores = mean_scores[true_mask]
    false_scores = mean_scores[~true_mask]

    # Same AUC-style diagnostic, restricted to node A's own candidate
    # edges only (dst == a) -- the fair comparison given target_node_mask
    # (B17 fix, see precision_a/recall_a above).
    a_dst_mask = (edge_index[1].numpy() == a)
    true_mask_a = true_mask & a_dst_mask
    false_mask_a = (~true_mask) & a_dst_mask
    true_scores_a = mean_scores[true_mask_a]
    false_scores_a = mean_scores[false_mask_a]

    print(f"\n=== result (FULL graph, includes every node's self-loop -- "
          f"kept for comparability with pre-B17-fix numbers) ===")
    print(f"precision={precision:.3f}  recall={recall:.3f}  "
          f"({len(predicted_edges)} predicted vs {len(true_edges)} true)")
    print(f"true-edge mean score={true_scores.mean():.4f}  false-edge mean score={false_scores.mean():.4f}")

    print(f"\n=== result (NODE A ONLY -- what target_node_mask actually "
          f"trains for; this is the metric that should gate Phase 4) ===")
    print(f"precision={precision_a:.3f}  recall={recall_a:.3f}  "
          f"({len(predicted_edges_a_only)} predicted vs {len(true_edges_a_only)} true)")
    if true_scores_a.size and false_scores_a.size:
        print(f"true-edge mean score={true_scores_a.mean():.4f}  false-edge mean score={false_scores_a.mean():.4f}")
    print(f"predicted edges (dst=A): {sorted(predicted_edges_a_only)}")
    print(f"true edges (dst=A):      {sorted(true_edges_a_only)}")

    report = {
        "n_clusters": args.n_clusters,
        "n_samples_cap": args.n_samples_cap,
        "epochs": args.epochs,
        "nonlinear": args.nonlinear,
        "a": int(a), "b": int(b), "c": int(c),
        "r": r,
        "full_graph": {
            "precision": precision, "recall": recall,
            "n_predicted": len(predicted_edges), "n_true": len(true_edges),
            "true_edge_mean_score": float(true_scores.mean()),
            "false_edge_mean_score": float(false_scores.mean()),
        },
        "node_a_only": {
            "precision": precision_a, "recall": recall_a,
            "n_predicted": len(predicted_edges_a_only), "n_true": len(true_edges_a_only),
            "true_edge_mean_score": float(true_scores_a.mean()) if true_scores_a.size else None,
            "false_edge_mean_score": float(false_scores_a.mean()) if false_scores_a.size else None,
        },
    }
    results_dir = Path(__file__).resolve().parents[1] / "results"
    results_dir.mkdir(exist_ok=True)
    out_path = results_dir / f"step3_semisynthetic_n{args.n_clusters}{'_nonlinear' if args.nonlinear else ''}.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
