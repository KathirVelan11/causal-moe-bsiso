"""B18 root-cause probe (2026-09-24): tests whether the n=15+ discrimination
collapse found in the semisynthetic ladder (see PROJECT_PLAN.md, Bug 18) is
caused by the SharedEncoder processing a much larger n^2 candidate-edge pool
at higher n_clusters, most of which (every non-A self-loop) carries zero
training signal under target_node_mask.

This is a near-verbatim copy of train_splitter_semisynthetic.py's pipeline
(same data loading, same model, same training loop) with exactly ONE change:
edge_index passed to the model is restricted to only edges with dst == a
(node A's own candidate set: its true 2 parents + every other node as a
distractor source, but NOT the other (n_clusters - 1) nodes' own self-loops
as separate destination targets). If node-A-only precision/recall/gap
recovers toward n=5's level under this restriction, that confirms the
candidate-pool-size (not n_clusters itself) is what drives the collapse --
i.e. confirms the shared-encoder-capacity/dilution hypothesis. If it does
NOT recover, the problem is something else (e.g. genuinely about how many
OTHER real regions of spatial diversity exist at that patch size, not raw
candidate count).

Usage: python scripts/train_splitter_semisynthetic_restricted.py --n-clusters 15 --epochs 30
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
    patch_slices_for_n_clusters,
    pick_abc_avoiding_confounds,
)
from causal_moe.data.windows import build_windowed_dataset, pool_windowed_dataset_to_clusters
from causal_moe.splitter.dirgnn import DIRGNNSplitter, LambdaWarmupSchedule
from scripts.train_splitter_semisynthetic import build_generator_windows, sample_swaps, precision_recall


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-clusters", type=int, default=15)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--r", type=float, default=None)
    parser.add_argument("--n-swaps", type=int, default=6)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--generator-window", type=int, default=10)
    parser.add_argument("--n-samples-cap", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    t0 = time.time()
    raw = load_raw_dataset()
    print(f"loaded raw dataset in {time.time()-t0:.1f}s")

    lat_slice, lon_slice = patch_slices_for_n_clusters(args.n_clusters)
    patch = extract_patch(raw, lat_slice=lat_slice, lon_slice=lon_slice)
    assignment = cluster_patch(patch, n_clusters=args.n_clusters, random_state=args.seed)

    windowed_raw = build_windowed_dataset(patch, lead_time_days=1)
    pooled = pool_windowed_dataset_to_clusters(windowed_raw, assignment)

    cluster_olr = pool_features_to_clusters(
        patch.fields[FIELDS.index("olr")].reshape(patch.n_time, -1), assignment
    )
    a, b, c = pick_abc_avoiding_confounds(cluster_olr, rng=np.random.default_rng(args.seed))
    print(f"injected rule: cluster {a}'s target := f(cluster {b}, cluster {c}), self-persist elsewhere")

    ds, true_edge_index_np = inject_causal_rule(pooled, a, b, c, nonlinear=False)
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

    # THE ONE CHANGE: restrict edge_index to dst == a only (node A's own
    # candidate set -- its 2 true parents plus every other node as a
    # distractor source), instead of the full n^2 fully-connected graph.
    # This drops the OTHER (n_clusters - 1) nodes' self-loops entirely,
    # which target_node_mask means the model was never rewarded for
    # scoring correctly anyway (this is exactly what B17 already showed
    # is unfair to *evaluate*; this script additionally never *shows*
    # the model those candidates during training/scoring at all).
    src = np.arange(n_clusters)
    dst = np.full(n_clusters, a)
    edge_index = torch.from_numpy(np.stack([src, dst], axis=0).astype(np.int64))
    n_true_edges_a = sum(1 for e in true_edges if e[1] == a)
    n_possible_a = n_clusters  # only dst==a edges now
    r = args.r if args.r is not None else n_true_edges_a / n_possible_a
    print(f"RESTRICTED candidate set: {n_clusters} edges (dst=={a} only), "
          f"n_true_edges_a={n_true_edges_a}, r={r:.3f} "
          f"(vs full-graph r={ (n_clusters + 2) / (n_clusters*n_clusters):.3f} for reference)")

    cluster_olr_full = cluster_olr
    gen_windows_all = build_generator_windows(cluster_olr_full, ds.sample_time_index, args.generator_window)
    gen_windows_t_full = torch.from_numpy(gen_windows_all)

    x_all_full = torch.from_numpy(ds.features)
    y_all_full = torch.from_numpy(ds.targets)

    x_all = x_all_full[keep_idx]
    y_all = y_all_full[keep_idx]
    gen_windows_t = gen_windows_t_full[keep_idx]
    features_bank = ds.features[keep_idx]

    n_features = ds.features.shape[-1]
    model = DIRGNNSplitter(
        in_channels=n_features, hidden_channels=16, r=r,
        generator_window_len=args.generator_window, generator_in_channels=1,
        use_self_features=False,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

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
                    entropy_weight=0.0,
                    prior_rate_weight=0.0,
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
    predicted_edges_a_only = set(
        zip(edge_index[0].numpy()[top_idx].tolist(), edge_index[1].numpy()[top_idx].tolist())
    )
    true_edges_a_only = {(src_, dst_) for src_, dst_ in true_edges if dst_ == a}
    precision_a, recall_a = precision_recall(predicted_edges_a_only, true_edges_a_only)

    true_mask_a = np.array([e in true_edges_a_only for e in zip(edge_index[0].numpy().tolist(), edge_index[1].numpy().tolist())])
    true_scores_a = mean_scores[true_mask_a]
    false_scores_a = mean_scores[~true_mask_a]

    print(f"\n=== RESTRICTED candidate set result (dst=={a} only, {n_clusters} candidates "
          f"vs {n_clusters*n_clusters} in the full-graph version) ===")
    print(f"precision={precision_a:.3f}  recall={recall_a:.3f}  "
          f"({len(predicted_edges_a_only)} predicted vs {len(true_edges_a_only)} true)")
    print(f"true-edge mean score={true_scores_a.mean():.4f}  false-edge mean score={false_scores_a.mean():.4f}")
    print(f"predicted edges (dst=A): {sorted(predicted_edges_a_only)}")
    print(f"true edges (dst=A):      {sorted(true_edges_a_only)}")

    report = {
        "n_clusters": args.n_clusters,
        "n_samples_cap": args.n_samples_cap,
        "epochs": args.epochs,
        "restricted_candidate_set": True,
        "n_candidates": n_clusters,
        "a": int(a), "b": int(b), "c": int(c),
        "r": r,
        "node_a_only": {
            "precision": precision_a, "recall": recall_a,
            "n_predicted": len(predicted_edges_a_only), "n_true": len(true_edges_a_only),
            "true_edge_mean_score": float(true_scores_a.mean()) if true_scores_a.size else None,
            "false_edge_mean_score": float(false_scores_a.mean()) if false_scores_a.size else None,
        },
    }
    results_dir = Path(__file__).resolve().parents[1] / "results"
    results_dir.mkdir(exist_ok=True)
    out_path = results_dir / f"step3_semisynthetic_n{args.n_clusters}_restricted.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
