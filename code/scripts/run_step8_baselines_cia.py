"""SS9 step 8 re-run with CIA enabled (§10's "CIA-for-regression" follow-up
task): identical to `run_step8_baselines.py` -- same persistence/plain-GNN/
random-subset ablations, same budget -- except the "full model" row is
trained with `train_step_cia_single_place.train_place_with_cia` (CIA
alignment term added to the splitter+expert loss) instead of step 4's plain
`train_place`. Everything else (data, budget, other three baselines) is
bit-for-bit the same code path as `run_step8_baselines.py`, so the two
result files are directly comparable.

Usage:
    python scripts/run_step8_baselines_cia.py [--target 22] [--variant direct]
        [--epochs 3] [--n-samples-cap 8000] [--cia-weight 0.1]
        [--cia-bandwidth 0.3]
"""

from __future__ import annotations

import argparse
import functools
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

print = functools.partial(print, flush=True)  # noqa: A001

import numpy as np
import torch

from causal_moe.baselines.simple import (
    PersistenceBaseline,
    PlainGNNBaseline,
    RandomSubsetBaseline,
)
from causal_moe.data.candidate_edges import (
    build_candidate_source_set,
    expand_source_clusters_to_edge_index,
)
from causal_moe.data.splits import chronological_split
from scripts.run_step8_baselines import (
    eval_fixed_edge_baseline,
    train_fixed_edge_baseline,
)
from scripts.train_step4_single_place import CACHE_PATH as DEFAULT_CACHE_PATH
from scripts.train_step4_single_place import load_cache
from scripts.train_step_cia_single_place import train_place_with_cia

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, default=22)
    parser.add_argument("--variant", type=str, default="direct")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--n-samples-cap", type=int, default=8000)
    parser.add_argument("--r", type=float, default=None)
    parser.add_argument("--n-swaps", type=int, default=6)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--generator-window", type=int, default=10)
    parser.add_argument("--entropy-weight", type=float, default=0.0)
    parser.add_argument("--cia-weight", type=float, default=0.1,
                         help="best-of-sweep setting from the CIA before/after run (§10) -- see writeup for why")
    parser.add_argument("--cia-bandwidth", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cache-path", type=Path, default=None)
    parser.add_argument("--val-start", type=str, default="2005-01-01")
    parser.add_argument("--test-start", type=str, default="2013-01-01")
    args = parser.parse_args()
    args.cap_contiguous = True
    args.use_self_features = False
    args.self_bypass = False

    RESULTS_DIR.mkdir(exist_ok=True)
    cache = load_cache(args.cache_path or DEFAULT_CACHE_PATH)
    target = args.target
    n_clusters = cache["n_clusters"]
    n_features = cache["features"].shape[-1]
    olr_lag0_channel = cache["olr_lag0_channel"]

    candidate_set = build_candidate_source_set(cache["edge_index"], target, args.variant, n_clusters=n_clusters)
    sources = candidate_set.source_clusters
    edge_index = torch.from_numpy(expand_source_clusters_to_edge_index(sources, target))
    n_edges = edge_index.shape[1]
    r = args.r if args.r is not None else min(0.5, 3.0 / n_edges)

    # B4 audit fix: honest out-of-sample split, same convention as step 8.
    from datetime import date as _date
    sample_dates = cache["sample_dates"].astype("datetime64[D]")
    split_masks = chronological_split(sample_dates, _date.fromisoformat(args.val_start), _date.fromisoformat(args.test_start))
    train_pool_idx = np.nonzero(split_masks.train_mask | split_masks.val_mask)[0]
    test_idx = np.nonzero(split_masks.test_mask)[0]
    if args.n_samples_cap > 0 and args.n_samples_cap < train_pool_idx.shape[0]:
        train_idx = train_pool_idx[: args.n_samples_cap]
    else:
        train_idx = train_pool_idx
    n = train_idx.shape[0]

    x_train = torch.from_numpy(cache["features"][train_idx])
    y_train = torch.from_numpy(cache["targets"][train_idx][:, target])
    x_test = torch.from_numpy(cache["features"][test_idx])
    y_test = torch.from_numpy(cache["targets"][test_idx][:, target])
    print(f"train pool: {n}/{train_pool_idx.shape[0]}  test (out-of-sample): {test_idx.shape[0]}")

    results = {}

    # 1. persistence
    pers = PersistenceBaseline(olr_channel=olr_lag0_channel)
    pers_pred = pers.predict(cache["features"][test_idx], target)
    results["persistence"] = float(np.mean((pers_pred - y_test.numpy()) ** 2))
    print(f"persistence MSE (out-of-sample) = {results['persistence']:.5f}")

    # 2. full model WITH CIA (splitter + expert + CIA alignment term)
    print(f"\ntraining FULL model + CIA (weight={args.cia_weight}, bandwidth={args.cia_bandwidth})...")
    t0 = time.time()
    full_cia = train_place_with_cia(
        variant=args.variant, target=target,
        features_all=cache["features"], targets_all=cache["targets"],
        sample_time_index=cache["sample_time_index"],
        real_edge_index=cache["edge_index"], n_clusters=n_clusters, args=args,
        sample_dates=sample_dates, olr_lag0_channel=olr_lag0_channel,
    )
    results["full_causal_splitter_cia"] = full_cia["expert_mse"]
    print(f"full model + CIA MSE (out-of-sample) = {full_cia['expert_mse']:.5f} ({time.time()-t0:.0f}s)")

    # 3. plain-GNN ablation (all edges, uniform weight) -- unchanged from step 8
    print(f"\ntraining PLAIN-GNN ablation (all {n_edges} edges, uniform weight)...")
    t0 = time.time()
    plain = PlainGNNBaseline(in_channels=n_features, edge_index=edge_index, hidden_channels=16)
    train_fixed_edge_baseline(plain, x_train, y_train, target, args.epochs, args.lr, args.batch_size, args.seed)
    results["plain_gnn_all_edges"] = eval_fixed_edge_baseline(plain, x_test, y_test, target)
    print(f"plain-GNN MSE (out-of-sample) = {results['plain_gnn_all_edges']:.5f} ({time.time()-t0:.0f}s)")

    # 4. random-subset ablation -- unchanged from step 8
    k = max(1, int(round(r * n_edges)))
    print(f"\ntraining RANDOM-SUBSET ablation ({k} of {n_edges} edges, fixed)...")
    t0 = time.time()
    rand = RandomSubsetBaseline(in_channels=n_features, edge_index=edge_index, r=r, hidden_channels=16, seed=args.seed)
    train_fixed_edge_baseline(rand, x_train, y_train, target, args.epochs, args.lr, args.batch_size, args.seed)
    results["random_subset"] = eval_fixed_edge_baseline(rand, x_test, y_test, target)
    print(f"random-subset MSE (out-of-sample) = {results['random_subset']:.5f} ({time.time()-t0:.0f}s)")

    # --- report -----------------------------------------------------------
    base = results["persistence"]
    print(f"\n=== step 8 + CIA: ablation comparison (place {target}, variant {args.variant}) ===")
    print(f"{'model':<28} {'MSE':>10} {'skill vs persistence':>22}")
    for name in ["persistence", "plain_gnn_all_edges", "random_subset", "full_causal_splitter_cia"]:
        mse = results[name]
        skill = 1.0 - mse / base if base > 0 else 0.0
        print(f"{name:<28} {mse:>10.5f} {skill:>+22.4f}")

    report = {
        "target": target,
        "variant": args.variant,
        "n_candidate_edges": int(n_edges),
        "r": r,
        "n_samples": int(n),
        "epochs": args.epochs,
        "cia_weight": args.cia_weight,
        "cia_bandwidth": args.cia_bandwidth,
        "results_mse": results,
        "skill_vs_persistence": {
            k2: (1.0 - v / base if base > 0 else 0.0) for k2, v in results.items()
        },
        "comparison_to_step8_no_cia": {
            "note": "compare against results/step8_baselines_place{target}_{variant}.json's "
                    "own full_causal_splitter MSE -- same data/split/budget/other-baselines, "
                    "only the full-model row differs (CIA added). Read that file at analysis "
                    "time rather than hardcoding a number here, since B4's honest-split fix "
                    "changes step 8's numbers run to run.",
        },
    }
    out_path = RESULTS_DIR / f"step8_baselines_cia_place{target}_{args.variant}.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
