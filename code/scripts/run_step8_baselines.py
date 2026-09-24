"""SS9 step 8: baselines + evaluation (SS6).

Runs the ABLATIONS that isolate this project's own contribution -- the
causal splitter -- under identical data, capacity, and training budget:

  full model     : splitter-selected causal subgraph -> expert (steps 4-5)
  plain-GNN      : ALL candidate edges, uniform weight -> same expert
  random-subset  : random top-r-sized fixed edge subset -> same expert
  persistence    : tomorrow's OLR = today's OLR

Scope note (honest, per SS6): the three PUBLISHED-METHOD baselines SS6 also
lists -- GC-MoE, GeoMoE (GraphMoRE substitute), DyMoE -- are separate
external codebases (SS7) requiring their own environments and data
adapters. They are NOT run here; this step covers the plain-GNN and
no-causal-selection ablations SS6 lists alongside them.

Usage:
    python scripts/run_step8_baselines.py [--target 22] [--variant direct]
        [--epochs 3] [--n-samples-cap 8000]
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
from scripts.train_step4_single_place import (
    load_cache,
    train_place,
)

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def train_fixed_edge_baseline(
    model: torch.nn.Module,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    target: int,
    epochs: int,
    lr: float,
    batch_size: int,
    seed: int,
) -> torch.nn.Module:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = x_train.shape[0]
    for epoch in range(epochs):
        order = rng.permutation(n)
        total = 0.0
        nb = 0
        for start in range(0, n - batch_size + 1, batch_size):
            idx = order[start : start + batch_size]
            opt.zero_grad()
            loss = torch.zeros(())
            for i in idx:
                pred = model(x_train[i], target)
                loss = loss + (pred - y_train[i]) ** 2
            loss = loss / len(idx)
            loss.backward()
            opt.step()
            total += float(loss.detach())
            nb += 1
        print(f"    epoch {epoch+1}/{epochs} loss={total/max(nb,1):.4f}")
    return model


@torch.no_grad()
def eval_fixed_edge_baseline(model, x_eval, y_eval, target) -> float:
    preds = np.array([float(model(x_eval[i], target)) for i in range(x_eval.shape[0])])
    return float(np.mean((preds - y_eval.numpy()) ** 2))


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
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cache-path", type=Path, default=None)
    parser.add_argument("--val-start", type=str, default="2005-01-01")
    parser.add_argument("--test-start", type=str, default="2013-01-01")
    args = parser.parse_args()
    args.cap_contiguous = True
    args.use_self_features = False
    args.self_bypass = False

    RESULTS_DIR.mkdir(exist_ok=True)
    from scripts.train_step4_single_place import CACHE_PATH as DEFAULT_CACHE_PATH
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

    # B4 audit fix: honest out-of-sample split -- train the ablation
    # baselines on the SAME train pool train_place() uses, evaluate on the
    # SAME held-out test span. Previously all baselines here trained AND
    # evaluated on x_all[:n], the training rows themselves.
    from datetime import date as _date
    sample_dates = cache["sample_dates"].astype("datetime64[D]")
    # B22 audit fix: pass the real lead time so no training/val sample's
    # target leaks into the next split's window (see splits.py).
    split_masks = chronological_split(
        sample_dates, _date.fromisoformat(args.val_start), _date.fromisoformat(args.test_start),
        lead_time_days=int(cache["lead_time_days"]),
    )
    train_pool_idx = np.nonzero(split_masks.train_mask | split_masks.val_mask)[0]
    test_idx = np.nonzero(split_masks.test_mask)[0]
    if args.n_samples_cap > 0 and args.n_samples_cap < train_pool_idx.shape[0]:
        train_idx = train_pool_idx[: args.n_samples_cap]  # contiguous prefix, B5
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

    # 2. full model (splitter + expert), same budget
    print(f"\ntraining FULL model (splitter + expert)...")
    t0 = time.time()
    full = train_place(
        variant=args.variant, target=target,
        features_all=cache["features"], targets_all=cache["targets"],
        sample_time_index=cache["sample_time_index"],
        real_edge_index=cache["edge_index"], n_clusters=n_clusters, args=args,
        sample_dates=sample_dates, olr_lag0_channel=olr_lag0_channel,
        lags_days=cache["lags_days"], lead_time_days=int(cache["lead_time_days"]),
    )
    results["full_causal_splitter"] = full["expert_mse"]
    print(f"full model MSE (out-of-sample) = {full['expert_mse']:.5f} ({time.time()-t0:.0f}s)")

    # 3. plain-GNN ablation (all edges, uniform weight)
    print(f"\ntraining PLAIN-GNN ablation (all {n_edges} edges, uniform weight)...")
    t0 = time.time()
    plain = PlainGNNBaseline(in_channels=n_features, edge_index=edge_index, hidden_channels=16)
    train_fixed_edge_baseline(plain, x_train, y_train, target, args.epochs, args.lr, args.batch_size, args.seed)
    results["plain_gnn_all_edges"] = eval_fixed_edge_baseline(plain, x_test, y_test, target)
    print(f"plain-GNN MSE (out-of-sample) = {results['plain_gnn_all_edges']:.5f} ({time.time()-t0:.0f}s)")

    # 4. random-subset ablation
    k = max(1, int(round(r * n_edges)))
    print(f"\ntraining RANDOM-SUBSET ablation ({k} of {n_edges} edges, fixed)...")
    t0 = time.time()
    rand = RandomSubsetBaseline(in_channels=n_features, edge_index=edge_index, r=r, hidden_channels=16, seed=args.seed)
    train_fixed_edge_baseline(rand, x_train, y_train, target, args.epochs, args.lr, args.batch_size, args.seed)
    results["random_subset"] = eval_fixed_edge_baseline(rand, x_test, y_test, target)
    print(f"random-subset MSE (out-of-sample) = {results['random_subset']:.5f} ({time.time()-t0:.0f}s)")

    # --- report -----------------------------------------------------------
    base = results["persistence"]
    print(f"\n=== step 8: ablation comparison (place {target}, variant {args.variant}) ===")
    print(f"{'model':<26} {'MSE':>10} {'skill vs persistence':>22}")
    for name in ["persistence", "plain_gnn_all_edges", "random_subset", "full_causal_splitter"]:
        mse = results[name]
        skill = 1.0 - mse / base if base > 0 else 0.0
        print(f"{name:<26} {mse:>10.5f} {skill:>+22.4f}")

    report = {
        "target": target,
        "variant": args.variant,
        "n_candidate_edges": int(n_edges),
        "r": r,
        "n_samples": int(n),
        "epochs": args.epochs,
        "results_mse": results,
        "skill_vs_persistence": {
            k2: (1.0 - v / base if base > 0 else 0.0) for k2, v in results.items()
        },
        "note": "Published-method baselines (GC-MoE, GeoMoE/GraphMoRE, DyMoE) are external "
                "codebases per SS6/SS7 and are not run here; these are the plain-GNN and "
                "no-causal-selection ablations SS6 lists.",
    }
    out_path = RESULTS_DIR / f"step8_baselines_place{target}_{args.variant}.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
