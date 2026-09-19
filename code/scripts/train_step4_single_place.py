"""SS9 step 4 / SS5.3 stage 2: splitter (SS4.2) + per-place expert (SS4.3)
trained TOGETHER on real, single-cluster BSISO data -- the first production
run, no synthetic/injected ground truth (unlike steps 2-3).

Uses the REAL 50-cluster full mesh already built in step 1
(cache/windowed_clustered50_lead1.npz) -- NOT step 3's semi-synthetic
patches (those were test scaffolding only, see architecture doc SS10 step 3
"IMPORTANT" note).

No ground-truth causal graph exists for real BSISO data (that's the whole
premise of SS5 -- real climate data is the "unknown field"). So this script
reports, instead of precision/recall against a known answer key:
  1. A raw lagged-correlation AUC-style baseline computed directly from the
     data (same diagnostic idea as diagnose_edge_scores.py / step 3's
     confound checks) -- tells us how much real signal EXISTS for the
     splitter to find, independent of any model.
  2. The trained splitter's edge-score separation (same true/false-style
     summary, but "true" here means "top-r selected", since there's no
     external ground truth -- reported as score spread/entropy instead of
     against an answer key).
  3. Forecast accuracy: expert MSE vs a naive persistence baseline
     (predict tomorrow's OLR = today's OLR) -- a real, measurable quantity
     even without causal ground truth, and the actual quantity SS4.3/SS6
     care about.

Runs all three candidate-edge-set variants (direct / 2hop / full,
causal_moe/data/candidate_edges.py) back to back per user decision
2026-09-19 ("try all 3, take best").

Usage:
    python scripts/train_step4_single_place.py [--target 22] [--epochs 5]
        [--variants direct,2hop,full] [--r 0.3] [--n-swaps 6] [--lr 1e-3]
        [--entropy-weight 0.0] [--n-samples-cap 2000]
"""

from __future__ import annotations

import argparse
import functools
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Long CPU runs are monitored by tailing a redirected log file; Python's
# default block buffering hides all progress until the process exits.
print = functools.partial(print, flush=True)  # noqa: A001

import numpy as np
import torch

from causal_moe.data.candidate_edges import (
    build_candidate_source_set,
    expand_source_clusters_to_edge_index,
)
from causal_moe.experts.expert import PlaceExpert, compute_expert_loss
from causal_moe.splitter.dirgnn import DIRGNNSplitter, LambdaWarmupSchedule

CACHE_PATH = Path(__file__).resolve().parents[1] / "cache" / "windowed_clustered50_lead1.npz"
# Channel layout (windows.py): 3 lags x 7 channels [sst,h850,u200,pw,u850,olr,is_ocean],
# lag0 (today) is the first 7-wide block -> lag0's olr is absolute channel index 5.
OLR_LAG0_CHANNEL = 5


def load_cache(path: Path = CACHE_PATH):
    d = np.load(path, allow_pickle=True)
    return {
        "features": d["features"],  # (n_samples, 50, 21)
        "targets": d["targets"],  # (n_samples, 50)
        "sample_time_index": d["sample_time_index"],
        "sample_dates": d["sample_dates"],
        "edge_index": d["edge_index"],  # (2, n_edges) real lattice adjacency
        "n_clusters": int(d["n_clusters"]),
    }


def build_generator_windows(
    olr_lag0_series: np.ndarray, sample_time_index: np.ndarray, generator_window_len: int
) -> np.ndarray:
    """Vectorized sliding window (no per-sample Python loop, matches step
    2/3's convention exactly -- CPU-only dev machine constraint).

    olr_lag0_series: (n_samples, n_clusters) -- the lag-0 (today) OLR value
        already present in `features`, reused directly rather than
        re-deriving from raw data (this cache is already at cluster
        resolution and already lag-aligned per sample_time_index).
    Returns: (n_samples, n_clusters, generator_window_len, 1) float32.
    """
    n_samples, n_clusters = olr_lag0_series.shape
    W = generator_window_len
    pad = np.repeat(olr_lag0_series[:1, :], W - 1, axis=0) if W > 1 else olr_lag0_series[:0, :]
    padded = np.concatenate([pad, olr_lag0_series], axis=0)  # (n_samples + W - 1, n_clusters)
    windows = np.lib.stride_tricks.sliding_window_view(padded, W, axis=0)  # (n_samples, n_clusters, W)
    return windows[..., None].astype(np.float32)


def sample_swaps(features: np.ndarray, anchor_row: int, n_swaps: int, rng: np.random.Generator) -> np.ndarray:
    """Same place's own history, both time directions (SS4.2/SS8)."""
    n_t = features.shape[0]
    candidates = np.delete(np.arange(n_t), anchor_row)
    chosen = rng.choice(candidates, size=min(n_swaps, candidates.shape[0]), replace=False)
    return features[chosen]


def raw_lagged_correlation_auc(
    olr_lag0_series: np.ndarray, target: int, source_clusters: np.ndarray, lags=(0, 1, 5, 10)
) -> float:
    """Data-hardness baseline, independent of any model (same idea as step
    2's direct diagnostic that first confirmed AO's signal existed in the
    raw data even when the old generator couldn't see it). For each
    candidate source, computes |correlation(source[t-lag], target[t])| at
    several lags and takes the best; then reports how well "is this source
    the strongest-correlated one" separates from "is this some other
    source" via a same style of pairwise AUC used elsewhere in this
    project. With no ground truth, this can't be a true/false-edge AUC --
    instead it reports the max |correlation| found for ANY candidate
    source, as a ceiling on how much raw linear signal exists at all for
    the splitter to discover."""
    n_t = olr_lag0_series.shape[0]
    best_per_source = {}
    for src in source_clusters:
        if src == target:
            continue
        best = 0.0
        for lag in lags:
            if lag == 0:
                a, b = olr_lag0_series[:, src], olr_lag0_series[:, target]
            else:
                a, b = olr_lag0_series[: n_t - lag, src], olr_lag0_series[lag:, target]
            if a.std() < 1e-8 or b.std() < 1e-8:
                continue
            corr = float(np.corrcoef(a, b)[0, 1])
            best = max(best, abs(corr))
        best_per_source[int(src)] = best
    return best_per_source


def train_place(
    variant: str,
    target: int,
    features_all: np.ndarray,
    targets_all: np.ndarray,
    sample_time_index: np.ndarray,
    real_edge_index: np.ndarray,
    n_clusters: int,
    args,
) -> dict:
    """Trains the splitter + expert jointly for ONE place and returns both
    the evaluation summary AND the trained modules, so later steps (SS9
    steps 5-7) can reuse the trained pair instead of retraining inline."""
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    candidate_set = build_candidate_source_set(real_edge_index, target, variant, n_clusters=n_clusters)
    sources = candidate_set.source_clusters
    n_sources = sources.shape[0]
    print(f"\n=== variant={variant}  target={target}  n_candidate_sources={n_sources} ===")

    causal_edge_index_full = torch.from_numpy(expand_source_clusters_to_edge_index(sources, target))

    n_samples_full = features_all.shape[0]
    if args.n_samples_cap > 0 and args.n_samples_cap < n_samples_full:
        if getattr(args, "cap_contiguous", False):
            # Chronological prefix, NOT a random subset: any downstream
            # time-series analysis (SS4.4's drift channels -- ADWIN and
            # STARS both assume time-ordered input) is meaningless on a
            # shuffled subsample, since "adjacent" rows would no longer be
            # adjacent days.
            keep_idx = np.arange(args.n_samples_cap)
        else:
            keep_idx = rng.choice(n_samples_full, size=args.n_samples_cap, replace=False)
            keep_idx.sort()
    else:
        keep_idx = np.arange(n_samples_full)
    n_samples = keep_idx.shape[0]
    print(f"using {n_samples}/{n_samples_full} samples")

    x_all = torch.from_numpy(features_all[keep_idx])  # (n_samples, 50, 21)
    y_all = torch.from_numpy(targets_all[keep_idx])  # (n_samples, 50)
    y_target_all = y_all[:, target]  # (n_samples,)
    features_bank = features_all[keep_idx]  # numpy, for swap sampling (whole 50-node graph)

    # OLR lag-0 channel for the generator's history window: channel layout
    # per windows.py is [lag0(7ch), lag5(7ch), lag10(7ch)], olr is index 5
    # within each 7-wide block -> lag0's olr is absolute channel index 5.
    olr_lag0_all = features_all[:, :, OLR_LAG0_CHANNEL]  # (n_samples_full, 50)
    gen_windows_full = build_generator_windows(olr_lag0_all, sample_time_index, args.generator_window)
    gen_windows = torch.from_numpy(gen_windows_full[keep_idx])  # (n_samples, 50, W, 1)

    n_causal_candidates = causal_edge_index_full.shape[1]
    r = args.r if args.r is not None else min(0.5, 3.0 / n_causal_candidates)

    splitter = DIRGNNSplitter(
        in_channels=21, hidden_channels=16, r=r,
        generator_window_len=args.generator_window, generator_in_channels=1,
    )
    expert = PlaceExpert(in_channels=21, hidden_channels=16)
    optimizer = torch.optim.Adam(list(splitter.parameters()) + list(expert.parameters()), lr=args.lr)

    target_mask = torch.zeros(n_clusters, dtype=torch.bool)
    target_mask[target] = True

    total_steps = args.epochs * n_samples
    schedule = LambdaWarmupSchedule(lambda_max=1.0, total_steps=max(total_steps, 1))

    step = 0
    t0 = time.time()
    for epoch in range(args.epochs):
        order = rng.permutation(n_samples)
        epoch_splitter_loss = 0.0
        epoch_expert_loss = 0.0
        n_batches = 0
        for start in range(0, n_samples - args.batch_size + 1, args.batch_size):
            idx_batch = order[start : start + args.batch_size]
            batch_loss = torch.zeros(())
            optimizer.zero_grad()
            for idx in idx_batch:
                env = sample_swaps(features_bank, int(idx), args.n_swaps, rng)
                env_t = torch.from_numpy(env)
                lam = schedule.value(step)

                splitter_out = splitter.compute_loss(
                    x_all[idx], y_all[idx], env_t, causal_edge_index_full, lam,
                    x_generator_window=gen_windows[idx],
                    entropy_weight=args.entropy_weight,
                    target_node_mask=target_mask,
                )

                split = splitter.split(gen_windows[idx], causal_edge_index_full)
                expert_out = compute_expert_loss(
                    expert, x_all[idx], target, y_target_all[idx],
                    split.causal_edge_index, split.causal_edge_weight,
                )

                sample_loss = splitter_out.total_loss + expert_out.total_loss
                batch_loss = batch_loss + sample_loss
                step += 1

            batch_loss = batch_loss / len(idx_batch)
            batch_loss.backward()
            optimizer.step()

            epoch_splitter_loss += splitter_out.total_loss.item()
            epoch_expert_loss += expert_out.total_loss.item()
            n_batches += 1

        elapsed = time.time() - t0
        print(f"  epoch {epoch+1}/{args.epochs}  "
              f"splitter_loss~={epoch_splitter_loss/max(n_batches,1):.4f}  "
              f"expert_loss~={epoch_expert_loss/max(n_batches,1):.4f}  "
              f"lambda={schedule.value(step):.3f}  elapsed={elapsed:.1f}s")

    elapsed = time.time() - t0
    print(f"training done in {elapsed:.1f}s ({n_samples * args.epochs} sample-steps)")

    # --- Evaluation: forecast accuracy vs naive persistence baseline -----
    splitter.eval()
    expert.eval()
    with torch.no_grad():
        eval_idx = np.arange(n_samples)
        preds = []
        for idx in eval_idx:
            split = splitter.split(gen_windows[idx], causal_edge_index_full)
            pred = expert(x_all[idx], target, split.causal_edge_index, split.causal_edge_weight)
            preds.append(pred.item())
        preds = np.array(preds)

    y_true = y_target_all.numpy()
    expert_mse = float(np.mean((preds - y_true) ** 2))
    persistence_pred = features_all[keep_idx][:, target, OLR_LAG0_CHANNEL]  # today's own OLR (lag0) as N=1 persistence forecast
    persistence_mse = float(np.mean((persistence_pred - y_true) ** 2))

    # --- Edge-score separation diagnostic (no ground truth -- reports
    # spread/entropy of the trained generator's scores, and which sources
    # got selected) ---
    with torch.no_grad():
        sample_idx = rng.choice(n_samples, size=min(200, n_samples), replace=False)
        all_scores = []
        for idx in sample_idx:
            split = splitter.split(gen_windows[idx], causal_edge_index_full)
            all_scores.append(split.edge_scores.numpy())
        mean_scores = np.mean(all_scores, axis=0)

    n_causal = max(1, round(r * n_causal_candidates))
    top_idx = np.argpartition(-mean_scores, n_causal - 1)[:n_causal]
    selected_sources = sorted(sources[i] for i in top_idx if sources[i] != target)
    # keep target's own self-loop visibility explicit if selected
    self_selected = target in [int(sources[i]) for i in top_idx]

    # data-hardness ceiling: best raw |correlation| per candidate source
    corr_ceiling = raw_lagged_correlation_auc(olr_lag0_all, target, sources)
    top3_by_corr = sorted(corr_ceiling.items(), key=lambda kv: -kv[1])[:3]

    print(f"\n--- variant={variant} result ---")
    print(f"r={r:.3f}  n_candidate_sources={n_sources}  n_selected={n_causal}")
    print(f"expert MSE={expert_mse:.4f}  persistence-baseline MSE={persistence_mse:.4f}  "
          f"(expert beats persistence: {expert_mse < persistence_mse})")
    print(f"selected non-self sources: {selected_sources}  self-loop selected: {self_selected}")
    print(f"mean edge score: {mean_scores.mean():.4f}  std: {mean_scores.std():.4f} "
          f"(low std ~ scores clustering near one value, no discrimination)")
    print(f"data-hardness ceiling, top-3 sources by raw |lagged correlation|: {top3_by_corr}")

    return {
        "variant": variant,
        "n_candidate_sources": n_sources,
        "r": r,
        "expert_mse": expert_mse,
        "persistence_mse": persistence_mse,
        "score_std": float(mean_scores.std()),
        "selected_sources": selected_sources,
        "corr_ceiling_top3": top3_by_corr,
        "elapsed_s": elapsed,
        "splitter": splitter,
        "expert": expert,
        "keep_idx": keep_idx,
        "edge_index": causal_edge_index_full,
    }


# Backwards-compatible alias: the original name used by this script's own
# main() before steps 5-7 needed the trained modules back.
run_variant = train_place


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, default=22, help="target cluster id (default 22 = equatorial Sumatra/Maritime Continent)")
    parser.add_argument("--variants", type=str, default="direct,2hop,full")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--r", type=float, default=None, help="top-r fraction; default = min(0.5, 3/n_candidates)")
    parser.add_argument("--n-swaps", type=int, default=6)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--generator-window", type=int, default=10)
    parser.add_argument("--entropy-weight", type=float, default=0.0)
    parser.add_argument("--n-samples-cap", type=int, default=0, help="if >0, subsample this many rows (fast dry run before the full ~16k-sample run)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    t0 = time.time()
    cache = load_cache()
    print(f"loaded cache in {time.time()-t0:.1f}s: features {cache['features'].shape}, "
          f"{cache['n_clusters']} clusters, {cache['edge_index'].shape[1]} directed real edges")

    variants = args.variants.split(",")
    results = []
    for variant in variants:
        result = run_variant(
            variant, args.target, cache["features"], cache["targets"],
            cache["sample_time_index"], cache["edge_index"], cache["n_clusters"], args,
        )
        results.append(result)

    print("\n=== summary across variants ===")
    for r in results:
        beats = "YES" if r["expert_mse"] < r["persistence_mse"] else "no"
        print(f"{r['variant']:>6}: n_sources={r['n_candidate_sources']:>3}  "
              f"expert_mse={r['expert_mse']:.4f}  persistence_mse={r['persistence_mse']:.4f}  "
              f"beats_persistence={beats}  score_std={r['score_std']:.4f}  time={r['elapsed_s']:.1f}s")

    results_dir = Path(__file__).resolve().parents[1] / "results"
    results_dir.mkdir(exist_ok=True)
    serializable = [
        {k: v for k, v in r.items() if k not in ("splitter", "expert", "keep_idx", "edge_index")}
        for r in results
    ]
    for r in serializable:
        r["selected_sources"] = [int(s) for s in r["selected_sources"]]
        r["corr_ceiling_top3"] = [[int(s), float(c)] for s, c in r["corr_ceiling_top3"]]
    out_path = results_dir / f"step4_place{args.target}.json"
    out_path.write_text(json.dumps({"target": args.target, "variants": serializable}, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
