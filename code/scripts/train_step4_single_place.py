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
from causal_moe.data.splits import chronological_split
from causal_moe.data.windows import LAGS_DAYS, N_CHANNELS_PER_LAG, channel_index
from causal_moe.experts.expert import PlaceExpert, compute_expert_loss
from causal_moe.splitter.dirgnn import DIRGNNSplitter, LambdaWarmupSchedule

CACHE_PATH = Path(__file__).resolve().parents[1] / "cache" / "windowed_clustered50_lead1.npz"


def load_cache(path: Path = CACHE_PATH):
    d = np.load(path, allow_pickle=True)
    lags_days = tuple(d["lags_days"].tolist()) if "lags_days" in d else (0, 5, 10)
    return {
        "features": d["features"],  # (n_samples, 50, n_features)
        "targets": d["targets"],  # (n_samples, 50)
        "sample_time_index": d["sample_time_index"],
        "sample_dates": d["sample_dates"],
        "edge_index": d["edge_index"],  # (2, n_edges) real lattice adjacency
        "n_clusters": int(d["n_clusters"]),
        "lags_days": lags_days,
        # olr lag-0 (today) absolute channel index within a place's feature
        # row -- computed from the cache's OWN lag set (B16-style coupling
        # avoided: this cache may be the legacy 21-feature/3-lag cache or
        # the widened 49-feature/7-lag cache, never hardcoded).
        "olr_lag0_channel": channel_index(0, "olr", lags_days=lags_days),
    }


def build_generator_windows(
    lag0_series: np.ndarray, sample_time_index: np.ndarray, generator_window_len: int
) -> np.ndarray:
    """Vectorized sliding window (no per-sample Python loop, matches step
    2/3's convention exactly -- CPU-only dev machine constraint).

    lag0_series: (n_samples, n_clusters) for the single-channel (OLR-only)
        case, or (n_samples, n_clusters, n_channels) for the B7 audit fix
        (feed the rationale generator all 6 fields, not just OLR -- at lead
        7, self-OLR alone gives R^2=0.065 while all six fields give 0.206;
        independently corroborated by Maeda et al. 2025, GRL, which finds
        PW/SST/U200 dominate BSISO predictive skill beyond 15 days).
        Reused directly from `features` rather than re-deriving from raw
        data (already at cluster resolution and lag-aligned per
        sample_time_index).
    Returns: (n_samples, n_clusters, generator_window_len, n_channels)
        float32 (n_channels=1 for the single-channel input case).
    """
    if lag0_series.ndim == 2:
        lag0_series = lag0_series[:, :, None]
    n_samples, n_clusters, n_channels = lag0_series.shape
    W = generator_window_len
    pad = np.repeat(lag0_series[:1], W - 1, axis=0) if W > 1 else lag0_series[:0]
    padded = np.concatenate([pad, lag0_series], axis=0)  # (n_samples + W - 1, n_clusters, n_channels)
    # sliding_window_view appends the window axis at the end for EACH axis
    # given; only slide along axis 0 (time), keep cluster/channel intact.
    windows = np.lib.stride_tricks.sliding_window_view(padded, W, axis=0)  # (n_samples, n_clusters, n_channels, W)
    return np.moveaxis(windows, -1, 2).astype(np.float32)  # (n_samples, n_clusters, W, n_channels)


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
    sample_dates: np.ndarray | None = None,
    olr_lag0_channel: int = 5,
    lags_days: tuple = (0, 5, 10),
    lead_time_days: int = 1,
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

    n_features = features_all.shape[-1]

    n_samples_full = features_all.shape[0]
    sample_dates_full = sample_dates  # (n_samples_full,) datetime64, or None

    # B4/Phase 3 audit fix: honest chronological train/test split instead of
    # evaluating on the training rows. train < val_start, val in
    # [val_start, test_start), test >= test_start (test span deliberately
    # contains the 2013-2022 gradual-drift window -- that's the OOD test).
    if sample_dates_full is not None and getattr(args, "test_start", None) is not None:
        from datetime import date as _date
        val_start = _date.fromisoformat(args.val_start)
        test_start = _date.fromisoformat(args.test_start)
        # B22 audit fix: pass the real lead time so no training/val sample's
        # TARGET (lead_time_days after its "today") falls inside the next
        # split's window -- see splits.py's chronological_split docstring.
        split_masks = chronological_split(sample_dates_full, val_start, test_start, lead_time_days=lead_time_days)
        train_pool_idx = np.nonzero(split_masks.train_mask | split_masks.val_mask)[0]
        test_idx_full = np.nonzero(split_masks.test_mask)[0]
    else:
        # No dates available (e.g. legacy cache) -- fall back to a plain
        # chronological 70/15/15 split so eval is still out-of-sample.
        n70 = int(0.70 * n_samples_full)
        n85 = int(0.85 * n_samples_full)
        train_pool_idx = np.arange(0, n85)
        test_idx_full = np.arange(n85, n_samples_full)

    # B5 audit fix: --n-samples-cap now defaults to a CONTIGUOUS chronological
    # prefix of the training pool (cap_contiguous=True by default), not a
    # random subsample -- a random subsample both destroys temporal
    # structure and, since it was drawn from the FULL record pre-split,
    # guaranteed train/test contamination.
    if args.n_samples_cap > 0 and args.n_samples_cap < train_pool_idx.shape[0]:
        if args.cap_contiguous:
            keep_idx = train_pool_idx[: args.n_samples_cap]
        else:
            keep_idx = rng.choice(train_pool_idx, size=args.n_samples_cap, replace=False)
            keep_idx.sort()
    else:
        keep_idx = train_pool_idx
    n_samples = keep_idx.shape[0]
    print(f"using {n_samples}/{train_pool_idx.shape[0]} train-pool samples "
          f"({test_idx_full.shape[0]} held out out-of-sample for test)")

    x_all_train = torch.from_numpy(features_all[keep_idx])  # (n_samples, 50, n_features)
    y_all_train = torch.from_numpy(targets_all[keep_idx])  # (n_samples, 50)
    y_target_train = y_all_train[:, target]  # (n_samples,)
    features_bank = features_all[keep_idx]  # numpy, for swap sampling (whole 50-node graph)

    x_all_test = torch.from_numpy(features_all[test_idx_full])
    y_all_test = torch.from_numpy(targets_all[test_idx_full])
    y_target_test = y_all_test[:, target]

    olr_lag0_all = features_all[:, :, olr_lag0_channel]  # (n_samples_full, 50)
    # B7 audit fix: when --generator-all-fields, feed the rationale
    # generator ALL fields' lag-0 values (not just OLR) -- at lead 7,
    # self-OLR alone gives R^2=0.065 while all six fields give 0.206
    # (independently corroborated by Maeda et al. 2025, GRL: PW/SST/U200
    # dominate BSISO skill beyond 15 days). Default off (backward compat).
    if getattr(args, "generator_all_fields", False):
        from causal_moe.data.raw import FIELDS
        all_field_channels = np.stack(
            [features_all[:, :, channel_index(0, f, lags_days=lags_days)] for f in FIELDS], axis=-1
        )  # (n_samples_full, 50, n_fields)
        gen_source = all_field_channels
        gen_in_channels = len(FIELDS)
    else:
        gen_source = olr_lag0_all
        gen_in_channels = 1
    gen_windows_full = build_generator_windows(gen_source, sample_time_index, args.generator_window)
    gen_windows = torch.from_numpy(gen_windows_full[keep_idx])  # (n_samples, 50, W, gen_in_channels)
    gen_windows_test = torch.from_numpy(gen_windows_full[test_idx_full])

    n_causal_candidates = causal_edge_index_full.shape[1]
    r = args.r if args.r is not None else min(0.5, 3.0 / n_causal_candidates)

    splitter = DIRGNNSplitter(
        in_channels=n_features, hidden_channels=16, r=r,
        generator_window_len=args.generator_window, generator_in_channels=gen_in_channels,
        use_self_features=args.use_self_features,
    )
    expert = PlaceExpert(in_channels=n_features, hidden_channels=16, self_bypass=args.self_bypass)
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
                    x_all_train[idx], y_all_train[idx], env_t, causal_edge_index_full, lam,
                    x_generator_window=gen_windows[idx],
                    entropy_weight=args.entropy_weight,
                    target_node_mask=target_mask,
                )

                # Reuse compute_loss's own internal split instead of calling
                # splitter.split(...) again (perf fix, 2026-09-23: profiling
                # showed the redundant second rationale-generator forward
                # pass was a meaningful share of per-step cost).
                split = splitter_out.split
                expert_out = compute_expert_loss(
                    expert, x_all_train[idx], target, y_target_train[idx],
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

    # --- Evaluation: OUT-OF-SAMPLE test span, forecast accuracy vs
    # persistence AND climatology (B4/B6 audit fix -- eval_idx used to be
    # arange(n_samples) over the TRAINING rows; now it's the held-out test
    # split from chronological_split, computed once above). ------------
    splitter.eval()
    expert.eval()
    n_test = x_all_test.shape[0]
    with torch.no_grad():
        preds = []
        for idx in range(n_test):
            split = splitter.split(gen_windows_test[idx], causal_edge_index_full)
            pred = expert(x_all_test[idx], target, split.causal_edge_index, split.causal_edge_weight)
            preds.append(pred.item())
        preds = np.array(preds)

    y_true = y_target_test.numpy()
    expert_mse = float(np.mean((preds - y_true) ** 2))
    persistence_pred = features_all[test_idx_full][:, target, olr_lag0_channel]  # today's own OLR (lag0) as N=1 persistence forecast
    persistence_mse = float(np.mean((persistence_pred - y_true) ** 2))
    # Climatology baseline: predict the TRAINING span's mean target value for
    # every test row (B6 audit fix -- "beats persistence by X%" alone hides
    # how small the persistence residual itself is; R^2 vs climatology is
    # the honest denominator).
    climatology_pred = float(y_target_train.numpy().mean())
    climatology_mse = float(np.mean((climatology_pred - y_true) ** 2))
    target_var = float(np.var(y_true))
    r2_vs_climatology = 1.0 - expert_mse / max(climatology_mse, 1e-12)
    r2_vs_persistence = 1.0 - expert_mse / max(persistence_mse, 1e-12)

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
    # Per-source mean score, keyed by source cluster id -- B9 audit fix:
    # lets a caller (e.g. run_pcmci_crosscheck.py) compute Spearman rank
    # agreement against PCMCI's effect-size ranking, not just set overlap.
    source_scores = {int(sources[i]): float(mean_scores[i]) for i in range(len(sources))}

    # data-hardness ceiling: best raw |correlation| per candidate source
    corr_ceiling = raw_lagged_correlation_auc(olr_lag0_all, target, sources)
    top3_by_corr = sorted(corr_ceiling.items(), key=lambda kv: -kv[1])[:3]

    print(f"\n--- variant={variant} result (OUT-OF-SAMPLE test, n_test={n_test}) ---")
    print(f"r={r:.3f}  n_candidate_sources={n_sources}  n_selected={n_causal}")
    print(f"expert MSE={expert_mse:.4f}  persistence MSE={persistence_mse:.4f}  climatology MSE={climatology_mse:.4f}  "
          f"target_var={target_var:.4f}")
    print(f"R^2 vs persistence={r2_vs_persistence:.4f}  R^2 vs climatology={r2_vs_climatology:.4f}  "
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
        "climatology_mse": climatology_mse,
        "target_var": target_var,
        "r2_vs_persistence": r2_vs_persistence,
        "r2_vs_climatology": r2_vs_climatology,
        "n_test": n_test,
        "score_std": float(mean_scores.std()),
        "selected_sources": selected_sources,
        "source_scores": source_scores,
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
    parser.add_argument("--n-samples-cap", type=int, default=0, help="if >0, cap the TRAINING pool to this many rows (fast dry run before the full run)")
    parser.add_argument("--cap-contiguous", type=lambda s: s.lower() != "false", default=True,
                         help="B5 audit fix: default True -- cap is a chronological prefix, not a random subsample. Pass --cap-contiguous false to restore the old (buggy) random-subsample behaviour.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cache-path", type=Path, default=CACHE_PATH)
    parser.add_argument("--val-start", type=str, default="2005-01-01", help="B4/Phase 3 split: train < val_start")
    parser.add_argument("--test-start", type=str, default="2013-01-01", help="B4/Phase 3 split: test >= test_start (contains the gradual-drift OOD window)")
    parser.add_argument("--use-self-features", type=lambda s: s.lower() != "false", default=False,
                         help="Bug 2 audit fix: default False -- splitter's shared encoder no longer force-fed the node's own raw features, so Var_swaps[risk] is load-bearing. Pass true to restore the old self-bypass ablation.")
    parser.add_argument("--self-bypass", type=lambda s: s.lower() != "false", default=False,
                         help="Bug 2 audit fix: default False -- expert MLP no longer force-fed the target's own raw features unconditionally. Pass true to restore the old self-bypass ablation.")
    parser.add_argument("--generator-all-fields", action="store_true",
                         help="B7 audit fix: feed the rationale generator all 6 fields' lag-0 values "
                              "instead of OLR alone (at lead 7, OLR-only gives R^2=0.065 vs 0.206 for all "
                              "six fields; corroborated by Maeda et al. 2025, GRL). Default off (backward compat).")
    args = parser.parse_args()

    t0 = time.time()
    cache = load_cache(args.cache_path)
    print(f"loaded cache in {time.time()-t0:.1f}s: features {cache['features'].shape}, "
          f"{cache['n_clusters']} clusters, {cache['edge_index'].shape[1]} directed real edges, "
          f"lags_days={cache['lags_days']}")

    sample_dates = cache["sample_dates"].astype("datetime64[D]") if "sample_dates" in cache else None

    variants = args.variants.split(",")
    results = []
    for variant in variants:
        result = run_variant(
            variant, args.target, cache["features"], cache["targets"],
            cache["sample_time_index"], cache["edge_index"], cache["n_clusters"], args,
            sample_dates=sample_dates, olr_lag0_channel=cache["olr_lag0_channel"],
            lags_days=cache["lags_days"], lead_time_days=int(cache["lead_time_days"]),
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
