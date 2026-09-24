"""CIA-for-regression, wired into an actual training loop and run on real
data -- the task flagged as "next session's first task" at the end of §10
step 2/4/8 and in `causal_moe/splitter/cia.py`'s own docstring.

Reuses `scripts/train_step4_single_place.py`'s exact data loading, cache,
generator-window construction, and candidate-edge-set variants (direct/
2hop/full) so the CIA-enabled run is comparable apples-to-apples with step
4/8's numbers -- only the loss adds one extra term.

**What CIA adds on top of the existing splitter+expert loss (per-sample):**
the existing loop already draws `n_swaps` environments per anchor sample to
compute VREx's mean/variance-across-swaps risk (§4.2). CIA (§10 step 2's
"CIA-for-regression", `causal_moe/splitter/cia.py`) is a DIFFERENT kind of
term: it needs a BATCH of anchors' hidden representations, all computed
under the SAME swap index, so it can compare "do same-outcome anchors look
alike regardless of which environment got swapped in" -- that comparison is
across anchors within a batch, not across swaps within one anchor (the
existing VREx term already covers swap-to-swap variance, per the CIA module
docstring). Concretely, per mini-batch:
  1. Draw ONE shared swap environment per anchor in the batch (each anchor
     still uses its OWN memory bank -- same place, different times, §4.2 --
     just one swap draw each, not `n_swaps`).
  2. Run `predict_causal(..., return_hidden=True)` for every anchor in the
     batch under its one shared-index swap, collecting (hidden, y) pairs.
  3. `cia_alignment_loss(hidden_batch, y_batch, bandwidth)` -- pulls
     same-outcome anchors' representations together.
  4. Add `cia_weight * cia_loss` to the batch loss, alongside the existing
     per-sample splitter+expert losses (unchanged).

This keeps the existing VREx mechanism fully intact and adds CIA as a
strictly additive batch-level term, matching how `cia.py` was designed
(§10: "an alignment term ... additive to the existing objective").

Usage:
    python scripts/train_step_cia_single_place.py [--target 22]
        [--epochs 3] [--variant direct] [--cia-weight 0.0]
        [--cia-bandwidth 1.0] [--n-samples-cap 8000]

Run with --cia-weight 0.0 to reproduce step 4's number exactly (sanity
check that CIA is additive, not disruptive, when switched off) and with
--cia-weight > 0 for the real before/after comparison.
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

from causal_moe.data.candidate_edges import (
    build_candidate_source_set,
    expand_source_clusters_to_edge_index,
)
from causal_moe.data.splits import chronological_split
from causal_moe.experts.expert import PlaceExpert, compute_expert_loss
from causal_moe.splitter.cia import cia_alignment_loss
from causal_moe.splitter.dirgnn import DIRGNNSplitter, LambdaWarmupSchedule
from scripts.train_step4_single_place import (
    build_generator_windows,
    load_cache,
    raw_lagged_correlation_auc,
    sample_swaps,
)

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def train_place_with_cia(
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
) -> dict:
    """Same structure as train_step4_single_place.train_place, with one
    addition: a CIA alignment term computed once per mini-batch (see module
    docstring). When args.cia_weight == 0.0 this reduces to step 4's loop
    exactly (kept as a literal sanity check before trusting the >0 numbers)."""
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    candidate_set = build_candidate_source_set(real_edge_index, target, variant, n_clusters=n_clusters)
    sources = candidate_set.source_clusters
    n_sources = sources.shape[0]
    print(f"\n=== variant={variant}  target={target}  n_candidate_sources={n_sources}  cia_weight={args.cia_weight} ===")

    causal_edge_index_full = torch.from_numpy(expand_source_clusters_to_edge_index(sources, target))

    n_features = features_all.shape[-1]
    n_samples_full = features_all.shape[0]

    # B4/Phase 3 audit fix: honest out-of-sample split, same convention as
    # train_step4_single_place.train_place.
    if sample_dates is not None and getattr(args, "test_start", None) is not None:
        from datetime import date as _date
        split_masks = chronological_split(
            sample_dates, _date.fromisoformat(args.val_start), _date.fromisoformat(args.test_start)
        )
        train_pool_idx = np.nonzero(split_masks.train_mask | split_masks.val_mask)[0]
        test_idx_full = np.nonzero(split_masks.test_mask)[0]
    else:
        n85 = int(0.85 * n_samples_full)
        train_pool_idx = np.arange(0, n85)
        test_idx_full = np.arange(n85, n_samples_full)

    if args.n_samples_cap > 0 and args.n_samples_cap < train_pool_idx.shape[0]:
        keep_idx = train_pool_idx[: args.n_samples_cap]  # B5: chronological prefix
    else:
        keep_idx = train_pool_idx
    n_samples = keep_idx.shape[0]
    print(f"using {n_samples}/{train_pool_idx.shape[0]} train-pool samples "
          f"({test_idx_full.shape[0]} held out out-of-sample for test)")

    x_all = torch.from_numpy(features_all[keep_idx])  # (n_samples, 50, n_features)
    y_all = torch.from_numpy(targets_all[keep_idx])  # (n_samples, 50)
    y_target_all = y_all[:, target]  # (n_samples,)
    features_bank = features_all[keep_idx]  # numpy, for swap sampling (whole 50-node graph)

    x_all_test = torch.from_numpy(features_all[test_idx_full])
    y_target_test = torch.from_numpy(targets_all[test_idx_full][:, target])

    olr_lag0_all = features_all[:, :, olr_lag0_channel]  # (n_samples_full, 50)
    gen_windows_full = build_generator_windows(olr_lag0_all, sample_time_index, args.generator_window)
    gen_windows = torch.from_numpy(gen_windows_full[keep_idx])  # (n_samples, 50, W, 1)
    gen_windows_test = torch.from_numpy(gen_windows_full[test_idx_full])

    n_causal_candidates = causal_edge_index_full.shape[1]
    r = args.r if args.r is not None else min(0.5, 3.0 / n_causal_candidates)

    splitter = DIRGNNSplitter(
        in_channels=n_features, hidden_channels=16, r=r,
        generator_window_len=args.generator_window, generator_in_channels=1,
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
        epoch_cia_loss = 0.0
        n_batches = 0
        for start in range(0, n_samples - args.batch_size + 1, args.batch_size):
            idx_batch = order[start : start + args.batch_size]
            batch_loss = torch.zeros(())
            optimizer.zero_grad()

            cia_hidden_list = []
            cia_y_list = []

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

                # Reuse compute_loss's own internal split (perf fix,
                # 2026-09-23, see train_step4_single_place.py).
                split = splitter_out.split
                expert_out = compute_expert_loss(
                    expert, x_all[idx], target, y_target_all[idx],
                    split.causal_edge_index, split.causal_edge_weight,
                )

                sample_loss = splitter_out.total_loss + expert_out.total_loss
                batch_loss = batch_loss + sample_loss

                if args.cia_weight > 0.0:
                    # One shared-index swap per anchor (independent of the
                    # n_swaps VREx draws above) -- see module docstring for
                    # why this is a separate draw, not reused from env_t.
                    cia_env = sample_swaps(features_bank, int(idx), 1, rng)[0]
                    cia_env_t = torch.from_numpy(cia_env)
                    _, hidden = splitter.predict_causal(x_all[idx], split, cia_env_t, return_hidden=True)
                    cia_hidden_list.append(hidden[target])  # this place's own row of the hidden embedding
                    cia_y_list.append(y_target_all[idx])

                step += 1

            if args.cia_weight > 0.0 and len(cia_hidden_list) > 1:
                hidden_batch = torch.stack(cia_hidden_list, dim=0)  # (batch, hidden_channels)
                y_batch = torch.stack(cia_y_list, dim=0)  # (batch,)
                cia_out = cia_alignment_loss(hidden_batch, y_batch, bandwidth=args.cia_bandwidth)
                batch_loss = batch_loss + args.cia_weight * cia_out.loss * len(idx_batch)
                epoch_cia_loss += float(cia_out.loss.detach())

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
              f"cia_loss~={epoch_cia_loss/max(n_batches,1):.4f}  "
              f"lambda={schedule.value(step):.3f}  elapsed={elapsed:.1f}s")

    elapsed = time.time() - t0
    print(f"training done in {elapsed:.1f}s ({n_samples * args.epochs} sample-steps)")

    # --- Evaluation: OUT-OF-SAMPLE test span (B4 audit fix), same
    # convention as step 4/8 so numbers stay comparable across scripts. ----
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
    persistence_pred = features_all[test_idx_full][:, target, olr_lag0_channel]
    persistence_mse = float(np.mean((persistence_pred - y_true) ** 2))
    climatology_pred = float(y_target_all.numpy().mean())
    climatology_mse = float(np.mean((climatology_pred - y_true) ** 2))
    r2_vs_climatology = 1.0 - expert_mse / max(climatology_mse, 1e-12)

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
    self_selected = target in [int(sources[i]) for i in top_idx]

    corr_ceiling = raw_lagged_correlation_auc(olr_lag0_all, target, sources)
    top3_by_corr = sorted(corr_ceiling.items(), key=lambda kv: -kv[1])[:3]

    print(f"\n--- variant={variant} cia_weight={args.cia_weight} result (OUT-OF-SAMPLE, n_test={n_test}) ---")
    print(f"r={r:.3f}  n_candidate_sources={n_sources}  n_selected={n_causal}")
    print(f"expert MSE={expert_mse:.4f}  persistence MSE={persistence_mse:.4f}  climatology MSE={climatology_mse:.4f}  "
          f"R^2 vs climatology={r2_vs_climatology:.4f}  "
          f"(expert beats persistence: {expert_mse < persistence_mse})")
    print(f"selected non-self sources: {selected_sources}  self-loop selected: {self_selected}")
    print(f"mean edge score: {mean_scores.mean():.4f}  std: {mean_scores.std():.4f}")
    print(f"data-hardness ceiling, top-3 sources by raw |lagged correlation|: {top3_by_corr}")

    return {
        "variant": variant,
        "cia_weight": args.cia_weight,
        "cia_bandwidth": args.cia_bandwidth,
        "n_candidate_sources": n_sources,
        "r": r,
        "expert_mse": expert_mse,
        "persistence_mse": persistence_mse,
        "climatology_mse": climatology_mse,
        "r2_vs_climatology": r2_vs_climatology,
        "n_test": n_test,
        "score_std": float(mean_scores.std()),
        "selected_sources": [int(s) for s in selected_sources],
        "self_loop_selected": bool(self_selected),
        "corr_ceiling_top3": [[int(s), float(c)] for s, c in top3_by_corr],
        "elapsed_s": elapsed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, default=22)
    parser.add_argument("--variant", type=str, default="direct")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--r", type=float, default=None)
    parser.add_argument("--n-swaps", type=int, default=6)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--generator-window", type=int, default=10)
    parser.add_argument("--entropy-weight", type=float, default=0.0)
    parser.add_argument("--cia-weight", type=float, default=1.0, help="0.0 reproduces step 4's loop exactly (sanity check)")
    parser.add_argument("--cia-bandwidth", type=float, default=1.0)
    parser.add_argument("--n-samples-cap", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cache-path", type=Path, default=None)
    parser.add_argument("--val-start", type=str, default="2005-01-01")
    parser.add_argument("--test-start", type=str, default="2013-01-01")
    parser.add_argument("--use-self-features", type=lambda s: s.lower() != "false", default=False)
    parser.add_argument("--self-bypass", type=lambda s: s.lower() != "false", default=False)
    args = parser.parse_args()

    t0 = time.time()
    from scripts.train_step4_single_place import CACHE_PATH as DEFAULT_CACHE_PATH
    cache = load_cache(args.cache_path or DEFAULT_CACHE_PATH)
    print(f"loaded cache in {time.time()-t0:.1f}s: features {cache['features'].shape}, "
          f"{cache['n_clusters']} clusters, {cache['edge_index'].shape[1]} directed real edges")

    sample_dates = cache["sample_dates"].astype("datetime64[D]")
    result = train_place_with_cia(
        args.variant, args.target, cache["features"], cache["targets"],
        cache["sample_time_index"], cache["edge_index"], cache["n_clusters"], args,
        sample_dates=sample_dates, olr_lag0_channel=cache["olr_lag0_channel"],
    )

    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / f"step_cia_place{args.target}_{args.variant}_w{args.cia_weight}.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
