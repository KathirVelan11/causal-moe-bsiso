"""GSINA (differentiable Sinkhorn selection), wired into an actual training
loop and run on real data -- Task 2 of this session, built after CIA was
tested and found not to help (§10, "CIA-for-regression" entry).

Reuses `scripts/train_step4_single_place.py`'s exact data loading, cache,
generator-window construction, and candidate-edge-set variants, so the
GSINA-enabled run is comparable apples-to-apples with step 4/8's numbers.
The ONLY change: `splitter.split()`'s hard top-r selection
(`split_graph_top_r`, gradient-detached) is replaced by
`causal_moe.splitter.gsina.gsina_split` (fully differentiable Sinkhorn
selection, `causal_moe/splitter/gsina.py`). This script does NOT touch
CIA at all -- the two fixes are kept independently testable, per the
original plan (§10 GSINA entry: "not run combined in the same experiment").

Because GSINA's split() has a different signature/output shape than
DIRGNNSplitter's own `split()` (every edge appears in BOTH the causal and
spurious groups, weighted by keep-probability, rather than a disjoint hard
partition -- see gsina.py's docstring), this script reimplements the
training loop directly against `DIRGNNSplitter`'s lower-level pieces
(`generator`, `encoder`, `causal_head`, `spurious_head`) rather than
calling `DIRGNNSplitter.compute_loss`/`.split()`, which are hard-coded to
`split_graph_top_r`. This keeps GSINA a genuine drop-in ALTERNATIVE
selection mechanism, not a fork of the splitter class itself.

Usage:
    python scripts/train_step_gsina_single_place.py [--target 22]
        [--epochs 2] [--variant direct] [--n-samples-cap 4000]
        [--gsina-iters 20] [--gsina-temperature 1.0]
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
import torch.nn.functional as F

from causal_moe.data.candidate_edges import (
    build_candidate_source_set,
    expand_source_clusters_to_edge_index,
)
from causal_moe.data.splits import chronological_split
from causal_moe.experts.expert import PlaceExpert, compute_expert_loss
from causal_moe.splitter.cia import cia_alignment_loss
from causal_moe.splitter.dirgnn import LambdaWarmupSchedule, RationaleGenerator, RegressionHead, SharedEncoder
from causal_moe.splitter.gsina import gsina_split
from scripts.train_step4_single_place import (
    build_generator_windows,
    load_cache,
    raw_lagged_correlation_auc,
    sample_swaps,
)

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def train_place_with_gsina(
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
    lead_time_days: int = 1,
) -> dict:
    """Same structure/budget/evaluation as train_step4_single_place.train_place
    and train_step_cia_single_place.train_place_with_cia, so all three are
    directly comparable. Rebuilds the splitter's pieces directly (generator,
    encoder, causal/spurious heads) instead of using DIRGNNSplitter.split(),
    since that method is hard-coded to split_graph_top_r (see module
    docstring)."""
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    candidate_set = build_candidate_source_set(real_edge_index, target, variant, n_clusters=n_clusters)
    sources = candidate_set.source_clusters
    n_sources = sources.shape[0]
    print(f"\n=== variant={variant}  target={target}  n_candidate_sources={n_sources}  GSINA(iters={args.gsina_iters}, temp={args.gsina_temperature}) ===")

    causal_edge_index_full = torch.from_numpy(expand_source_clusters_to_edge_index(sources, target))

    n_features = features_all.shape[-1]
    n_samples_full = features_all.shape[0]

    if sample_dates is not None and getattr(args, "test_start", None) is not None:
        from datetime import date as _date
        # B22 audit fix: pass the real lead time so no split boundary leaks
        # a training target into the held-out window (see splits.py).
        split_masks = chronological_split(
            sample_dates, _date.fromisoformat(args.val_start), _date.fromisoformat(args.test_start),
            lead_time_days=lead_time_days,
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

    x_all = torch.from_numpy(features_all[keep_idx])
    y_all = torch.from_numpy(targets_all[keep_idx])
    y_target_all = y_all[:, target]
    features_bank = features_all[keep_idx]

    x_all_test = torch.from_numpy(features_all[test_idx_full])
    y_target_test = torch.from_numpy(targets_all[test_idx_full][:, target])

    olr_lag0_all = features_all[:, :, olr_lag0_channel]
    gen_windows_full = build_generator_windows(olr_lag0_all, sample_time_index, args.generator_window)
    gen_windows = torch.from_numpy(gen_windows_full[keep_idx])
    gen_windows_test = torch.from_numpy(gen_windows_full[test_idx_full])

    n_causal_candidates = causal_edge_index_full.shape[1]
    r = args.r if args.r is not None else min(0.5, 3.0 / n_causal_candidates)

    in_channels = n_features
    hidden_channels = 16
    generator = RationaleGenerator(1, args.generator_window, hidden_channels)
    encoder = SharedEncoder(in_channels, hidden_channels, use_self_features=args.use_self_features)
    causal_head = RegressionHead(hidden_channels)
    expert = PlaceExpert(in_channels=in_channels, hidden_channels=hidden_channels, self_bypass=args.self_bypass)
    optimizer = torch.optim.Adam(
        list(generator.parameters()) + list(encoder.parameters())
        + list(causal_head.parameters()) + list(expert.parameters()),
        lr=args.lr,
    )

    target_idx = target  # single-target risk (star graph, same convention as target_node_mask elsewhere)

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

            cia_hidden_list = []
            cia_y_list = []

            for idx in idx_batch:
                lam = schedule.value(step)
                raw_scores = generator(gen_windows[idx], causal_edge_index_full)  # (n_edges,)
                gsina_out = gsina_split(
                    raw_scores, causal_edge_index_full, r=r,
                    n_iters=args.gsina_iters, temperature=args.gsina_temperature,
                    gumbel_noise=args.gsina_gumbel,
                )

                x_anchor = x_all[idx]
                env = sample_swaps(features_bank, int(idx), args.n_swaps, rng)
                env_t = torch.from_numpy(env)
                n_swaps_i = env_t.shape[0]

                # Vectorized across all n_swaps environments in one encoder
                # forward pass (Bug 15/Phase -1 audit fix, 2026-09-23), same
                # node-id-offset tiling as DIRGNNSplitter.predict_causal_batch.
                offsets = torch.arange(n_swaps_i) * n_clusters
                c_idx, c_w = gsina_out.causal_edge_index, gsina_out.causal_edge_weight
                s_idx, s_w = gsina_out.spurious_edge_index, gsina_out.spurious_edge_weight
                c_idx_b = (c_idx.unsqueeze(0) + offsets.view(n_swaps_i, 1, 1)).permute(1, 0, 2).reshape(2, -1)
                s_idx_b = (s_idx.unsqueeze(0) + offsets.view(n_swaps_i, 1, 1)).permute(1, 0, 2).reshape(2, -1)
                c_w_b = c_w.unsqueeze(0).expand(n_swaps_i, -1).reshape(-1)
                s_w_b = s_w.unsqueeze(0).expand(n_swaps_i, -1).reshape(-1)
                x_anchor_tiled = x_anchor.unsqueeze(0).expand(n_swaps_i, -1, -1).reshape(n_swaps_i * n_clusters, in_channels)
                x_env_flat = env_t.reshape(n_swaps_i * n_clusters, in_channels)
                edge_groups_batched = [
                    (c_idx_b, c_w_b, x_anchor_tiled),
                    (s_idx_b, s_w_b, x_env_flat),
                ]
                h_batched = encoder(x_anchor_tiled, edge_groups_batched, n_swaps_i * n_clusters)
                pred_batched = causal_head(h_batched)  # (n_swaps * n_clusters,)
                predictions_per_swap = pred_batched.reshape(n_swaps_i, n_clusters)

                if args.cia_weight > 0.0:
                    # CIA task 2 combined-check only (--cia-weight > 0):
                    # reuse swap-index-0's hidden state from the batched
                    # forward pass above -- no extra pass needed.
                    h0 = h_batched.reshape(n_swaps_i, n_clusters, -1)[0]
                    cia_hidden_list.append(h0[target_idx])
                    cia_y_list.append(y_target_all[idx])

                sq_err = (predictions_per_swap - y_all[idx].unsqueeze(0)) ** 2
                risks = sq_err[:, target_idx]  # (n_swaps,) -- single-target risk, same convention as target_node_mask
                mean_risk = risks.mean()
                variance_risk = risks.var(unbiased=False) if risks.shape[0] > 1 else torch.zeros_like(mean_risk)
                splitter_loss = mean_risk + lam * variance_risk

                # spurious leakage gauge (measurement only, gradient-blocked
                # from generator/encoder -- same role as DIRGNNSplitter's own
                # predict_spurious_gauge, reimplemented here since GSINA's
                # split() shape differs from SplitterOutput)
                with torch.no_grad():
                    h_s = encoder(x_anchor.detach(), [(gsina_out.spurious_edge_index, gsina_out.spurious_edge_weight.detach(), x_anchor.detach())], n_clusters)

                expert_out = compute_expert_loss(
                    expert, x_anchor, target, y_target_all[idx],
                    gsina_out.causal_edge_index, gsina_out.causal_edge_weight,
                )

                sample_loss = splitter_loss + expert_out.total_loss
                batch_loss = batch_loss + sample_loss
                step += 1

            if args.cia_weight > 0.0 and len(cia_hidden_list) > 1:
                hidden_batch = torch.stack(cia_hidden_list, dim=0)
                y_batch = torch.stack(cia_y_list, dim=0)
                cia_out = cia_alignment_loss(hidden_batch, y_batch, bandwidth=args.cia_bandwidth)
                batch_loss = batch_loss + args.cia_weight * cia_out.loss * len(idx_batch)

            batch_loss = batch_loss / len(idx_batch)
            batch_loss.backward()
            optimizer.step()

            epoch_splitter_loss += float(splitter_loss.detach())
            epoch_expert_loss += float(expert_out.total_loss.detach())
            n_batches += 1

        elapsed = time.time() - t0
        print(f"  epoch {epoch+1}/{args.epochs}  "
              f"splitter_loss~={epoch_splitter_loss/max(n_batches,1):.4f}  "
              f"expert_loss~={epoch_expert_loss/max(n_batches,1):.4f}  "
              f"lambda={schedule.value(step):.3f}  elapsed={elapsed:.1f}s")

    elapsed = time.time() - t0
    print(f"training done in {elapsed:.1f}s ({n_samples * args.epochs} sample-steps)")

    # --- Evaluation: OUT-OF-SAMPLE test span (B4 audit fix), same
    # convention as step 4/8/CIA for comparability. --------------------
    generator.eval()
    encoder.eval()
    causal_head.eval()
    expert.eval()
    n_test = x_all_test.shape[0]
    with torch.no_grad():
        preds = []
        all_keep_probs = []
        sample_score_idx = rng.choice(n_test, size=min(200, n_test), replace=False)
        for idx in range(n_test):
            raw_scores = generator(gen_windows_test[idx], causal_edge_index_full)
            gsina_out = gsina_split(
                raw_scores, causal_edge_index_full, r=r,
                n_iters=args.gsina_iters, temperature=args.gsina_temperature,
                gumbel_noise=False,  # deterministic at eval
            )
            pred = expert(x_all_test[idx], target, gsina_out.causal_edge_index, gsina_out.causal_edge_weight)
            preds.append(pred.item())
            if idx in sample_score_idx:
                all_keep_probs.append(gsina_out.keep_prob.numpy())
        preds = np.array(preds)
        mean_keep_prob = np.mean(all_keep_probs, axis=0)

    y_true = y_target_test.numpy()
    expert_mse = float(np.mean((preds - y_true) ** 2))
    persistence_pred = features_all[test_idx_full][:, target, olr_lag0_channel]
    persistence_mse = float(np.mean((persistence_pred - y_true) ** 2))
    climatology_pred = float(y_target_all.numpy().mean())
    climatology_mse = float(np.mean((climatology_pred - y_true) ** 2))
    r2_vs_climatology = 1.0 - expert_mse / max(climatology_mse, 1e-12)

    n_causal = max(1, round(r * n_causal_candidates))
    top_idx = np.argpartition(-mean_keep_prob, n_causal - 1)[:n_causal]
    selected_sources = sorted(sources[i] for i in top_idx if sources[i] != target)
    self_selected = target in [int(sources[i]) for i in top_idx]

    corr_ceiling = raw_lagged_correlation_auc(olr_lag0_all, target, sources)
    top3_by_corr = sorted(corr_ceiling.items(), key=lambda kv: -kv[1])[:3]

    print(f"\n--- variant={variant} GSINA result (OUT-OF-SAMPLE, n_test={n_test}) ---")
    print(f"r={r:.3f}  n_candidate_sources={n_sources}  n_selected={n_causal}")
    print(f"expert MSE={expert_mse:.4f}  persistence MSE={persistence_mse:.4f}  climatology MSE={climatology_mse:.4f}  "
          f"R^2 vs climatology={r2_vs_climatology:.4f}  "
          f"(expert beats persistence: {expert_mse < persistence_mse})")
    print(f"selected non-self sources: {selected_sources}  self-loop selected: {self_selected}")
    print(f"mean keep-probability: {mean_keep_prob.mean():.4f}  std: {mean_keep_prob.std():.4f}")
    print(f"data-hardness ceiling, top-3 sources by raw |lagged correlation|: {top3_by_corr}")

    return {
        "variant": variant,
        "gsina_iters": args.gsina_iters,
        "gsina_temperature": args.gsina_temperature,
        "n_candidate_sources": n_sources,
        "r": r,
        "expert_mse": expert_mse,
        "persistence_mse": persistence_mse,
        "climatology_mse": climatology_mse,
        "r2_vs_climatology": r2_vs_climatology,
        "n_test": n_test,
        "score_std": float(mean_keep_prob.std()),
        "selected_sources": [int(s) for s in selected_sources],
        "self_loop_selected": bool(self_selected),
        "corr_ceiling_top3": [[int(s), float(c)] for s, c in top3_by_corr],
        "elapsed_s": elapsed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, default=22)
    parser.add_argument("--variant", type=str, default="direct")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--r", type=float, default=None)
    parser.add_argument("--n-swaps", type=int, default=6)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--generator-window", type=int, default=10)
    parser.add_argument("--gsina-iters", type=int, default=20)
    parser.add_argument("--gsina-temperature", type=float, default=1.0)
    parser.add_argument("--gsina-gumbel", action="store_true")
    parser.add_argument("--cia-weight", type=float, default=0.0, help="0.0 (default) = GSINA alone; >0 = combined GSINA+CIA check (Task 2 follow-up, kept off by default so GSINA stays independently testable per the original plan)")
    parser.add_argument("--cia-bandwidth", type=float, default=0.3)
    parser.add_argument("--n-samples-cap", type=int, default=4000, help="default matches step 8's original budget for direct comparability")
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
    result = train_place_with_gsina(
        args.variant, args.target, cache["features"], cache["targets"],
        cache["sample_time_index"], cache["edge_index"], cache["n_clusters"], args,
        sample_dates=sample_dates, olr_lag0_channel=cache["olr_lag0_channel"],
        lead_time_days=int(cache["lead_time_days"]),
    )

    RESULTS_DIR.mkdir(exist_ok=True)
    suffix = f"_cia{args.cia_weight}" if args.cia_weight > 0.0 else ""
    out_path = RESULTS_DIR / f"step_gsina_place{args.target}_{args.variant}_seed{args.seed}{suffix}.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
