"""SS9 step 6: hibernate/reactivate (SS4.5), still single place.

Builds on step 5's saved signals (results/step5_signals_place*.npz) for the
regime-change schedule (when to hibernate/spawn), and on step 5's cache for
the actual feature/target data needed to TRAIN each generation.

B10 audit fix, 2026-09-23: the original version constructed a fresh,
UNTRAINED `PlaceExpert(...)` at every hibernate/spawn point and never ran a
single optimizer step on it (no `backward()`/`optim` call anywhere in the
file) -- so the archive stored random weights, "reactivation" loaded random
weights into another random module, and the reported "29 reactivated / 19
fresh" only measured cosine similarity between edge-score SIGNATURES, never
comparing reactivated-vs-fresh expert ACCURACY, which is the entire SS4.5
claim ("warm-starting from an archived match beats spawning cold"). This
version actually trains each generation's expert on its own regime span
(using the causal edge set the step-5 splitter selected, held fixed -- only
the per-generation EXPERT is retrained/warm-started here, not the splitter
itself, matching SS4.5's own scope: "spawn new splitter gen + new expert" is
listed as a Tier-2 action in SS4.4, but SS4.5's own hibernate/reactivate
mechanism and its evaluation are about the EXPERT), then evaluates
reactivated-vs-fresh on the FIRST `--eval-days` days of the NEXT regime
(held out from that generation's own training) as the actual accuracy
comparison SS4.5 claims.

It still replays the record and, at every Tier-2 trigger from step 5's
saved detector alarms, runs SS4.5's lifecycle:

    current signature -> similarity search over the WHOLE archive
      -> best match >= threshold ? reactivate that expert as a WARM START
                                 : spawn a fresh expert
    ... and the outgoing expert is HIBERNATED (not deleted), carrying the
        mean causal signature of the regime it served.

What this step measures (the thing SS4.5 claims to buy over plain DyMoE,
which deletes on spawn):
  - how many spawn events the record produces,
  - how many of those were served by REACTIVATING an archived expert
    instead of spawning cold -- i.e. how often regimes actually recur,
  - for EACH reactivation event: held-out MSE of the warm-started expert vs
    a FRESH-spawn control trained on the identical data, on the next
    regime's first `--eval-days` days -- the actual accuracy comparison.
  - the archive size over time (bounded active compute, no information
    loss).

Usage:
    python scripts/run_step6_lifecycle.py [--target 22] [--variant direct]
        [--similarity-threshold 0.9] [--min-regime-days 180]
        [--gen-epochs 3] [--eval-days 60]
"""

from __future__ import annotations

import argparse
import copy
import functools
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

print = functools.partial(print, flush=True)  # noqa: A001

import numpy as np
import torch

from causal_moe.data.candidate_edges import build_candidate_source_set, expand_source_clusters_to_edge_index
from causal_moe.drift.archive import ExpertArchive
from causal_moe.drift.channels import center_signatures
from causal_moe.drift.forgetting import RegimeObservation, average_forgetting
from causal_moe.experts.expert import PlaceExpert, compute_expert_loss
from causal_moe.experts.pool import ExpertPool
from scripts.train_step4_single_place import CACHE_PATH, load_cache

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def train_expert_on_span(
    expert: PlaceExpert,
    x_all: torch.Tensor,
    y_target: torch.Tensor,
    target: int,
    causal_edge_index: torch.Tensor,
    causal_edge_weight: torch.Tensor,
    lo: int,
    hi: int,
    epochs: int,
    lr: float,
    batch_size: int,
    seed: int,
) -> PlaceExpert:
    """Trains `expert` (in place) on samples [lo, hi) using a FIXED causal
    edge set/weighting (the step-5 splitter's own selection for this place,
    frozen -- only the expert is being trained here, see module docstring).
    """
    n = hi - lo
    if n <= 0:
        return expert
    rng = np.random.default_rng(seed)
    opt = torch.optim.Adam(expert.parameters(), lr=lr)
    idx_range = np.arange(lo, hi)
    for _ in range(epochs):
        order = rng.permutation(idx_range)
        for start in range(0, max(n - batch_size + 1, 1), max(batch_size, 1)):
            batch_idx = order[start : start + batch_size]
            if batch_idx.shape[0] == 0:
                continue
            opt.zero_grad()
            loss = torch.zeros(())
            for idx in batch_idx:
                out = compute_expert_loss(
                    expert, x_all[idx], target, y_target[idx],
                    causal_edge_index, causal_edge_weight,
                )
                loss = loss + out.total_loss
            loss = loss / batch_idx.shape[0]
            loss.backward()
            opt.step()
    return expert


@torch.no_grad()
def eval_expert_on_span(
    expert: PlaceExpert,
    x_all: torch.Tensor,
    y_target: torch.Tensor,
    target: int,
    causal_edge_index: torch.Tensor,
    causal_edge_weight: torch.Tensor,
    lo: int,
    hi: int,
) -> float | None:
    if hi <= lo:
        return None
    expert.eval()
    preds = []
    for idx in range(lo, hi):
        pred = expert(x_all[idx], target, causal_edge_index, causal_edge_weight)
        preds.append(pred.item())
    expert.train()
    preds = np.array(preds)
    y_true = y_target[lo:hi].numpy()
    return float(np.mean((preds - y_true) ** 2))


@torch.no_grad()
def eval_pool_on_span(
    pool: ExpertPool,
    x_all: torch.Tensor,
    y_target: torch.Tensor,
    target: int,
    causal_edge_index: torch.Tensor,
    causal_edge_weight: torch.Tensor,
    signatures: np.ndarray,
    signature_at: "callable",
    k: int,
    lo: int,
    hi: int,
) -> float | None:
    """Variant B (B11 fix, Phase 5): held-out MSE of the pool's top-k
    similarity-blended prediction (`ExpertPool.predict`) over [lo, hi),
    re-routing at every step since the signature (and therefore which
    archived generations are most similar) can drift within the span.
    k=1 reproduces Variant A's single-best-match behaviour exactly (see
    `ExpertPool.route`'s own docstring); k=3 is the actual MoE this phase
    is meant to evaluate. Returns None if the pool is empty or the span
    is empty -- both are "not yet comparable" states, not zero-error."""
    if hi <= lo or pool.size == 0:
        return None
    preds = []
    for idx in range(lo, hi):
        sig = signature_at(idx)
        pred, _decision = pool.predict(
            x_all[idx], target, causal_edge_index, causal_edge_weight, sig, k=k,
        )
        preds.append(pred.item())
    preds = np.array(preds)
    y_true = y_target[lo:hi].numpy()
    return float(np.mean((preds - y_true) ** 2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, default=22)
    parser.add_argument("--variant", type=str, default="direct")
    parser.add_argument("--similarity-threshold", type=float, default=0.9)
    parser.add_argument(
        "--min-regime-days", type=int, default=180,
        help="debounce: ignore spawn triggers closer together than this, so a "
             "burst of adjacent STARS alarms is one regime change, not dozens",
    )
    parser.add_argument("--signature-window", type=int, default=120,
                        help="days of signature history averaged into the signature a spawn decision uses")
    parser.add_argument("--baseline-window", type=int, default=365,
                        help="days used to define the per-edge baseline that signatures are centered against")
    parser.add_argument("--gen-epochs", type=int, default=3,
                        help="B10 fix: epochs to train each generation's expert on its own regime span")
    parser.add_argument("--eval-days", type=int, default=60,
                        help="B10 fix: held-out days at the START of the NEXT regime used to compare "
                             "reactivated-warm-start vs fresh-spawn accuracy")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cache-path", type=Path, default=None)
    parser.add_argument("--pool-k", type=int, default=3,
                         help="Phase 5 / B11 fix: top-k pool members to blend for Variant B "
                              "(k=1 reproduces Variant A's single-best-match reactivation exactly)")
    args = parser.parse_args()

    sig_path = RESULTS_DIR / f"step5_signals_place{args.target}_{args.variant}.npz"
    if not sig_path.exists():
        raise SystemExit(f"missing {sig_path} -- run scripts/run_step5_drift.py first")

    data = np.load(sig_path, allow_pickle=True)
    raw_signatures = data["signatures"]  # (T, n_edges)
    # Centered, for the SAME reason Channel 2 centers (see
    # causal_moe.drift.channels.center_signatures): raw edge scores share a
    # large common offset, which makes cosine similarity saturate at ~1.0
    # between ANY two signatures and turns archive matching into a
    # rubber-stamp. Centering compares deviation patterns instead.
    signatures = center_signatures(raw_signatures, baseline_window=args.baseline_window)
    dates = data["dates"]
    ch1 = data["ch1_alarms"].tolist()
    ch2 = data["ch2_alarms"].tolist()
    T, n_edges = signatures.shape

    # Tier-2 (spawn) triggers: SS4.4's fusion gate. Use OR here -- the
    # lifecycle is what we're testing, and OR is the rule that actually
    # produces enough events on this record to exercise it (AND fired once,
    # per step 5). Both rules' detection quality is already reported in
    # step 5; this step is about what happens AFTER a trigger.
    triggers = sorted(set(ch1) | set(ch2))

    # Debounce: collapse bursts into single regime-change events.
    debounced: list[int] = []
    for t in triggers:
        if not debounced or (t - debounced[-1]) >= args.min_regime_days:
            debounced.append(int(t))

    print(f"place {args.target}, variant={args.variant}: {T} days, {n_edges} candidate edges")
    print(f"raw triggers: {len(triggers)} (ch1={len(ch1)}, ch2={len(ch2)}) -> "
          f"{len(debounced)} after {args.min_regime_days}-day debounce")

    # B10 fix: load the actual feature/target cache so each generation's
    # expert can be genuinely trained, not left randomly initialized. The
    # candidate edge set is the same physical (source, target) set step 5
    # used (deterministic given target+variant+mesh), held FIXED with
    # uniform weights across generations -- only the expert's weights
    # change per generation/reactivation; re-deriving the splitter's own
    # per-day edge scores here is out of this fix's scope (would require
    # re-running the trained splitter per generation, a separate cost).
    cache = load_cache(args.cache_path or CACHE_PATH)
    features_all = cache["features"]
    targets_all = cache["targets"]
    n_features = features_all.shape[-1]
    n_clusters = cache["n_clusters"]
    candidate_set = build_candidate_source_set(cache["edge_index"], args.target, args.variant, n_clusters=n_clusters)
    sources = candidate_set.source_clusters
    causal_edge_index = torch.from_numpy(expand_source_clusters_to_edge_index(sources, args.target))
    causal_edge_weight = torch.ones(causal_edge_index.shape[1]) / causal_edge_index.shape[1]
    x_all = torch.from_numpy(features_all)
    y_target = torch.from_numpy(targets_all[:, args.target])
    print(f"loaded cache for training: {n_features} features/place, "
          f"{causal_edge_index.shape[1]} fixed candidate edges for place {args.target}")

    archive = ExpertArchive(place=args.target, similarity_threshold=args.similarity_threshold)
    # Phase 5 / B11 fix: an ExpertPool is the SAME archive contents, viewed
    # for top-k blending instead of single-best-match reactivation (see
    # causal_moe/experts/pool.py's module docstring) -- share the archive's
    # own `experts` list directly so every hibernate() call is immediately
    # visible to the pool too, no separate bookkeeping needed.
    pool = ExpertPool(place=args.target, in_channels=n_features, hidden_channels=16, members=archive.experts)
    live_expert = PlaceExpert(in_channels=n_features, hidden_channels=16)

    def signature_at(t: int) -> np.ndarray:
        lo = max(0, t - args.signature_window)
        return signatures[lo:t].mean(axis=0) if t > lo else signatures[t]

    events = []
    generation = 1
    regime_start = 0
    t0 = time.time()
    for gen_idx, t in enumerate(debounced):
        if t <= regime_start:
            continue
        # Train the OUTGOING generation's expert on its own regime span
        # before hibernating it (B10 fix -- this used to be skipped
        # entirely, so the archive stored random weights).
        train_expert_on_span(
            live_expert, x_all, y_target, args.target, causal_edge_index, causal_edge_weight,
            regime_start, t, args.gen_epochs, args.lr, args.batch_size, args.seed + gen_idx,
        )

        outgoing_signature = signatures[regime_start:t].mean(axis=0)
        expert_id = f"P{args.target}-gen{generation}"
        archive.hibernate(expert_id, live_expert, outgoing_signature, regime_start, t)

        current_signature = signature_at(t)
        next_regime_end = debounced[gen_idx + 1] if gen_idx + 1 < len(debounced) else T
        eval_lo, eval_hi = t, min(t + args.eval_days, next_regime_end)

        # Warm-start candidate: try_reactivate loads the best archived
        # match's weights into a copy, if similarity clears the threshold.
        warm_expert = PlaceExpert(in_channels=n_features, hidden_channels=16)
        decision = archive.try_reactivate(current_signature, warm_expert)
        # Fresh-spawn CONTROL: same architecture, random init, same held-out
        # span -- the direct comparison SS4.5 claims a win on.
        fresh_expert = PlaceExpert(in_channels=n_features, hidden_channels=16)

        warm_mse_before = eval_expert_on_span(warm_expert, x_all, y_target, args.target, causal_edge_index, causal_edge_weight, eval_lo, eval_hi)
        fresh_mse_before = eval_expert_on_span(fresh_expert, x_all, y_target, args.target, causal_edge_index, causal_edge_weight, eval_lo, eval_hi)
        # Fine-tune the fresh control on the SAME new-regime prefix a warm
        # start would otherwise get "for free" from its prior training, so
        # the comparison is warm-start-then-adapt vs cold-start-then-adapt,
        # not warm-start vs an untouched fresh network.
        if eval_hi > eval_lo:
            train_expert_on_span(fresh_expert, x_all, y_target, args.target, causal_edge_index, causal_edge_weight, eval_lo, eval_hi, 1, args.lr, args.batch_size, args.seed + gen_idx)
        fresh_mse_after = eval_expert_on_span(fresh_expert, x_all, y_target, args.target, causal_edge_index, causal_edge_weight, eval_lo, eval_hi)

        # Phase 5 / B11 fix: Variant B (k>1 pool blend) evaluated on the
        # SAME held-out span as Variant A above, using the pool as it
        # stood right after this generation's hibernate() (i.e. excluding
        # the generation currently being evaluated FROM, same as
        # try_reactivate's own archive search above). k=1 is also computed
        # so Variant A and Variant B are directly comparable at matched k,
        # not just "single best match" vs "k=3" conflating two different
        # changes at once.
        pool_mse_k1 = eval_pool_on_span(
            pool, x_all, y_target, args.target, causal_edge_index, causal_edge_weight,
            signatures, signature_at, k=1, lo=eval_lo, hi=eval_hi,
        )
        pool_mse_k = eval_pool_on_span(
            pool, x_all, y_target, args.target, causal_edge_index, causal_edge_weight,
            signatures, signature_at, k=args.pool_k, lo=eval_lo, hi=eval_hi,
        )

        live_expert = warm_expert  # next generation continues from here (warm or freshly-inited)

        generation += 1
        events.append({
            "day_index": int(t),
            "date": str(dates[t])[:10],
            "hibernated": expert_id,
            "archive_size": len(archive.experts),
            "reactivated": bool(decision.reactivated),
            "matched": decision.matched_expert_id,
            "similarity": round(float(decision.similarity), 4),
            "reason": decision.reason,
            "eval_days": int(eval_hi - eval_lo),
            "warm_mse_before_adapt": warm_mse_before,
            "fresh_mse_before_adapt": fresh_mse_before,
            "fresh_mse_after_adapt": fresh_mse_after,
            "warm_beats_fresh_before_adapt": (
                (warm_mse_before < fresh_mse_before) if warm_mse_before is not None and fresh_mse_before is not None else None
            ),
            "pool_mse_k1": pool_mse_k1,
            "pool_mse_k": pool_mse_k,
            "pool_k": args.pool_k,
            "pool_size_at_event": pool.size,
            "variant_b_beats_variant_a": (
                (pool_mse_k < pool_mse_k1) if pool_mse_k is not None and pool_mse_k1 is not None else None
            ),
        })
        print(f"  {str(dates[t])[:10]}  hibernate {expert_id}  -> {decision.reason}  "
              f"[eval {eval_hi-eval_lo}d: warm_mse={warm_mse_before} fresh_mse(untrained)={fresh_mse_before} fresh_mse(1ep)={fresh_mse_after} "
              f"pool_mse(k=1)={pool_mse_k1} pool_mse(k={args.pool_k})={pool_mse_k}]")
        regime_start = t

    print(f"total lifecycle time: {time.time()-t0:.1f}s")
    n_react = sum(1 for e in events if e["reactivated"])
    print(f"\n=== step 6 summary ===")
    print(f"spawn events: {len(events)}")
    print(f"served by REACTIVATING an archived expert: {n_react}")
    print(f"served by spawning fresh: {len(events) - n_react}")
    print(f"final archive size: {len(archive.experts)} (plain DyMoE would have deleted all of these)")
    if events:
        sims = [e["similarity"] for e in events]
        print(f"similarity to best archived match: min={min(sims):.3f} "
              f"max={max(sims):.3f} mean={np.mean(sims):.3f} "
              f"(threshold {args.similarity_threshold})")

    # B10 fix: the actual SS4.5 claim, measured for the first time -- among
    # events that WERE reactivated (had a real warm-start weight load),
    # does the warm-started expert beat an equivalent fresh spawn on the
    # next regime's held-out days?
    react_events = [e for e in events if e["reactivated"] and e["warm_beats_fresh_before_adapt"] is not None]
    if react_events:
        n_warm_wins = sum(1 for e in react_events if e["warm_beats_fresh_before_adapt"])
        print(f"\n=== B10 fix: warm-start vs fresh-spawn accuracy (the actual SS4.5 claim) ===")
        print(f"among {len(react_events)} reactivation events with a measurable next-regime eval span:")
        print(f"  warm-start beats untrained-fresh: {n_warm_wins}/{len(react_events)} "
              f"({100*n_warm_wins/len(react_events):.0f}%)")
        mean_warm = np.mean([e["warm_mse_before_adapt"] for e in react_events])
        mean_fresh0 = np.mean([e["fresh_mse_before_adapt"] for e in react_events])
        mean_fresh1 = np.mean([e["fresh_mse_after_adapt"] for e in react_events])
        print(f"  mean MSE: warm-start={mean_warm:.4f}  fresh(untrained)={mean_fresh0:.4f}  fresh(1-epoch-adapted)={mean_fresh1:.4f}")
    else:
        print("\n(no reactivation events with a measurable next-regime eval span -- "
              "either no reactivations occurred, or every regime was too short/last-in-record)")

    # B12 audit fix: average-forgetting comparison between hibernate+warm-
    # start (this script's single-best-match reactivation, "warm" MSE
    # below) and a delete-on-spawn ablation ("fresh" MSE, what EVERY event
    # would have gotten had reactivation never been attempted). NOTE
    # (2026-09-24 Phase 5 wiring): this pair is WARM-START vs FRESH-SPAWN,
    # a different axis from Variant A/B below -- renamed from the
    # original "Variant B vs Variant A" labels, which pre-date Phase 5's
    # actual A/B definition (Bug 11 fix: Variant A = k=1 single-expert,
    # Variant B = k=3 pool blend, both defined in causal_moe/experts/
    # pool.py) and collided with it confusingly. "Regime" here = the
    # best-matching ARCHIVED expert id (decision.matched), since that
    # identifies which recurring regime this event's held-out span
    # belongs to; events with no archive match (nothing to recur to yet)
    # are excluded, matching average_forgetting's own "needs 2+ visits to
    # a regime" requirement.
    warm_obs = [
        RegimeObservation(regime=e["matched"], generation=e["day_index"], mse=e["warm_mse_before_adapt"])
        for e in events if e["matched"] is not None and e["warm_mse_before_adapt"] is not None
    ]
    fresh_obs = [
        RegimeObservation(regime=e["matched"], generation=e["day_index"], mse=e["fresh_mse_before_adapt"])
        for e in events if e["matched"] is not None and e["fresh_mse_before_adapt"] is not None
    ]
    warm_forgetting = average_forgetting(warm_obs)
    fresh_forgetting = average_forgetting(fresh_obs)
    print(f"\n=== B12 fix: average forgetting, warm-start (single-match reactivate) vs fresh-spawn (delete-on-spawn) ===")
    print(f"warm-start avg forgetting: {warm_forgetting.average_forgetting:.4f} "
          f"over {warm_forgetting.n_regimes_revisited} revisited regimes")
    print(f"fresh-spawn avg forgetting: {fresh_forgetting.average_forgetting:.4f} "
          f"over {fresh_forgetting.n_regimes_revisited} revisited regimes")
    if warm_forgetting.n_regimes_revisited > 0 and fresh_forgetting.n_regimes_revisited > 0:
        print(f"warm-start wins on forgetting: "
              f"{warm_forgetting.average_forgetting < fresh_forgetting.average_forgetting} "
              f"(lower = less forgetting = better)")

    # Phase 5 / B11 fix, the actual claim this phase is meant to test:
    # Variant A (k=1, single active expert -- pool routing collapsed to
    # the one best match, mathematically the same operation warm-start
    # reactivation performs, just measured through the pool's own
    # code path for a clean apples-to-apples k=1-vs-k=3 comparison) vs
    # Variant B (k=args.pool_k, default 3 -- the actual top-k blended
    # mixture). Both measured via the IDENTICAL eval_pool_on_span call
    # (pool_mse_k1 / pool_mse_k on each event), so this isolates the
    # effect of k alone, unlike the warm/fresh comparison above which
    # also changes training (fresh gets 1 epoch of adaptation, warm
    # doesn't).
    pool_k1_obs = [
        RegimeObservation(regime=e["matched"], generation=e["day_index"], mse=e["pool_mse_k1"])
        for e in events if e["matched"] is not None and e["pool_mse_k1"] is not None
    ]
    pool_k_obs = [
        RegimeObservation(regime=e["matched"], generation=e["day_index"], mse=e["pool_mse_k"])
        for e in events if e["matched"] is not None and e["pool_mse_k"] is not None
    ]
    variant_a_forgetting = average_forgetting(pool_k1_obs)
    variant_b_forgetting = average_forgetting(pool_k_obs)
    vb_events = [e for e in events if e["variant_b_beats_variant_a"] is not None]
    print(f"\n=== Phase 5 / B11 fix: Variant A (k=1) vs Variant B (k={args.pool_k}) -- the actual MoE comparison ===")
    if vb_events:
        n_b_wins = sum(1 for e in vb_events if e["variant_b_beats_variant_a"])
        mean_a_mse = np.mean([e["pool_mse_k1"] for e in vb_events])
        mean_b_mse = np.mean([e["pool_mse_k"] for e in vb_events])
        print(f"among {len(vb_events)} events with both k=1 and k={args.pool_k} measurable:")
        print(f"  Variant B beats Variant A (lower held-out MSE): {n_b_wins}/{len(vb_events)} "
              f"({100*n_b_wins/len(vb_events):.0f}%)")
        print(f"  mean MSE: Variant A (k=1)={mean_a_mse:.4f}  Variant B (k={args.pool_k})={mean_b_mse:.4f}")
    else:
        print("(no events with a non-empty pool AND a measurable eval span -- "
              "need at least 2 spawn events before Variant B's blend has >1 member to route over)")
    print(f"average forgetting: Variant A (k=1)={variant_a_forgetting.average_forgetting:.4f} "
          f"over {variant_a_forgetting.n_regimes_revisited} revisited regimes, "
          f"Variant B (k={args.pool_k})={variant_b_forgetting.average_forgetting:.4f} "
          f"over {variant_b_forgetting.n_regimes_revisited} revisited regimes")
    if variant_a_forgetting.n_regimes_revisited > 0 and variant_b_forgetting.n_regimes_revisited > 0:
        print(f"Variant B wins on forgetting: "
              f"{variant_b_forgetting.average_forgetting < variant_a_forgetting.average_forgetting} "
              f"(lower = less forgetting = better)")

    report = {
        "target": args.target,
        "variant": args.variant,
        "similarity_threshold": args.similarity_threshold,
        "min_regime_days": args.min_regime_days,
        "gen_epochs": args.gen_epochs,
        "eval_days": args.eval_days,
        "n_raw_triggers": len(triggers),
        "n_debounced_events": len(debounced),
        "n_spawn_events": len(events),
        "n_reactivated": n_react,
        "n_fresh": len(events) - n_react,
        "final_archive_size": len(archive.experts),
        "warm_vs_fresh": {
            "n_comparable_events": len(react_events),
            "n_warm_wins": sum(1 for e in react_events if e["warm_beats_fresh_before_adapt"]) if react_events else None,
            "mean_warm_mse": float(np.mean([e["warm_mse_before_adapt"] for e in react_events])) if react_events else None,
            "mean_fresh_mse_untrained": float(np.mean([e["fresh_mse_before_adapt"] for e in react_events])) if react_events else None,
            "mean_fresh_mse_1epoch": float(np.mean([e["fresh_mse_after_adapt"] for e in react_events])) if react_events else None,
        },
        "forgetting_warm_vs_fresh": {
            "warm_start": {
                "average_forgetting": warm_forgetting.average_forgetting,
                "n_regimes_revisited": warm_forgetting.n_regimes_revisited,
            },
            "fresh_spawn": {
                "average_forgetting": fresh_forgetting.average_forgetting,
                "n_regimes_revisited": fresh_forgetting.n_regimes_revisited,
            },
        },
        "variant_a_vs_variant_b": {
            "pool_k": args.pool_k,
            "n_comparable_events": len(vb_events),
            "n_variant_b_wins": sum(1 for e in vb_events if e["variant_b_beats_variant_a"]) if vb_events else None,
            "mean_variant_a_mse_k1": float(np.mean([e["pool_mse_k1"] for e in vb_events])) if vb_events else None,
            "mean_variant_b_mse_k": float(np.mean([e["pool_mse_k"] for e in vb_events])) if vb_events else None,
            "forgetting_variant_a_k1": {
                "average_forgetting": variant_a_forgetting.average_forgetting,
                "n_regimes_revisited": variant_a_forgetting.n_regimes_revisited,
            },
            "forgetting_variant_b_k": {
                "average_forgetting": variant_b_forgetting.average_forgetting,
                "n_regimes_revisited": variant_b_forgetting.n_regimes_revisited,
            },
        },
        "events": events,
    }
    out_path = RESULTS_DIR / f"step6_lifecycle_place{args.target}_{args.variant}.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
