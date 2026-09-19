"""SS9 step 6: hibernate/reactivate (SS4.5), still single place.

Builds directly on step 5's saved signals (results/step5_signals_place*.npz)
-- no retraining needed, so this step is cheap: it replays the record,
and every time the drift detector fires a Tier-2 event (SS4.4's "spawn new
splitter gen + new expert"), it runs SS4.5's lifecycle:

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
    which is SS4.5's entire premise ("climate systems have real recurring
    regimes ... recurrence is the norm, not an edge case"),
  - the archive size over time (bounded active compute, no information
    loss).

Usage:
    python scripts/run_step6_lifecycle.py [--target 22] [--variant direct]
        [--similarity-threshold 0.9] [--min-regime-days 180]
"""

from __future__ import annotations

import argparse
import functools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

print = functools.partial(print, flush=True)  # noqa: A001

import numpy as np
import torch

from causal_moe.drift.archive import ExpertArchive
from causal_moe.drift.channels import center_signatures
from causal_moe.experts.expert import PlaceExpert

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


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

    archive = ExpertArchive(place=args.target, similarity_threshold=args.similarity_threshold)
    live_expert = PlaceExpert(in_channels=21, hidden_channels=16)

    def signature_at(t: int) -> np.ndarray:
        lo = max(0, t - args.signature_window)
        return signatures[lo:t].mean(axis=0) if t > lo else signatures[t]

    events = []
    generation = 1
    regime_start = 0
    for t in debounced:
        if t <= regime_start:
            continue
        outgoing_signature = signatures[regime_start:t].mean(axis=0)
        expert_id = f"P{args.target}-gen{generation}"
        archive.hibernate(expert_id, live_expert, outgoing_signature, regime_start, t)

        current_signature = signature_at(t)
        # Fresh module for the next generation; try_reactivate loads
        # archived weights into it on a hit (warm start, SS4.5).
        live_expert = PlaceExpert(in_channels=21, hidden_channels=16)
        decision = archive.try_reactivate(current_signature, live_expert)

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
        })
        print(f"  {str(dates[t])[:10]}  hibernate {expert_id}  -> {decision.reason}")
        regime_start = t

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

    report = {
        "target": args.target,
        "variant": args.variant,
        "similarity_threshold": args.similarity_threshold,
        "min_regime_days": args.min_regime_days,
        "n_raw_triggers": len(triggers),
        "n_debounced_events": len(debounced),
        "n_spawn_events": len(events),
        "n_reactivated": n_react,
        "n_fresh": len(events) - n_react,
        "final_archive_size": len(archive.experts),
        "events": events,
    }
    out_path = RESULTS_DIR / f"step6_lifecycle_place{args.target}_{args.variant}.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
