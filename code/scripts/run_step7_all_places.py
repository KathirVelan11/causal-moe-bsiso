"""SS9 step 7: scale from one place to many (all 50 clusters).

Repeats the step-4 splitter+expert training per place, then reports the
distribution of results across places rather than a single number -- which
is the actual question step 7 answers: does the mechanism hold up
everywhere, or only at the one place step 4 happened to pick?

COST WARNING (SS8 "Compute/hardware"): the architecture doc's own estimate
for 50 lineages is ~1.9h for a 10-epoch full-record run on this CPU-only
machine. Defaults here are deliberately cheaper (fewer epochs, optional
sample cap) so the step produces a real cross-place distribution within a
practical budget; `--epochs`/`--n-samples-cap` scale it up when time allows.
Per-place cost is dominated by the candidate-set size, so `direct` (mean
degree ~5-10) is the default variant.

Usage:
    python scripts/run_step7_all_places.py [--variant direct] [--epochs 2]
        [--n-samples-cap 4000] [--places 0-49]
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

from scripts.train_step4_single_place import load_cache, train_place

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def parse_places(spec: str, n_clusters: int) -> list[int]:
    if spec.strip() == "all":
        return list(range(n_clusters))
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-")
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(part))
    return [p for p in out if 0 <= p < n_clusters]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", type=str, default="direct")
    parser.add_argument("--places", type=str, default="all")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--n-samples-cap", type=int, default=4000)
    parser.add_argument("--cap-contiguous", action="store_true", default=True)
    parser.add_argument("--r", type=float, default=None)
    parser.add_argument("--n-swaps", type=int, default=6)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--generator-window", type=int, default=10)
    parser.add_argument("--entropy-weight", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)
    cache = load_cache()
    n_clusters = cache["n_clusters"]
    places = parse_places(args.places, n_clusters)
    print(f"step 7: {len(places)} places, variant={args.variant}, epochs={args.epochs}, "
          f"cap={args.n_samples_cap}")

    rows = []
    t_start = time.time()
    for k, place in enumerate(places):
        t0 = time.time()
        res = train_place(
            variant=args.variant,
            target=place,
            features_all=cache["features"],
            targets_all=cache["targets"],
            sample_time_index=cache["sample_time_index"],
            real_edge_index=cache["edge_index"],
            n_clusters=n_clusters,
            args=args,
        )
        beats = res["expert_mse"] < res["persistence_mse"]
        skill = 1.0 - (res["expert_mse"] / res["persistence_mse"]) if res["persistence_mse"] > 0 else 0.0
        rows.append({
            "place": place,
            "n_candidate_sources": int(res["n_candidate_sources"]),
            "expert_mse": res["expert_mse"],
            "persistence_mse": res["persistence_mse"],
            "skill_vs_persistence": skill,
            "beats_persistence": bool(beats),
            "score_std": res["score_std"],
            "selected_sources": [int(s) for s in res["selected_sources"]],
        })
        elapsed_all = time.time() - t_start
        eta = (elapsed_all / (k + 1)) * (len(places) - k - 1)
        print(f"[{k+1}/{len(places)}] place {place:>2}: mse={res['expert_mse']:.4f} "
              f"persist={res['persistence_mse']:.4f} skill={skill:+.3f} "
              f"beats={'Y' if beats else 'n'} ({time.time()-t0:.0f}s, ETA {eta/60:.0f}m)")

        # checkpoint after every place so a long run is never lost
        out_path = RESULTS_DIR / f"step7_all_places_{args.variant}.json"
        skills = [r["skill_vs_persistence"] for r in rows]
        out_path.write_text(json.dumps({
            "variant": args.variant,
            "epochs": args.epochs,
            "n_samples_cap": args.n_samples_cap,
            "n_places_done": len(rows),
            "n_beating_persistence": sum(1 for r in rows if r["beats_persistence"]),
            "skill_mean": float(np.mean(skills)),
            "skill_median": float(np.median(skills)),
            "places": rows,
        }, indent=2))

    skills = [r["skill_vs_persistence"] for r in rows]
    n_beat = sum(1 for r in rows if r["beats_persistence"])
    print(f"\n=== step 7 summary ({len(rows)} places) ===")
    print(f"places beating persistence: {n_beat}/{len(rows)} ({100*n_beat/max(len(rows),1):.0f}%)")
    print(f"skill vs persistence: mean={np.mean(skills):+.4f} median={np.median(skills):+.4f} "
          f"min={np.min(skills):+.4f} max={np.max(skills):+.4f}")
    worst = sorted(rows, key=lambda r: r["skill_vs_persistence"])[:5]
    best = sorted(rows, key=lambda r: -r["skill_vs_persistence"])[:5]
    print(f"best places:  {[(r['place'], round(r['skill_vs_persistence'],3)) for r in best]}")
    print(f"worst places: {[(r['place'], round(r['skill_vs_persistence'],3)) for r in worst]}")
    print(f"total time: {(time.time()-t_start)/60:.1f} min")


if __name__ == "__main__":
    main()
