"""GSINA version of SS9 step 7: scale the GSINA-based splitter (Task 2 fix,
§10 "GSINA" entry) from the small validation set (places 22, 4, 38, 44) to
all 50 clusters -- checks whether the real improvement found on those 4
places holds mesh-wide, or was specific to that small sample.

Repeats `train_step_gsina_single_place.train_place_with_gsina` per place,
same pattern as `run_step7_all_places.py` did for the original hard-top-r
splitter. Checkpoints its JSON after every place (this is a long run, never
let a crash lose completed places).

COST WARNING: GSINA's Sinkhorn iterations (~20/sample) make each
sample-step ~2x the cost of the original hard-top-r splitter (measured:
~10ms/sample-step here vs ~5ms for train_step4_single_place, on this
CPU-only machine). At the step-7 default budget (4000 samples/2 epochs per
place = 8000 sample-steps), that is ~80s/place x 50 places =~ 65-70 minutes
total -- flagged before launching per this session's compute constraints.
Run in the BACKGROUND; check results/step7_all_places_gsina.json for
progress (checkpointed after every place) rather than waiting on this
process to exit.

Usage:
    python scripts/run_step7_all_places_gsina.py [--variant direct]
        [--epochs 2] [--n-samples-cap 4000] [--places 0-49]
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

from scripts.run_step7_all_places import parse_places
from scripts.train_step4_single_place import load_cache
from scripts.train_step_gsina_single_place import train_place_with_gsina

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", type=str, default="direct")
    parser.add_argument("--places", type=str, default="all")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--n-samples-cap", type=int, default=4000)
    parser.add_argument("--r", type=float, default=None)
    parser.add_argument("--n-swaps", type=int, default=6)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--generator-window", type=int, default=10)
    parser.add_argument("--gsina-iters", type=int, default=20)
    parser.add_argument("--gsina-temperature", type=float, default=1.0)
    parser.add_argument("--gsina-gumbel", action="store_true")
    parser.add_argument("--cia-weight", type=float, default=0.0)
    parser.add_argument("--cia-bandwidth", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)
    cache = load_cache()
    n_clusters = cache["n_clusters"]
    places = parse_places(args.places, n_clusters)
    print(f"step 7 (GSINA): {len(places)} places, variant={args.variant}, epochs={args.epochs}, "
          f"cap={args.n_samples_cap}, gsina_iters={args.gsina_iters}")

    rows = []
    t_start = time.time()
    for k, place in enumerate(places):
        t0 = time.time()
        res = train_place_with_gsina(
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
            "self_loop_selected": res["self_loop_selected"],
        })
        elapsed_all = time.time() - t_start
        eta = (elapsed_all / (k + 1)) * (len(places) - k - 1)
        print(f"[{k+1}/{len(places)}] place {place:>2}: mse={res['expert_mse']:.4f} "
              f"persist={res['persistence_mse']:.4f} skill={skill:+.3f} "
              f"beats={'Y' if beats else 'n'} ({time.time()-t0:.0f}s, ETA {eta/60:.0f}m)")

        # checkpoint after every place so a long run is never lost
        out_path = RESULTS_DIR / f"step7_all_places_gsina_{args.variant}.json"
        skills = [r["skill_vs_persistence"] for r in rows]
        out_path.write_text(json.dumps({
            "variant": args.variant,
            "epochs": args.epochs,
            "n_samples_cap": args.n_samples_cap,
            "gsina_iters": args.gsina_iters,
            "n_places_done": len(rows),
            "n_beating_persistence": sum(1 for r in rows if r["beats_persistence"]),
            "skill_mean": float(np.mean(skills)),
            "skill_median": float(np.median(skills)),
            "places": rows,
        }, indent=2))

    skills = [r["skill_vs_persistence"] for r in rows]
    n_beat = sum(1 for r in rows if r["beats_persistence"])
    print(f"\n=== step 7 (GSINA) summary ({len(rows)} places) ===")
    print(f"places beating persistence: {n_beat}/{len(rows)} ({100*n_beat/max(len(rows),1):.0f}%)")
    print(f"skill vs persistence: mean={np.mean(skills):+.4f} median={np.median(skills):+.4f} "
          f"min={np.min(skills):+.4f} max={np.max(skills):+.4f}")
    print(f"total time: {(time.time()-t_start)/60:.1f} min")

    # comparison note against the original (non-GSINA) step 7 result, if present
    orig_path = RESULTS_DIR / f"step7_all_places_{args.variant}.json"
    if orig_path.exists():
        orig = json.loads(orig_path.read_text())
        print(f"\noriginal (hard top-r) step 7: {orig['n_beating_persistence']}/{orig['n_places_done']} beat "
              f"persistence, mean skill {orig['skill_mean']:+.4f}")
        print(f"GSINA step 7:                 {n_beat}/{len(rows)} beat persistence, "
              f"mean skill {np.mean(skills):+.4f}")


if __name__ == "__main__":
    main()
