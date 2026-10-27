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


INFORMATIVE_SCORE_STD = 0.02  # README SS9 discrimination gate

_W: dict = {}


def _init_worker(cache_path, args, cache=None) -> None:
    import torch

    torch.set_num_threads(1)
    _W["cache"] = cache if cache is not None else load_cache(cache_path)
    _W["args"] = args


def _run_one(place: int) -> dict:
    cache, args = _W["cache"], _W["args"]
    res = train_place(
        variant=args.variant,
        target=place,
        features_all=cache["features"],
        targets_all=cache["targets"],
        sample_time_index=cache["sample_time_index"],
        real_edge_index=cache["edge_index"],
        n_clusters=cache["n_clusters"],
        args=args,
        sample_dates=cache["sample_dates"].astype("datetime64[D]"),
        olr_lag0_channel=cache["olr_lag0_channel"],
        lags_days=cache["lags_days"],
        lead_time_days=int(cache["lead_time_days"]),
    )
    beats = res["expert_mse"] < res["persistence_mse"]
    skill = 1.0 - (res["expert_mse"] / res["persistence_mse"]) if res["persistence_mse"] > 0 else 0.0
    return {
        "place": place,
        "n_candidate_sources": int(res["n_candidate_sources"]),
        "prior_regions": res["prior_regions"],
        "n_test": int(res["n_test"]),
        "expert_mse": res["expert_mse"],
        "persistence_mse": res["persistence_mse"],
        "climatology_mse": res.get("climatology_mse"),
        "r2_vs_persistence": res.get("r2_vs_persistence"),
        "r2_vs_climatology": res.get("r2_vs_climatology"),
        "skill_vs_persistence": skill,
        "beats_persistence": bool(beats),
        "score_std": res["score_std"],
        "selected_sources": [int(s) for s in res["selected_sources"]],
        "elapsed_s": res["elapsed_s"],
    }


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
    parser.add_argument("--cache-path", type=Path, default=None)
    parser.add_argument("--val-start", type=str, default="2005-01-01")
    parser.add_argument("--test-start", type=str, default="2013-01-01")
    parser.add_argument("--use-self-features", type=lambda s: s.lower() != "false", default=False)
    parser.add_argument("--self-bypass", type=lambda s: s.lower() != "false", default=False)
    parser.add_argument("--generator-all-fields", action="store_true",
                         help="B7 audit fix: feed the rationale generator all 6 fields, not just OLR "
                              "(train_place reads this via getattr; this script never exposed it before Phase 9.2).")
    parser.add_argument("--season-path", type=str, default=None,
                         help="Phase 1: results/phase1/bsiso_season.npz -- train/test only on samples whose target day is in the BSISO season")
    parser.add_argument("--prior", type=str, default="none", choices=("none", "pcmci", "random", "corr"),
                         help="Phase 1 Stage C: source-region prior handed to the splitter")
    parser.add_argument("--prior-mode", type=str, default="restrict", choices=("restrict", "init", "regularize"))
    parser.add_argument("--prior-path", type=str, default="results/phase1/causal_graph_50.json")
    parser.add_argument("--prior-weight", type=float, default=1.0, help="regularize mode: penalty weight on off-shortlist edge scores")
    parser.add_argument("--prior-init-scale", type=float, default=2.0, help="init mode: logit bias for the strongest prior region")
    parser.add_argument("--workers", type=int, default=1, help="places trained in parallel processes")
    parser.add_argument("--out", type=str, default=None, help="output JSON (default results/step7_all_places_<variant>.json)")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)
    from scripts.train_step4_single_place import CACHE_PATH as DEFAULT_CACHE_PATH
    cache_path = args.cache_path or DEFAULT_CACHE_PATH
    cache = load_cache(cache_path)
    n_clusters = cache["n_clusters"]
    places = parse_places(args.places, n_clusters)
    print(f"step 7: {len(places)} places, variant={args.variant}, epochs={args.epochs}, "
          f"cap={args.n_samples_cap}, prior={args.prior}/{args.prior_mode}, season={'yes' if args.season_path else 'no'}, "
          f"workers={args.workers}, OUT-OF-SAMPLE eval (test>={args.test_start})")
    out_path = Path(args.out) if args.out else RESULTS_DIR / f"step7_all_places_{args.variant}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    t_start = time.time()
    if args.workers > 1:
        import multiprocessing as mp
        pool = mp.get_context("fork").Pool(args.workers, initializer=_init_worker, initargs=(cache_path, args))
        results = pool.imap_unordered(_run_one, places)
    else:
        _init_worker(cache_path, args, cache)
        results = map(_run_one, places)
    for k, row in enumerate(results):
        rows.append(row)
        rows.sort(key=lambda r: r["place"])
        elapsed_all = time.time() - t_start
        eta = (elapsed_all / (k + 1)) * (len(places) - k - 1)
        print(f"[{k+1}/{len(places)}] place {row['place']:>2}: mse={row['expert_mse']:.4f} "
              f"persist={row['persistence_mse']:.4f} skill={row['skill_vs_persistence']:+.3f} "
              f"R2vsClim={row['r2_vs_climatology']:+.3f} score_std={row['score_std']:.4f} "
              f"beats={'Y' if row['beats_persistence'] else 'n'} ({row['elapsed_s']:.0f}s, ETA {eta/60:.0f}m)")

        # checkpoint after every place so a long run is never lost
        skills = [r["skill_vs_persistence"] for r in rows]
        r2_clims = [r["r2_vs_climatology"] for r in rows if r["r2_vs_climatology"] is not None]
        out_path.write_text(json.dumps({
            "variant": args.variant,
            "epochs": args.epochs,
            "n_samples_cap": args.n_samples_cap,
            "prior": args.prior,
            "prior_mode": args.prior_mode if args.prior != "none" else None,
            "season": bool(args.season_path),
            "n_places_done": len(rows),
            "n_beating_persistence": sum(1 for r in rows if r["beats_persistence"]),
            "n_informative": sum(1 for r in rows if r["score_std"] >= INFORMATIVE_SCORE_STD),
            "skill_mean": float(np.mean(skills)),
            "skill_median": float(np.median(skills)),
            "r2_vs_climatology_mean": float(np.mean(r2_clims)) if r2_clims else None,
            "places": rows,
        }, indent=2))

    skills = [r["skill_vs_persistence"] for r in rows]
    r2_clims = [r["r2_vs_climatology"] for r in rows if r["r2_vs_climatology"] is not None]
    n_beat = sum(1 for r in rows if r["beats_persistence"])
    print(f"\n=== step 7 summary ({len(rows)} places, OUT-OF-SAMPLE) ===")
    print(f"places beating persistence: {n_beat}/{len(rows)} ({100*n_beat/max(len(rows),1):.0f}%)")
    print(f"skill vs persistence: mean={np.mean(skills):+.4f} median={np.median(skills):+.4f} "
          f"min={np.min(skills):+.4f} max={np.max(skills):+.4f}")
    if r2_clims:
        print(f"R^2 vs climatology: mean={np.mean(r2_clims):+.4f} median={np.median(r2_clims):+.4f} "
              f"min={np.min(r2_clims):+.4f} max={np.max(r2_clims):+.4f}")
    worst = sorted(rows, key=lambda r: r["skill_vs_persistence"])[:5]
    best = sorted(rows, key=lambda r: -r["skill_vs_persistence"])[:5]
    print(f"best places:  {[(r['place'], round(r['skill_vs_persistence'],3)) for r in best]}")
    print(f"worst places: {[(r['place'], round(r['skill_vs_persistence'],3)) for r in worst]}")
    print(f"total time: {(time.time()-t_start)/60:.1f} min")


if __name__ == "__main__":
    main()
