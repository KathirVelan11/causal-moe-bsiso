"""Phase 1 Stage C: train the splitter + expert at all 50 regions with the
frozen causal-graph prior and its controls, on BSISO-season target days,
tested on 2013-2022 (spec SS5).

Grid (per candidate pool direct / 2hop / full):
  none                              (no prior -- the season-day rerun of the
                                     existing direct/2hop/full results)
  pcmci, random, corr  x  restrict, init, regularize

Same training settings as the README's headline 7-day sweep (5 epochs,
4000-sample contiguous training cap, generator window 10), so the only
changes are the season restriction and the prior. One worker pool runs
every (config, region) job; each config's JSON is checkpointed after every
region and finished regions are skipped on restart.

Usage:
    python scripts/phase1_run_stage_c.py [--workers 12] [--pools full,direct,2hop]
        [--priors none,random,pcmci,corr] [--modes restrict,init,regularize]
"""

from __future__ import annotations

import argparse
import functools
import json
import sys
import time
from argparse import Namespace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

print = functools.partial(print, flush=True)  # noqa: A001

import numpy as np

from scripts import run_step7_all_places as step7
from scripts.train_step4_single_place import load_cache

CODE = Path(__file__).resolve().parents[1]
OUT = CODE / "results" / "phase1" / "stage_c"

TRAIN_DEFAULTS = dict(
    epochs=5, n_samples_cap=4000, cap_contiguous=True, r=None, n_swaps=6, lr=1e-3, batch_size=16,
    generator_window=10, entropy_weight=0.0, seed=0, val_start="2005-01-01", test_start="2013-01-01",
    use_self_features=False, self_bypass=False, generator_all_fields=False, max_candidates=0,
    season_path=str(CODE / "results" / "phase1" / "bsiso_season.npz"),
    prior_path=str(CODE / "results" / "phase1" / "causal_graph_50.json"),
    prior_weight=1.0, prior_init_scale=2.0,
)


def config_name(pool: str, prior: str, mode: str | None) -> str:
    return f"{pool}_{prior}" + (f"_{mode}" if mode else "")


def _job(job):
    name, args, place = job
    step7._W["args"] = args
    row = step7._run_one(place)
    return name, row


def summarize(name: str, args: Namespace, rows: list[dict]) -> dict:
    skills = [r["skill_vs_persistence"] for r in rows]
    r2 = [r["r2_vs_climatology"] for r in rows]
    return {
        "config": name,
        "variant": args.variant,
        "prior": args.prior,
        "prior_mode": args.prior_mode if args.prior != "none" else None,
        "season": True,
        "epochs": args.epochs,
        "n_samples_cap": args.n_samples_cap,
        "n_places_done": len(rows),
        "n_beating_persistence": sum(1 for r in rows if r["beats_persistence"]),
        "n_beating_climatology": sum(1 for r in rows if r["r2_vs_climatology"] > 0),
        "n_informative": sum(1 for r in rows if r["score_std"] >= step7.INFORMATIVE_SCORE_STD),
        "skill_mean": float(np.mean(skills)),
        "r2_vs_climatology_mean": float(np.mean(r2)),
        "score_std_mean": float(np.mean([r["score_std"] for r in rows])),
        "places": sorted(rows, key=lambda r: r["place"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--pools", default="full,direct,2hop")
    parser.add_argument("--priors", default="none,random,pcmci,corr")
    parser.add_argument("--modes", default="restrict,init,regularize")
    parser.add_argument("--places", default="all")
    parser.add_argument("--cache-path", type=Path, default=CODE / "cache" / "windowed_clustered50_lead7.npz")
    parser.add_argument("--epochs", type=int, default=TRAIN_DEFAULTS["epochs"])
    parser.add_argument("--n-samples-cap", type=int, default=TRAIN_DEFAULTS["n_samples_cap"])
    parser.add_argument("--prior-path", default=TRAIN_DEFAULTS["prior_path"])
    parser.add_argument("--out-dir", type=Path, default=OUT)
    cli = parser.parse_args()

    out_dir = cli.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    overrides = dict(epochs=cli.epochs, n_samples_cap=cli.n_samples_cap, prior_path=cli.prior_path)
    cache = load_cache(cli.cache_path)
    places = step7.parse_places(cli.places, cache["n_clusters"])

    configs = {}
    for pool in cli.pools.split(","):
        for prior in cli.priors.split(","):
            for mode in ([None] if prior == "none" else cli.modes.split(",")):
                configs[config_name(pool, prior, mode)] = Namespace(
                    **{**TRAIN_DEFAULTS, **overrides}, variant=pool, prior=prior, prior_mode=mode or "restrict"
                )

    done = {}
    for name in configs:
        path = out_dir / f"{name}.json"
        done[name] = json.loads(path.read_text())["places"] if path.exists() else []
    jobs = [(name, args, p) for name, args in configs.items() for p in places
            if p not in {r["place"] for r in done[name]}]
    print(f"stage C: {len(configs)} configs x {len(places)} places, {len(jobs)} jobs to run, workers={cli.workers}")

    step7._init_worker(cli.cache_path, None, cache)  # loaded before fork -> shared pages
    import multiprocessing as mp

    t0 = time.time()
    with mp.get_context("fork").Pool(cli.workers, initializer=step7._init_worker,
                                      initargs=(cli.cache_path, None, cache)) as pool:
        for k, (name, row) in enumerate(pool.imap_unordered(_job, jobs, chunksize=1)):
            done[name].append(row)
            (out_dir / f"{name}.json").write_text(json.dumps(summarize(name, configs[name], done[name]), indent=1))
            eta = (time.time() - t0) / (k + 1) * (len(jobs) - k - 1)
            print(f"[{k + 1}/{len(jobs)}] {name} place {row['place']}: R2clim={row['r2_vs_climatology']:+.3f} "
                  f"skill={row['skill_vs_persistence']:+.3f} score_std={row['score_std']:.4f} "
                  f"({row['elapsed_s']:.0f}s, ETA {eta / 3600:.1f}h)")

    print("\n=== stage C summary ===")
    for name in configs:
        path = out_dir / f"{name}.json"
        if path.exists():
            s = json.loads(path.read_text())
            print(f"{name:>28}: n={s['n_places_done']} informative={s['n_informative']} "
                  f"R2clim={s['r2_vs_climatology_mean']:+.4f} beatsClim={s['n_beating_climatology']} "
                  f"skill={s['skill_mean']:+.4f}")


if __name__ == "__main__":
    main()
