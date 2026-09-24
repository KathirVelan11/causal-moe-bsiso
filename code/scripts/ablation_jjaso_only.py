"""B23 ablation (2026-09-24): does restricting training/eval to the BSISO
season (May-Oct, "JJASO") change skill or edge-discrimination versus the
project's existing year-round result?

This is a ONE-OFF empirical check, not a permanent pipeline change -- B23
is an open scope question (see PROJECT_PLAN.md), not a bug with an agreed
fix. Deliberately kept as a standalone script rather than adding a
--season-filter flag to train_step4_single_place.py, so it doesn't touch
any code path the existing (already-committed, already-cited) results
depend on.

Method: load the exact same cache train_step4_single_place.py uses, mask
every sample array (features, targets, sample_time_index, sample_dates)
to May-Oct days BEFORE calling the same train_place() function with the
same args/seed -- so this is a true apples-to-apples comparison against
results/step4_place22.json, with the only difference being which rows
are visible to training/eval at all. Masking is safe here because each
row's lag features and lead target were already fully assembled when the
cache was built (windows.py) -- dropping a row doesn't corrupt any other
row's lag/lead computation, it just removes that day from consideration.

Usage:
    python scripts/ablation_jjaso_only.py --target 22 --variants direct,2hop,full
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from scripts.train_step4_single_place import load_cache, run_variant, CACHE_PATH

JJASO_MONTHS = (5, 6, 7, 8, 9, 10)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, default=22)
    parser.add_argument("--variants", type=str, default="direct,2hop,full")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--r", type=float, default=None)
    parser.add_argument("--n-swaps", type=int, default=6)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--generator-window", type=int, default=10)
    parser.add_argument("--entropy-weight", type=float, default=0.0)
    parser.add_argument("--n-samples-cap", type=int, default=0)
    parser.add_argument("--cap-contiguous", type=lambda s: s.lower() != "false", default=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cache-path", type=Path, default=CACHE_PATH)
    parser.add_argument("--val-start", type=str, default="2005-01-01")
    parser.add_argument("--test-start", type=str, default="2013-01-01")
    parser.add_argument("--use-self-features", type=lambda s: s.lower() != "false", default=False)
    parser.add_argument("--self-bypass", type=lambda s: s.lower() != "false", default=False)
    parser.add_argument("--generator-all-fields", action="store_true")
    args = parser.parse_args()

    t0 = time.time()
    cache = load_cache(args.cache_path)
    print(f"loaded cache in {time.time()-t0:.1f}s: features {cache['features'].shape}, "
          f"{cache['n_clusters']} clusters, lags_days={cache['lags_days']}")

    sample_dates_full = cache["sample_dates"].astype("datetime64[D]")
    months = sample_dates_full.astype("datetime64[M]").astype(int) % 12 + 1
    jjaso_mask = np.isin(months, JJASO_MONTHS)
    n_full = sample_dates_full.shape[0]
    n_jjaso = int(jjaso_mask.sum())
    print(f"JJASO filter: keeping {n_jjaso}/{n_full} samples ({100*n_jjaso/n_full:.1f}%)")

    # Filter every sample-indexed array identically. sample_time_index is
    # an index into the ORIGINAL raw time axis (used only for cross-
    # referencing, e.g. raw correlation diagnostics) -- filtering it
    # alongside the others keeps it aligned with the filtered rows; it is
    # not used to re-slice features_all downstream, so this is safe.
    features_jjaso = cache["features"][jjaso_mask]
    targets_jjaso = cache["targets"][jjaso_mask]
    sample_time_index_jjaso = cache["sample_time_index"][jjaso_mask]
    sample_dates_jjaso = sample_dates_full[jjaso_mask]

    variants = args.variants.split(",")
    results = []
    for variant in variants:
        result = run_variant(
            variant, args.target, features_jjaso, targets_jjaso,
            sample_time_index_jjaso, cache["edge_index"], cache["n_clusters"], args,
            sample_dates=sample_dates_jjaso, olr_lag0_channel=cache["olr_lag0_channel"],
            lags_days=cache["lags_days"], lead_time_days=int(cache["lead_time_days"]),
        )
        results.append(result)

    print("\n=== JJASO-only ablation summary ===")
    for r in results:
        beats = "YES" if r["expert_mse"] < r["persistence_mse"] else "no"
        print(f"{r['variant']:>6}: n_sources={r['n_candidate_sources']:>3}  "
              f"expert_mse={r['expert_mse']:.4f}  persistence_mse={r['persistence_mse']:.4f}  "
              f"climatology_mse={r['climatology_mse']:.4f}  r2_vs_climatology={r['r2_vs_climatology']:.4f}  "
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
    out_path = results_dir / f"ablation_jjaso_place{args.target}.json"
    out_path.write_text(json.dumps({
        "target": args.target,
        "n_full": n_full,
        "n_jjaso": n_jjaso,
        "variants": serializable,
    }, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
