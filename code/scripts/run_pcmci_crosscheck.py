"""PCMCI independent cross-check (SS4.2) on the real BSISO data.

SS4.2 is explicit about PCMCI's role here: "an independent cross-check ...
Nothing else -- it is not a pre-filter, not a replacement for Channel 2,
not a source of ground truth." If two completely different approaches --
PCMCI's conditional-independence statistics and the splitter's neural
invariance training -- land on the same causal structure, that agreement is
much stronger evidence than either method checking its own consistency.

This runs PCMCI (tigramite, SS7) over the same cluster-level OLR series the
splitter sees for one target place, restricted to the same candidate source
set, and compares the two rankings.

SS4.2's disagreement protocol is followed in the reporting:
  1. report WHERE they disagree (same edges scored oppositely, or entirely
     different structure),
  2. note that PCMCI assumes roughly STATIONARY causal structure -- the
     exact assumption this project's premise says breaks during a regime
     shift, so splitter-only findings are not automatically errors,
  3. report both results and the disagreement itself, rather than forcing a
     winner.

Usage:
    python scripts/run_pcmci_crosscheck.py [--target 22] [--variant direct]
        [--tau-max 10] [--pc-alpha 0.05]
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

from causal_moe.data.candidate_edges import build_candidate_source_set
from scripts.train_step4_single_place import OLR_LAG0_CHANNEL, load_cache

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, default=22)
    parser.add_argument("--variant", type=str, default="direct")
    parser.add_argument("--tau-max", type=int, default=10,
                        help="max lag tested; 10 matches SS4.1's own lag structure (today/5d/10d)")
    parser.add_argument("--pc-alpha", type=float, default=0.05)
    parser.add_argument("--n-samples", type=int, default=6000,
                        help="chronological prefix used; PCMCI cost grows fast with T and N")
    args = parser.parse_args()

    from tigramite import data_processing as pp
    from tigramite.independence_tests.parcorr import ParCorr
    from tigramite.pcmci import PCMCI

    RESULTS_DIR.mkdir(exist_ok=True)
    cache = load_cache()
    target = args.target

    candidate_set = build_candidate_source_set(
        cache["edge_index"], target, args.variant, n_clusters=cache["n_clusters"]
    )
    sources = candidate_set.source_clusters
    # PCMCI works on a multivariate series; use the candidate sources (which
    # already include the target itself) as the variable set.
    var_ids = list(int(s) for s in sources)
    target_pos = var_ids.index(target)

    olr = cache["features"][: args.n_samples, :, OLR_LAG0_CHANNEL]  # (T, 50)
    series = olr[:, var_ids].astype(np.float64)  # (T, n_vars)
    T, N = series.shape
    print(f"PCMCI on place {target}, variant={args.variant}: T={T}, N={N} vars {var_ids}, "
          f"tau_max={args.tau_max}, pc_alpha={args.pc_alpha}")

    dataframe = pp.DataFrame(series, var_names=[f"c{v}" for v in var_ids])
    pcmci = PCMCI(dataframe=dataframe, cond_ind_test=ParCorr(), verbosity=0)
    results = pcmci.run_pcmci(tau_max=args.tau_max, pc_alpha=args.pc_alpha)

    p_matrix = results["p_matrix"]  # (N, N, tau_max+1)
    val_matrix = results["val_matrix"]

    # Drivers of the TARGET: links (source -> target) at any lag >= 1.
    # Rank each source by its strongest significant link into the target.
    pcmci_scores = {}
    for i, v in enumerate(var_ids):
        best_val, best_lag, best_p = 0.0, None, 1.0
        for tau in range(1, args.tau_max + 1):
            p = float(p_matrix[i, target_pos, tau])
            val = abs(float(val_matrix[i, target_pos, tau]))
            if val > best_val:
                best_val, best_lag, best_p = val, tau, p
        pcmci_scores[v] = {"strength": best_val, "lag": best_lag, "p_value": best_p}

    significant = {v: s for v, s in pcmci_scores.items() if s["p_value"] < args.pc_alpha}
    ranked = sorted(pcmci_scores.items(), key=lambda kv: -kv[1]["strength"])

    print("\n--- PCMCI: drivers of the target, ranked by |link strength| ---")
    for v, s in ranked:
        flag = "SIG" if s["p_value"] < args.pc_alpha else "   "
        self_note = " (self/persistence)" if v == target else ""
        print(f"  {flag} c{v:<3} strength={s['strength']:.4f} lag={s['lag']} "
              f"p={s['p_value']:.4g}{self_note}")

    # --- compare against the splitter's selection, if step 4 ran ----------
    step4_path = RESULTS_DIR / f"step4_place{target}.json"
    comparison = None
    if step4_path.exists():
        step4 = json.loads(step4_path.read_text())
        variant_entry = next((v for v in step4["variants"] if v["variant"] == args.variant), None)
        if variant_entry:
            splitter_selected = set(int(s) for s in variant_entry["selected_sources"])
            pcmci_selected = set(v for v in significant if v != target)
            agree = splitter_selected & pcmci_selected
            only_splitter = splitter_selected - pcmci_selected
            only_pcmci = pcmci_selected - splitter_selected
            comparison = {
                "splitter_selected": sorted(splitter_selected),
                "pcmci_significant": sorted(pcmci_selected),
                "agreement": sorted(agree),
                "only_splitter": sorted(only_splitter),
                "only_pcmci": sorted(only_pcmci),
            }
            print("\n--- SS4.2 cross-check ---")
            print(f"splitter selected: {sorted(splitter_selected)}")
            print(f"PCMCI significant: {sorted(pcmci_selected)}")
            print(f"AGREEMENT (both):  {sorted(agree)}")
            print(f"only splitter:     {sorted(only_splitter)}")
            print(f"only PCMCI:        {sorted(only_pcmci)}")
            print("\nSS4.2 disagreement protocol: PCMCI assumes roughly STATIONARY causal")
            print("structure. Edges the splitter finds but PCMCI misses are not automatically")
            print("errors -- they may be non-stationary/regime-dependent links, which is the")
            print("exact case this project's premise says PCMCI cannot represent.")
    else:
        print(f"\n(no {step4_path.name} yet -- run scripts/train_step4_single_place.py "
              f"to enable the side-by-side comparison)")

    report = {
        "target": target,
        "variant": args.variant,
        "tau_max": args.tau_max,
        "pc_alpha": args.pc_alpha,
        "n_samples": T,
        "variables": var_ids,
        "pcmci_scores": {str(k): v for k, v in pcmci_scores.items()},
        "comparison": comparison,
    }
    out_path = RESULTS_DIR / f"pcmci_place{target}_{args.variant}.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
