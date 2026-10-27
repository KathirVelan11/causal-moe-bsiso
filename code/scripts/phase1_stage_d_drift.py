"""Phase 1 Stage D: is the causal structure stable over time, and do its
changes line up with known regime changes? (spec SS6)

For each target, PCMCI is rerun on sliding windows of BSISO seasons, using
only that target's candidates that passed B1 + B2
(results/phase1/causal_graph_50.json::candidates_after_b2). A window's
parent set is the FDR-significant (source region, field) pairs; adjacent
lags of the same pair count as one link (spec SS7). The causal signature
distance between consecutive windows is 1 - Jaccard(parents). That series
feeds the existing Rodionov/STARS detector (causal_moe/drift/rodionov.py),
which is the regime-shift test the drift channel already uses.

Checks against ENSO, using a Nino-3.4 index computed from our own SST
field (5S-5N, 170W-120W, season mean, standardized across years;
El Nino > +0.5, La Nina < -0.5):
  - is the distance larger when the ENSO phase changed between the two
    windows than when it did not (Mann-Whitney, one-sided)?
  - share of STARS alarms in years with an ENSO phase change, and the
    number of alarms in stable years (false drift).

This runs over the whole record (1979-2022): it describes structure change
and is not used to score forecasts.

Usage:
    python scripts/phase1_stage_d_drift.py [--window-seasons 1] [--targets all]
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
import pandas as pd
from scipy.stats import mannwhitneyu

from causal_moe.causal_graph.nodes import node_id
from causal_moe.causal_graph.pcmci import LaggedDesign, PCMCIConfig, run_targets
from causal_moe.drift.rodionov import detect_regime_shifts

CODE = Path(__file__).resolve().parents[1]
OUT = CODE / "results" / "phase1"


def nino34_by_year(time: np.ndarray, in_season: np.ndarray) -> pd.Series:
    from causal_moe.data.raw import FIELDS, load_raw_dataset

    raw = load_raw_dataset()
    li = (raw.lat >= -5) & (raw.lat <= 5)
    lj = (raw.lon >= 190) & (raw.lon <= 240)
    sst = np.nanmean(raw.fields[FIELDS.index("sst")][:, li][:, :, lj].reshape(raw.n_time, -1), axis=1)
    years = pd.DatetimeIndex(time).year
    s = pd.Series(sst[in_season]).groupby(years[in_season]).mean()
    # Anomalies are relative to a fixed 1979-2014 climatology, so the warming
    # trend would push late years toward "El Nino"; remove it first.
    x = s.index.to_numpy(dtype=float)
    s = s - np.polyval(np.polyfit(x, s.to_numpy(), 1), x)
    return (s - s.mean()) / s.std()


def enso_phase(z: float) -> str:
    return "nino" if z > 0.5 else ("nina" if z < -0.5 else "neutral")


def parent_pairs(links, node_region_field) -> set:
    return {node_region_field(l.node) for l in links}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--window-seasons", type=int, default=1)
    parser.add_argument("--targets", type=str, default="all")
    parser.add_argument("--fdr-alpha", type=float, default=0.05)
    parser.add_argument("--pc-alpha", type=float, default=0.05)
    parser.add_argument("--stars-cutoff", type=int, default=5)
    parser.add_argument("--stars-p", type=float, default=0.05)
    args = parser.parse_args()

    graph = json.loads((OUT / "causal_graph_50.json").read_text())
    tau_max = graph["meta"]["stage_a"]["tau_max"]
    series = np.load(CODE / "cache" / "phase1_nodes50.npz")["series"]
    season = np.load(OUT / "bsiso_season.npz")
    time, in_season = season["time"], season["in_season"]
    years = pd.DatetimeIndex(time).year.to_numpy()
    all_years = sorted(set(years[in_season & (np.arange(time.size) >= 2 * tau_max)]))
    targets = list(range(50)) if args.targets == "all" else [int(t) for t in args.targets.split(",")]
    cfg = PCMCIConfig(tau_max=tau_max, pc_alpha=args.pc_alpha, fdr_alpha=args.fdr_alpha)
    from causal_moe.causal_graph.nodes import node_region_field

    # Candidate (node, lag) per target, plus the target's own OLR past.
    cand = {}
    for t in targets:
        edges = graph["targets"][str(t)]["candidates_after_b2"]
        pairs = {(node_id(e["src_region"], e["field"]), int(e["lag"])) for e in edges}
        pairs |= {(node_id(t, "olr"), lag) for lag in (1, 2, 3)}
        cand[node_id(t, "olr")] = sorted(pairs)
    nodes = np.array(sorted({n for c in cand.values() for n, _ in c} | set(cand)))

    windows = [all_years[i : i + args.window_seasons] for i in range(len(all_years) - args.window_seasons + 1)]
    parents = {t: [] for t in targets}
    for w in windows:
        rows = np.nonzero(in_season & np.isin(years, w) & (np.arange(time.size) >= 2 * tau_max))[0]
        design = LaggedDesign(series, rows, nodes, cfg)
        cand_cols = {tn: np.array([design.col_of[p] for p in c]) for tn, c in cand.items()}
        res = run_targets(design, list(cand), cfg, cand_cols=cand_cols)
        for t in targets:
            parents[t].append(parent_pairs(res[node_id(t, "olr")][0], node_region_field))
        print(f"window {w[0]}-{w[-1]}: mean parents/target "
              f"{np.mean([len(parents[t][-1]) for t in targets]):.1f} ({rows.size} days)")

    end_years = [w[-1] for w in windows]
    nino = nino34_by_year(time, in_season)
    phase = {int(y): enso_phase(z) for y, z in nino.items()}

    per_target, dist_matrix = {}, []
    for t in targets:
        d = []
        for a, b in zip(parents[t][:-1], parents[t][1:]):
            union = a | b
            d.append(1.0 - len(a & b) / len(union) if union else 0.0)
        d = np.array(d)
        dist_matrix.append(d)
        shifts = detect_regime_shifts(d, cut_off_length=args.stars_cutoff, p=args.stars_p).shift_indices
        per_target[str(t)] = {
            "distance": d.round(4).tolist(),
            "alarm_years": [int(end_years[i + 1]) for i in shifts],
            "mean_parents": float(np.mean([len(p) for p in parents[t]])),
        }

    pair_years = end_years[1:]
    changed = np.array([phase[y] != phase[y - args.window_seasons] for y in pair_years])
    mean_d = np.mean(dist_matrix, axis=0)
    mw = mannwhitneyu(mean_d[changed], mean_d[~changed], alternative="greater")
    agg_shifts = detect_regime_shifts(mean_d, cut_off_length=args.stars_cutoff, p=args.stars_p).shift_indices
    alarm_years = [y for v in per_target.values() for y in v["alarm_years"]]
    changed_years = {y for y, c in zip(pair_years, changed) if c}
    in_changed = sum(1 for y in alarm_years if y in changed_years)

    summary = {
        "window_seasons": args.window_seasons,
        "years": [int(y) for y in pair_years],
        "enso_phase": phase,
        "enso_phase_changed": [bool(c) for c in changed],
        "mean_distance": mean_d.round(4).tolist(),
        "mean_distance_phase_change": float(mean_d[changed].mean()),
        "mean_distance_no_change": float(mean_d[~changed].mean()),
        "mannwhitney_p": float(mw.pvalue),
        "aggregate_alarm_years": [int(pair_years[i]) for i in agg_shifts],
        "n_target_alarms": len(alarm_years),
        "share_alarms_in_phase_change_years": in_changed / max(len(alarm_years), 1),
        "share_years_with_phase_change": float(changed.mean()),
        "n_alarms_in_stable_years": len(alarm_years) - in_changed,
    }
    (OUT / f"stage_d_drift_w{args.window_seasons}.json").write_text(json.dumps({"summary": summary, "targets": per_target}, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k not in ("years", "enso_phase", "enso_phase_changed", "mean_distance")}, indent=1))


if __name__ == "__main__":
    main()
