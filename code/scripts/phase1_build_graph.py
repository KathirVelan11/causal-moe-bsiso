"""Phase 1 Stages A + B: build the causal graph with PCMCI on 1979-2012
season target days, validate it with B1-B4, freeze the per-target shortlist.

Stages (each writes to results/phase1/ and is skipped by later stages if
its output already exists, unless re-requested with --stages):
  season    BSISO index + per-year active season from our OLR/u850
  a50       Stage A: PCMCI, 300 nodes x lags 1-60, targets = 50 OLR nodes
  b1        decoy pass: same PCMCI with phase-randomized decoys in the pool
  a100      Stage A at 100 nested regions (for B4)
  validate  B1 strength filter -> B2 lag mask -> B3 top-k -> B4 persistence
            -> results/phase1/causal_graph_50.json (the frozen prior)

Usage:
    python scripts/phase1_build_graph.py [--stages season,a50,b1,a100,validate]
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

from causal_moe.causal_graph import validate as V
from causal_moe.causal_graph.nodes import (
    N_FIELDS,
    build_node_series,
    node_id,
    node_region_field,
    region_min_distance_m,
    split_regions_in_two,
    valid_nodes,
)
from causal_moe.causal_graph.pcmci import LaggedDesign, PCMCIConfig, run_targets
from causal_moe.causal_graph.season import compute_bsiso_season

CODE = Path(__file__).resolve().parents[1]
OUT = CODE / "results" / "phase1"
CACHE = CODE / "cache"
DISCOVERY_END = np.datetime64("2012-12-31")
VAL_START = np.datetime64("2005-01-01")
CLUSTER_FIT_END = np.datetime64("2005-01-01")


class Context:
    """Lazy loaders shared by the stages."""

    def __init__(self, args):
        self.args = args
        self._raw = None

    @property
    def raw(self):
        if self._raw is None:
            from causal_moe.data.raw import load_raw_dataset

            t0 = time.time()
            self._raw = load_raw_dataset()
            print(f"raw loaded ({time.time() - t0:.0f}s)")
        return self._raw

    def season(self):
        d = np.load(OUT / "bsiso_season.npz")
        return d["time"], d["in_season"]

    def labels50(self):
        return np.load(self.args.cache_path)["cluster_labels"]

    def nodes(self, n_regions: int):
        path = CACHE / f"phase1_nodes{n_regions}.npz"
        if not path.exists():
            time_, _ = self.season()
            labels = self.labels50()
            if n_regions == 100:
                labels = split_regions_in_two(self.raw, labels, self.raw.time < CLUSTER_FIT_END)
            series = build_node_series(self.raw, labels, time_ <= DISCOVERY_END)
            np.savez(path, series=series, labels=labels)
        d = np.load(path)
        return d["series"], d["labels"]

    def rows(self, tau_max: int, lo=None, hi=DISCOVERY_END):
        time_, in_season = self.season()
        keep = in_season & (time_ <= hi) & (np.arange(time_.size) >= 2 * tau_max)
        if lo is not None:
            keep &= time_ >= lo
        return np.nonzero(keep)[0]


def cfg_from(args) -> PCMCIConfig:
    return PCMCIConfig(tau_max=args.tau_max, pc_alpha=args.pc_alpha, fdr_alpha=args.fdr_alpha)


def to_edges(links, n_real_nodes: int) -> list[dict]:
    out = []
    for l in links:
        e = {"lag": l.lag, "val": l.val, "p": l.p, "q": l.q}
        if l.node >= n_real_nodes:
            e.update(src_region=-1, field="decoy", decoy_id=l.node - n_real_nodes)
        else:
            region, field = node_region_field(l.node)
            e.update(src_region=region, field=field)
        out.append(e)
    return out


def run_stage_a(ctx, series, n_regions, extra=None, label=""):
    args = ctx.args
    cfg = cfg_from(args)
    rows = ctx.rows(cfg.tau_max)
    n_real = series.shape[1]
    nodes = valid_nodes(series)
    if extra is not None:
        series = np.concatenate([series, extra], axis=1)
        nodes = np.concatenate([nodes, n_real + np.arange(extra.shape[1])])
    t0 = time.time()
    design = LaggedDesign(series, rows, nodes, cfg)
    print(f"[{label}] design {design.X.shape} ({len(nodes)} nodes x {cfg.tau_max} lags, {rows.size} season target days)")
    targets = [node_id(r, "olr") for r in range(n_regions)]
    res = run_targets(design, targets, cfg, log=print if args.verbose else None)
    graph = {}
    for t in targets:
        links, n_pc1 = res[t]
        graph[str(t // N_FIELDS)] = {"n_pc1_survivors": n_pc1, "edges": to_edges(links, n_real)}
    print(f"[{label}] done in {(time.time() - t0) / 60:.1f} min; "
          f"mean significant parents/target = {np.mean([len(g['edges']) for g in graph.values()]):.1f}")
    meta = {"n_regions": n_regions, "n_nodes": int(len(nodes)), "n_candidates_per_target": int(design.X.shape[1]),
            "n_target_days": int(rows.size), "tau_min": cfg.tau_min, "tau_max": cfg.tau_max,
            "pc_alpha": cfg.pc_alpha, "fdr_alpha": cfg.fdr_alpha, "years": "1979-2012, BSISO-season target days"}
    return meta, graph


def stage_season(ctx):
    s = compute_bsiso_season(ctx.raw)
    np.savez(OUT / "bsiso_season.npz", time=s.time, pc=s.pc, amplitude=s.amplitude, in_season=s.in_season)
    import pandas as pd

    per_year = pd.Series(s.in_season).groupby(pd.DatetimeIndex(s.time).year).sum()
    summary = {
        "explained_variance_pc1_pc2": s.explained_variance.tolist(),
        "fallback_years": s.fallback_years,
        "season_days_per_year": {int(k): int(v) for k, v in per_year.items()},
        "onset": {y: str(v) for y, v in s.onset.items()},
        "end": {y: str(v) for y, v in s.end.items()},
    }
    (OUT / "bsiso_season.json").write_text(json.dumps(summary, indent=1))
    print(f"season: EOF var {np.round(s.explained_variance, 3)}, days/yr mean {per_year.mean():.0f} "
          f"(min {per_year.min()}, max {per_year.max()}), fallback years {s.fallback_years}")


def stage_a50(ctx):
    series, _ = ctx.nodes(50)
    meta, graph = run_stage_a(ctx, series, 50, label="A50")
    (OUT / "stage_a_50.json").write_text(json.dumps({"meta": meta, "graph": graph}, indent=1))


def stage_b1(ctx):
    series, _ = ctx.nodes(50)
    decoys, origin = V.make_decoys(series, valid_nodes(series), ctx.args.n_decoys, seed=ctx.args.seed)
    meta, graph = run_stage_a(ctx, series, 50, extra=decoys, label="B1 decoys")
    report = V.decoy_report({k: g["edges"] for k, g in graph.items()}, ctx.args.n_decoys)
    print(f"B1: {report['n_decoys_picked']}/{report['n_decoys']} decoys picked "
          f"(rate {report['decoy_rate']:.1%}, gate < {V.DECOY_RATE_GATE:.0%}: "
          f"{'PASS' if report['passes_gate'] else 'FAIL'}), max decoy |val| {report['max_decoy_strength']:.3f}")
    meta["decoy_origin_nodes"] = origin.tolist()
    (OUT / "b1_decoy_pass.json").write_text(json.dumps({"meta": meta, "report": report, "graph": graph}, indent=1))


def stage_a100(ctx):
    series, _ = ctx.nodes(100)
    meta, graph = run_stage_a(ctx, series, 100, label="A100")
    (OUT / "stage_a_100.json").write_text(json.dumps({"meta": meta, "graph": graph}, indent=1))


def stage_validate(ctx):
    args = ctx.args
    a50 = json.loads((OUT / "stage_a_50.json").read_text())
    b1 = json.loads((OUT / "b1_decoy_pass.json").read_text())
    a100 = json.loads((OUT / "stage_a_100.json").read_text())
    series, labels = ctx.nodes(50)
    dist = region_min_distance_m(ctx.raw, labels)

    threshold = b1["report"]["max_decoy_strength"]
    if not b1["report"]["passes_gate"]:
        print("WARNING: B1 decoy gate FAILED -- tighten PCMCI settings (pc_alpha / fdr_alpha) and rerun a50,b1")

    counts, after_b2 = {}, {}
    for t_str, g in a50["graph"].items():
        t = int(t_str)
        edges = [dict(e) for e in g["edges"]]
        c = {"pcmci": len(edges)}
        edges = V.apply_b1(edges, threshold)
        c["b1"] = len(edges)
        all_b2 = [dict(e) for e in edges]
        edges = V.apply_b2(all_b2, t, dist)
        c["b2"] = len(edges)
        c["b2_failed"] = [e for e in all_b2 if not e["b2_pass"]]
        counts[t], after_b2[t] = c, edges

    fit_rows = ctx.rows(args.tau_max, hi=VAL_START - np.timedelta64(1, "D"))
    val_rows = ctx.rows(args.tau_max, lo=VAL_START)
    node_of = lambda e: node_id(e["src_region"], e["field"])  # noqa: E731
    k, curve = V.choose_k(series, node_of, {t: node_id(t, "olr") for t in after_b2}, after_b2, fit_rows, val_rows,
                          k_max=args.k_max)
    if args.k is not None:
        k = args.k
    print(f"B3: k = {k} (validation R^2 by k: { {kk: round(v, 4) for kk, v in curve.items()} })")

    fine = {int(t): g["edges"] for t, g in a100["graph"].items()}
    targets, persist_scores = {}, []
    for t in sorted(after_b2):
        own, drivers = V.apply_b3(after_b2[t], t, k)
        score = V.apply_b4(drivers, t, fine)
        if score is not None:
            persist_scores.append(score)
        counts[t]["b3"] = len(drivers)
        targets[str(t)] = {
            "n_pcmci_edges": counts[t]["pcmci"],
            "n_after_b1": counts[t]["b1"],
            "n_after_b2": counts[t]["b2"],
            "b2_failed": counts[t]["b2_failed"],
            "candidates_after_b2": [{k: e[k] for k in ("src_region", "field", "lag", "val")} for e in after_b2[t]],
            "own_region_edges": own,
            "drivers": drivers,
            "persistence_score": score,
            "persistence_pass": None if score is None else score >= V.PERSISTENCE_PASS,
            "prior": V.region_prior(drivers),
        }

    n_driver = [len(v["drivers"]) for v in targets.values()]
    n_regions = [len(v["prior"]) for v in targets.values()]
    summary = {
        "decoy_report": b1["report"],
        "b1_strength_threshold": threshold,
        "k": k,
        "k_validation_r2": curve,
        "edges_mean": {s: float(np.mean([counts[t][s] for t in counts])) for s in ("pcmci", "b1", "b2", "b3")},
        "prior_regions_per_target_mean": float(np.mean(n_regions)),
        "targets_with_no_driver": int(sum(1 for n in n_driver if n == 0)),
        "persistence_mean": float(np.mean(persist_scores)) if persist_scores else None,
        "persistence_pass_count": int(sum(1 for s in persist_scores if s >= V.PERSISTENCE_PASS)),
        "persistence_scored_targets": len(persist_scores),
        "speed_limits_ms": V.SPEED_LIMIT_MS,
    }
    out = {"meta": {"stage_a": a50["meta"], "stage_a_100": a100["meta"]}, "summary": summary, "targets": targets}
    (OUT / "causal_graph_50.json").write_text(json.dumps(out, indent=1))
    print("edges per target (mean): " + ", ".join(f"{s}={v:.1f}" for s, v in summary["edges_mean"].items()))
    print(f"prior regions per target (mean) {summary['prior_regions_per_target_mean']:.2f}, "
          f"targets with no driver {summary['targets_with_no_driver']}, "
          f"B4 persistence mean {summary['persistence_mean']}, pass {summary['persistence_pass_count']}/"
          f"{summary['persistence_scored_targets']}")


STAGES = {"season": stage_season, "a50": stage_a50, "b1": stage_b1, "a100": stage_a100, "validate": stage_validate}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stages", default="season,a50,b1,a100,validate")
    parser.add_argument("--tau-max", type=int, default=60)
    parser.add_argument("--pc-alpha", type=float, default=0.05)
    parser.add_argument("--fdr-alpha", type=float, default=0.05)
    parser.add_argument("--n-decoys", type=int, default=250)
    parser.add_argument("--k-max", type=int, default=10)
    parser.add_argument("--k", type=int, default=None, help="override the data-chosen top-k")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cache-path", type=Path, default=CACHE / "windowed_clustered50_lead7.npz")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    ctx = Context(args)
    for name in args.stages.split(","):
        t0 = time.time()
        print(f"=== stage {name} ===")
        STAGES[name](ctx)
        print(f"=== stage {name} done ({(time.time() - t0) / 60:.1f} min) ===")


if __name__ == "__main__":
    main()
