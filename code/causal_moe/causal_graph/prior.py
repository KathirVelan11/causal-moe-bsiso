"""Stage C: hand a per-target source-region prior to the splitter
(Phase 1 spec SS5).

Prior kinds (all region-level, decision 1):
  pcmci   validated shortlist from results/phase1/causal_graph_50.json
  random  same number of regions drawn at random from the same pool
          (strengths: the PCMCI strengths, shuffled)
  corr    same number of regions ranked by raw lagged |correlation| with
          the target's OLR on TRAINING rows only (the prior that already
          failed as a pool cap, B18)

Modes:
  restrict    candidate pool = prior regions (+ the target itself)
  init        pool unchanged; per-edge learnable bias starts at
              init_scale * strength / max strength for prior regions, 0 else
  regularize  pool unchanged; penalty on scores of non-prior regions

The target region itself is always a candidate (existing convention in
candidate_edges.py) and is neutral in every mode: never biased, never
penalized, never counted in the prior size.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from causal_moe.data.candidate_edges import CandidateSourceSet, rank_sources_by_lagged_correlation

PRIOR_KINDS = ("none", "pcmci", "random", "corr")
PRIOR_MODES = ("restrict", "init", "regularize")


def load_pcmci_prior(path: Path, target: int) -> dict[int, float]:
    graph = json.loads(Path(path).read_text())
    entry = graph["targets"].get(str(target), {"prior": {}})
    return {int(r): float(v["strength"]) for r, v in entry["prior"].items() if int(r) != target}


def build_prior(
    kind: str,
    target: int,
    pool: np.ndarray,
    pcmci_prior: dict[int, float],
    olr_train: np.ndarray | None = None,
    seed: int = 0,
) -> dict[int, float]:
    """Returns {source_region: strength} restricted to `pool`, excluding
    the target. Every kind has the same size as the PCMCI prior in this pool."""
    in_pool = {r: s for r, s in pcmci_prior.items() if r in set(pool.tolist()) and r != target}
    size = len(in_pool)
    if kind == "pcmci":
        return in_pool
    others = np.array([r for r in pool if r != target])
    if size == 0:
        return {}
    if kind == "random":
        rng = np.random.default_rng(seed * 1000 + target)
        chosen = rng.choice(others, size=size, replace=False)
        strengths = rng.permutation(list(in_pool.values()))
        return {int(r): float(s) for r, s in zip(chosen, strengths)}
    if kind == "corr":
        ranked = rank_sources_by_lagged_correlation(olr_train, target, pool)
        return {int(r): float(c) for r, c in ranked[:size]}
    raise ValueError(f"unknown prior kind {kind!r}")


def apply_prior(
    mode: str,
    candidate_set: CandidateSourceSet,
    prior: dict[int, float],
    init_scale: float = 2.0,
) -> tuple[CandidateSourceSet, np.ndarray | None, np.ndarray | None]:
    """Returns (candidate set, per-edge init bias or None, off-prior bool
    mask or None), edges ordered as candidate_set.source_clusters."""
    target = candidate_set.target
    if mode == "restrict":
        sources = np.array(sorted(set(prior) | {target}), dtype=np.int64)
        return CandidateSourceSet(target, f"{candidate_set.variant}_restrict", sources), None, None
    sources = candidate_set.source_clusters
    if mode == "init":
        top = max(prior.values()) if prior else 1.0
        bias = np.array([init_scale * prior[s] / top if s in prior else 0.0 for s in sources], dtype=np.float32)
        return candidate_set, bias, None
    if mode == "regularize":
        mask = np.array([(s not in prior) and (s != target) for s in sources], dtype=bool)
        return candidate_set, None, mask
    raise ValueError(f"unknown prior mode {mode!r}")
