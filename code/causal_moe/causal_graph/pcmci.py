"""Stage A: PCMCI (Runge et al. 2019) with partial-correlation tests,
restricted to links INTO the target nodes (Phase 1 spec SS3).

Equivalent to tigramite's `PCMCI.run_pcmci(tau_min=1, tau_max, pc_alpha,
link_assumptions=...)` with ParCorr where each target may have any lagged
parent from the candidate pool and every other node may only have its own
past as parents (that "own past" is what MCI conditions the source on, SS3
step 2). Checked link-for-link against tigramite in tests/test_pcmci.py.

Why not call tigramite directly: at 300 nodes x 60 lags = 18,000
candidates per target, tigramite builds a fresh array for every single
test. Here the lagged design matrix is built once and shared by all
targets, and each PC1 iteration tests every candidate against the same
conditioning set in one batched projection.

Two stages per target:
  1. PC1 (screening): start with every candidate (node, lag); iteration p
     conditions each candidate on the p strongest current parents (other
     than itself, max_combinations=1 as in tigramite) and drops those with
     p > pc_alpha. Stops when fewer than p+1 parents remain.
  2. MCI (final test): each survivor is retested conditioning on the
     target's other parents AND the source's own-past parents shifted by
     the link's lag. P-values are Benjamini-Hochberg corrected over ALL
     candidates of that target (dropped candidates count as p = 1).

Lag 0 is excluded: direction comes from time order.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass
class PCMCIConfig:
    tau_min: int = 1
    tau_max: int = 60
    pc_alpha: float = 0.05
    fdr_alpha: float = 0.05
    max_conds_dim: int | None = None
    max_conds_px: int | None = None
    chunk: int = 4096


@dataclass
class Link:
    node: int
    lag: int
    val: float  # MCI partial correlation (signed)
    p: float
    q: float  # BH-FDR adjusted


def _orthonormal(Z: np.ndarray) -> np.ndarray | None:
    if Z is None or Z.shape[1] == 0:
        return None
    q, _ = np.linalg.qr(Z)
    return q


def _resid(A: np.ndarray, Q: np.ndarray | None) -> np.ndarray:
    return A if Q is None else A - Q @ (Q.T @ A)


def _parcorr(ry: np.ndarray, RX: np.ndarray, n_cond: int) -> tuple[np.ndarray, np.ndarray]:
    """ry (n,), RX (n, m): residuals after removing the same conditioning set."""
    num = RX.T @ ry
    den = np.linalg.norm(RX, axis=0) * np.linalg.norm(ry)
    val = np.clip(num / np.maximum(den, 1e-12), -1 + 1e-12, 1 - 1e-12)
    df = ry.shape[0] - 2 - n_cond
    t = val * np.sqrt(df / (1 - val**2))
    return val, 2 * stats.t.sf(np.abs(t), df)


def _standardize(a: np.ndarray) -> np.ndarray:
    a = a - a.mean(axis=0)
    sd = a.std(axis=0)
    return a / np.where(sd > 1e-12, sd, 1.0)


class LaggedDesign:
    """All (node, lag) candidate columns evaluated on the target-day rows.

    series: (T, N) node series. rows: (n,) target-day time indices, all
    >= 2 * tau_max so shifted own-past conditions stay inside the record.
    """

    def __init__(self, series: np.ndarray, rows: np.ndarray, nodes: np.ndarray, cfg: PCMCIConfig):
        if rows.min() < 2 * cfg.tau_max:
            raise ValueError("rows must start at >= 2 * tau_max")
        self.series, self.rows, self.cfg = series, rows, cfg
        lags = np.arange(cfg.tau_min, cfg.tau_max + 1)
        self.col_node = np.repeat(nodes, lags.size)
        self.col_lag = np.tile(lags, nodes.size)
        self.X = np.empty((rows.size, self.col_node.size), dtype=np.float32)
        for k, nd in enumerate(nodes):
            block = series[rows[:, None] - lags[None, :], nd]
            self.X[:, k * lags.size : (k + 1) * lags.size] = _standardize(block.astype(np.float64))
        self.col_of = {(int(n), int(l)): c for c, (n, l) in enumerate(zip(self.col_node, self.col_lag))}

    def column(self, node: int, lag: int) -> np.ndarray:
        c = self.col_of.get((node, lag))
        if c is not None:
            return self.X[:, c].astype(np.float64)
        return _standardize(self.series[self.rows - lag, node].astype(np.float64))

    def target(self, node: int) -> np.ndarray:
        return _standardize(self.series[self.rows, node].astype(np.float64))


def pc1(y: np.ndarray, design: LaggedDesign, cand_cols: np.ndarray, cfg: PCMCIConfig) -> list[int]:
    """PC1 condition selection. Returns surviving columns, strongest first."""
    parents = np.asarray(cand_cols)
    val_min = np.full(design.X.shape[1], np.inf)
    p = 0
    while len(parents) - 1 >= p and (cfg.max_conds_dim is None or p <= cfg.max_conds_dim):
        order = parents[np.argsort(-val_min[parents], kind="stable")]
        Zcols = order[:p]
        Q = _orthonormal(design.X[:, Zcols].astype(np.float64))
        ry = _resid(y, Q)
        pvals = np.empty(design.X.shape[1])
        rest = order[p:]
        for s in range(0, rest.size, cfg.chunk):
            cols = rest[s : s + cfg.chunk]
            val, pv = _parcorr(ry, _resid(design.X[:, cols].astype(np.float64), Q), p)
            val_min[cols] = np.minimum(val_min[cols], np.abs(val))
            pvals[cols] = pv
        for c in Zcols:  # a conditioning parent is tested against the next p strongest
            conds = [z for z in order[: p + 1] if z != c]
            Qc = _orthonormal(design.X[:, conds].astype(np.float64))
            val, pv = _parcorr(_resid(y, Qc), _resid(design.X[:, [c]].astype(np.float64), Qc), p)
            val_min[c] = min(val_min[c], abs(val[0]))
            pvals[c] = pv[0]
        parents = order[pvals[order] <= cfg.pc_alpha]
        p += 1
    return [int(c) for c in parents[np.argsort(-val_min[parents], kind="stable")]]


def own_past_parents(design: LaggedDesign, node: int, cfg: PCMCIConfig) -> list[tuple[int, int]]:
    """PC1 for a source node over its own lags only (its MCI conditions)."""
    lags = np.arange(cfg.tau_min, cfg.tau_max + 1)
    sub = LaggedDesign(design.series, design.rows, np.array([node]), cfg)
    cols = pc1(sub.target(node), sub, np.arange(lags.size), cfg)
    return [(node, int(sub.col_lag[c])) for c in cols]


def mci(
    y: np.ndarray,
    design: LaggedDesign,
    parent_cols: list[int],
    px_cache: dict,
    cfg: PCMCIConfig,
) -> list[tuple[int, float, float]]:
    """Returns [(col, val, p)] for every PC1 survivor."""
    out = []
    for c in parent_cols:
        node, lag = int(design.col_node[c]), int(design.col_lag[c])
        if node not in px_cache:
            px_cache[node] = own_past_parents(design, node, cfg)
        px = px_cache[node] if cfg.max_conds_px is None else px_cache[node][: cfg.max_conds_px]
        conds = [design.X[:, z].astype(np.float64) for z in parent_cols if z != c]
        conds += [design.column(n, l + lag) for n, l in px]
        Z = np.stack(conds, axis=1) if conds else None
        Q = _orthonormal(Z)
        val, pv = _parcorr(_resid(y, Q), _resid(design.X[:, [c]].astype(np.float64), Q), 0 if Z is None else Z.shape[1])
        out.append((c, float(val[0]), float(pv[0])))
    return out


def bh_fdr(pvals: np.ndarray) -> np.ndarray:
    m = pvals.size
    order = np.argsort(pvals)
    ranked = pvals[order] * m / np.arange(1, m + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(m)
    out[order] = np.minimum(q, 1.0)
    return out


def run_targets(
    design: LaggedDesign,
    targets: list[int],
    cfg: PCMCIConfig,
    cand_cols: dict | None = None,
    log=None,
) -> dict[int, tuple[list[Link], int]]:
    """PCMCI for several target nodes: PC1 for every target first, then
    MCI, so a source that is itself a target (another region's OLR) is
    conditioned on its full PC1 parent set, exactly as tigramite does; all
    other sources are conditioned on their own-past parents.

    cand_cols: optional {target: candidate column array} (default: all).
    Returns {target: (FDR-significant links sorted by |val| desc,
    number of PC1 survivors)}.
    """
    all_cols = np.arange(design.X.shape[1])
    cands, parents = {}, {}
    for k, t in enumerate(targets):
        cands[t] = all_cols if cand_cols is None else np.asarray(cand_cols[t])
        parents[t] = pc1(design.target(t), design, cands[t], cfg)
        if log:
            log(f"  PC1 {k + 1}/{len(targets)} target node {t}: {len(parents[t])} survivors")
    px_cache = {t: [(int(design.col_node[c]), int(design.col_lag[c])) for c in parents[t]] for t in targets}

    out = {}
    for t in targets:
        tested = mci(design.target(t), design, parents[t], px_cache, cfg)
        pvals = np.ones(cands[t].size)
        pos = {int(c): k for k, c in enumerate(cands[t])}
        for c, _, pv in tested:
            pvals[pos[c]] = pv
        q = bh_fdr(pvals)
        links = [
            Link(int(design.col_node[c]), int(design.col_lag[c]), val, pv, float(q[pos[c]]))
            for c, val, pv in tested
            if q[pos[c]] <= cfg.fdr_alpha
        ]
        links.sort(key=lambda l: -abs(l.val))
        out[t] = (links, len(parents[t]))
    return out


def run_target(design: LaggedDesign, target: int, cfg: PCMCIConfig, cand_cols: np.ndarray | None = None):
    """Single-target convenience wrapper around run_targets."""
    return run_targets(design, [target], cfg, None if cand_cols is None else {target: cand_cols})[target]
