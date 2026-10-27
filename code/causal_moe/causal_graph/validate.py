"""Stage B: the four checks applied to the Stage-A PCMCI graph, in order
(Phase 1 spec SS4).

  B1 decoys      phase-randomized copies of real nodes (same spectrum and
                 autocorrelation, no real relationship) are added to the
                 pool in a SEPARATE PCMCI pass. Decoy rate = share of decoys
                 picked as a parent of any target (gate: < 30%). Real edges
                 no stronger than the strongest selected decoy are dropped.
  B2 lag mask    implied speed = closest-cell distance between regions /
                 lag; edges needing a speed the source field cannot carry
                 fail. Lags near the 30-60 day oscillation are only flagged.
  B3 top-k       per target, keep the k strongest surviving edges from OTHER
                 regions (the target region itself is always a candidate
                 for the splitter, so its own edges do not spend the
                 budget). k is chosen from the data: the k whose edges best
                 predict the target on 2005-2012 season days when fitted on
                 1979-2004 (both inside the discovery years).
  B4 resolution  the same pipeline at 100 nested regions; an edge persists
                 if any child pair carries the same field within +-2 days
                 of the lag. Persistence score per target >= 0.5 passes;
                 failures are flagged, not deleted.

Edges are dicts: {src_region, field, lag, val, p, q} (+ flags added here).
"""

from __future__ import annotations

import numpy as np

# Generous per-field speed bounds, m/s (SS4 B2): SST effects ~1 m/s or less;
# convective fields (OLR, PW) up to convectively coupled Kelvin-wave speeds;
# dynamical fields (winds, height) up to dry Kelvin / Rossby-wave-train speeds.
SPEED_LIMIT_MS = {"sst": 1.0, "olr": 20.0, "pw": 20.0, "h850": 50.0, "u850": 50.0, "u200": 50.0}
OSCILLATION_ALIAS_MIN_LAG = 25
DECOY_RATE_GATE = 0.30
PERSISTENCE_PASS = 0.50
PERSISTENCE_LAG_TOL = 2


# ---------------------------------------------------------------- B1 ------
def phase_randomize(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Surrogate with the same power spectrum (hence autocorrelation,
    including the 30-60 day peak) and random Fourier phases."""
    n = x.shape[0]
    f = np.fft.rfft(x - x.mean())
    phases = rng.uniform(0, 2 * np.pi, f.shape[0])
    phases[0] = 0.0
    if n % 2 == 0:
        phases[-1] = 0.0
    out = np.fft.irfft(np.abs(f) * np.exp(1j * phases), n=n)
    return ((out - out.mean()) / out.std()).astype(np.float32)


def make_decoys(series: np.ndarray, source_nodes: np.ndarray, n_decoys: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Returns (decoy series (T, n_decoys), the real node each was built from)."""
    rng = np.random.default_rng(seed)
    origin = rng.choice(source_nodes, size=n_decoys, replace=True)
    return np.stack([phase_randomize(series[:, n], rng) for n in origin], axis=1), origin


def decoy_report(decoy_pass: dict, n_decoys: int) -> dict:
    """decoy_pass: {target_region: [edge dicts]} from the pass with decoys,
    where decoy edges carry src_region == -1."""
    picked, decoy_vals, n_real = set(), [], 0
    for edges in decoy_pass.values():
        for e in edges:
            if e["src_region"] == -1:
                picked.add(e["decoy_id"])
                decoy_vals.append(abs(e["val"]))
            else:
                n_real += 1
    rate = len(picked) / n_decoys
    return {
        "n_decoys": n_decoys,
        "n_decoys_picked": len(picked),
        "decoy_rate": rate,
        "gate": DECOY_RATE_GATE,
        "passes_gate": rate < DECOY_RATE_GATE,
        "n_decoy_edges": len(decoy_vals),
        "n_real_edges_in_decoy_pass": n_real,
        "decoy_edge_share": len(decoy_vals) / max(len(decoy_vals) + n_real, 1),
        "max_decoy_strength": max(decoy_vals) if decoy_vals else 0.0,
        "p95_decoy_strength": float(np.percentile(decoy_vals, 95)) if decoy_vals else 0.0,
    }


def apply_b1(edges: list[dict], max_decoy_strength: float) -> list[dict]:
    for e in edges:
        e["b1_pass"] = abs(e["val"]) > max_decoy_strength
    return [e for e in edges if e["b1_pass"]]


# ---------------------------------------------------------------- B2 ------
def apply_b2(edges: list[dict], target_region: int, dist_m: np.ndarray) -> list[dict]:
    kept = []
    for e in edges:
        d = float(dist_m[e["src_region"], target_region])
        speed = d / (e["lag"] * 86400.0)
        e["distance_km"] = d / 1e3
        e["implied_speed_ms"] = speed
        e["b2_pass"] = speed <= SPEED_LIMIT_MS[e["field"]]
        e["flag_oscillation_alias"] = e["lag"] >= OSCILLATION_ALIAS_MIN_LAG
        if e["b2_pass"]:
            kept.append(e)
    return kept


# ---------------------------------------------------------------- B3 ------
def _validation_r2(series, node_of, target_node, edges, fit_rows, val_rows) -> float:
    def design(rows):
        cols = [series[rows - e["lag"], node_of(e)] for e in edges]
        return np.column_stack([np.ones(rows.size)] + cols) if cols else np.ones((rows.size, 1))

    y_fit, y_val = series[fit_rows, target_node], series[val_rows, target_node]
    A = design(fit_rows)
    coef = np.linalg.solve(A.T @ A + 1e-3 * np.eye(A.shape[1]), A.T @ y_fit)
    resid = y_val - design(val_rows) @ coef
    return 1.0 - resid.var() / y_val.var()


def choose_k(series, node_of, target_nodes: dict, kept: dict, fit_rows, val_rows, k_max: int = 10) -> tuple[int, dict]:
    """kept: {target_region: B2-surviving edges}. Returns (k, curve) where
    curve[k] = mean validation R^2 across targets using the target region's
    own edges + its top-k other-region edges."""
    curve = {}
    for k in range(k_max + 1):
        r2 = []
        for t, edges in kept.items():
            own = [e for e in edges if e["src_region"] == t]
            other = sorted((e for e in edges if e["src_region"] != t), key=lambda e: -abs(e["val"]))[:k]
            r2.append(_validation_r2(series, node_of, target_nodes[t], own + other, fit_rows, val_rows))
        curve[k] = float(np.mean(r2))
    best = max(curve, key=curve.get)
    return best, curve


def apply_b3(edges: list[dict], target_region: int, k: int) -> tuple[list[dict], list[dict]]:
    """Returns (own-region edges, top-k other-region edges)."""
    own = [e for e in edges if e["src_region"] == target_region]
    other = sorted((e for e in edges if e["src_region"] != target_region), key=lambda e: -abs(e["val"]))[:k]
    return own, other


# ---------------------------------------------------------------- B4 ------
def apply_b4(drivers: list[dict], target_region: int, fine_graph: dict) -> float | None:
    """fine_graph: {child_target_region: [edge dicts at 100 regions]}.
    Marks each driver edge with `b4_persists`; returns the persistence score
    (None when the target has no drivers)."""
    fine = []
    for child in (2 * target_region, 2 * target_region + 1):
        fine += fine_graph.get(child, [])
    for e in drivers:
        e["b4_persists"] = any(
            f["src_region"] // 2 == e["src_region"]
            and f["field"] == e["field"]
            and abs(f["lag"] - e["lag"]) <= PERSISTENCE_LAG_TOL
            for f in fine
        )
    if not drivers:
        return None
    return sum(e["b4_persists"] for e in drivers) / len(drivers)


# ------------------------------------------------------------ region prior --
def region_prior(drivers: list[dict]) -> dict:
    """Merge validated (field, lag) edges into a per-source-region prior:
    strength = max |MCI val| over that region's edges (decision 1: the
    splitter works at region level; field/lag detail stays in the graph
    report)."""
    out: dict = {}
    for e in drivers:
        r = out.setdefault(int(e["src_region"]), {"strength": 0.0, "edges": []})
        r["strength"] = max(r["strength"], abs(e["val"]))
        r["edges"].append({"field": e["field"], "lag": e["lag"], "val": e["val"]})
    return out
