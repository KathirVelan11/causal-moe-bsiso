"""Mid-scale semi-synthetic check (§5.2, §9 step 3): a real contiguous raw
patch, clustered with the same method planned for the full mesh, real input
features, but a hand-written injected causal rule replacing the target so
the true answer is known by construction.

Patch -- REVISED 2026-09-19 after the first 5-cluster run: 13x21 raw cells,
lat 30N..0 (raw lat index 0..12 inclusive), lon 70E..120E (raw lon index
28..48 inclusive) -- 273 raw cells, ~53% ocean, spans the Arabian
Sea/western coast through the Bay of Bengal to Indochina. Original patch
(7x7, Bay of Bengal only, lat 20N..5N idx 4..10, lon 82.5E..97.5E idx
33..39, 49 cells) is kept as ORIGINAL_PATCH_* below since it's still a
valid real patch -- it was replaced only because it's too spatially small
for 5 CLUSTERS specifically to contain a confound-free triple (see
pick_abc_avoiding_confounds docstring for the full root-cause writeup):
exhaustive check over all 60 possible (A,B,C) triples on the 7x7 patch at
5 clusters found none with both confound checks under 0.3 (best was
max_other_corr=0.606, a_self_corr=0.431) -- real Bay-of-Bengal convection
is too spatially coherent at that scale/cluster-count for any 3
sub-regions to be genuinely independent. The wider patch gives 5 cluster
means enough spatial diversity for a clean triple to exist (best found:
max_other_corr=0.104, a_self_corr=0.082).

Injected rule design (confirmed with user, 2026-09-19, after directly
checking DIR-GNN's own Spurious-Motif generation code
(Wuyxin/DIR-GNN/spmotif_gen/spmotif.ipynb) -- their labels are a fully
DETERMINISTIC function of the causal part only, zero label noise;
spuriousness comes from correlating causal/non-causal parts during
sampling, never from noise added to the label itself):
  y_A[t+1] = w1 * olr_B[t] + w2 * olr_C[t]   (deterministic, no added noise
             -- real physical noise already present in olr_B/olr_C
             themselves satisfies §5.2's "real noise" requirement)
  y_X[t+1] = olr_X[t]   for every other cluster X (pure self-persistence,
             no cross-cluster causal edge -- so the ONLY true off-diagonal
             causal edges in the whole candidate set are B->A and C->A)

Confound check (added per user's concern 2026-09-19: a linear rule is only
a fair test if B and C aren't trivially replaceable by some OTHER cluster
D that's incidentally almost-as-correlated with B or C from real climate
physics -- nonlinearity would not fix that, only picking A/B/C so no such
near-duplicate exists does). `pick_abc_avoiding_confounds` below checks
this explicitly before injection.

A nonlinear variant (adds a B*C interaction term) is also provided
(`inject_causal_rule(..., nonlinear=True)`) as a second run at the same
cluster size, per user's "try both" decision 2026-09-19.
"""

from __future__ import annotations

import numpy as np

from causal_moe.data.clustering import ClusterAssignment, compute_cluster_assignment
from causal_moe.data.mesh import LatticeMesh, build_cluster_mesh, build_raw_lattice_mesh
from causal_moe.data.raw import FIELDS, BSISORawData
from causal_moe.data.windows import WindowedDataset

# Arabian Sea -> Bay of Bengal -> Indochina 13x21 patch, raw grid indices
# into the full 25x144 lattice (revised 2026-09-19, see module docstring):
# lat 30N..0, lon 70E..120E. Default / smallest patch, used for n_clusters=5.
PATCH_LAT_SLICE = slice(0, 13)  # 13 rows
PATCH_LON_SLICE = slice(28, 49)  # 21 cols

# Original 7x7 Bay-of-Bengal-only patch (2026-09-19) -- kept for reference;
# too spatially small even for 5 clusters (see module docstring), not used
# by default anywhere.
ORIGINAL_PATCH_LAT_SLICE = slice(4, 11)  # lat 20N..5N
ORIGINAL_PATCH_LON_SLICE = slice(33, 40)  # lon 82.5E..97.5E

# Per-cluster-count patch sizes (confirmed with user 2026-09-19: "grow patch
# per size, keep going" -- finer clustering = smaller regions = more
# neighbor correlation, so the confound check needs more raw spatial extent
# to keep finding a clean triple as n_clusters grows). Each entry verified
# BEFORE use by exhaustively checking at least one confound-free triple
# exists at that (patch, n_clusters) combination -- see
# Causal_MoE_Architecture.md §10 step 3 for the verification numbers.
#   5  clusters -> 13x21 (lat 30N..0,     lon 70E..120E)  -- 273 cells
#   15 clusters -> 18x37 (lat 30N..-15N,  lon 50E..140E)  -- 666 cells
PATCH_SIZE_BY_N_CLUSTERS: dict[int, tuple[slice, slice]] = {
    5: (slice(0, 13), slice(28, 49)),
    15: (slice(0, 18), slice(20, 57)),
}


def patch_slices_for_n_clusters(n_clusters: int) -> tuple[slice, slice]:
    """Looks up the verified (lat_slice, lon_slice) for a given cluster
    count from PATCH_SIZE_BY_N_CLUSTERS. Raises KeyError with a clear
    message if that size hasn't been set up/verified yet -- deliberately
    no silent fallback, since an unverified patch size is exactly how the
    5-cluster/7x7 confound bug happened the first time."""
    if n_clusters not in PATCH_SIZE_BY_N_CLUSTERS:
        raise KeyError(
            f"no verified patch size for n_clusters={n_clusters} -- add an entry to "
            "PATCH_SIZE_BY_N_CLUSTERS after checking a confound-free (A,B,C) triple "
            "exists there (see module docstring / pick_abc_avoiding_confounds)"
        )
    return PATCH_SIZE_BY_N_CLUSTERS[n_clusters]


DEFAULT_RULE_WEIGHTS = (0.6, 0.4)  # (w_B, w_C), sums to 1 -- matches olr's own scale


def extract_patch(
    raw: BSISORawData,
    lat_slice: slice = PATCH_LAT_SLICE,
    lon_slice: slice = PATCH_LON_SLICE,
) -> BSISORawData:
    """Slices a contiguous raw-cell patch out of the full BSISO field (§5.2
    step 1: "a real contiguous patch of raw grid cells"). Returns a
    BSISORawData with the same shape/field conventions as the full-grid
    loader (just smaller n_lat/n_lon) so every downstream function
    (clustering, windowing, mesh-building) works unchanged -- no duplicate
    patch-specific logic needed."""
    return BSISORawData(
        fields=raw.fields[:, :, lat_slice, lon_slice],
        ocean_mask=raw.ocean_mask[lat_slice, lon_slice],
        time=raw.time,
        lat=raw.lat[lat_slice],
        lon=raw.lon[lon_slice],
    )


def cluster_patch(patch: BSISORawData, n_clusters: int, random_state: int = 0) -> ClusterAssignment:
    """Clusters a patch's raw cells using the SAME method planned for the
    full mesh (§5.2 step 2: k-means on OLR anomaly correlation structure,
    causal_moe.data.clustering.compute_cluster_assignment) -- this tests the
    clustering step itself, not just the splitter."""
    olr_idx = FIELDS.index("olr")
    olr_patch = patch.fields[olr_idx]  # (T, n_lat_patch, n_lon_patch)
    return compute_cluster_assignment(olr_patch, n_clusters=n_clusters, random_state=random_state)


def build_patch_mesh(patch: BSISORawData, assignment: ClusterAssignment) -> LatticeMesh:
    """Cluster-level adjacency for a patch, same derivation as the full mesh
    (lattice adjacency -> cluster adjacency, causal_moe.data.mesh)."""
    raw_mesh = build_raw_lattice_mesh(patch.lat, patch.lon)
    return build_cluster_mesh(raw_mesh, assignment)


def pick_abc_avoiding_confounds(
    cluster_olr: np.ndarray,
    max_other_corr: float = 0.3,
    rng: np.random.Generator | None = None,
    max_attempts: int = 200,
) -> tuple[int, int, int]:
    """Picks (A, B, C) cluster ids for the injected rule (§5.2 step 4), with
    an explicit confound check (added per user's concern 2026-09-19, TIGHTENED
    2026-09-19 after the first 5-cluster/7x7-patch run failed at chance:
    root-caused to this check being too loose -- see below).

    Two confound sources, BOTH checked:
      1. B or C near-duplicated by some OTHER cluster D (original check) --
         if D is almost as correlated with B or C as they are with each
         other, a model could match y_A using D instead of B/C.
      2. A itself already naturally correlated with B or C BEFORE injection
         (added 2026-09-19 -- the original check missed this entirely). If
         A's own real OLR already resembles B/C's real OLR (common in
         spatially coherent real climate data -- adjacent/nearby regions'
         convection moves together), then after injection the model cannot
         tell "A depends on B/C" from "A's own persistence looks like B/C
         anyway", even with zero other-cluster confounds.
    A nonlinear rule would NOT fix either source (neither changes which
    clusters correlate with which), so this is a selection check, not a
    rule-shape choice.

    First 5-cluster run (7x7 Bay-of-Bengal patch, max_other_corr=0.8):
    exhaustive check over all 60 possible (A,B,C) triples found NONE
    confound-free -- every triple had max_other_corr or a_self_corr in
    0.4-0.7. Root cause: the whole 7x7 patch is too spatially coherent at
    only 5 clusters (real convection correlates smoothly over that small an
    area) for any hand-written rule to be cleanly separable, not a
    threshold-tuning issue. Fix used: grow the raw patch (more spatial
    diversity among the 5 cluster means) rather than loosen the check
    further -- confirmed with user 2026-09-19.

    cluster_olr: (T, n_clusters) float32 -- pooled cluster-level OLR series
        (same-day, used only to screen candidates -- the actual injected
        rule below uses today's B/C to predict tomorrow's A).
    max_other_corr: reject a triple if any OTHER cluster D has
        |corr(D, B)| or |corr(D, C)| >= this threshold, OR if A's own
        |corr(A, B)| or |corr(A, C)| >= this threshold (i.e. D -- or A
        itself -- would be almost as good a "proxy" for B or C as B/C are
        for the rule). Tightened default 0.3 (was 0.8) -- 0.8 let through
        triples with real correlations of 0.6-0.7, nowhere near "distinct".

    Returns (A, B, C) cluster ids, all distinct. Raises ValueError if no
    confound-free triple is found within max_attempts random draws.
    """
    rng = rng if rng is not None else np.random.default_rng(0)
    n_clusters = cluster_olr.shape[1]
    if n_clusters < 3:
        raise ValueError(f"need >= 3 clusters to pick A, B, C; got {n_clusters}")

    corr = np.corrcoef(cluster_olr.T)  # (n_clusters, n_clusters)
    corr = np.nan_to_num(corr, nan=0.0)

    all_ids = np.arange(n_clusters)
    for _ in range(max_attempts):
        a, b, c = rng.choice(all_ids, size=3, replace=False)
        if abs(corr[a, b]) >= max_other_corr or abs(corr[a, c]) >= max_other_corr:
            continue
        others = [d for d in all_ids if d not in (a, b, c)]
        ok = True
        for d in others:
            if abs(corr[d, b]) >= max_other_corr or abs(corr[d, c]) >= max_other_corr:
                ok = False
                break
        if ok:
            return int(a), int(b), int(c)

    raise ValueError(
        f"no confound-free (A, B, C) triple found in {max_attempts} attempts "
        f"(max_other_corr={max_other_corr}); every candidate had a near-duplicate "
        "proxy among the other clusters, or A itself already correlated with B/C -- "
        "try a bigger/more spatially diverse patch, or relax max_other_corr"
    )


def inject_causal_rule(
    ds: WindowedDataset,
    a: int,
    b: int,
    c: int,
    weights: tuple[float, float] = DEFAULT_RULE_WEIGHTS,
    nonlinear: bool = False,
    interaction_weight: float = 0.2,
) -> tuple[WindowedDataset, np.ndarray]:
    """Replaces targets with a hand-written rule (§5.2 step 4), keeping real
    input features untouched. True answer known by construction:
      - cluster A's target := w_B * olr_B[today] + w_C * olr_C[today]
        (+ interaction_weight * olr_B[today] * olr_C[today] if nonlinear)
      - every other cluster X's target := olr_X[today]  (self-persistence,
        no cross-cluster causal edge)

    "today's OLR" is read straight back out of ds.features (lag-0 block,
    channel = olr) rather than re-loading raw data, so the injected target
    is built from the SAME already-lagged/pooled feature values the model
    will see as input (§5.2: "keep the real input features ... only the
    target changes").

    Returns (new WindowedDataset with injected targets, true_edge_index)
    where true_edge_index is (2, n_true_edges) int64 -- the ground-truth
    causal edges for this synthetic mesh: (B,A) and (C,A) if nonlinear=False,
    same two edges if nonlinear=True (the interaction term doesn't add a
    third causal SOURCE, both B and C are already causal parents of A), plus
    (X,X) self-loops for every other cluster (self-persistence = each node
    causally depends on its own past).
    """
    n_clusters = ds.features.shape[1]
    if not (0 <= a < n_clusters and 0 <= b < n_clusters and 0 <= c < n_clusters):
        raise ValueError(f"a={a}, b={b}, c={c} must all be in [0, {n_clusters})")
    if len({a, b, c}) != 3:
        raise ValueError(f"a, b, c must be distinct, got a={a}, b={b}, c={c}")

    olr_channel = FIELDS.index("olr")  # lag-0 block starts at offset 0, channel order matches FIELDS
    today_olr = ds.features[:, :, olr_channel]  # (n_samples, n_clusters) -- lag0 block, olr channel

    w_b, w_c = weights
    new_targets = today_olr.copy()  # self-persistence default for every cluster
    rule_value = w_b * today_olr[:, b] + w_c * today_olr[:, c]
    if nonlinear:
        rule_value = rule_value + interaction_weight * today_olr[:, b] * today_olr[:, c]
    new_targets[:, a] = rule_value

    new_ds = WindowedDataset(
        features=ds.features,
        targets=new_targets.astype(np.float32),
        sample_time_index=ds.sample_time_index,
        lead_time_days=ds.lead_time_days,
        lags_days=ds.lags_days,
        field_names=ds.field_names,
    )

    src = [b, c] + [x for x in range(n_clusters) if x != a]
    dst = [a, a] + [x for x in range(n_clusters) if x != a]
    true_edge_index = np.array([src, dst], dtype=np.int64)

    return new_ds, true_edge_index


def make_fully_connected_edge_index(n_nodes: int) -> np.ndarray:
    """Candidate edge set for the splitter (matches
    scripts/train_splitter_causaldynamics.py's convention): every possible
    node pair including self-loops (n^2 pairs) -- the injected rule's true
    self-persistence edges require self-loops to be a candidate, so no
    physical-adjacency pre-filter is used here either, same as step 2."""
    src, dst = np.meshgrid(np.arange(n_nodes), np.arange(n_nodes), indexing="ij")
    return np.stack([src.reshape(-1), dst.reshape(-1)], axis=0).astype(np.int64)
