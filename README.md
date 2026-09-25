# Causal-MoE for Tropical Intraseasonal Forecasting

---

## 1. The idea, in plain terms

We forecast tropical weather (specifically, a signal called OLR — see the
data section below) a few days ahead, at 50 regions across the Indo-Pacific
monsoon domain. Two things make this project different from a standard
forecasting model:

1. **The model explains itself.** Before forecasting a region's future
   weather, a sub-model looks at other regions that *could* plausibly
   influence it, and tries to pick out which ones are the real physical
   drivers — as opposed to regions that just happen to look correlated by
   coincidence. The forecast is then made using only those selected
   drivers. This is the **causal splitter**.
2. **The model adapts when the physics changes.** Climate isn't static —
   which regions drive which other regions can shift over months or years
   (e.g. during El Niño). Instead of one frozen model, each region has its
   own chain of "expert" models over time; the system watches for signs
   that the old causal relationships have broken down and, when they have,
   retrains or spawns a new expert — while keeping old experts on file in
   case that old pattern comes back later (a recurring season, a repeat
   El Niño). A router can also **blend several archived experts** together
   rather than committing to just one. This is the **Dynamic
   Mixture-of-Experts (MoE)** half.

The name **Causal-MoE** is these two ideas combined: causal structure
driving when and how the expert pool changes, instead of a fixed retraining
schedule.

**Honest one-line summary of where the project stands (2026-09-24, after a
full audit-and-fix pass — see `PROJECT_PLAN.md` for the complete,
chronological record of 23 bugs found and fixed):** the *forecasting* half
works well and is now honestly evaluated on genuinely unseen future data —
it beats a naive "tomorrow = today" baseline at every one of the 50 regions
at a 7-day lead, and beats climatology (a weaker baseline) even at leads
where persistence is unbeatable. The *causal discovery* half genuinely
discriminates real drivers from noise for the `direct` (physically
adjacent) candidate set — a real result this project did not have before
this pass — but that discrimination degrades sharply as the candidate pool
grows (`2hop`, `full`), and a first attempt at fixing that did not work
(see §7.3). Nothing here claims to have found the atmosphere's true causal
graph with certainty — no such ground truth exists for real climate data
(see §3.4) — but the causal-edge scores are no longer indistinguishable
from random, which was this project's central, now-corrected problem.

---

## 2. The data

**Source:** BSISO (Boreal Summer Intraseasonal Oscillation) real-time
monitoring data, supplied by the supervisor. BSISO is in the same family
as the MJO (Madden-Julian Oscillation) — a slow-moving pattern of tropical
convection that drives monsoon rainfall on a ~30-60 day cycle.

**Shape:** a 25×144 latitude/longitude grid (2.5° resolution) over the
Indo-Pacific, one value per day, **1979-01-01 to 2022-12-31** (16,071 days,
zero missing days), for **6 physical fields**:

| field | meaning | how predictable it is (autocorrelation at 10 days) |
|---|---|---|
| sst | sea surface temperature | high (0.36) — easiest |
| h850 | mid-level geopotential height | low (0.01) |
| u200 | upper-level jet wind | low (0.05) |
| pw | column water vapour | low (0.03) |
| u850 | low-level monsoon wind | low (0.01) |
| **olr** | outgoing longwave radiation | **lowest (0.01) — hardest** |

**Why OLR is the forecast target.** OLR is a standard proxy for
convection: low OLR = cold, high cloud tops = active rainfall; high OLR =
clear skies = suppressed rainfall. It's also the *hardest* of the 6 fields
to predict from its own past, which matters because it rules out the
trivial "just predict tomorrow = today" shortcut from looking artificially
good. **Important framing point:** the model never sees rainfall data and
is never trained on rainfall labels — it forecasts OLR, and a forecasted
OLR drop is *interpreted* as rainfall intensifying, because that physical
relationship is well established. The correct description is "OLR-based
rainfall/convection forecasting," not "rainfall prediction."

A second file (BSISO PC1/PC2 index) was investigated as a possible target
but is computed for the whole domain at once, with no per-place breakdown
— unusable as the per-region target this project needs, so it's kept only
as a diagnostic reference.

**Preprocessing already done before we received it:** seasonal climatology
and harmonics removed, then normalized (per grid cell, verified directly
— not a single global mean/std, which would have conflated spatial
pattern with temporal anomaly) — i.e. every field is already an anomaly
series, not raw physical units.

**Scope note (added 2026-09-24, checked empirically, not assumed):** BSISO
is, by definition, a boreal-summer (May-Oct) phenomenon; the model trains
and evaluates on the full 12-month record. A direct check — retraining one
region restricted to May-Oct only and comparing against the year-round
result — found no consistent, meaningful difference in skill or
discrimination. So year-round training is kept, and "BSISO" throughout
this document is shorthand for "the tropical intraseasonal variability
this dataset captures," not a claim that training was season-restricted.

---

## 3. Architecture

### 3.1 What's borrowed vs. what's ours

Two published-method families are combined here. What is ours is the
causal-drift lifecycle wiring connecting them, and the specific bug fixes
that made the causal half's scores mean something:

```
 ┌─────────────────────────────┐        ┌──────────────────────────────┐
 │  BORROWED — DIR-GNN           │        │  BORROWED — DyMoE-style        │
 │  (Wu et al., ICLR 2022)       │        │  Mixture-of-Experts             │
 │                                │        │                                │
 │  Splits a graph into a        │        │  A growing pool of per-place    │
 │  causal part and a            │        │  experts; sparse top-k          │
 │  non-causal part; trains      │        │  cosine-similarity routing      │
 │  for invariance to swapping   │        │  across the pool.                │
 │  the non-causal part.         │        │                                  │
 └───────────────┬───────────────┘        └────────────────┬───────────────┘
                 │  causal edge scores                       │  expert pool
                 │  (which edges are "real")                 │  mechanics
                 v                                            v
     ┌───────────────────────────────────────────────────────────────────┐
     │             OURS — the causal-drift lifecycle                      │
     │                                                                     │
     │  • Turns DIR-GNN's per-edge causal scores into a per-place         │
     │    "causal signature" vector, tracked over time.                   │
     │  • Feeds that signature into a dedicated change-point channel      │
     │    (STARS/Rodionov) running ALONGSIDE a standard error-based       │
     │    channel — a drift signal built from causal structure, not       │
     │    just forecast error.                                            │
     │  • Fuses both channels (OR / AND) into freeze / retrain-in-place /  │
     │    hibernate+spawn decisions.                                      │
     │  • Hibernate + similarity-matched reactivate, so a retired expert  │
     │    can be warm-started if its old regime recurs, instead of        │
     │    deleting on spawn.                                              │
     │  • A k=3 cosine-similarity POOL BLEND across several archived      │
     │    experts (not just the single nearest match) — this is the       │
     │    part with the most robust, cross-place-confirmed evidence       │
     │    behind it (see §7.4).                                           │
     └───────────────────────────────────────────────────────────────────┘
```

### 3.2 From 3,600 grid cells to 50 regions

The raw data is 3,600 grid cells (25×144). The 3,600 cells are grouped
into **50 regions** via k-means clustering on OLR correlation structure
(climatologically similar cells group together automatically, not
hand-drawn boxes). Each region's feature value is the mean of its member
cells (~72 cells on average). 46 of 50 clusters are single spatially
contiguous blobs (verified, not assumed).

```
   3,600 raw grid cells (25 lat x 144 lon)
                 |
      [k-means on OLR correlation, fit on TRAINING SPAN ONLY --
       fixed 2026-09-23, was previously fit on the full record]
                 |
                 v
   50 regions ("places"), each = mean of ~72 member cells
                 |
   [lattice adjacency, longitude wraparound]
                 |
                 v
      50-node graph mesh  <-- this is the "place" the rest
                                of the system operates on
```

**A known, documented limitation of this clustering choice:** grouping
cells by correlation, then later asking "does region A causally drive
region B," carries a theoretical risk that the clustering's own similarity
metric could echo back as a fake "causal" signal. Checked directly: the
strongest real "causal" pair found in this project (region 22's top
source) has raw correlation 0.34 — a moderate, physically ordinary
teleconnection strength, not the near-1.0 value that scenario would
predict. Not exhaustively ruled out across all 50 regions, but the
available evidence argues against it dominating the results.

### 3.3 The mesh (graph structure)

- **Nodes** = the 50 regions.
- **Edges** = physical adjacency between regions (a lattice graph, with
  longitude wraparound), used to define the `direct`/`2hop` candidate sets
  in §3.4 below — kept deliberately independent of how the clusters
  themselves were formed (geography vs. correlation), so the "who can be a
  candidate" question doesn't inherit the clustering step's own bias.
- **Node features** = the 6 physical fields, plus a land/ocean flag (SST
  is undefined over land), at **7 time lags** — today, 1, 2, 3, 5, 7, and
  10 days ago (49 numbers per region per day). Widened from an original
  3-lag set (0/5/10) once a cross-check (PCMCI) found real driving
  relationships at lags the old set couldn't represent at all.

```
one region's node vector, per day
┌───────────────────────────────────────────────────────────┐
│  6 fields (sst, h850, u200, pw, u850, olr)                  │
│    x 7 lags (today, -1, -2, -3, -5, -7, -10 days)             │
│    + is_ocean flag                                             │
│  = 49 values, one node, one day                                │
└───────────────────────────────────────────────────────────┘
```

### 3.4 The causal splitter (DIR-GNN-based)

For one target region, the splitter is offered one of three candidate
pools (§3.6 explains why there are three) and scores every candidate edge
for how causally relevant it is to the target's future OLR. It keeps only
the top-scoring fraction of edges (the **causal set**) and discards the
rest (the **non-causal set**).

**There is no ground truth for real climate causal graphs — this matters
enough to say plainly.** Nobody has ever directly confirmed "region 27
causally drives region 22" as an observed fact; only correlated/lagged
patterns are ever observable. The project therefore checks the mechanism
three different, each individually weaker, ways: (a) does using this edge
improve genuinely out-of-sample forecast accuracy (real ground truth for
*prediction*, not for *causation* — a spurious correlation can help
prediction too); (b) does an independent, differently-built method
(PCMCI) tend to agree on which sources matter (agreement between two
guessers is supporting evidence, not proof); (c) on a *synthetic* dataset
where a causal rule was deliberately injected by us, so the true answer is
actually known, does the splitter recover it (real ground truth, but only
for fake data).

**How it's trained without ever being told the true answer:** the model is
shown the same target region's history at *different points in time*, and
made to swap out the non-causal part of the candidate information for a
different time's non-causal part, while keeping the causal part fixed. If
the causal part really is sufficient to explain the outcome, the
prediction should stay accurate and *stable* no matter which time period's
non-causal background got swapped in. Training rewards both accuracy and
that stability (low variance across swaps).

```
one region, one day
        |
        v
[ Rationale Generator ] --- scores every candidate edge
        |
   splits into:
        |
   +----+----------------+
   |                      |
[causal edges c̃]    [non-causal edges s̃]
   |                      |
   |                [swap for a different
   |                 day's non-causal edges]
   |                      |
   +----------+-----------+
              v
       [ Shared Encoder ]        <-- no longer force-fed the target's
              |                      own raw features unconditionally
      +-------+--------+            (self-bypass bug, fixed 2026-09-23 --
      v                v            this is what was silently making the
[Causal Classifier] [Spurious      swap-variance term always ~0 before)
 (real prediction,   Classifier]
  gradients flow)    (leakage check
      |               only, gradient-
      v               blocked)
Loss = average error across swaps
       + λ · variance across swaps
```

**The bug this project spent this session finding and fixing:** for most
of this project's history, `target_features = x_all[target]` (the
target's own current values, INCLUDING today's OLR) was concatenated into
the expert's input unconditionally, regardless of which edges the
splitter selected — an unconditional shortcut so strong (self-persistence
alone explains ~92-99% of variance at short leads) that the model's
optimal strategy was to ignore the selected edges entirely. Because the
swap-variance term only produces a gradient when the prediction actually
depends on the swapped information, this shortcut also silently zeroed
out the ENTIRE causal-vs-spurious training signal — not just hurt it.
Removing the shortcut (`self_bypass: bool = False`, self is now just
another candidate edge that can be *selected*, not force-fed) is what
took edge-score discrimination std from ≤0.005 (indistinguishable from
uniform noise, every prior version of this project) to a genuinely
discriminating 0.02+ for the `direct` variant.

### 3.5 The expert (forecaster)

A separate small model, `PlaceExpert`, takes the region's own causal edge
set `c̃` from the splitter and forecasts its OLR N days ahead — a
different model from the splitter's own internal classifier, so the
forecasting model and the causal-discovery model can be developed,
frozen, retrained, or replaced independently.

```
causal edge set c̃ (from splitter)
        |
        v
┌────────────────────────────┐
│  PlaceExpert                 │
│  weighted-mean aggregation    │
│  over c̃'s source regions      │
│  -> small MLP -> forecast     │
└────────────────────────────┘
        |
        v
  OLR forecast, N days ahead, for THIS region only
```

### 3.6 Three candidate-pool variants, and a real, unresolved trade-off

The splitter isn't restricted to physically adjacent regions only — three
variants are tried and compared, on purpose:

| variant | candidate pool | typical size |
|---|---|---|
| `direct` | target's immediate physical neighbours (+ self) | ~6 |
| `2hop` | neighbours, plus neighbours-of-neighbours (+ self) | ~15-20 |
| `full` | all other 49 regions, ignoring geography entirely (+ self) | 50 |

**The honest finding (Bug 18, confirmed at the full 50-region mesh-wide
level):** forecast accuracy stays essentially flat across all three (~40%
skill vs. persistence regardless of pool size) — bigger pools don't hurt
accuracy. But edge-score discrimination collapses sharply as the pool
grows:

```
        n_candidates   discrimination-gate pass rate (out of 50 regions)
direct       ~6              27/50  (54%)
2hop        ~15-20             8/50  (16%)
full          50                0/50  (0%)
```

**Mechanism (why, not just what):** the splitter has one shared encoder
that must score every candidate in the pool in a single forward pass. A
small, fixed, consistent pool (like `direct`'s physical neighbours,
identical in shape every training run) is a much easier discrimination
problem than a large pool of many similar-looking competing hypotheses.
This was confirmed independently on a synthetic harness with a known true
answer (discrimination present at 5 candidate nodes, gone from 15 upward,
never recovering through 50) before being confirmed on the real mesh.

**An attempted fix (2026-09-24) did not work, and this is reported
honestly rather than rounded up to "solved":** correlation-based capping
— shrinking `full`'s 50 candidates down to the top-8 most-correlated
sources, matching `direct`'s own pool size — was tried on real data. It
did NOT recover `direct`-level discrimination (capped std 0.0102 vs.
`direct`'s 0.0203, even below the uncapped `full`'s 0.0137). A real,
physically-coherent, non-arbitrary mid-sized pool (`2hop`, ~15-20
candidates) also fails to clear the gate, which weakens "it's about pool
composition, not just count" as an explanation too. **Practical
consequence: use `direct` for any claim about "which regions causally
drive this one" — it is the only variant with reliable discrimination.
`2hop`/`full` are accuracy-only ablations, not causal-discovery results,
until this gap is closed.** The capping mechanism was kept in the
codebase (`--max-candidates`, opt-in, zero effect when unset) as a tested
starting point for whoever investigates this further — see
`PROJECT_PLAN.md`, Bug 18, for the full experimental trail.

### 3.7 Dynamic Mixture-of-Experts: drift detection + hibernate/reactivate/blend

Each of the 50 regions runs its own independent chain of experts over
time, like a version history, with an added pool-blending layer:

```
Region A:  Expert A-1 --[drift]--> Expert A-2 --[drift]--> Expert A-3 (ACTIVE)
                |                        |
          [hibernate]              [hibernate]
                v                        v
             Archive A  <---- searched on every future drift event,
                                and BLENDED (top-k=3, cosine similarity)
                                rather than only single-best-matched
```

**Detecting drift — two independent channels, fused into one decision:**

```
┌──────────────────────────┐      ┌──────────────────────────────┐
│ Channel 1 — error-based    │      │ Channel 2 — causal-structure   │
│                              │      │ based (STARS/Rodionov applied   │
│ watches the expert's        │      │ to the splitter's own edge      │
│ rolling forecast error      │      │ scores, turned into a per-      │
│                              │      │ region "causal signature")      │
└──────────────┬───────────────┘      └───────────────┬──────────────┘
               │                                        │
               └───────────────┬────────────────────────┘
                                v
                    ┌───────────────────────┐
                    │  Fusion gate             │
                    │  OR  --  fast, noisier   │
                    │  AND --  slow, cleaner    │
                    └───────────┬───────────┘
                                v
              ┌─────────────────────────────────┐
              │  3-tier response                    │
              │  Tier 0: freeze                     │
              │  Tier 1: retrain expert in place     │
              │  Tier 2: hibernate + spawn/reactivate │
              └─────────────────────────────────┘
```

**Two DIFFERENT claims that must not be conflated (a real, checked
distinction, not a simplification for this doc):**

```
   drift event fires for Region A
              |
              v
   compute Region A's current causal signature
              |
              v
   compare (cosine similarity) against EVERY archived
   signature in the whole system
              |
       +------+---------------------+
       |                            |
  Variant A (k=1)              Variant B (k=3)
  single best match             BLEND of the top-3
  reactivated as warm start     matches, weighted by similarity
```

- **Variant B (the k=3 pool blend) beats Variant A (single best match) on
  both forecast accuracy AND average forgetting, at all 3 places tested —
  a robust, cross-place-confirmed result.** This is the strongest MoE
  finding in the project.
- **A separate, easily-confused claim — does single-best-match
  warm-starting beat spawning fresh — is genuinely place-dependent**
  (wins at 1 of 3 places tested on the forgetting metric, loses at 2).
  **Do not state "warm-starting always helps forgetting"** — the pool
  blend is the part with robust support, not single-match warm-starting.

---

## 4. Code map

All code lives under `DIR_GNN + DyMoE/code/`.

```
code/
├── causal_moe/                  the actual library
│   ├── data/
│   │   ├── raw.py               loads the 6 fields + ocean mask
│   │   ├── clustering.py        k-means -> 50 regions (fit on train span only)
│   │   ├── mesh.py              lattice adjacency (raw cells + regions)
│   │   ├── windows.py           49-value lagged features, OLR-N-days target
│   │   ├── splits.py            chronological split (leak-free, B22) +
│   │   │                        El Niño / gradual-drift windows
│   │   ├── candidate_edges.py   direct/2hop/full variants + candidate-pool
│   │   │                        capping (B18 fix attempt)
│   │   ├── causaldynamics.py    loads validation data w/ known causal graphs
│   │   └── semisynthetic.py     real features + a hand-injected causal rule
│   │                            (synthetic ground truth, see §3.4)
│   ├── splitter/
│   │   ├── dirgnn.py            the causal splitter itself (self-bypass fixed)
│   │   ├── cia.py               attempted fix #1 (re-verified post-bugfix:
│   │   │                        worse than plain DIR-GNN, kept as ablation)
│   │   └── gsina.py             attempted fix #2 (same verdict as CIA)
│   ├── experts/
│   │   ├── expert.py            the forecaster (self-bypass fixed)
│   │   └── pool.py              Mixture-of-Experts: top-k cosine-similarity
│   │                            router/blend (Variant A vs Variant B, §3.7)
│   ├── drift/
│   │   ├── rodionov.py          STARS/Rodionov change-point test
│   │   ├── channels.py          both drift channels + fusion gate
│   │   ├── archive.py           hibernate/reactivate store
│   │   └── forgetting.py        average-forgetting / backward-transfer metric
│   └── baselines/simple.py      persistence / plain-GNN / random-subset ablations
├── scripts/                      one runnable script per experiment
├── tests/                        144 tests, all passing
├── cache/                        preprocessed mesh caches (not in git --
│                                  regenerable via build_mesh_cache.py; several
│                                  lead times: lead1, lead1_v2, lead7)
└── results/                      every experiment's result as JSON, plus
    └── phase9_figures/            5 generated PNG charts of the current findings
```

**Reference implementations used** (real published code, not reimplemented
from scratch where a maintained version exists): DIR-GNN splitter
mechanics from [Wuyxin/DIR-GNN](https://github.com/Wuyxin/DIR-GNN);
PCMCI cross-check from
[tigramite](https://github.com/jakobrunge/tigramite); validation data from
[kausable/CausalDynamics](https://github.com/kausable/CausalDynamics).

---

## 5. Testing

144 automated tests (`pytest tests -q`, all passing), covering every module
above. A standing project practice: every real bug found is locked in with
a dedicated regression test, not just fixed in place — e.g. the self-bypass
fix (§3.4) has a bit-for-bit verification test against the old, unfixed
behaviour; the chronological-split boundary-leak fix (Bug 22) has a test
asserting the exact number of dropped boundary samples; the `load_cache`
regression found while building an ablation script (Bug 22c) has two tests
covering both a normal cache and a legacy cache missing the newer field.

---

## 6. Key results — honestly reported, in order of how much to trust them

### 6.1 Forecasting accuracy — the most solid result

All 50 regions, out-of-sample test period (2013-2022, chosen because it
deliberately contains a documented gradual climate-drift window — an
out-of-distribution stress test, not just a random holdout), honest
chronological train/val/test split (no split-boundary leak, Bug 22 fixed):

| lead | beats persistence | skill vs persistence (mean) | R² vs climatology (mean) | discrimination gate pass rate |
|---|---|---|---|---|
| 7 days | **50/50 (100%)** | +0.401 | −0.014 (see note below) | 27/50 (54%), `direct` variant |
| 1 day | 0/50 (0%) — expected | −1.622 | +0.729 | 37/50 (74%) — better than lead-7 |

**Read this table carefully — two real nuances, not omitted:**
- At lead-1, self-persistence ("tomorrow = today") is such a strong
  baseline (~93-99% of variance) that nothing beats it — this is expected,
  not a failure, and the model still clearly beats the weaker climatology
  baseline there.
- At lead-7, "beats persistence" and "beats climatology" are DIFFERENT
  claims, and only the first one holds on average — R² vs climatology is
  slightly negative (31-39 of 50 regions individually score worse than
  predicting the training-period's mean value). This was found during a
  later audit pass (checked directly from the result files, not assumed)
  and had been omitted from an earlier draft of this project's reporting
  — it does not invalidate the "beats persistence" claim, which is a
  separate, true, honestly-computed result, but the two must not be
  conflated.
- Discrimination quality (whether the selected edges are trustworthy, not
  just whether the forecast is accurate) and forecast accuracy are
  DIFFERENT axes that do not move together — lead-1 has worse accuracy
  but BETTER discrimination than lead-7. This is the clearest evidence in
  the whole project that "accurate forecast" and "trustworthy causal
  graph" are separate claims.

### 6.2 Causal discrimination quality — real, but pool-size-dependent

See §3.6 above for the full direct/2hop/full breakdown and the honest
report of the failed fix attempt. Bottom line: `direct` is the only
variant with reliable discrimination; treat `2hop`/`full` as
accuracy-only.

### 6.3 CIA and GSINA — both re-tested post-bugfix, both worse than plain

Both were originally built to solve a symptom (near-uniform edge scores)
that turned out to BE the self-bypass bug in §3.4, not an inherent
limitation of this training objective. Re-tested after the fix, on
identical config (region 22, `direct`, lead-7):

| method | expert MSE | discrimination std |
|---|---|---|
| plain DIR-GNN (post-bugfix) | **0.3207** | **0.0203 (clears gate)** |
| CIA | 0.3324 (worse) | 0.0026 (far below gate) |
| GSINA | 0.3364 (worse) | 0.0087 (far below gate) |

Both are kept in the codebase as tested, working, but non-beneficial
ablations — not deleted, not recommended for use.

### 6.4 Drift detection — a real, measured speed-vs-reliability trade-off

Two deliberately chosen test windows: an abrupt shift (the 1997-98 El
Niño, evaluated with a false-alarm ground truth that correctly accounts
for OTHER major El Niño events too — a real bug found and fixed, Bug 13)
and a gradual shift (2013-2022).

| detector | catches abrupt shift? | catches gradual shift? |
|---|---|---|
| Channel 1 (error-based) | missed | yes, but slow |
| Channel 2 (causal structure) | yes, fast | yes |
| Fused, OR | yes, fast | yes |
| Fused, AND | missed | yes, slow |

Reported as a genuine trade-off (OR reacts fast but false-alarms more
often; AND is more trustworthy but slower and can miss fast shifts), not
resolved to one "correct" rule.

### 6.5 Mixture-of-Experts — the most robust MoE finding

See §3.7 above. Variant B (k=3 pool blend) beats Variant A (single-best
warm start) on both MSE and forgetting, at all 3 places tested — the
strongest, most consistently-replicated finding among the MoE results.

---

## 7. Honest state of the project, and what's genuinely still open

**What's solid, tested, and demonstrated end to end:** the full pipeline —
mesh → splitter → expert → drift detection → hibernate/reactivate/blend →
50-region scale-up — runs, is tested (144 tests), and produces real,
honestly out-of-sample forecast skill beating persistence at every region
at lead-7, with the causal splitter now genuinely discriminating real
signal from noise (not indistinguishable from random, which was every
prior version of this project's actual state before this session's fixes).

**What's genuinely still open, not glossed over:**

1. **Bug 18 — discrimination collapses as candidate-pool size grows, and
   a first fix attempt did not work** (§3.6). This is the single biggest
   open item. Whoever picks this up next has a tested `--max-candidates`
   tool to start from, plus a clear finding of what DIDN'T work
   (naive size-matching), narrowing the search.
2. **Bug 23 — the clustering-forms-regions-then-tests-causation
   theoretical concern (§3.2)** has one supporting data point (a moderate,
   not suspiciously high, correlation) but was not exhaustively checked
   across all 50 regions.
3. **The R² vs climatology gap at lead-7 (§6.1)** should be stated
   plainly in any future writeup, not silently dropped from a headline
   table the way an earlier draft of this project's own reporting did.
4. **Published-method baselines** (other researchers' MoE-for-graphs
   methods, listed in the original evaluation spec) were never run —
   only internal ablations. Real remaining work if a like-for-like
   comparison against other published work is needed.

**Where to find the full, unabridged history:** `PROJECT_PLAN.md` (project
root) is the complete, dated, append-only log of every bug found (23
total, numbered B1-B23), every experiment run, and every result — this
document is a current-state summary of it, not a replacement for it.
`code/README.md` carries the most detailed current numbers table.
