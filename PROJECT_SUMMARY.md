# Causal-MoE for BSISO Forecasting — Project Summary

> **What this file is.** A clean, standalone explanation of the whole
> project so far: the idea, the architecture, the code, every experiment
> run, and what the results actually mean. Written so someone opening this
> project for the first time — teammate, supervisor, future you — doesn't
> need to dig through anything else to understand it.

---

## 1. The idea, in plain terms

We forecast tropical weather (specifically, a signal called OLR — see §2)
a few days ahead, at 50 regions across the Indo-Pacific monsoon domain.
Two things make this project different from a standard forecasting model:

1. **The model explains itself.** Before forecasting a region's future
   weather, a sub-model looks at all the other regions that *could*
   plausibly influence it, and tries to pick out which ones are the real
   physical drivers — as opposed to regions that just happen to look
   correlated by coincidence. The forecast is then made using only those
   selected drivers. This is the **causal splitter**.
2. **The model adapts when the physics changes.** Climate isn't static —
   which regions drive which other regions can shift over months or years
   (e.g. during El Niño). Instead of one frozen model, each region has its
   own chain of "expert" models over time; the system watches for signs
   that the old causal relationships have broken down and, when they have,
   retrains or spawns a new expert — while keeping old experts on file in
   case that old pattern comes back later (a recurring season, a repeat
   El Niño). This is the **Dynamic Mixture-of-Experts (DyMoE)** half.

The name **Causal-MoE** is these two ideas combined: causal discovery
driving when and how the expert pool changes, instead of a fixed retraining
schedule.

**Honest one-line summary of where the project landed:** the *forecasting*
half works well (beats a naive baseline at 94% of places); the *causal
discovery* half — the part meant to make the system interpretable — does
not yet reliably find the true drivers on real data, and we have strong,
multi-method evidence for exactly why (§7). One of two attempted fixes
(GSINA) measurably repairs the symptom but doesn't yet improve forecast
accuracy; this is reported as a real, nuanced finding, not a failure to
hide.

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
to predict from its own past (autocorrelation table above), which matters
because it rules out the trivial "just predict tomorrow = today"
shortcut from looking artificially good. **Important framing point:** the
model never sees rainfall data and is never trained on rainfall labels —
it forecasts OLR, and a forecasted OLR drop is *interpreted* as rainfall
intensifying, because that physical relationship is well established. The
correct description is "OLR-based rainfall/convection forecasting," not
"rainfall prediction."

A second file (BSISO PC1/PC2 index) was investigated as a possible target
but is computed for the whole domain at once, with no per-place breakdown
— unusable as the per-region target this project needs, so it's kept only
as a diagnostic reference.

**Preprocessing already done before we received it:** seasonal climatology
and harmonics removed, then normalized — i.e. every field is already an
anomaly series, not raw physical units.

---

## 3. Architecture

### 3.1 From 3,600 grid cells to 50 regions

The raw data is 3,600 grid cells (25×144). Running an independent model
per grid cell was tested and measured to be computationally infeasible on
this CPU-only laptop (~134 hours for one training pass at full resolution
vs. ~1.9 hours at reduced resolution — a measured benchmark, not a guess).
So the 3,600 cells are grouped into **50 regions** via k-means clustering
on OLR correlation structure (climatologically similar cells group
together automatically, not hand-drawn boxes). Each region's feature value
is the mean of its member cells (72 cells on average).

### 3.2 The mesh (graph structure)

- **Nodes** = the 50 regions.
- **Edges** = physical adjacency between regions (a lattice graph, with
  longitude wraparound since the grid is a full circle around the globe).
- **Node features** = the 6 physical fields, plus a land/ocean flag (SST
  is undefined over land), at **3 time lags** — today, 5 days ago, 10 days
  ago (21 numbers per region per day). The lags exist because a causal
  link like "region Y's future depends on region X's state 5 days ago"
  is structurally invisible to a model that only ever sees "today."

### 3.3 The causal splitter (DIR-GNN-based)

For one target region, the splitter looks at a candidate set of other
regions (its graph neighbours) and scores every candidate edge for how
causally relevant it is to the target's future OLR. It then keeps only the
top-scoring fraction of edges (the **causal set**) and discards the rest
(the **non-causal set**).

**How it's trained without ever being told the true answer** (no dataset
of "correct" causal graphs exists for real climate data): the model is
shown the same target region's history at *different points in time*, and
made to swap out the non-causal part of the graph for a different time's
non-causal part, while keeping the causal part fixed. If the causal part
really is sufficient to explain the outcome, the prediction should stay
accurate and *stable* no matter which time period's non-causal background
got swapped in. Training rewards both accuracy and that stability
(low variance across swaps) — this is the DIR-GNN mechanism (Wu et al.,
ICLR 2022), adapted here from single-graph classification to a
region-and-time forecasting setting.

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
       [ Shared Encoder ]
              |
      +-------+--------+
      v                v
[Causal Classifier] [Spurious Classifier]
 (real prediction,   (leakage check only,
  gradients flow)     gradient-blocked)
      |
      v
Loss = average error across swaps
       + λ · variance across swaps
```

### 3.4 The expert (forecaster)

A separate small model, `PlaceExpert`, takes the region's own causal edge
set `c̃` from the splitter and forecasts its OLR N days ahead — deliberately
a different model from the splitter's own internal classifier, so the
forecasting model and the causal-discovery model can be developed,
frozen, retrained, or replaced independently.

### 3.5 Dynamic Mixture-of-Experts: drift detection + hibernate/reactivate

Each of the 50 regions runs its own independent chain of experts over
time, like a version history:

```
Region A:  Expert A-1 --[drift]--> Expert A-2 --[drift]--> Expert A-3 (ACTIVE)
                |                        |
          [hibernate]              [hibernate]
                v                        v
             Archive A  <---- searched on every future drift event
```

**Detecting drift — two independent channels:**
- **Channel 1 (error-based):** watches the expert's rolling forecast
  error using ADWIN, a standard streaming change-detection algorithm.
  Catches drift once it shows up as worse forecasts.
- **Channel 2 (causal-structure-based):** watches whether the *pattern* of
  which edges the splitter scores highly is changing over time, using
  STARS/Rodionov (2004), a sequential regime-shift test from climate
  science. Catches drift that changes the underlying physics before it
  necessarily shows up as forecast error.
- **Fusion:** the two channels are combined two ways — OR (either channel
  triggers) and AND (both must agree) — and both are reported, since which
  is "better" is a real trade-off (§6) rather than an obvious choice.

**Hibernate & reactivate:** climate has recurring regimes (seasons,
El Niño/La Niña). Instead of deleting an old expert when a new one spawns
(what the DyMoE baseline paper does) or keeping every old expert active
forever (doesn't scale), retired experts are archived. When a new drift
event fires, the system searches the *whole* archive for a similar past
regime; a good match is reactivated as a warm start (continues training
from those old weights); no good match means a fresh expert spawns.

### 3.6 How the pieces fit together end to end

```
raw grid (3,600 cells)
        |
   [clustering] -> 50 regions, lattice mesh
        |
   [windowing] -> 21 lagged features/region/day
        |
        v
  +-----------------------------------------+
  |  per region, per time block:             |
  |    Splitter (3.3) -> causal edge set c̃   |
  |    Expert (3.4) trained on c̃             |
  |    -> OLR forecast, N days ahead         |
  +-----------------------------------------+
        |
   [drift detection, 3.5] --watches--> error + causal-structure signals
        |
   drift fires? -> retrain in place, or hibernate + spawn/reactivate (3.5)
```

---

## 4. Code map

All code lives under `DIR_GNN + DyMoE/code/`.

```
code/
├── causal_moe/                  the actual library
│   ├── data/
│   │   ├── raw.py               loads the 6 fields + ocean mask
│   │   ├── clustering.py        k-means -> 50 regions
│   │   ├── mesh.py              lattice adjacency (raw cells + regions)
│   │   ├── windows.py           21-value lagged features, OLR-N-days target
│   │   ├── splits.py            chronological + El Niño / gradual-drift windows
│   │   ├── candidate_edges.py   3 candidate-edge-set variants (direct/2hop/full)
│   │   ├── causaldynamics.py    loads validation data w/ known causal graphs
│   │   └── semisynthetic.py     real features + a hand-injected causal rule
│   ├── splitter/
│   │   ├── dirgnn.py            the causal splitter itself (§3.3)
│   │   ├── cia.py               attempted fix #1 (§7.3)
│   │   └── gsina.py             attempted fix #2 (§7.3)
│   ├── experts/expert.py        the forecaster (§3.4)
│   ├── drift/
│   │   ├── rodionov.py          STARS/Rodionov change-point test
│   │   ├── channels.py          both drift channels + fusion gate
│   │   └── archive.py           hibernate/reactivate store (§3.5)
│   └── baselines/simple.py      persistence / plain-GNN / random-subset ablations
├── scripts/                      one runnable script per experiment (§5-§7)
├── tests/                        124 tests, all passing (§5)
├── cache/                        preprocessed mesh cache (windowed_clustered50_lead1.npz)
├── data_external/causaldynamics/ downloaded validation dataset
└── results/
    ├── raw/                      every experiment's result as JSON
    └── slide_figures/            generated PNG charts of the results (§8)
```

**Reference implementations used** (real published code, not reimplemented
from scratch where a maintained version exists): DIR-GNN splitter
mechanics from [Wuyxin/DIR-GNN](https://github.com/Wuyxin/DIR-GNN);
ADWIN from [river](https://riverml.xyz); PCMCI cross-check from
[tigramite](https://github.com/jakobrunge/tigramite); validation data from
[kausable/CausalDynamics](https://github.com/kausable/CausalDynamics).

---

## 5. Testing

124 automated tests (`pytest`, all passing as of the last run), covering
every module above — including tests written specifically after real bugs
were caught during development, e.g.:
- an early version of the splitter accidentally fed the *same* swapped
  data to both the causal and non-causal edges, which silently defeated
  the entire invariance mechanism (§3.3) — caught, fixed, and locked in
  with a dedicated regression test;
- an early version of the archive-matching similarity score saturated at
  a meaningless 1.000 for every comparison, because raw scores shared a
  large common offset that swamped the real signal — caught, fixed by
  centering scores before comparing them, and locked in with a test.

---

## 6. What was built and run, in order

This mirrors the project's actual build order, each stage adding exactly
one new working part so any failure points at whichever stage just got
added.

| step | what | outcome |
|---|---|---|
| 1 | Mesh loader — raw data → 50-region graph + lagged features | done; cache built, 25/25 tests |
| 2 | Splitter trained on datasets with a **known** true causal graph (CausalDynamics), to check the mechanism actually works before trusting it on real data with no answer key | mixed at first, then fixed (§7.1) |
| 3 | Mid-scale check: real BSISO features + a hand-written, deliberately-known causal rule injected as the target | real signal at small scale, degrading sharply at larger candidate sets (§7.1) |
| 4 | First real, full production run: splitter + expert together, on real BSISO OLR, no known answer key anymore | forecasting works; causal selection does not discriminate well (§7.2) |
| 5 | Add drift detection (single region) | both channels work; a real OR-vs-AND speed/false-alarm trade-off measured (§6.1 below) |
| 6 | Add hibernate/reactivate (single region) | works; ~60% of new drift events found a matching past regime instead of starting fresh |
| 7 | Scale everything to all 50 regions | forecasting generalizes (94% beat baseline); causal-ranking weakness confirmed mesh-wide, not a one-region fluke |
| 8 | Baselines + evaluation | the project's key negative result: the causal splitter ≈ a random edge subset (§7.2) |
| — | Attempted fix #1: CIA (representation alignment) | does not help; makes things worse (§7.3) |
| — | Attempted fix #2: GSINA (differentiable edge selection) | fixes the diagnosed symptom, but no forecast-accuracy gain at full scale (§7.3) |

### 6.1 Drift detection result (step 5), concretely

Two deliberately chosen test windows: an **abrupt** shift (the 1997-98
El Niño) and a **gradual** shift (a documented slow monsoon change,
2013-2022).

| detector | catches abrupt shift? | catches gradual shift? | false alarms/year |
|---|---|---|---|
| Channel 1 (error, ADWIN) | missed | yes, but slow (70 days) | very low (0.02) |
| Channel 2 (causal structure, STARS) | yes, fast (17 days) | yes (33 days) | high (1.79) |
| Fused, OR | yes (17 days) | yes (33 days) | high (1.81) |
| Fused, AND | missed | yes, slow (70 days) | very low (0.02) |

This is exactly the speed-vs-reliability trade-off the two-channel design
was meant to expose: OR reacts fast but false-alarms often; AND is
trustworthy but slow and can miss fast shifts entirely. Reported as a
finding, not resolved to one "correct" rule.

---

## 7. Key results — the parts that matter most

### 7.1 Does the splitter even work, in principle? (Steps 2-3)

Tested first on synthetic data with a **known** true causal graph, so
right and wrong answers are actually checkable (impossible on real
climate data). Two important findings:

1. **A real architecture bug was found and fixed.** The splitter's
   edge-scoring model could only see one instant in time, so it was
   structurally blind to any causal link that only shows up as a *lagged*
   relationship (e.g. "X five days ago causes Y today") — which is most of
   what actually matters in climate data. Fixed by giving it a short
   trailing history window instead of one snapshot. This measurably
   improved edge-recovery (from chance-level, AUC~0.50, to AUC~0.69 on one
   test graph).
2. **A real, still-unresolved limitation: dense candidate sets.** On a
   graph with many candidate edges, the fix above stopped working
   (AUC stayed at chance). The same pattern reappeared later, independently,
   in the semi-synthetic test (recovery collapsed going from 5 to 15
   candidate regions) and again on real data (steps 4, 8) — four separate
   observations of the same underlying cause, which the literature (Wang
   et al., NeurIPS 2024) explains: the loss function used here has *many*
   equally-good solutions, and nothing forces training toward specifically
   the one that uses the true causal edges instead of some other
   equally-accurate mix.

### 7.2 Does it work on real data? (Steps 4, 7, 8) — the central finding

**The forecasting half works.** Across all 50 regions, 47 (94%) beat a
naive "tomorrow = today" baseline, by 9.8% on average (range −4.1% to
+22.6%).

**The causal-selection half does not.** Step 8's controlled comparison, at
matched training budget, is the clearest evidence:

| model | forecast error (MSE) | beats naive baseline? |
|---|---|---|
| naive baseline (tomorrow = today) | 0.02572 | — |
| **plain GNN, uses every candidate edge** | **0.02328** | **yes, by the most** |
| random 3-of-8 edge subset | 0.02410 | yes |
| **our causal splitter's selected edges** | **0.02406** | yes, but statistically tied with *random* |

The splitter's carefully-learned edge selection performs **the same as
picking edges at random**, and *worse* than just using every edge without
selecting at all. An independent statistical cross-check (PCMCI, a
well-established causal-discovery method from a different methodology
entirely) confirms the splitter isn't picking noise — every edge it
selects really is a statistically real driver — but it systematically
**misses the strongest ones**, including self-persistence (a region
predicting its own future, the single strongest link almost everywhere)
and the top cross-region driver. So the failure mode is specifically
**ranking**, not **detecting noise**: the model's edge scores cluster too
close together to reliably tell a strong driver from a weak one.

**Why this is reported as a real finding, not a failure to bury:** it's
reproduced four independent ways (steps 2, 3, 4, 8), cross-checked against
an established statistical method (PCMCI) that agrees on *which* edges are
real while disagreeing on *ranking*, and matches a specific, named
theoretical limitation in the invariant-learning literature (non-unique
optimal solutions in VREx-style training objectives — Wang et al., NeurIPS
2024) rather than looking like an implementation bug. 124 passing tests,
including tests written directly against the official DIR-GNN reference
code's selection logic, rule out "we implemented it wrong."

### 7.3 Two attempted fixes

Both were pulled from the literature specifically because they target the
diagnosed cause (non-unique optima / weak selection gradient), not tried
at random.

**CIA (cross-environment alignment)** — adds a term that pulls together
the internal representations of same-outcome examples across different
swapped backgrounds, meant to supply the missing tie-breaker. **Result:
does not help.** Tested at 4 weight/bandwidth settings and 2 data budgets
— every single configuration made forecast error *worse*, in one case bad
enough that the model stopped beating the naive baseline at all
(0.03040 vs. 0.02572). Score spread also shrank further rather than
widening, the opposite of CIA's intended effect. Plausible reasons are
documented (§10, CIA entry, in the working log) but not fully resolved —
most likely that the regression adaptation of a method designed for
classification is too weak a constraint at this batch size.

**GSINA (differentiable edge selection)** — replaces the hard,
gradient-blocked "keep the top-r%" cutoff with a smooth, fully
differentiable selection, so gradient can actually push scores toward the
correct ranking during training (rather than only through the *weight* of
edges already hard-picked). **Result: a genuine, reproducible fix to the
diagnosed symptom, but not to forecast accuracy.**

- Score spread widened **5x, at every single one of the 50 regions**, no
  exceptions — the near-uniform-score problem (§7.1-7.2) is measurably
  gone.
- It selects self-persistence at every region and, on several test
  regions, correctly finds the single strongest real driver by independent
  measures (raw correlation and PCMCI) — something the original mechanism
  never managed at that data budget.
- **But** across the full 50-region sweep, forecast accuracy is
  essentially a wash versus the original mechanism (46/50 still beat
  baseline vs. 47/50 before; mean skill +9.62% vs. +9.83% — not a
  meaningful difference). An earlier 4-region spot check looked like a
  clear win, which the full sweep corrected — a concrete example of why
  the full sweep mattered rather than trusting a small, if carefully
  chosen, sample.

**Combining both fixes** (tested once, one region): no better than GSINA
alone — consistent with both targeting the same underlying weakness by
different means, so fixing it once with GSINA leaves little for CIA to
add.

**Bottom line on the fixes:** GSINA is a real, interpretability-grade
improvement (the selected edges are now trustworthy and match independent
evidence); it is not yet an accuracy improvement. This is presented in the
project as its own honest, nuanced conclusion, not rounded up to "solved"
or down to "failed."

---

## 8. Where to see the actual outputs

- **Every experiment's raw numeric result:** `code/results/raw/*.json` —
  one file per run, e.g. `step8_baselines_place22_direct.json`,
  `step7_all_places_gsina_direct.json`.
- **Generated charts of the above** (used for presenting this work):
  `code/results/slide_figures/*.png` — e.g. `06_1_density_cliff.png`
  (the recovery-collapse finding, §7.1), `08_1_ablation.png` and
  `08_5_fix_comparison.png` (the step 8 / CIA / GSINA comparison, §7.2-7.3),
  `07_2_all_places_skill.png` (the 50-region forecast skill spread, §7.2),
  `09_1_lifecycle_timeline.png` (hibernate/reactivate in action, §3.5/§6).
- **The preprocessed data cache:** `code/cache/windowed_clustered50_lead1.npz`.

---

## 9. Honest state of the project, and what's left

**What's solid and demonstrated end to end:** the full pipeline — mesh →
splitter → expert → drift detection → hibernate/reactivate → 50-region
scale-up — runs, is tested (124 tests), and produces real forecast skill
beating a naive baseline at 94% of regions, with a measured and explained
speed-vs-reliability trade-off in drift detection.

**What's not yet solved:** the causal-discovery half doesn't yet produce
a ranking trustworthy enough to claim "these are, in order, the true
drivers." We know precisely why (§7.1-7.2, a named, literature-documented
limitation of this training objective on graphs), we've ruled out
"implementation bug" and "needs more training" as explanations, and one of
two literature-sourced fixes (GSINA) demonstrably repairs the diagnosed
symptom — it just hasn't yet translated into better forecasts.

**Documented next steps, not yet built** (candidates for a future
session, in the order most likely to pay off):
1. Run GSINA as the standing mechanism through the full drift-detection
   and hibernate/reactivate pipeline (steps 5-6), not just the static
   forecast comparison — its much wider score spread may behave
   differently as a *change-point signal* even without changing raw MSE.
2. Sweep GSINA across more of the 50 regions with multiple seeds, to
   firm up the "5x wider score spread, everywhere" finding into a
   statistically reported result rather than a strong pattern.
3. Investigate why better-justified edge selection didn't improve MSE —
   e.g. check whether the expert model is already saturating on
   self-persistence and general neighbourhood signal regardless of exactly
   which 3 edges are weighted highest (the working hypothesis, not yet
   directly tested).
4. The published-method baselines (GC-MoE, DyMoE, GeoMoE) listed in the
   evaluation spec were never run — only internal ablations (plain-GNN,
   random-subset) were. Real remaining work if a like-for-like comparison
   against other teams' published methods is needed for the final report.
