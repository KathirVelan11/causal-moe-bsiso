# Causal-MoE — code

DIR-GNN causal splitter + Dynamic MoE for BSISO forecasting. See
`../Causal_MoE_Architecture.md` for the full spec (architecture, dataset,
decisions log, build order).

## Environment

This dev machine already has a working CPU-only PyTorch + PyTorch Geometric
install at:

```
C:\Users\ok\AppData\Local\hermes\hermes-agent\venv\Scripts\python.exe
```

Reuse it instead of downloading another copy of torch (large, slow). Run
everything with that interpreter, e.g.:

```powershell
& "C:\Users\ok\AppData\Local\hermes\hermes-agent\venv\Scripts\python.exe" -m pytest tests -q
```

Missing from that venv but needed later (§4.4 Channel 1 drift only —
`river.drift.ADWIN`):

```powershell
& "C:\Users\ok\AppData\Local\hermes\hermes-agent\venv\Scripts\python.exe" -m pip install river==0.22.0
```

**On another machine** (no hermes-agent venv available): create a normal
venv and install `requirements.txt`:

```powershell
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```

## Layout

```
causal_moe/
  data/       mesh construction, raw .npz loading, windowing (§4.1)
  splitter/   DIR-GNN causal rationale generator (§4.2)
  experts/    per-place time-block Dynamic MoE (§4.3)
  drift/      ADWIN + STARS drift detection, fusion gate (§4.4), hibernate/reactivate (§4.5)
  baselines/  GC-MoE, DyMoE, GeoMoE(GraphMoRE substitute), plain-GNN (§6)
tests/        pytest suite, mirrors causal_moe/ layout
scripts/      CLI entry points (build mesh cache, train, evaluate)
```

## Build order (§9 of the architecture doc)

- [x] 1. Mesh loader (§4.1) — raw loader, 50-cluster k-means (OLR
      correlation), cluster adjacency, 21-value lagged windows, OLR-N-days
      target, train/val/test + deliberate-shift splits. 25/25 tests
      passing, verified against the real dataset. Cached:
      `cache/windowed_clustered50_lead1.npz` (71 MB).
- [x] 2. Validation dataset (CausalDynamics, §5.1) — DONE 2026-09-19 (with a
      caveat). Loader + splitter built and tested (see
      `../Causal_MoE_Architecture.md` §10 for full detail). Root cause of the
      earlier AUC~0.50 (no-signal) result found and fixed 2026-09-18: the
      rationale generator scored edges from a single timestep only,
      structurally unable to see cross-time (lagged-correlation-style)
      signal — confirmed the signal exists in the raw data (AUC 0.62–0.74 via
      direct lagged correlation) but was unreachable by the old generator.
      Fixed by feeding the generator a 10-step trailing window per node
      instead of one instant. Added an entropy/polarization penalty
      (`entropy_weight`) on generator scores for the remaining density gap
      (NRI-style sparsity-rate prior also tried, `prior_rate_weight`, but
      made things worse combined with entropy — noted, not used). **Final:
      NONE AUC~0.694 (no penalty needed), AO AUC~0.586 (with
      `entropy_weight=0.5`)** — both clear chance, satisfying §5.1's gate.
      55/55 tests passing. Literature review (NeurIPS 2024 "Dissecting the
      Failure of Invariant Learning on Graphs") explains AO's residual
      weakness theoretically — our loss is VREx, which has non-unique
      optima on graphs; noted as a caveat carried into step 3/4, with
      stronger fixes (CIA-style alignment, GSINA-style selection)
      identified but not built — see architecture doc §10 for full detail
      and citations.
- [x] 3. Mid-scale semi-synthetic check (§5.2) — DONE 2026-09-19 (scope
      decided, not exhaustive — see below). Real Bay-of-Bengal-region
      patches, clustered with the same method as the full mesh, injected
      hand-written causal rule (DIR-GNN-precedent deterministic rule,
      confound-checked B/C selection). **5 clusters: precision/recall
      0.333** (13x21 patch, above 0.24 chance baseline). **15 clusters:
      0.000** (18x37 patch, chance-level) — a real, sharp drop, only
      partially recovered by `entropy_weight=0.5` (0.062). Answers §5.2's
      "smooth degradation vs. cliff" question: looks like a cliff between
      5 and 15. **Decision: stopped the sweep here (30/50 and the
      nonlinear rule variant deliberately not run)** — the 5-vs-15 pair
      already answers the question; further sizes cost more compute for
      an unlikely-to-change result. CIA/GSINA-style fixes (identified in
      step 2) kept as documented future work, not built. **Important:**
      step 3's patches/clusters are test scaffolding only — step 4 does
      NOT reuse them, it uses the real 50-cluster full mesh from step 1.
      Full detail + all numbers: `../Causal_MoE_Architecture.md` §10 step
      3. 67/67 tests passing.
- [x] 4. One expert, one real place (§5.3) — DONE 2026-09-19. Splitter +
      new `PlaceExpert` (`causal_moe/experts/expert.py`) trained together on
      the real 50-cluster mesh, place 22 (equatorial Sumatra / Maritime
      Continent). All three candidate-edge-set variants run and compared
      (`causal_moe/data/candidate_edges.py`: direct=8, 2hop=19, full=50
      sources). **All three beat the persistence baseline** (MSE
      0.0240/0.0250/0.0238 vs 0.0276) — the step-4 gate passes, the
      pipeline produces real forecast skill on real data. **But the causal
      selection does not discriminate:** the variants pick almost
      *disjoint* source sets ({13,37,44} / {11,35,37} / {0,31,36}), none
      picks cluster 38 (the strongest real driver by raw lagged
      correlation, 0.545), and score spread shrinks as the candidate set
      grows (0.0051→0.0025→0.0019) — the **steps 2/3 density cliff
      reproducing on real data**. Read honestly: the *forecasting* half
      works (likely via self-persistence + neighbourhood info), the
      *causal-discovery* half does not yet earn its place. `direct` carried
      forward (best score spread, cheapest, matches §4.1's physical
      adjacency). Full detail + numbers: `../Causal_MoE_Architecture.md`
      §10 step 4. 96/96 tests passing.
- [x] 5. Drift detection (§4.4), single place — DONE 2026-09-19. Both
      channels built (`causal_moe/drift/`): Channel 1 = `river.drift.ADWIN`
      over the expert's rolling error; Channel 2 = a from-scratch
      STARS/Rodionov (2004) regime-shift test over a **causal-signature
      distance** series (cosine vs a baseline reference — the same
      similarity mechanism §4.5 reuses). **Both** fusion rules (OR and AND)
      implemented and reported, per §4.4's "reportable experiment".
      **Two methodology bugs caught before the real run:** (a) the sample
      cap was a *random* subsample, which destroys the time ordering both
      detectors assume — fixed to a chronological prefix; (b) training
      across the whole record let the model absorb the very shifts the
      detector should catch — fixed to train on a pre-shift baseline
      (`--train-years`) then detect *forward*. **Result** (place 22,
      `direct`, 10-yr baseline, detection 1979-2022): OR detects both §8
      shift windows fast (17d/33d latency) at ~1.8 false alarms/yr; AND is
      near-false-alarm-free (0.02/yr) but misses the abrupt El Niño window.
      That latency-vs-false-alarm trade-off is the §6 deliverable.
      **Negative result recorded honestly:** §8's specific prediction —
      Channel 2 accumulating while Channel 1 stays flat — is **not**
      confirmed (contrasts ~0.0002, negligible). 17/17 drift tests.
- [x] 6. Hibernate/reactivate (§4.5), single place — DONE 2026-09-19.
      `causal_moe/drift/archive.py` (whole-archive cosine search, warm-start
      reactivation, deep-copied weights) + `scripts/run_step6_lifecycle.py`.
      **Real bug found by the first run:** it reported a too-good 48/48
      reactivations at similarity *exactly 1.000* — an artifact, because on
      real data all candidate edges score near the same mean (~0.463), so
      raw signature vectors are nearly parallel and cosine saturates
      (cos(1979, 2022) = 0.99997). Fixed with `center_signatures()`
      (subtract the per-edge baseline so matching compares *deviation
      patterns*, not the shared offset). **Result after the fix:** 48 regime
      events → **29 reactivated / 19 fresh spawns**, reusing 17 distinct
      archived experts, similarities spanning −0.368 to 0.997. This is the
      first direct evidence for §4.5 over plain DyMoE (which deletes on
      spawn): regimes *do* recur ~60% of the time, but not always — which
      is exactly why §4.5 specifies a threshold plus a fresh-spawn
      fallback. 5/5 archive tests.
- [x] 7. Scale to all 50 clusters — **DONE 2026-09-19**
      (`scripts/run_step7_all_places.py`, 4,000 samples/2 epochs per
      place, 38.5 min total, no contention). **47/50 places (94%) beat
      persistence**, mean skill +9.8% (median +10.2%, range −4.1% to
      +22.6%). Forecasting generalizes across the mesh — place 22 wasn't
      cherry-picked. Score spread stayed low (mean 0.0041) across all 50
      places regardless of local degree (4-11 neighbours) — the
      causal-ranking weakness from steps 4/8 is mesh-wide, not
      place-specific. Full per-place table:
      `results/step7_all_places_direct.json`.
- [x] 8. Baselines + evaluation (§6) — **ablations DONE 2026-09-19**
      (`causal_moe/baselines/simple.py`, `scripts/run_step8_baselines.py`).
      Published-method baselines (GC-MoE, GeoMoE/GraphMoRE, DyMoE) are
      external codebases per §6/§7 and were **not** run — don't present
      them as done. **Result, place 22, identical budget:** plain-GNN
      (all edges) **0.02328** < full causal splitter **0.02406** ≈ random
      subset **0.02410** < persistence 0.02572. **The causal splitter
      performs the same as a random edge subset (0.2% apart) and slightly
      worse than just using all edges.** The forecasting pipeline is sound
      (everything beats persistence) but the skill is coming from the
      GNN+expert, *not* from causal rationale selection. With the PCMCI
      cross-check (step 4: all splitter picks are PCMCI-significant, but it
      misses the two strongest drivers), the diagnosis is a **ranking
      failure** driven by near-uniform edge scores — the on-real-data
      confirmation of the VREx non-unique-optima limitation (NeurIPS 2024)
      already cited for steps 2/3. Next levers stay CIA / GSINA (design
      work, not tuning).

### CIA-for-regression (attempted fix) — TRAINED, EVALUATED, DOES NOT HELP 2026-09-19

First named fix (§10 step 2) for the ranking weakness confirmed in steps
4/7/8: `causal_moe/splitter/cia.py` adapts CIA (Wang et al., NeurIPS 2024)
from classification to regression — a Gaussian-kernel closeness weight on
true targets replaces the hard same-class test. 9/9 tests, suite 110/110.

Wired into a real training loop this session
(`scripts/train_step_cia_single_place.py`, `scripts/run_step8_baselines_cia.py`)
and run on place 22's real data. **Result: CIA does not help — it makes
forecast MSE worse at every setting tried.** Sweep of 4 weight/bandwidth
combos (place 22, direct, 8000 samples/3 epochs): best case still +8%
worse MSE than CIA off (0.0227 vs 0.0210), score spread shrank further
rather than widening. Step 8's ablation table re-run at the *exact*
original budget (4000 samples/2 epochs, matching
`step8_baselines_place22_direct.json`) with the mildest-degradation
setting (`cia_weight=0.1, bandwidth=0.3`): the causal splitter's MSE rose
from 0.02406 (beats persistence 0.02572) to **0.03040 (does NOT beat
persistence)**. Sanity-checked first: `cia_weight=0.0` reproduces the
no-CIA loop bit-for-bit (`cia_loss~=0.0000` throughout), so this is a real
effect of the added term, not a bug in the wiring. Full numbers + honest
interpretation (batch-size/target-scale/gradient-competition hypotheses,
none confirmed): `../Causal_MoE_Architecture.md` §10, "CIA-for-regression"
entry. Result files: `results/step_cia_place22_direct_w*.json`,
`results/step8_baselines_cia_place22_direct.json`.

### GSINA (differentiable selection) — TRAINED, EVALUATED, REAL IMPROVEMENT 2026-09-19

Second named fix. Replaces the hard top-r argpartition selection with a
differentiable Sinkhorn-based one (`causal_moe/splitter/gsina.py`,
Liu et al. arXiv 2402.07191, adapted to a per-edge 2-column keep/drop
assignment since this project's splitter has one scalar score per edge,
not a full attention matrix) so gradient reaches *which* edges get
emphasized, not only their weight. 14/14 new tests, suite **124/124**.

Wired into a training loop (`scripts/train_step_gsina_single_place.py`)
and run on place 22, same exact budget as step 8 (4000 samples/2 epochs),
2 seeds for a robustness check. **Result: a real improvement, unlike
CIA.** MSE 0.0235/0.0240 (both seeds beat persistence 0.0257) vs the
original hard-top-r splitter's 0.02406 and CIA's 0.03040. More notably:
**GSINA is the only mechanism in the whole project, at this budget, to
select cluster 38** (the strongest real driver by raw correlation, 0.545,
and by PCMCI, 0.153) — in *both* seeds — plus the self-loop (PCMCI's
actual dominant driver, 0.877) in both seeds too. Score spread (std
0.021-0.040) is 4-19x wider than every other mechanism tried
(0.0016-0.0051), i.e. genuinely discriminating between edges rather than
clustering near one value — directly addressing the root symptom
diagnosed in step 4/8. Caveats: 2 seeds/1 place only (not yet a step-7-
style mesh-wide sweep), and self-loop dominance means this may partly be
"correctly identifying persistence matters most" rather than proof of
fine-grained cross-cluster ranking — not yet checked at a place with
weaker self-persistence. Not run combined with CIA (deliberately, per
plan). Full numbers + caveats: `../Causal_MoE_Architecture.md` §10,
"GSINA" entry. Result files:
`results/step_gsina_place22_direct_seed{0,1}.json`.

**Follow-up, same session:** validated on 3 more places (4, 38, 44 —
step 7's worst/best/#2 places). **Holds up on 38 and 44** (both beat
persistence, both select the top-2 real drivers by raw correlation
exactly, score std stays 4-7x wider than hard-top-r). **Does not manufacture
skill on place 4** (step 7's weakest place — GSINA still doesn't beat
persistence there, though it comes close and still picks a real driver,
not noise) — reasonable reading: GSINA improves selection quality but
can't create signal where the underlying forecasting problem is
genuinely hard. Also ran **GSINA+CIA combined** (place 22): MSE 0.0234,
essentially identical to GSINA alone — CIA adds nothing once GSINA has
already fixed the discrimination problem, a sensible null result, not a
concerning one. **Full 50-place GSINA sweep — COMPLETE**
(`scripts/run_step7_all_places_gsina.py`, ~70 min). **Corrects the
4-place read above: forecast accuracy is a wash at full scale, not an
improvement** — 46/50 beat persistence vs the original's 47/50, mean
skill +9.62% vs +9.83% (GSINA better on only 24/50 places). **What DID
hold up mesh-wide: score spread is 5x wider on every single place**
(0.0206 vs 0.0041) and the self-loop is selected at all 50 places,
matching PCMCI's finding that self-persistence usually dominates. Final,
honest claim: **GSINA fixes the diagnosed symptom (uniform,
non-discriminating scores) and produces a more independently-consistent
subgraph, but this does not yet translate into better MSE** — a genuine,
nuanced finding, not the clean win the small sample suggested. Full
numbers: `../Causal_MoE_Architecture.md` §10, GSINA follow-up entry.
Result file: `results/step7_all_places_gsina_direct.json`.

### PCMCI cross-check (§4.2) — DONE 2026-09-19

`scripts/run_pcmci_crosscheck.py` (tigramite 5.2.10.1). Independent second
opinion on place 22, exactly the scope §4.2 allows (not a pre-filter, not
ground truth). All 3 splitter-selected sources are PCMCI-significant (no
false positives), but the splitter misses the two dominant drivers —
self-persistence (strength 0.877) and cluster 38 (0.153). Full table in
`../Causal_MoE_Architecture.md` §10 step 4.
