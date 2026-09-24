# Causal-MoE — code

DIR-GNN causal splitter + Dynamic MoE for BSISO forecasting. See
`../Causal_MoE_Architecture.md` for the full spec (architecture, dataset,
decisions log, build order) and `../PROJECT_PLAN.md` for the complete,
chronological record of every bug found and fixed, every experiment run,
and every number reported below.

## Status (2026-09-24, Phase 9 complete)

Both halves of the pipeline work and are honestly evaluated end-to-end:
the forecasting half beats persistence at every place tested (at leads
where persistence is beatable at all — see the lead-1 result below),
and the causal-discovery half genuinely discriminates edges for the
`direct` candidate-set variant — not perfectly everywhere, but
substantially and reproducibly, which was not true of any version of
this project before this session's fixes.

**This corrects the project's own earlier claims.** The build-order
section below (steps 1-8, CIA, GSINA, PCMCI) is kept as a historical
record of the original build, but its "causal splitter = random
selection" conclusion has been superseded: that conclusion was itself
caused by three concrete, fixable bugs (a self-bypass shortcut in both
the encoder and the expert, no honest train/test split, and a diluted
loss/metric in the semi-synthetic validation ladder), not a fundamental
limitation of VREx-style invariant learning on graphs. Fixing those bugs
(plus 22 others found along the way — full list and evidence in
`../PROJECT_PLAN.md`) changed the result. All numbers below are final —
every follow-up flagged as "optional" or "untested" in the original
bug-fix pass has now been run (Phase 9, 2026-09-24): both candidate-set
variants and the hardest lead mesh-wide, CIA/GSINA re-verified
post-bugfix, drift/lifecycle re-run at 3 places, and figures generated.
See `results/phase9_figures/` for the visual versions of the numbers
below.

**One open scope question, flagged 2026-09-24, not yet resolved (B23):**
this project trains and evaluates on the full 12-month record, but
BSISO is by definition a boreal-summer (May-Oct) phenomenon — outside
that window the dominant tropical intraseasonal mode is the MJO, with
different dynamics. Every number below is therefore an average over
two different physical regimes, not a clean BSISO-only result. The
seasonal difference is real but modest where checked (place 22/lead-7:
persistence R² vs climatology -0.656 JJASO vs -0.607 other-season).
Not fixed — this is a scope call for whoever owns the writeup: either
add a season filter and re-run, or soften the framing to "tropical
intraseasonal variability" rather than "BSISO" specifically. See B23 in
`../PROJECT_PLAN.md` for the full analysis.

### Headline numbers

**Forecasting, lead-7 (all 50 places, out-of-sample test ≥ 2013-01-01,
honest chronological split, self-bypass off, 5 epochs):**

| variant | ~candidates | beat persistence | skill mean | R² vs climatology mean | discrimination gate (std≥0.02) |
|---|---|---|---|---|---|
| **direct** | 6 | 50/50 (100%) | +0.401 | **−0.014** (31/50 places negative) | **27/50 (54%)** |
| 2hop | 15-20 | 50/50 (100%) | +0.395 | −0.023 (34/50 places negative) | 8/50 (16%) |
| full | 50 | 50/50 (100%) | +0.396 | −0.020 (39/50 places negative) | 0/50 (0%) |

(Sources: `results/step7_all_places_direct_lead7.json`,
`step7_all_places_2hop.json`, `step7_all_places_full.json`.)

**Bug found during a later audit pass (2026-09-24), not caught by Phase
9: R² vs climatology at lead-7 is *negative on average*, for all three
variants — omitted from this table until now.** This is not a
regression from a code change; it was never checked here (only skill-
vs-persistence and score_std were compared when the sweep moved from
epochs=2 to epochs=5 — see `PROJECT_PLAN.md`'s Progress Log). Mechanism,
confirmed directly from the result files: `climatology_mse` (predicting
the training-period's mean target value for every test row) averages
~0.314 at BOTH lead-1 and lead-7 — expected, since a constant
prediction's error against the 2013-2022 test span doesn't depend on
forecast lead. But `expert_mse` at lead-7 averages ~0.317, essentially
tied with climatology, because lead-7 OLR has very little short-lag
structure left to exploit relative to its own variance — unlike
persistence (MSE 0.535 at lead-7), which climatology beats easily since
persistence has no way to track the slow drift a training-mean constant
implicitly captures. **Read plainly: at lead-7, "beats persistence"
and "beats climatology" are not the same claim, and this project's
model only clearly wins the first one.** The lead-1 result already
disclosed climatology honestly (+0.729, strongly positive there); this
lead-7 gap should have been disclosed the same way and was not. Not a
correctness bug in the model or the metric computation itself (verified
by recomputing `r2_vs_climatology_mean` directly from each `results/*.json`
file's per-place data — it matches the file's own summary field exactly),
purely a reporting omission, now fixed here.

**The central, honest finding from the full variant sweep: forecasting
accuracy is essentially flat regardless of candidate-pool size (~40%
skill vs persistence whichever variant you pick), but edge-discrimination
quality collapses sharply as the pool grows** (Bug 18, candidate-pool
dilution of the shared encoder — confirmed on the real mesh, not just
the semi-synthetic harness where it was first found). **Practical
consequence: use `direct` for any claim about "which regions causally
drive this one" — it is the only variant with real discrimination.
`2hop`/`full` are accuracy-only ablations, not causal-discovery
results.**

**Forecasting, lead-1 (hardest lead — self-persistence is ~93% of
variance here, Bug 1's own original diagnosis, now confirmed mesh-wide):**

- **0/50 places beat persistence** — exactly as Bug 1 predicted at the
  start of this whole audit: no amount of causal-edge information beats
  "tomorrow = today" when today alone already explains 93% of the
  variance. This is not a regression or a failure; it is the honest
  result at the lead where the task is nearly unwinnable against that
  specific baseline.
- **Every place still clearly beats climatology** (mean R² vs
  climatology = +0.729) — the model is genuinely skillful, just not
  more skillful than the trivial baseline at this lead.
- **Discrimination is actually better at lead-1 than lead-7**: 37/50
  (74%) clear the std≥0.02 gate, vs 27/50 at lead-7. Accuracy-vs-
  persistence and discrimination quality are two different axes and do
  not move together — this is the clearest evidence for that in the
  whole project. (Source: `results/step7_all_places_direct_lead1.json`.)

**Causal discrimination, CIA/GSINA re-verified post-bugfix (place 22,
`direct`, lead-7, identical config to plain DIR-GNN):**

| method | expert MSE | discrimination std |
|---|---|---|
| plain DIR-GNN (post all 19 bug fixes) | **0.3207** | **0.0203** (clears gate) |
| CIA (cia_weight=1.0) | 0.3324 (worse) | 0.0026 (far below gate) |
| GSINA (iters=20) | 0.3364 (worse) | 0.0087 (far below gate) |

Both literature-derived fixes were built to solve a problem (near-
uniform edge scores) that turned out to be this project's own self-
bypass bug, not an inherent VREx-on-graphs limitation. Post-fix, plain
DIR-GNN beats both on accuracy **and** discrimination. Kept as tested,
working, documented ablations — not deleted, not recommended for use.

**Mixture-of-Experts (3 places tested: 22, 12, 29 — low/mid/high
discrimination std — real training budget, not a smoke test):**

| | place 22 | place 12 | place 29 |
|---|---|---|---|
| Variant B beats Variant A (MSE) | 30/49 (61%) | 42/61 (69%) | 41/61 (67%) |
| Variant B wins on forgetting | **yes** | **yes** | **yes** |
| warm-start beats fresh-spawn (MSE) | 28/34 (82%) | mixed | 24/36 (67%) |
| warm-start wins on forgetting | no | no | **yes** |

- **Variant B (k=3 cosine-similarity blended pool) beats Variant A
  (k=1, single nearest expert) on both MSE and forgetting at all 3
  places tested — a robust, cross-place-confirmed result.** This is
  the Phase 5 gate this project set for itself, and it is met, not just
  at one place.
- **A separate, easily-confused claim — whether warm-starting a
  reactivated expert beats spawning fresh — is genuinely place-
  dependent**: warm-start loses on forgetting at 2/3 places but wins at
  the third. **Do not state "warm-starting helps forgetting" as a
  general claim** — it doesn't, reliably. What is robust is "the k=3
  pool blend helps," which is a different, narrower, and better-
  supported claim.

### What changed this session (short version)

Three root causes made the original causal splitter indistinguishable
from a random edge subset:

1. **Both the shared encoder and the per-place expert could see the
   target node's own raw features directly**, including near-perfect
   self-persistence at short lead times — so gradient into the
   edge-selection mechanism collapsed to ~0 (the model had no reason to
   use the selected edges at all). Fixed by making self-features opt-in
   (`use_self_features` / `self_bypass`, both default `False` now).
2. **Every training/eval entry point evaluated in-sample** (no
   chronological train/test split), so headline numbers didn't measure
   generalization. Fixed via `causal_moe.data.splits.chronological_split`
   everywhere (train < 2005, val 2005–2012, test ≥ 2013).
3. **The semi-synthetic validation ladder's loss and metric didn't
   match**: `target_node_mask` (meant to focus loss on one node) existed
   but two of three semi-synthetic scripts never used it, and even after
   fixing that, the precision/recall metric itself was still scored
   against every node's untrained self-loop instead of just the node
   that was actually trained on — mechanically worsening as cluster
   count grew regardless of real splitter quality. Both fixed.

15 further bugs (compute cost, stale test fixtures, a vacuous PCMCI
cross-check, a lifecycle loop that never trained anything, a missing
MoE pool/router, a missing forgetting metric, wrong drift ground truth,
clustering leakage, an unvectorized 64%-of-runtime Python loop, and
others) were found and fixed in the same pass. **Full list, evidence,
and before/after numbers for all 18 bugs: `../PROJECT_PLAN.md`.**

### Known remaining gaps (report honestly, do not paper over)

All items below were re-investigated in Phase 9 (2026-09-24) — this
section states the FINAL disposition of each, not an open TODO.

- **Discrimination is real but incomplete, and now explained mesh-wide,
  not just hypothesized.** 27/50 places clear the 0.02 edge-score-std
  gate at 5 epochs for `direct`; the other 23/50 still don't clear it
  even at this budget. Root cause (`Bug 18`) is confirmed as
  candidate-pool-size dilution of the shared encoder — **confirmed
  directly on the real mesh** via the full variant sweep (discrimination
  gate pass rate 27/50 → 8/50 → 0/50 as candidate pool grows
  direct→2hop→full, while accuracy stays flat at ~40% skill regardless).
  This is no longer an inferred-from-the-synthetic-harness hypothesis.
  On the synthetic harness itself, restricting the candidate pool
  recovered precision/recall from 0.0/0.0 to 0.5/0.5 at n=15 — a real
  but partial mitigation (half of true causal parents still missed even
  in that best case). Phase 4's semisynthetic gate is formally closed as
  **not met, root-caused, partially mitigated, not pursued further**
  (see `../PROJECT_PLAN.md` Phase 9 §9.8 for the full disposition and
  why a third harness attempt wasn't worth the compute).
  `--generator-all-fields` WAS tested mesh-wide (Phase 9.2): no
  meaningful difference either way (skill +0.4007→+0.3985,
  gate 27/50→26/50) — closed as a flat, non-beneficial follow-up, not
  a remaining gap.
- **All 3 candidate-edge variants (`direct`/`2hop`/`full`) are now
  swept across all 50 places**, at both lead-1 and lead-7 for `direct`.
  No longer place-22-only. See the Headline numbers table above.
- **CIA and GSINA are re-tested against the fixed baseline** (Phase
  9.6): both measurably worse than plain DIR-GNN on MSE and
  discrimination std at place 22. Confirmed, not just hypothesized —
  see the table above.
- **Drift detection and hibernate/reactivate are re-run at 3 places**
  (22, 12, 29 — Phase 9.7), under this session's honest-eval, real
  training-budget conditions. Variant B beating Variant A is now a
  cross-place-confirmed finding; the warm-start-vs-fresh-spawn
  forgetting result is honestly reported as place-dependent (wins at
  1/3 places), not generalized from a single site. See the table above.
- **Genuinely still open, not closed by Phase 9**: whether the
  candidate-pool-restriction mitigation for Bug 18 (proven on the
  synthetic harness) would also help the real mesh's `2hop`/`full`
  variants if wired in as an explicit per-target cap — this was
  deliberately not built (Phase 9 §9.3, gated on 9.1's initial 2-place
  read which under-detected the effect; 9.4's full sweep later
  confirmed the effect is real, but building the fix was judged lower
  priority than finishing the sweep, CIA/GSINA, and lifecycle checks
  first). A natural next step for anyone continuing this project, not
  a bug in the current numbers.

---

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
  experts/    per-place time-block Dynamic MoE (§4.3), ExpertPool router (Bug 11)
  drift/      ADWIN + STARS drift detection, fusion gate (§4.4), hibernate/reactivate (§4.5),
              forgetting/backward-transfer metrics (Bug 12)
  baselines/  GC-MoE, DyMoE, GeoMoE(GraphMoRE substitute), plain-GNN (§6)
tests/        pytest suite, mirrors causal_moe/ layout (138/138 passing)
scripts/      CLI entry points (build mesh cache, train, evaluate)
```

## Reproducing the headline numbers

```powershell
# 50-place sweep, direct variant, lead-7, honest eval, 5 epochs (the citable result)
python scripts/run_step7_all_places.py --variant direct --epochs 5 \
    --n-samples-cap 4000 --cache-path cache/windowed_clustered50_lead7.npz \
    --generator-window 10
# -> results/step7_all_places_direct_lead7.json

# Same sweep, 2hop/full variants (candidate-pool-size comparison, Bug 18)
python scripts/run_step7_all_places.py --variant 2hop --epochs 5 \
    --cache-path cache/windowed_clustered50_lead7.npz
python scripts/run_step7_all_places.py --variant full --epochs 5 \
    --cache-path cache/windowed_clustered50_lead7.npz

# Lead-1 sweep (hardest lead, Bug 1's own prediction)
python scripts/run_step7_all_places.py --variant direct --epochs 5 \
    --cache-path cache/windowed_clustered50_lead1_v2.npz

# CIA / GSINA re-verification, place 22 (matches plain DIR-GNN's config exactly)
python scripts/train_step_cia_single_place.py --target 22 --variant direct \
    --epochs 5 --n-samples-cap 4000 --cache-path cache/windowed_clustered50_lead7.npz
python scripts/train_step_gsina_single_place.py --target 22 --variant direct \
    --epochs 5 --n-samples-cap 4000 --cache-path cache/windowed_clustered50_lead7.npz

# MoE Variant A (k=1) vs Variant B (k=3) comparison (needs step 5's signal
# file for the target place first, run_step5_drift.py)
python scripts/run_step5_drift.py --target 22 --variant direct \
    --cache-path cache/windowed_clustered50_lead7.npz
python scripts/run_step6_lifecycle.py --target 22 --variant direct \
    --gen-epochs 3 --eval-days 60 --cache-path cache/windowed_clustered50_lead7.npz

# Figures for all of the above (pure plotting, no training)
python scripts/generate_phase9_figures.py
# -> results/phase9_figures/*.png
```

---

## Build order (§9 of the architecture doc) — original build, historical record

*The entries below describe the pipeline as it was built and evaluated
before this session's bug-fix pass. They are kept for history and
because most of the mechanism descriptions (what each script does) are
still accurate. Their **conclusions about causal-discrimination quality
are superseded** by the Status section above — re-read that section
first for the current, honest picture.*

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
      55/55 tests passing.
- [x] 3. Mid-scale semi-synthetic check (§5.2) — original run DONE
      2026-09-19, **superseded by this session's B3/B17/B18 fixes** (see
      Status section above — the original "sharp cliff between 5 and 15
      clusters" finding was substantially a target_node_mask dilution bug
      (B3) plus a metric-scoring bug (B17), not purely a modeling
      limitation. Even after both fixes, a genuine but smaller residual
      gap remains at larger cluster counts (B18) — see `../PROJECT_PLAN.md`
      for the full corrected ladder.
- [x] 4. One expert, one real place (§5.3) — original run DONE 2026-09-19,
      **superseded by this session's honest re-evaluation.** Original
      result used in-sample evaluation and self-bypass features (both since
      fixed). Current honest out-of-sample result for place 22: see Status
      section above and `results/step4_place22.json`.
- [x] 5. Drift detection (§4.4), single place — DONE 2026-09-19. Both
      channels built (`causal_moe/drift/`): Channel 1 = `river.drift.ADWIN`
      over the expert's rolling error; Channel 2 = a from-scratch
      STARS/Rodionov (2004) regime-shift test over a **causal-signature
      distance** series (cosine vs a baseline reference — the same
      similarity mechanism §4.5 reuses). **Both** fusion rules (OR and AND)
      implemented and reported, per §4.4's "reportable experiment".
      Result (place 22, `direct`, 10-yr baseline, detection 1979-2022): OR
      detects both §8 shift windows fast (17d/33d latency) at ~1.8 false
      alarms/yr; AND is near-false-alarm-free (0.02/yr) but misses the
      abrupt El Niño window. **Re-run at 3 places under the honest-eval
      pipeline (Phase 9.7, 2026-09-24)** — places 12 and 29 both detect
      both shift windows on all channels/fusions too, with place-specific
      latency differences (e.g. place 29's causal-signature channel
      catches the gradual drift far faster than its error channel,
      117d vs 1158d) — see `../PROJECT_PLAN.md` Phase 9 log for full
      numbers. No longer a place-22-only mechanism demonstration.
- [x] 6. Hibernate/reactivate (§4.5) + MoE pool (Bug 11) — mechanism DONE
      2026-09-19, **MoE pool routing added and validated this session.**
      `causal_moe/drift/archive.py` (whole-archive cosine search, warm-start
      reactivation) + `causal_moe/experts/pool.py` (Bug 11: `ExpertPool`,
      cosine-similarity top-k routing, softmax blend) + `scripts/
      run_step6_lifecycle.py` (rewritten this session to actually train each
      generation, Bug 10). See Status section above for the Variant A vs B
      and warm-start vs fresh-spawn results, now confirmed at 3 places
      (22, 12, 29), not just place 22.
- [x] 7. Scale to all 50 clusters — original 2026-09-19 run superseded;
      **current honest result is the Status section's headline number
      above**, now covering all 3 candidate-set variants
      (`direct`/`2hop`/`full`) and both lead-1 and lead-7, not just
      `direct`/lead-7 (`results/step7_all_places_direct_lead7.json`,
      `_2hop.json`, `_full.json`, `_direct_lead1.json`, all epochs=5).
- [x] 8. Baselines + evaluation (§6) — ablations DONE 2026-09-19. Published-
      method baselines (GC-MoE, GeoMoE/GraphMoRE, DyMoE) are external
      codebases per §6/§7 and were **not** run — don't present them as done.
      Original result (plain-GNN ≈ causal splitter ≈ random subset, all
      beat persistence) reflected the pre-fix self-bypass/in-sample-eval
      bugs; not yet re-run under the current honest-eval pipeline.

### CIA-for-regression and GSINA — attempted fixes, pre-dates root-cause bugs

Both are literature-derived fixes for "edge scores don't discriminate"
that were built and evaluated **before** this session found the actual
mechanical causes (self-bypass, in-sample eval, loss/metric dilution).
Original (pre-bugfix) numbers: GSINA showed a real, reproducible
discrimination improvement (score std 5x wider mesh-wide) without
reliably improving MSE; CIA made MSE worse in every configuration
tried. Full numbers and citations: `../Causal_MoE_Architecture.md` §10.

**Re-tested against the current fixed baseline (Phase 9.6, 2026-09-24):
confirmed no longer necessary.** Place 22, `direct`, lead-7, identical
config to plain DIR-GNN: plain DIR-GNN MSE=0.3207/std=0.0203 (clears
gate) vs CIA MSE=0.3324/std=0.0026 (worse on both) vs GSINA
MSE=0.3364/std=0.0087 (worse on both). See the Status section's table
above. Kept in the codebase as tested, working, non-beneficial
ablations — not deleted, not recommended for use going forward.

### PCMCI cross-check (§4.2) — DONE 2026-09-19, Bug 9 fix applied this session

`scripts/run_pcmci_crosscheck.py` (tigramite 5.2.10.1). Independent second
opinion on place 22, exactly the scope §4.2 allows (not a pre-filter, not
ground truth). This session fixed Bug 9 (the original set-overlap-only
comparison was vacuous when most candidates are PCMCI-significant);
`run_pcmci_crosscheck.py` now computes real Spearman rank agreement
between splitter scores and PCMCI `|strength|`. Not yet re-run against
the current honest/fixed splitter — the original entry
(`../Causal_MoE_Architecture.md` §10 step 4) predates the self-bypass fix.
