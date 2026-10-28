# Final Review deck: slide-by-slide guide

This guide walks through all 28 slides of `Final_Review_Updated.pdf`. For each slide it explains:

- **what the slide says**;
- **where each number comes from** (which experiment, which result file);
- **how each threshold or setting was chosen**;
- **likely questions and how to answer them**.

All numbers come from the project's result files in `code/results/`. The `step5_*` and `step6_*` files are on the main branch; the `phase1/` files are on the `phase1-causal-graph` branch. Where a slide shows a summary, the guide gives the calculation behind it.

---

## A few terms used throughout

| Term | Meaning |
|---|---|
| **Node** | One variable averaged over one region, e.g. "u850 in region 27". 50 regions × 6 variables = 300 nodes. |
| **Lag (τ)** | How many days before the target day the source value is taken. A lag-2 link means "the source 2 days ago is related to the target today". |
| **Lead (L)** | How far ahead the forecast is made: 7-day or 1-day. |
| **Link / edge** | A statistically significant dependence from a source node at some lag to a target OLR node. |
| **Own-region link** | A link from the target's own region (e.g. region 22's own OLR yesterday → region 22's OLR today). |
| **Outside link** | A link from a *different* region. These are what the causal graph adds beyond "use your own past". |
| **Anomaly / standardised** | Every value is "how far from normal for this place and day of year", in standard deviations. 0 = normal, +1 = one standard deviation above normal. |
| **Skill vs persistence** | 1 − MSE(model) / MSE(persistence). Persistence predicts "the value at forecast time stays the same". Positive = better than persistence. |
| **R² vs climatology** | 1 − MSE(model) / MSE(climatology). Climatology predicts the long-term average (≈ 0 for anomalies). Positive = better than climatology. |
| **Informative splitter** | A splitter whose source scores actually differ: the standard deviation of its mean scores across candidates is ≥ 0.02. Below that, all candidates get about the same score, so the selection is effectively random. |

---

## Slide 1: Title

Project title, team members, team number and supervisor.

The deck title, "Graph-MoE: A Topology-Aware Mixture-of-Experts Framework for Drift Adaptive Weather Forecasting", differs from the thesis title, "Causal Graph Discovery and Graph-Guided Mixture-of-Experts for Climate and Regional Weather Forecasting". Pick one before the review; examiners notice. If you keep "Topology-Aware", be ready to explain it. The splitter works on the region adjacency graph (the topology), and the causal graph decides *which* parts of that topology to use.

---

## Slide 2: Problem statement and motivation

The slide makes four claims; here is the reasoning behind each.

1. **Regimes shift.** El Niño years, La Niña years and neutral years change rainfall and convection patterns. A model trained on 1979–2004 has seen some of each, but the mix changes, so its error drifts.
2. **Full retraining is wasteful.** A shift usually affects some regions much more than others. Retraining all 50 regional models when only a few changed costs 50× the necessary compute and risks forgetting what still works.
3. **Correlation is not causation.** If two regions are both driven by the same large convective envelope (the BSISO), they look correlated even though neither drives the other. A model that relies on such a correlation breaks when the common driver changes. Causal discovery asks a stricter question: does the source still carry information once the other drivers are accounted for?
4. **Scale makes it worse.** More regions mean more things that can drift independently.

**Our approach** has two parts:
- validate real drivers first (Phase 1, Stages A–B);
- adapt only the regions that drifted (drift detection plus the expert lifecycle).

**Likely question: "Why not just retrain monthly?"** Scheduled retraining reacts late to abrupt shifts and wastes compute when nothing changed. The drift slides (21–23) show the detector reacting in about a month at region 22. They also show that most regime changes reuse an archived expert instead of retraining one.

---

## Slide 3: Phase 1 objectives

Each objective maps to a later slide:

| Objective | Evidence slide |
|---|---|
| 1. Causal graph with PCMCI | Slides 9, 13 |
| 2. Stress tests B1–B4 | Slides 10, 14, 15, 16 |
| 3. Graph fed to the splitter three ways | Slides 10, 17 |
| 4. Comparison on unseen years | Slides 17, 18 |
| 5. Drift-adaptive experts | Slides 11, 21, 22, 23 |

For objective 3, all three modes (restrict, initialise, regularise) are implemented and unit-tested. The results shown use **restrict** mode, which is the strongest form. Say so if asked.

---

## Slide 4: Phase 2 objectives

This slide is a plan; no Phase 2 results are claimed.
- **Backbone:** one shared graph neural network learns common physics for all of India.
- **Experts:** small per-region models that correct the backbone.
- **Causal gate:** decides which expert to trust, using the validated drivers (for example, "the BSISO is active over India right now").
- **72-hour forecasts:** rolled out step by step.
- **Station check:** forecasts are compared with real weather-station observations (NOAA ISD), not only with gridded reanalysis.
- **Benchmarks:** GraphCast (DeepMind, *Science* 2023) and WeatherNext (Google's operational successor models) are the external references.

---

## Slide 5: Consolidated literature review and research gap

Read each row as "what it does well, and what it cannot do for us".

| Paper | Why it matters to us | What it lacks |
|---|---|---|
| **DIR** (Wu et al., ICLR 2022) | Our splitter *is* DIR adapted to forecasting: split the inputs into a causal and a non-causal part, swap the non-causal part with another example's, and require the prediction to stay stable. | Built for one-off graph classification; no time series, no drift, no expert adaptation. |
| **GC-MoE** (Ghaffari et al., 2026) | A router blends pre-trained experts per node, using topology and recent input. | Experts are frozen, so there is no response when the data drift. |
| **GeoMoE** (Cao et al., 2026) | Routes nodes to experts by graph curvature (a geometric property). | Curvature is static; it knows nothing about causes or drift. |
| **GraphMETRO** (Wu et al., NeurIPS 2024) | Each expert handles one kind of distribution shift. | The shift types are hand-designed; it does not discover them from data. |

**The gap:** nobody combines (a) discovering causal structure, (b) detecting drift, and (c) adapting experts, for spatiotemporal weather forecasting. Our system does all three.

**Likely question: "How is this different from DyMoE?"** DyMoE (Kong et al., 2025, in the references) adds a new expert every time a new batch of data arrives, on a fixed schedule. We add, retrain or reactivate an expert only when a *measured* drift signal fires, and we can bring back an old expert when a past regime returns.

---

## Slide 6: Dataset

**Source:** BSISO monitoring fields, already converted to anomalies. The seasonal cycle for each grid cell and calendar day was removed, and each cell was normalised.

**Grid:** 2.5° × 2.5°, 30°S–30°N, all longitudes. That is 25 latitude rows × 144 longitude columns = 3,600 cells, and 2.5° ≈ 275 km at the equator.

**Time:** 1 January 1979 to 31 December 2022, 16,071 consecutive days with nothing missing.

**The six fields and why each is there:**

| Field | Meaning | Why it matters for convection |
|---|---|---|
| SST | Sea surface temperature | Warm ocean supplies heat and moisture |
| h850 | Height of the 850 hPa pressure surface (~1.5 km) | Low-level pressure pattern; shows the wave response to convection |
| u200 | East–west wind at 200 hPa (~12 km) | Outflow at the top of storms |
| PW | Precipitable water (total column water vapour) | Moist columns convect |
| u850 | East–west wind at 850 hPa | Monsoon low-level flow that carries moisture |
| OLR | Outgoing longwave radiation | Low OLR = cold, high cloud tops = active convection. **The target.** |

**Why OLR is the target:** it is the standard satellite proxy for tropical convection, and it is the hardest field to predict. Its day-to-day memory is the lowest of the six (autocorrelation at 10 days ≈ 0.01, against 0.36 for SST). Any skill must therefore come from understanding the system, not from copying yesterday.

**How the 50 regions were made:** k-means clustering groups grid cells whose OLR time series behave alike. It was fitted on the training years only. 46 of the 50 regions are single connected areas, averaging about 72 cells each. 50 is a compromise: fine enough to resolve different convective zones, coarse enough that 50 × 6 nodes × 60 lags is computationally tractable and each region mean is not too noisy.

**Adjacency mesh:** two regions are neighbours if any of their cells touch, with longitude wrapping around. This defines the "direct" candidate pool, meaning neighbours only.

**Split:** 1979–2012 is used for everything that is learned or decided; 2013–2022 is only ever used for scoring. Within 1979–2012 the forecaster trains on 1979–2004 and validates on 2005–2012. The split is chronological rather than random because neighbouring days are nearly identical. A random split would put almost-duplicate days in both training and test, and the scores would be inflated.

**Season:** see the next slide.

---

## Slide 7: Active BSISO season

**What the figure shows:** one horizontal bar per year (1980 at the top, 2022 at the bottom), spanning that year's active BSISO season. Blue bars are training years and orange bars are test years.

**How the BSISO index was obtained**, following the published recipe of Lee et al. (2013):
1. Take OLR and u850 anomalies over 10°S–30°N, 40°–160°E (the Indian Ocean to the western Pacific). Lee et al. use up to 40°N; our grid stops at 30°N.
2. Subtract each point's mean over the previous 120 days. This removes slow variations without using any future data.
3. Weight each point by √cos(latitude), so that grid cells shrinking toward the poles do not dominate, and scale each field so OLR and u850 count equally.
4. Run a combined EOF (principal component analysis) on May–October days of 1979–2012 only.
5. The first two principal components (PC1, PC2) describe the oscillation. Its **amplitude** is √(PC1² + PC2²).

**"EOF variance 11.2% / 7.7%":** the first two EOF modes explain 11.2% and 7.7% of the variance of the combined fields. Lee et al. report about 10% and 7%. Getting nearly the same split from our own data, on a slightly smaller domain, shows the index is reproduced correctly.

**How the season was decided:** in each year, look between 1 April and 30 November. The season starts at the first run of at least 20 consecutive days in which the 31-day average amplitude exceeds 1, and ends at the last such run. It is then padded by 21 days at each end.
- *Amplitude > 1* is the standard "active oscillation" threshold, because the PCs are scaled to unit variance.
- *20 days* rules out short noisy bursts.
- *21 days of padding* keeps the onset and decay phases.

**Results:** 208 days per year on average, ranging from 82 to 286 days; the season starts between April and July. This gives 7,153 discovery days (1979–2012) and 1,989 scored test days (2013–2022).

**Why not just May–October?** The bars show the season varies a lot. A fixed window would include dead weeks in weak years and cut off active weeks in strong years.

**Important detail:** the season restricts which **target days** are scored. Source values can come from any day. A target on 1 June can use a source from mid-April at lag 45.

---

## Slide 8: System architecture

This is the Phase 1 workflow. Follow the numbers.

1. **Data:** as on slide 6.
2. **Regions and nodes:** 3,600 cells → 50 regions → 300 nodes.
3. **Season and time split:** as on slide 7.
4. **Stage A, PCMCI:** 18,000 candidate links tested per target → **536 links** in total (slide 9).
5. **Stage B, validate:** four checks. Decoys 0/250, speed mask 3 failures, top-k with k = 10, persistence 0.62 (slides 10, 14–16).
6. **Freeze graph and prior:** validated outside links are merged into one entry per source region. The strength is the largest |MCI| among that region's links. The result is **2.06 source regions per target** on average. It is frozen (saved and never changed) before the test years are touched.
7. **Stage C, use the graph:** the frozen graph tells each region's splitter which source regions to consider (slide 17).
8. **OLR forecast:** direct forecasts at 1 and 7 days, scored on the 1,989 test-season days.
9. **Drift detection:** forecast errors feed channel 1; the causal signature feeds channel 2 (slide 11).
10. **Expert lifecycle:** retrain, spawn, or reactivate an archived expert (slides 22–23).
11. **Stage D, causal drift:** checks whether the PCMCI graph itself changes with El Niño. This is a supporting analysis, discussed in the thesis.

"**Frozen** before 2013" is the key integrity point. Nothing that shapes the graph has seen a test year.

---

## Slide 9: How PCMCI works

**The problem PCMCI solves:** correlation misleads in three ways.
- **Persistence.** If A really drives B at lag 2 and A changes slowly, then A at lags 1, 3 and 4 also *looks* related.
- **Common driver.** The BSISO moistens and reddens many regions at once, so they look linked even if they are not.
- **Indirect path.** If A → C → B, then A looks linked to B directly.

**Step 1, candidates.** For one OLR target, every node (300) at every lag from 1 to 60 days is a candidate: 300 × 60 = **18,000 candidates per target**. Over 50 targets that is 900,000.
- *Why 60 days?* A full BSISO cycle is 30–60 days, so 60 days covers precursors one cycle ahead.
- *Why no lag 0?* A same-day link has no time direction. Requiring the source to come first means every link points from past to future.

**Step 2, PC1 screening.** Each candidate is tested for whether it is still related to the target after removing the effect of the strongest candidates found so far. Round 0 has no conditioning, round 1 conditions on the strongest, round 2 on the top two, and so on. A candidate is dropped when its p-value exceeds 0.05. This removes most "persistence echoes" early, because the target's own recent past is conditioned on first.

**Step 3, MCI final test.** Each survivor is retested while removing (a) the target's other parents and (b) the source's own past.
- (a) removes indirect paths and common drivers.
- (b) removes the source's own persistence, so the test measures only what the source adds at that exact lag.

The resulting partial correlation is the link's **strength (MCI value)**.

**Step 4, false discovery rate.** With 18,000 tests at p < 0.05, about 900 false links per target would appear by chance alone. Benjamini–Hochberg correction sorts the p-values and keeps only links whose adjusted q-value is ≤ 0.05. The expected share of false links among those kept is then about 5%. Candidates dropped in step 2 count as p = 1, so the correction covers all 18,000 tests; this is the conservative choice.

**The test itself:** partial correlation (ParCorr). It is the correlation of the two residuals after regressing out the conditioning variables, with a Student-t test for significance.

**Why 0.05 for both levels?** It is the standard significance level. The 0.05 in step 2 is deliberately loose, because step 3 rechecks everything; the strict control happens in step 4.

**"Vectorised implementation":** the reference library, Tigramite, rebuilds data arrays for every single test, which is too slow at 18,000 candidates. Our version builds the lagged data matrix once and shares it across all 50 targets. It then tests all candidates that share a conditioning set in one matrix operation (a QR decomposition plus one matrix product).

**"Identical PC1 parent sets, MCI values within 0.002":** on synthetic data with known causes, including a fake common driver and a pure-noise variable, our code and Tigramite pick exactly the same parents. Their strengths agree to the third decimal place. Runtime is about 23 minutes for all 50 targets.

---

## Slide 10: Graph validation, four checks

There is no ground truth for real atmospheric causes, so each check is designed to be able to **reject** links. The tags under the boxes are the measured outcomes.

**B1, decoys (0 of 250 selected).** 250 fake series are made by phase-randomising real nodes: Fourier-transform the series, scramble the phases, transform back. The result has exactly the same rhythm and memory as a real series, including the 30–60-day BSISO peak, but no real relationship to anything. The fakes are added to the candidate pool (550 nodes, 33,000 candidates per target) and PCMCI is rerun. Any decoy that gets selected is a false positive by construction.
- *Gate < 30%* comes from the project plan. It is a warning line, not a statistical law: a decoy rate anywhere near 30% would mean the tests are badly fooled by memory.

**B2, lag mask (3 of 536 fail).** For a link from region A to region B at lag τ days: implied speed = (closest distance between A and B) ÷ τ. The limits by source field are:
- **SST: 1 m/s.** Ocean influences are slow.
- **OLR and PW: 20 m/s.** This is the top of the speed range of convectively coupled Kelvin waves (about 15–20 m/s).
- **Winds and h850: 50 m/s.** Fast dry equatorial waves and upper-level wave trains can approach this.

The "closest distance" is the smallest cell-to-cell distance, so the speed is a lower bound and the test is generous. Links at lags of 25 days or more are flagged as possible echoes of the 30–60-day oscillation, but not deleted.

**B3, top-k (k = 10, 157 kept).** Each target's outside links are ranked by |MCI|. For k = 0, 1, …, 10, a simple ridge regression is fitted on 1979–2004 using the target's own-region links plus its top-k outside links, and scored by R² on 2005–2012. The k with the best validation R² is chosen.

**B4, resolution (mean 0.62; 33 of 48).** Each region is split in two (100 regions, 596 nodes) and PCMCI is rerun. An outside link "persists" if some sub-region pair shows a link from the same field within ±2 days of the same lag.
- *±2 days* allows for slightly different timing at a finer scale.
- A target passes if at least 50% of its outside links persist; the 50% threshold comes from the project plan.
- Failures are flagged, not deleted, because a real large-scale signal can genuinely weaken when split.

**P, prior (2.06 regions per target):** see slide 8, step 6.

**Bottom row: three ways to use the frozen graph.**
- **Restrict:** the candidate pool becomes the prior regions plus the target itself. This is a hard constraint.
- **Initialise:** the full pool is kept, but prior regions start with a score bonus of 2 × strength/max strength, which training can override.
- **Regularise:** the full pool is kept, and scoring regions outside the prior is penalised.
- **Matched controls:** no prior; a *random* prior with the same number of regions; and a *correlation* prior made of the most correlated regions. These exist so that any gain can be credited to the graph's choice of regions rather than just to a smaller pool.

---

## Slide 11: Drift detectors and the expert lifecycle

**Setup for the drift experiment.** For each test region, the splitter and expert are trained on the first 15 years (1979 to early 1994). They then forecast every day of the full record. Two signals are recorded daily:
1. the forecast **error**;
2. the **causal signature**, which is the splitter's vector of scores for all candidate source edges.

**Channel 1, ADWIN on the error.** ADWIN keeps a window of recent errors and repeatedly asks whether the window can be split into an older and a newer part with significantly different average error. If so, it drops the older part and raises an alarm. The confidence parameter is δ = 0.002, the river library's default; smaller means fewer false alarms.

**Channel 2, STARS on the causal signature.**
1. Centre the signatures by subtracting the average signature over the first 365 days. This is needed because raw scores share a large common offset that makes every pair look about 99.99% similar.
2. Compute each day's cosine distance from that baseline. This is one number per day: how different today's choice of drivers is from normal.
3. Rodionov's STARS test looks for a shift in the mean of this series. It flags a point that falls outside the current regime's range, then confirms the shift only if the following points keep supporting it, using the regime-shift index.

The cut-off length is L = 120 days, the minimum regime length we care about (about one season), with significance p = 0.05.

**Fusion.**
- **OR:** an alarm from either channel counts. Faster, but more false alarms.
- **AND:** both channels must alarm within 90 days of each other. Slower, fewer false alarms.

**Tiers:** no alarm → keep (Tier 0); minor change → retrain (Tier 1); regime change → archive and reactivate or spawn (Tier 2).

**Lifecycle details:**
- Triggers are the OR of both channels, merged so that two events are at least **180 days** apart. This "debounce" treats a burst of alarms as one regime change.
- When a regime ends, the current expert is first trained on that regime (3 epochs). It is then **hibernated**, meaning stored in the archive with its regime's average causal signature.
- For the new regime, the current signature (mean of the last 120 days) is compared with every archived signature by cosine similarity.
- If the best match is **≥ 0.9**, that archived expert is reactivated with a **warm start**, continuing from its saved weights. Otherwise a fresh expert is spawned.
- 0.9 is a strict threshold, so an expert is only reused when the driver pattern is very close.
- Every reactivation is also compared with a fresh-spawn control trained on identical data, scored on the next regime's first **60 days**, which the expert has not seen.

---

## Slide 12: Model, configuration and parameters

**Left table, training settings.** These are identical for every compared condition, so any difference between conditions comes from the prior, not from the training budget.
- **5 epochs, 4,000 contiguous samples.** This is a compute budget: 50 regions × several configurations must finish in about 20 minutes each on CPU. Slide 20 shows why 5 rather than 2 epochs: discrimination needs it.
- **Adam, learning rate 1e-3, batch 16.** Standard, stable defaults.
- **Hidden width 16.** This is what the training scripts use. *Note: the old deck said 32, which is only the class default.* The model is deliberately small, so we test source *selection*, not model capacity.
- **Generator window 10 days.** The splitter judges each candidate from the last 10 days of both regions' histories. A single day cannot reveal a lagged relationship; a 10-day window can.
- **6 swaps; variance weight 0 → 1 after a 30% warm-up.** For each training example, the non-causal sources are replaced by their values on 6 other random days. The loss is the mean error plus λ × the variance of the error across the 6 versions. λ starts at 0 so the model first learns to forecast at all, then rises to 1 to enforce stability.

**Right table, PCMCI settings:** explained on slide 9.

**Bottom text:**
- **50 independent splitter + expert pairs.** Each region has its own models with no shared weights, so one region's drift cannot damage another's model.
- **Splitter.**
  1. It scores every candidate source region (a small MLP over the two 10-day windows).
  2. It keeps the top fraction as "causal", where the fraction is min(0.5, 3/number of candidates). That means about 3 sources in big pools, and half the pool in tiny ones.
  3. It swaps the rest with other days and trains for low and stable error.
- **Expert.**
  1. It takes the selected sources' 49-value feature vectors (6 fields × 7 lags of 0, 1, 2, 3, 5, 7 and 10 days, plus an ocean-fraction value).
  2. It averages them, weighted by the splitter's scores.
  3. A 3-layer MLP turns that average into the OLR forecast.
- The target's own history enters only if the splitter selects the target region. This forces the forecast to depend on the selection, otherwise the selection could be ignored.

---

## Slide 13: Discovered causal graph

**536 significant links** survive PCMCI with false-discovery-rate control (q ≤ 0.05), out of 900,000 candidates. That is 0.06%, so the graph is very sparse, an average of 10.7 links per target.

**160 links from other regions.** The other 376 are own-region links, mostly the target's own OLR at lags 1–7. The 160 outside links break down by field as follows:
- OLR 72: convection next door leads convection here;
- u850 37: low-level winds carrying moisture;
- u200 22: upper-level outflow and waves;
- h850 18: low-level pressure and wave response;
- PW 11: moisture;
- SST 0.

**1-day median lag.** Half of the outside links act at a lag of 1 day; 90% act at 2 days or less, and none at 25 days or more.

**Why no SST and no long lags?** PCMCI measures *direct* links. Once it conditions on the fast atmospheric drivers (winds, moisture, nearby convection), a slow driver like SST has already passed its influence through them, so it adds nothing directly. This does not say SST is unimportant; it says SST acts *through* the atmosphere.

**2.06 source regions per target.** The 157 outside links that survive B2 are merged by source region: 103 region-to-region pairs over 50 targets is about 2.06. Two targets, regions 13 and 42, have none.

**The map.** Arrows point from source to target, coloured by field. Most connect neighbours. A few long arrows (u200 and h850) cross thousands of km in 1–2 days. This is physically possible because equatorial waves travel at tens of metres per second.

**Region 22, the worked example** (western equatorial Pacific / Maritime Continent):
- PCMCI finds 13 links: 10 within the region, plus h850 from region 27 at lags 1 and 3, and PW from region 6 at lag 2.
- The prior is therefore region 27 (strength 0.21) and region 6 (0.05).
- Separately, the splitter trained **without any prior** on region 22's neighbours selects regions 6, 12 and 27.

Two completely different methods, one statistical and one neural, agree on regions 6 and 27. This is independent corroboration.

---

## Slide 14: B1 decoy test

**Left chart:** 0 decoys selected and 250 rejected.

**Right chart:** links after each check. Own-region links (blue) are 376 at every step. Outside links (orange) go 160 → 160 → 157:
- B1 removes nothing, because no decoy was selected, so there is no strength bar to clear;
- B2 removes 3 links.

**Why this matters:** the most common way causal discovery fails on climate data is mistaking a series' own memory for a relationship. The decoys have exactly that memory but no relationship, and none got through.

**"517 vs 536":** with 250 extra decoys in the pool there are 33,000 rather than 18,000 hypotheses per target, so the false-discovery correction is slightly stricter. The real graph therefore shrinks a little, from 536 to 517 links. The strength distributions of real links in the two runs are almost identical.

**Honest limit, if asked:** decoys don't copy every property of real data. In particular they don't share a hidden common driver with the target. A 0% decoy rate rules out memory-driven false links, not every possible false link.

---

## Slide 15: B2 physical lag mask

**Reading the figure:** one panel per source field. x = closest distance between the two regions; y = implied speed (log scale). The dashed line is the field's limit; dots are kept links and red crosses are rejected ones.

**The 3 failures:**
- h850 from region 22 → region 32: 5,531 km at lag 1, i.e. 64 m/s against a 50 m/s limit;
- OLR 28 → 33: 2,889 km at lag 1, i.e. 33 m/s against 20;
- OLR 3 → 47: 2,215 km at lag 1, i.e. 26 m/s against 20.

They are removed from the prior but kept, flagged, in the graph file.

**How speed is computed:** 5,531 km ÷ (1 day × 86,400 s) ≈ 64 m/s.

**Pattern:**
- OLR and PW links are mostly close neighbours at a few m/s, consistent with the convective envelope moving at about 5 m/s.
- Wind and height links reach further and faster, up to 46 m/s over about 5,500 km, consistent with equatorial waves.
- SST has no outside links at all.

**Why a speed test and not a distance cutoff?** A distance cutoff would delete real long-range links. The speed test only deletes links that would need to be *too fast*.

---

## Slide 16: B3 top-k and B4 resolution

**Figure (B4):** each bar is a target region's persistence score, sorted. Orange bars are below 0.5 and flagged. The right panel shows the share of links that persist, by field.

**0.62 mean persistence:** averaged over the 48 targets that have outside links. On average, 62% of a target's outside links reappear when its regions are split in two.

**33 / 48 targets ≥ 0.5:** these pass; 15 are flagged. Seven targets score 0; these are mostly targets with only one or two weak links.

**By field:**
- OLR 56/70 (80%) and u200 14/22: large-scale signals, robust to how regions are drawn;
- u850 17/37, h850 6/17, PW 4/11: more local, more sensitive to boundaries.

**k = 10, B3 keeps all 157:** validation R² rises with every extra link, from 0.9619 at k = 0 to 0.9633 at k = 10, so the best k is the largest tested. That equals the most outside links any target has. B3 therefore removes nothing: PCMCI with false-discovery-rate control had already made the graph sparse (about 3 outside links per target).

The R² is high (0.96) because own-region links at their own lags explain most of the target. The outside links each add a small but positive amount.

---

## Slide 17: Causal graph as the splitter's guide

**The table:** all runs use the **full pool**, where every one of the 50 regions is a candidate source. Scores are on the 1,989 held-out season days, 2013–2022.

| Column | How it is computed |
|---|---|
| Informative splitters | Number of regions (of 50) whose splitter score spread is ≥ 0.02 |
| Beat persistence | Number of regions where model MSE < persistence MSE |
| Mean skill vs pers. | Mean over the 50 regions of 1 − MSE / MSE_persistence |
| Mean R² vs clim. | Mean over the 50 regions of 1 − MSE / MSE_climatology |

**Reading it:**

**Informative, 0 → 24 (7 days) and 0 → 36 (1 day).** With 50 similar-looking candidates, the unguided splitter gives them all about the same score (slide 20 shows why). Restricting to the graph's regions plus the target (about 3 candidates) lets it actually choose. This shows the graph makes source selection meaningful.

**1-day skill, −1.527 → −1.327; R² 0.720 → 0.745.** The prior improves the forecast.
- *Why is skill vs persistence negative at 1 day?* Tomorrow's OLR is very close to today's, so "no change" is an extremely strong 1-day forecast.
- Both models still clearly beat climatology (R² > 0.7 in all 50 regions).

**7 days, essentially unchanged (+0.415 → +0.416).** The graph's outside links act within 1–2 days. To forecast 7 days ahead you need what a source was doing 7+ days before the target, and those links mostly don't exist at that lag. The graph identifies the right *neighbours*, but week-ahead forecasting needs week-ahead *precursors*. The fix is to rebuild the graph testing only lags ≥ 7.

**32 of 50 regions improve at 1 day:**
- median improvement +0.40 in skill vs persistence;
- median MSE ratio 0.83, i.e. about 17% lower error;
- Wilcoxon signed-rank test across the 50 regions: p = 0.049, borderline significant.

These are computed by pairing each region's score with and without the prior.

**If asked "is the gain from the graph or just from a smaller pool?"** That is exactly what the random-prior control answers, and it is the next run. Say so directly.

---

## Slide 18: 1-day gain, region by region

**The figure:** one dot per region. x = score without the prior; y = score with it. Above the diagonal means better with the prior. Colour = number of outside regions in the prior.

**Region 22:** skill vs persistence goes from −0.60 to −0.05, almost matching persistence at 1 day.

**Grouped by prior size** (mean 1-day skill vs persistence, without → with):

| Prior size | Targets | Change |
|---|---|---|
| 0 | 2 | −0.86 → **+0.52** |
| 1 | 13 | −1.63 → −2.48 |
| 2 | 24 | −1.53 → **−0.91** |
| ≥3 | 11 | −1.53 → **−1.22** |

- **0 outside drivers:** these targets see only their own region, which at 1 day is close to persistence and therefore good.
- **1 driver:** the hard case. With 2 candidates, the selection rule keeps 1 (half of 2). If the splitter picks the outside driver, the target's own strong 1-day memory is lost.
- **2 or more drivers:** improve on average.

**Fix for the 1-driver case:** always allow the target's own history, or use the softer initialise or regularise modes.

---

## Slide 19: Forecast accuracy

**Left: per-region MSE and RMSE of the 7-day forecast.** This is the year-round run with the direct (neighbour) pool, tested on 2013–2022.
- Mean MSE is 0.317. Because the data are standardised, RMSE = √0.317 ≈ 0.559 of one standard deviation.
- The best regions reach about 0.17 MSE and the hardest about 0.48.

**Right: region 22, 1-day forecast against observed OLR, 1979–80.** The red forecast follows the black observations, including sharp swings.
- *Be precise if asked:* this figure is illustrative. It comes from a quick retrain for plotting and shows a window inside the training years.
- The held-out scores are the ones in slide 17's table.

**"Beats persistence in all 50 regions at 7 days":** at 7 days, mean skill vs persistence is about +0.40 in every pool, i.e. roughly 40% lower squared error than "no change".

**Climatology is the harder bar at 7 days.** After a week OLR has almost no memory, so "no change" is a weak forecast and "predict normal" is a strong one. On average the model is slightly below climatology (R² ≈ −0.015), although 14–19 of the 50 regions beat it. Present both comparisons together.

---

## Slide 20: What makes source selection work

**Left: training budget.** With the direct pool, 2 epochs give 5 of 50 informative splitters and 5 epochs give 27 of 50, at the same forecast accuracy. The selection mechanism needs enough training before its scores separate. This is why all comparisons use 5 epochs.

**Right: pool size, no prior, year-round, 7 days.**

| Pool | Candidates | Skill vs persistence | Informative splitters |
|---|---|---|---|
| direct (neighbours) | ~6–8 | +0.401 | 27 / 50 |
| 2hop (neighbours of neighbours) | ~15–23 | +0.395 | 8 / 50 |
| full (all 50) | 50 | +0.396 | 0 / 50 |

Forecast accuracy stays flat, while the ability to *choose* collapses as the pool grows. One shared scoring network has to rank all candidates at once, and 50 similar-looking candidates are too many.

**The link to the graph:** the graph prior brings the full pool back from 0 to 24 or 36 informative splitters (slide 17). This is the gap the validated graph closes.

---

## Slide 21: Drift detection

**The two test shifts:**
- **Abrupt:** the 1997–98 El Niño, one of the strongest on record. The window runs from March 1997 to July 1998.
- **Gradual:** 2013–2022, a slow change across the test decade.

**Latency:** days from the start of the shift window to the detector's first alarm inside it. Smaller is faster.

**False alarms per year:** alarms outside **all known shift events**. Alarms during other major El Niños (1982–83, 1991–92, 2015–16) are not counted as false.

| Region | Shift | Error (ADWIN) | Causal (STARS) | OR | AND |
|---|---|---|---|---|---|
| 22 | El Niño | 159 d | **35 d** | 35 d | missed |
| 22 | gradual | 966 d | **181 d** | 181 d | 966 d |
| 29 | El Niño | 223 d | **92 d** | 92 d | 235 d |
| 29 | gradual | 1,158 d | **117 d** | 117 d | 1,195 d |
| 12 | El Niño | **95 d** | 123 d | 95 d | 123 d |
| 12 | gradual | **70 d** | 100 d | 70 d | 100 d |

**How the "×" factors are computed (error latency ÷ causal latency):**
- Region 22 El Niño: 159 / 35 = **4.5×**; region 29: 223 / 92 = **2.4×**.
- Region 22 gradual: 966 / 181 = **5.3×**; region 29: 1,158 / 117 = **9.9×**.

**Interpretation:**
- At regions 22 and 29 the causal channel sees the change much earlier: the choice of drivers shifts before the forecast error rises. That is the forward-looking signal we wanted.
- At region 12 the error channel is faster, so present the result as **2 of 3 regions**.
- **OR** catches every shift at every region, at the cost of 1.8–3.2 false alarms per year.
- **AND** is much quieter (0.2–0.9 per year) but slower, and at region 22 it missed the El Niño.

This is a real speed-versus-caution trade-off, and we report both.

**Why these three regions?** They span low, medium and high splitter discrimination (22, 12, 29), so the result is not cherry-picked from the easiest case.

---

## Slide 22: Expert lifecycle

**Timeline:** one row per region. Each vertical line is a detected regime change; green means an archived expert was reactivated and orange means a fresh one was spawned.

| Region | Regime changes | Reactivated | Spawned |
|---|---|---|---|
| 22 | 49 | 34 | 15 |
| 12 | 61 | 34 | 27 |
| 29 | 61 | 36 | 25 |
| **Total** | **171** | **104 (61%)** | 67 |

**What it means:** in 61% of regime changes, the new regime's driver pattern matched an archived expert's at a cosine similarity of 0.9 or more. Climate regimes recur, so experts trained on past regimes are worth keeping instead of deleting.

*Why are there about 50–60 changes in 44 years?* Triggers are the OR of both channels, and bursts are merged so changes are at least 180 days apart. That gives at most about two per year, roughly matching seasonal-to-interannual regime changes.

**The table: how to reactivate.** For each event, two ways of reactivating are compared on the next 60 unseen days:
- **Single best:** use the one most similar archived expert.
- **Top-3 blend:** average the predictions of the three most similar archived experts, weighted by similarity.

| | Region 22 | Region 12 | Region 29 |
|---|---|---|---|
| Blend wins (events) | 30/49 (61%) | 42/61 (69%) | 41/61 (67%) |
| Mean MSE | 0.368 → **0.352** | 0.422 → **0.402** | 0.384 → **0.376** |
| Forgetting | −0.010 → **−0.013** | 0.183 → **0.170** | 0.092 → **0.086** |

The blend is better on both error and forgetting at every region. This is the most consistent lifecycle result, so it is the default.

**Forgetting:** for each regime visited at least twice, take the error now minus the best error ever achieved on that regime, then average. 0 means no forgetting, positive means forgetting, and negative means the system got better on old regimes. Region 22's negative values mean positive backward transfer.

---

## Slide 23: Warm start vs fresh spawn

**The question:** when an archived expert is reactivated, should training continue from its saved weights (warm start), or should a brand-new expert be trained (fresh)?

**The figure:** average forgetting (lower is better).
- Region 12: warm 0.187 vs fresh 0.159; fresh better.
- Region 22: 0.025 vs 0.005; fresh better.
- Region 29: 0.105 vs 0.114; warm better.

**Forecast error:** warm start wins in 28 of 34 events at region 22 and 24 of 36 at region 29. Region 12 is mixed.

**Interpretation:**
- Warm-starting usually helps accuracy right after the switch.
- Whether it also preserves old knowledge depends on the region.
- That is why the **pooled top-3 blend** (slide 22), not warm start, is the recommended default: it is the only strategy that wins everywhere.

**If asked why fresh can forget less:** a fresh expert's error on old regimes isn't "pulled" by continued training on the new one.

---

## Slide 24: Key findings

Each card summarises one earlier slide:

| Card | Meaning | Slide |
|---|---|---|
| 0 / 250 | No fake series fooled the causal discovery | 14 |
| 3 / 536 | Only 3 links need a physically impossible speed | 15 |
| 62% | Links persist when regions are redrawn at double resolution | 16 |
| 32 / 50 | Regions whose 1-day forecast improves with the causal prior | 17–18 |
| up to 9.9× | Causal channel faster than the error channel (region 29, gradual shift; 2.4–9.9× at 2 of 3 regions) | 21 |
| 104 / 171 | Regime changes served by reactivating an archived expert | 22 |

The three takeaway sentences follow from these: the graph is trustworthy, it helps where its time scale matches, and causal structure detects drift early while archived experts pay off.

---

## Slide 25: Current limitations

1. **No ground truth.** Nobody can directly observe "region 27 causes region 22". All checks are indirect: they can reject bad links but cannot prove good ones.
2. **Time-scale mismatch.** The graph's links act in 1–2 days, but the main forecast is 7 days ahead. This explains the neutral 7-day result. The fix is a graph tested at lags of at least 7.
3. **Evaluation scope.**
   - Single training seed.
   - The comparison is against the unguided splitter.
   - The random-prior and correlation-prior controls are the next runs. They separate "the graph chose the right regions" from "the pool got smaller".
4. **Three regions** for the drift and lifecycle results; extending to all 50 is planned.

Stating these yourself is better than having them pointed out. Each comes with a concrete next step.

---

## Slides 26–27: Phase 2 plan

Follow the numbers in the workflow.
1. **India data:**
   - IMDAA, a regional reanalysis by Indian and UK agencies: about 12 km, hourly. It is the main input and target.
   - ERA5, a global reanalysis: about 31 km, hourly. It provides context from outside India and the large-scale drivers.
   - NOAA ISD: hourly station observations.
2. **Preprocessing:** regrid to a common mesh, align time stamps, and normalise with training-year statistics only.
3. **Causal detector:** the Phase 1 pipeline, with one change learned from Phase 1. Only lags ≥ the forecast lead are tested, so every link can actually be used at forecast time.
4. **Shared GNN backbone:** learns common physics over the regional mesh by message passing.
5. **Causal gate and experts:** region-specific experts; the gate decides which expert to trust using the validated drivers.
6. **Forecasts, 1–72 hours:** autoregressive, so each step feeds the next. Variables are 2 m temperature, 2 m humidity, 10 m winds and mean sea-level pressure, plus station temperature and dewpoint.
7. **Station validation:** RMSE, MAE and bias by lead time and region against ISD.
8. **Benchmark:** GraphCast and WeatherNext on the same data, horizons and protocol, with scores split by monsoon phase and El Niño state.
9–10. **Drift monitor and adaptation:** the Phase 1 detector per region; adapt only the affected expert, with pooled reactivation by default.
11. **Global scaling:** only after the regional system is competitive.

**Why the Phase 1 results support Phase 2:** the Phase 2 horizon is 1–72 hours, exactly the 1–2-day time scale on which the Phase 1 graph proved useful.

---

## Slide 28: References

[1]–[4] are the four papers in the literature table. [5] DyMoE and [6] the graph out-of-distribution survey complete the related work. [7] GraphCast is the benchmark. [8] PCMCI is the core method, [9] Lee et al. supplies the BSISO index recipe, and [10] STARS and [11] ADWIN are the two drift detectors.

---

## Quick reference: where every number comes from

| Number | Source file |
|---|---|
| 11.2% / 7.7%, 208 d, 7,153, 1,989 | `phase1/bsiso_season.json`, `phase1/causal_graph_50.json` |
| 536, 376/160, by-field counts, lags | `phase1/stage_a_50.json` |
| 0/250, 517 | `phase1/causal_graph_50.json` → `decoy_report` |
| 3/536, speeds | `phase1/causal_graph_50.json` (B2 flags) |
| k = 10, R² curve, 2.06 | `phase1/causal_graph_50.json` → `summary` |
| 0.62, 33/48, by field | `phase1/stage_a_100.json` + `causal_graph_50.json` |
| Stage C table, 32/50, Wilcoxon | `phase1/stage_c/full_*.json`, `phase1/stage_c_lead1/full_*.json` |
| Pool comparison | `step7_all_places_{direct_lead7,2hop,full}.json` |
| MSE 0.317 / RMSE 0.559 | `step7_all_places_direct_lead7.json` |
| Drift latencies | `step5_drift_place{22,12,29}_direct.json` |
| Lifecycle counts, blend, forgetting | `step6_lifecycle_place{22,12,29}_direct.json` |
