# Evidence and claim audit

Every empirical claim in the thesis mapped to its supporting artifact. Paths
are relative to `code/results/` on the `phase1-causal-graph` branch unless
stated. "Recomputed" means the number was recomputed from the per-region
values in that file while writing the thesis and matches the status document.

## Data and season

| Claim | Location | Source |
|---|---|---|
| Grid 25×144, 2.5°, 30°S–30°N, 0–357.5°E; 16,071 days; no missing days | §3.4, Table 3.2 | `causal_moe/data/raw.py` docstring; main README §3.1 |
| Upstream anomalies, normalised per cell, climatology 1979–2014 | §3.7 | status doc §2.1; README §3.3 |
| 50 regions by k-means on OLR; 46/50 contiguous; ~72 cells/region | §3.3 | README §4 |
| 100 regions, 596 nodes | §3.3, §5.6 | `phase1/causal_graph_50.json` meta |
| BSISO EOF variance 11.2% / 7.7% (Lee: ~10% / ~7%) | §7.2, Table 7.2 | status doc §3.1; `phase1/bsiso_season.json` |
| Season 208 days/yr (82–286), onset Apr–Jul | §3.6, §7.2 | status doc §2.1; Fig. 7.1 |
| 7,153 discovery season days; 1,989 test season days | §3.6, §3.10 | `causal_graph_50.json` meta; Stage C files (`n_test`) |
| Feature vector 49 values (6 fields × 7 lags + ocean) | §3.5 | README §5 |

## Stage A and B

| Claim | Location | Source |
|---|---|---|
| 18,000 candidates/target; τ 1–60; pc_alpha 0.05; BH 0.05 | §4, Table 6.2 | `causal_graph_50.json` meta |
| 536 links; 10.7/target; 376 own; 160 outside (OLR 72, u850 37, u200 22, h850 18, PW 11, SST 0) | Table 7.3 | status doc §3.2; `phase1/stage_a_50.json` |
| Outside lags median 1 d; 90% ≤ 2 d; none ≥ 25 d | Table 7.3 | status doc §3.2 |
| Region 22 walk-through (13 links; prior 27 (0.21), 6 (0.05); score 0.67; splitter picks 6, 12, 27) | Table 7.4 | README §13.2 |
| 0/250 decoys; 517 vs 536 real links (10.3 vs 10.7) | §7.4.1, Fig. 7.4 | `causal_graph_50.json` summary.decoy_report; status doc §3.3 |
| 3/536 B2 failures (64, 33, 26 m/s, lag 1, named pairs); 28 kept outside links 500–5,500 km, max 46 m/s | §7.4.2 | README §13.3; status doc §3.3 |
| k = 10; R² 0.9619 → 0.9633 | §7.4.3 | `causal_graph_50.json` summary.k_validation_r2 |
| Prior 2.06 regions/target; targets 13, 42 empty; distribution 2/13/24/5/3/3 | §7.4.3, Fig. 7.3 | summary; Fig. 03 |
| Persistence mean 0.62; 33/48 pass; by field OLR 56/70, u200 14/22, u850 17/37, h850 6/17, PW 4/11; 97/157 persist; 7 targets at 0 | §7.4.4 | summary; Fig. 07 |
| Vectorised PCMCI verified against Tigramite 5.2.10.1; ~23 min | §6.3 | `tests/test_pcmci.py`; status doc §2.2 |

## Stage C and forecasting

| Claim | Location | Source |
|---|---|---|
| No-prior season pools: 30/9/0 informative; skill +0.412/+0.413/+0.415; beat clim 15/18/14; R² −0.020/−0.018/−0.015 | Table 7.6 | `phase1/stage_c/{direct,2hop,full}_none.json` |
| Year-round pools: 27/8/0; +0.401/+0.395/+0.396; R² −0.014/−0.023/−0.020 | Table 7.6 | `step7_all_places_*.json`; README §9 |
| 7 d: informative 0 → 24; skill +0.415 → +0.416; beat clim 14 → 19; R² −0.0153 → −0.0151 | Table 7.7 | `phase1/stage_c/full_{none,pcmci_restrict}.json` |
| 7 d per region: 23 better / 27 worse; median R² Δ −0.010; Wilcoxon p ≈ 0.56 | §7.7.1 | Recomputed |
| 1 d: informative 0 → 36; beat pers 0 → 3; skill −1.527 → −1.327; R² 0.720 → 0.745 | Table 7.7 | `phase1/stage_c_lead1/full_{none,pcmci_restrict}.json` |
| 1 d per region: 32/50 better; median Δ skill +0.40, Δ R² +0.048; Wilcoxon p = 0.049; sign p = 0.065; median MSE ratio 0.83 | §7.7.2, Fig. 7.10 | Recomputed |
| Region 22 at 1 d: −0.60 → −0.05 | §7.7.2 | status doc §3.4; result file |
| 1 d by prior size: 0 → (2) −0.857/+0.519; 1 → (13) −1.628/−2.477; 2 → (24) −1.528/−0.908; ≥3 → (11) −1.530/−1.218 | Table 7.8 | Recomputed (rows 0–2 match status doc) |
| CIA / GSINA vs plain splitter (MSE 0.3207/0.3324/0.3364; σ 0.0203/0.0026/0.0087) | Table 7.9 | `step4_place22.json`, `step_cia_*`, `step_gsina_*` |
| 2 vs 5 epochs: 5/50 vs 27/50 informative | §7.9.2 | README §10.2 |

## Stage D and lifecycle

| Claim | Location | Source |
|---|---|---|
| Niño-3.4 proxy matches 22/24 years (1989, 2008 wrong) | §7.10 | status doc §2.5 |
| 1-season: 1.1–1.3 parents; 0.164 vs 0.151; p 0.20; alarms 2004, 2011, 2016; 55 per-target alarms, 76% vs 67%, 13 stable | Table 7.10 | `phase1/stage_d_drift_w1.json`; README §13.5 |
| 3-season: 1.6–2.0; 0.200 vs 0.207; p 0.76; none; 21 alarms, 43%, 12 stable | Table 7.10 | `phase1/stage_d_drift_w3.json` |
| Drift detector table region 22; regions 12, 29 detect both; region 29: 117 vs 1,158 days | Table 8.3 | `step5_drift_place{22,12,29}_direct.json`; README §11.1 |
| Pool blend vs single best 30/49, 42/61, 41/61; forgetting yes ×3; warm start 28/34, mixed, 24/36; forgetting no/no/yes | Table 8.4 | `step6_lifecycle_place{22,12,29}_direct.json`; README §11.3 |

## Literature claims

All literature citations are to published papers, books or widely cited arXiv
preprints with the stated authors, venues and years. No claim of "no prior
work" is made; novelty is positioned against the comparison in Table 2.1.
Reference details were written from the authors' knowledge of the literature
and **should be spot-checked** (DOIs) before submission, as no online
lookup was performed while drafting.

## Claims deliberately not made

- No number for any random-prior, correlation-prior, initialise or regularise
  run, other pools with the prior, multi-seed run, τ_min = 7 graph, full-year
  graph, time-half or second-dataset analysis.
- No claim that links are mechanistically causal.
- No Phase 2 (India) results; Chapter 8 is a design, plus groundwork on Phase 1 data.
- No comparison against GC-MoE, GeoMoE or DyMoE (never run).
