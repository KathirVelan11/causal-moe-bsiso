# Evidence and claim audit

Every empirical claim in the thesis mapped to its source. Paths are under
`code/results/` (main branch for `step*`, `phase1-causal-graph` branch for `phase1/`).

| Claim | Section | Source |
|---|---|---|
| Grid, period, 16,071 days, anomalies, climatology 1979–2014 | §3.2 | `causal_moe/data/raw.py`; status document |
| 50 regions (46 contiguous, ~72 cells); 596 nodes at 100 regions | §3.3 | README §4; `phase1/causal_graph_50.json` meta |
| BSISO EOF 11.2% / 7.7%; season 208 d (82–286); 7,153 / 1,989 days | §3.4, §7.1 | `phase1/bsiso_season.json`; graph meta; Stage C files |
| 536 links; 376 own / 160 outside by field; lags median 1 d | §7.2 | `phase1/stage_a_50.json` |
| Region 22 walk-through; splitter selects 6, 12, 27 | §7.2, Table 7.1 | README §13.2 |
| 0/250 decoys; 517 vs 536 real links | §7.3 | `causal_graph_50.json` summary |
| 3/536 B2 failures; 28 outside links 500–5,500 km, max 46 m/s | §7.3 | README §13.3 |
| k = 10; R² 0.9619 → 0.9633; 2.06 regions/target | §7.3 | `causal_graph_50.json` summary |
| Persistence 0.62, 33/48, by field | §7.3 | summary; Fig. 07 |
| Pool baselines (27/8/0; 30/9/0; skill +0.40) | Table 7.3 | `step7_all_places_*.json`; `phase1/stage_c/*_none.json` |
| Mean 7-day MSE 0.3166, RMSE 0.5585 (year-round direct) | Fig. 7.9 | `step7_all_places_direct_lead7.json` (recomputed) |
| 2 vs 5 epochs: 5/50 vs 27/50 informative | §7.4.1 | README §10.2; deck figure |
| Stage C 7 d and 1 d table; 32/50; median Δ; Wilcoxon p = 0.049; MSE ratio 0.83 | Table 7.4, §7.4.2 | `phase1/stage_c{,_lead1}/full_*.json` (recomputed) |
| Prior-size breakdown | §7.4.2 | recomputed from the 1-day files |
| Drift latencies and false-alarm rates (3 regions × 2 shifts × 4 rules) | Table 7.5 | `step5_drift_place{22,12,29}_direct.json` |
| 104/171 reactivated; per-region 34/15, 34/27, 36/25 | §7.6, Fig. 7.15 | `step6_lifecycle_place*_direct.json` (figure regenerated) |
| Pool blend vs single: wins, mean MSE, forgetting | Table 7.6 | `step6_lifecycle_*` → `variant_a_vs_variant_b` |
| Warm vs fresh: forgetting values; 28/34, 24/36 | §7.6, Fig. 7.16 | `step6_lifecycle_*` → `forgetting_warm_vs_fresh`; README §11.3 |
| Stage D: 22/24 ENSO years; 0.164 vs 0.151 (p 0.20); 0.200 vs 0.207 (p 0.76) | §7.7 | `phase1/stage_d_drift_w{1,3}.json` |

## Where the review deck and the result files differ

| Deck | Result files (used in thesis) |
|---|---|
| Region 22 lifecycle: 29 reactivated, 19 spawned (48) | 34 reactivated, 15 spawned (49) — timeline regenerated from the file |
| Drift table: error channel missed 1997–98; structure 17 d; gradual 70 d / 33 d | error 159 d, structure 35 d; gradual 966 d / 181 d |
| Hidden size 32 | Training scripts use 16 (32 is the class default) |
| Domain map label "25S–30N" | Grid is 30°S–30°N (25 latitudes); map not used |

The deck figures came from an earlier quick-retrain script (`slide_figures`),
before the 2026-09-24 audit fix that produced the current result files.

## Not claimed

No results for random/correlation priors, initialise/regularise modes, other
pools with the prior, multiple seeds, a τ ≥ 7 graph, or any Phase 2 experiment.
