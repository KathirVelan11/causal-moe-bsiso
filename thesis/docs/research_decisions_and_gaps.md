# Research decisions and gaps report

This report lists the unresolved dataset, methodology, implementation and
evaluation decisions behind the thesis, and the inputs that were not available
when it was written. The thesis's own decision log is Appendix C, Table C.2.

## 1. Inputs that were not available

| Requested input | Status | How the thesis handles it |
|---|---|---|
| `Metamaterials_Thesis.pdf` (sample thesis) | Read in full | Used for depth, structure and front-matter order; no content copied |
| Amrita LaTeX template (zip) | Read in full | Used as the base: `mythesis.tex`, class options, front matter, `ieeetr` |
| `Plan.pdf` (Phase 1 plan) | **Not supplied** | Phase 1 specification taken from your brief, the status document you pasted, and the repository (`phase1-causal-graph` branch: README §13, code, results) |
| Phase 2 design PDFs (two files) | **Not supplied** | Chapter 8 written from your brief; datasets and design marked as planned |
| `Phase_1_Causal_Graph_BSISO_OLR_Research_Report.md` | **Not supplied** | Not used |

**Action:** if `Plan.pdf` or the Phase 2 PDFs contain details that differ from
the brief (e.g. exact gate design, dataset periods, region definitions for
India), update Chapter 8 and the decision log accordingly.

## 2. Conflicts found and how they were resolved

| Conflict | Resolution in the thesis |
|---|---|
| Brief says "50 Indo-Pacific regions"; the data grid covers 30°S–30°N at **all** longitudes, and the region map includes Atlantic/African regions | Described as "50 tropical regions centred on the Indo-Pacific" with the full domain stated explicitly (Ch. 1 §1.11, Ch. 3 §3.4) |
| Brief says Phase 1 removes a smoothed climatology on training data; data actually arrive with an upstream climatology fitted on **1979–2014** | Stated as received, listed as a limitation (§3.7, §7.14), refit listed as future work (E23) |
| Status doc numbers the project in three phases; brief asks for two | Phase 3 (global) folded into future work (§8.14, §9.9) |
| Main-branch README (year-round, pre-Phase 1) vs Phase 1 README (season) give slightly different pool-size numbers | Both reported separately and labelled (Table 7.6) |
| Phase 1 README (older) says Stage C prior runs "not yet run"; status doc (newer) reports four 7-day and two 1-day configurations | Status doc and result files used (they agree with each other) |
| Earlier splitter-score drift signal detected two test shifts; Phase 1 PCMCI parent-set signal does not track ENSO | Both reported; reconciled in §8.9 (different signals, tests, windows) |
| Status doc 1-day table groups prior size 0/1/2/3; thesis groups 0/1/2/≥3 | Recomputed from the result files so all 50 targets are covered (Table 7.8); rows 0–2 match the status doc exactly |

## 3. Your instruction on flaws vs limitations

You asked that the thesis state genuine limitations but not list
implementation flaws. The thesis therefore:

- describes every method theoretically and in full;
- reports only measured results, with honest interpretation;
- lists the scientific limitations from your status document (§6 of it) in
  §7.14 and §9.6;
- presents experiments not yet run (random/correlation controls, other modes
  and pools, multi-seed runs, τ_min = 7 graph, sensitivity checks) as a
  **defined experimental grid and future work** (Table 6.5, Appendix B Table
  B.3, §9.8), not as "skipped" items;
- does not invent any number for experiments that have not been run.

Two points could not be softened further without misrepresenting the evidence,
and are stated plainly but neutrally:
1. RQ4 (causal prior vs random/correlation priors) is **open**.
2. The core Phase 1 hypothesis is **not yet confirmed** (the graph is
   reliable; its forecasting benefit is shown only vs no prior and only at 1 day).

## 4. Open decisions

### Dataset
- D14: originating products and licence of the BSISO monitoring archive [confirm].
- D11: refit the anomaly climatology on 1979–2012.
- Second reanalysis/OLR product for robustness (E22).
- Phase 2 datasets: IMDAA, ERA5, ISD versions, periods, licences, acquisition (D16).

### Methodology
- D5: lead-matched graph (τ_min = 7) — the most important change suggested by the results (E15).
- B3 criterion: forecast-horizon skill or "beats random links" (E16).
- B1: record decoy MCI strengths before FDR (E17).
- D8: whether B4-flagged links should leave the prior (E25).
- D2: nonlinear CI tests / LPCMCI for latent confounders.
- Stage D: strength-based distance and longer windows (E24).

### Implementation
- Phase 2 backbone architecture and expert structure (D15: regional, per-node, regime or hybrid).
- Rule that keeps the target's own history available when the prior has a single region (see §7.12).

### Evaluation
- Random- and correlation-prior controls (E13, E14) — answer RQ4.
- Remaining 27-configuration grid (E18) and 3–5 seeds (E19).
- Diebold–Mariano and block-bootstrap intervals on daily errors.
- Year-by-year and BSISO-phase breakdowns.

## 5. Suggested priority (≈ 4 h of compute for the first two)

1. Random and correlation priors, restrict, full pool, 1 and 7 days.
2. Rebuild graph with τ_min = 7; rerun item 1 at 7 days.
3. Multiple seeds for every reported Stage C cell.
4. Remaining modes/pools; sensitivity checks; climatology refit.
