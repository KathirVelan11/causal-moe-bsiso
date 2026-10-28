# Research decisions and open items

## Changes in this version
- Condensed from 135 to 67 pages; one appendix; no list of algorithms.
- References reduced to 35, built around the review deck's core papers
  (DIR, GC-MoE, GeoMoE, GraphMETRO, DyMoE, graph OOD survey, GraphCast).
- Literature review rebuilt on the deck's consolidated table and research-gap statement.
- Review-deck figures added; five new workflow images (overview, Phase 1,
  validation, lifecycle, Phase 2).
- Drift-detection and expert-lifecycle results added (Tables 7.5–7.6, Figs 7.15–7.16).
- Team details filled from the deck.

## To confirm
1. Advisor designation and submission month (`frontmatter/details.tex`).
2. Reference details supplied without full author lists: Kong et al. (DyMoE,
   arXiv:2508.09974) — full author list; the graph OOD survey — the deck cites
   "J. Li et al., ACM Computing Surveys, 2022", the thesis cites
   H. Li, X. Wang, Z. Zhang, W. Zhu, arXiv:2202.07987 (2022); confirm which is intended.
3. WeatherNext is named as a planned benchmark without a citation (reference limit).
4. Thesis title: the deck uses "Graph-MoE: A Topology-Aware Mixture-of-Experts
   Framework for Drift Adaptive Weather Forecasting"; change `\ThesisTitle` if
   that is the official title.
5. Archive provenance and licence of the BSISO monitoring data.

## Open methodology (future work in Ch. 9)
- Random and correlation priors of equal size; initialise/regularise modes; all pools; several seeds.
- Graph with lags ≥ forecast lead (7 days).
- Drift and lifecycle on all 50 regions; strength-based structural signals.
- Second data source; climatology refitted on 1979–2012.
