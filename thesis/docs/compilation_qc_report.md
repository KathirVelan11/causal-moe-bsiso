# Compilation and quality-control report

## Build

- Toolchain: TeX Live 2023 (Debian packages), pdfTeX, BibTeX `ieeetr`, `latexmk`.
- Command: `latexmk -pdf mythesis.tex` — exit status 0.
- Output: `mythesis.pdf`, 135 pages (front matter i–xx, 9 chapters, 3 appendices, references, publications page).

## Checks performed

| Check | Result |
|---|---|
| LaTeX errors | None |
| Undefined references / citations | None (no `LaTeX Warning` lines in the final log) |
| BibTeX warnings | None |
| Duplicate PDF destinations (roman/arabic page numbers) | Fixed with `hypertexnames=false` |
| Missing images / broken `\input` paths | None |
| Overfull boxes | 2 remaining, both ≤ 22 pt and not visible on the page (a table in Ch. 3 and one in Ch. 8); all visible overflows fixed |
| TikZ diagrams (Figs 1.1, 3.1, 3.2, 4.1, 5.1, 5.2, 8.1) | Rendered and inspected; overlapping labels fixed |
| Result figures (Figs 7.1–7.10) | All from saved result files; Fig 7.9 had a status footnote cropped (caption carries the context); Fig 7.10 generated for the thesis |
| Every figure/table referenced in text | Yes |
| Template preserved | Class, geometry, spacing, title page, certificate, declaration, page numbering, `ieeetr` |
| Placeholders | Names, roll numbers, team number, advisor, date (`frontmatter/details.tex`); archive provenance/licences `[confirm]` |

## Scientific-consistency checklist (from the brief)

- [x] Phase 1 is the primary contribution; Phase 2 is one combined extension; no separate Phase 3.
- [x] 300-node (50 regions) and 596-node (100 regions) representations explained, with the 4 dropped SST sub-region nodes.
- [x] Six variables and 1–60-day lags consistent throughout.
- [x] 1979–2012 development / 2013–2022 holdout respected.
- [x] Season restricts target days only; source history kept.
- [x] B1 → B2 → B3 → B4 in order.
- [x] MoE controls and evaluation answer the RQs; RQ4 explicitly open.
- [x] Phase 2 claims separated from Phase 1 results.

## Integrity checklist

- [x] No fabricated results (see `evidence_and_claim_audit.md`).
- [x] Unresolved decisions listed (Appendix C Table C.2; `research_decisions_and_gaps.md`).
- [x] Proposed experiments labelled *planned*.
- [x] Correlation, conditional dependence and causation distinguished.

## Remaining items for you

1. Fill `frontmatter/details.tex` and the bracketed advisor acknowledgement.
2. Confirm archive provenance and licences (Table 3.2, Table B.4).
3. Spot-check bibliography DOIs/page numbers (entries written from knowledge, not looked up online).
4. If `Plan.pdf` / Phase 2 PDFs differ from the brief, update Chapter 8.
5. When the random/correlation controls are run, add them to Tables 6.5, 7.7 and 7.11 and update RQ4/H4 and the abstract.
