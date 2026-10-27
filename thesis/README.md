# Thesis: Causal Graph Discovery and Graph-Guided Mixture-of-Experts

B.Tech thesis (Department of Artificial Intelligence, Amrita Vishwa Vidyapeetham,
Coimbatore), built on the department's LaTeX template (`Thesis_Template___Amrita_AID`).

The compiled thesis is `mythesis.pdf` (135 pages).

## Before submission: fill in your details

All personal and official fields are placeholders in **one file**:
`frontmatter/details.tex` (student names, roll numbers, team number, advisor
name and designation, submission month). The title page, bonafide certificate,
declaration and acknowledgements all read from it. If your team does not have
four members, delete the unused lines in `mythesis.tex`,
`frontmatter/certificate.tex`, `frontmatter/declaration.tex` and
`frontmatter/acknowledgements.tex`.

Also fill in the fields marked `[confirm]` in Chapter 3 (Table 3.2: data
archive provenance and licence) and Appendix B (Table B.4: licences). The
advisor-specific acknowledgement is a bracketed placeholder in
`frontmatter/acknowledgements.tex`.

## How to build the PDF

### Option 1: Overleaf (no installation)

1. Zip the whole `thesis/` folder.
2. In Overleaf, choose **New Project → Upload Project** and upload the zip.
3. Open **Menu** and set **Compiler: pdfLaTeX** and **Main document: `mythesis.tex`**.
4. Click **Recompile**. Overleaf runs BibTeX automatically.

### Option 2: local TeX installation

Install TeX Live (Linux/Windows), MacTeX (macOS) or MiKTeX (Windows). On
Debian/Ubuntu:

```bash
sudo apt-get install texlive-latex-extra texlive-fonts-recommended texlive-pictures texlive-science latexmk
```

Then, from this folder:

```bash
latexmk -pdf mythesis.tex
```

Without `latexmk`, run the four passes by hand (needed for citations, the
table of contents and cross-references):

```bash
pdflatex mythesis.tex
bibtex mythesis
pdflatex mythesis.tex
pdflatex mythesis.tex
```

To remove the auxiliary files afterwards: `latexmk -c`.

Engine: pdfLaTeX. Bibliography: BibTeX with `ieeetr` (the template's style).
Packages beyond the template: `lmodern`, `microtype`, `bm`, `mathtools`,
`booktabs`, `tabularx`, `multirow`, `makecell`, `xcolor`, `float`, `caption`,
`subcaption`, `enumitem`, `listings`, `algorithm`, `algpseudocode`, `tikz`,
`url`, `hyperref`. All are in TeX Live's `latex-extra`, `pictures` and `science` collections.

## Project structure

```
thesis/
  mythesis.tex             main file (template layout; \input's everything)
  preamble.tex             template packages + additions + notation macros
  mybib.bib                bibliography (BibTeX, ~120 entries; only cited ones print)
  frontmatter/
    details.tex            <-- your names, roll numbers, advisor, date
    certificate.tex        bonafide certificate (template wording)
    declaration.tex        declaration + countersignature (template wording)
    acknowledgements.tex, abbreviations.tex, symbols.tex, abstract.tex
  chapters/
    chapter1_introduction.tex
    chapter2_literature_review.tex
    chapter3_problem_and_data.tex
    chapter4_pcmci_methodology.tex
    chapter5_graph_validation_moe.tex
    chapter6_implementation_experiments.tex
    chapter7_results_discussion.tex
    chapter8_phase2_extension.tex
    chapter9_conclusion.tex
  appendices/
    appendixA_notation_derivations.tex
    appendixB_configuration.tex
    appendixC_reproducibility.tex
  figures/                 result figures (PNG) from code/results/phase1 and phase9
  images/univA.png         university logo (from the template)
  scripts/make_stage_c_figures.py   builds figures/10_stage_c_lead1.png
  docs/
    research_decisions_and_gaps.md
    evidence_and_claim_audit.md
    compilation_qc_report.md
```

All schematic diagrams are drawn in TikZ inside the chapter files. All result
figures are PNGs generated from the project's saved result files (see
Appendix C, Table C.1).

## Regenerating the 1-day Stage C figure

From the repository root, with the `phase1-causal-graph` results available:

```bash
python thesis/scripts/make_stage_c_figures.py code/results/phase1 thesis/figures
```

## Template conventions kept

- `report` class, A4, 12 pt, 0.8 in margins, one-and-a-half spacing.
- Title page, bonafide certificate, declaration and countersignature wording
  unchanged, with personal fields replaced by placeholders.
- Roman page numbers for the front matter, Arabic from Chapter 1.
- Abbreviations and List of Symbols pages, chapters, appendices, `ieeetr`
  references and "List of Publications" page.
- Front matter order follows the department's accepted sample thesis
  (Acknowledgements before the Contents).
- The template file `images/univA.png` is referenced with the correct
  capitalisation (the template's certificate used `univa.png`, which fails on
  case-sensitive systems such as Linux and Overleaf).
