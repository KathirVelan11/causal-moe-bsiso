# Thesis: Causal Graph Discovery and Graph-Guided Mixture-of-Experts

B.Tech thesis (Department of Artificial Intelligence, Amrita Vishwa Vidyapeetham,
Coimbatore), built on the department's LaTeX template (`Thesis_Template___Amrita_AID`).

The compiled thesis is `mythesis.pdf` (67 pages, 35 references).

## Before submission

Personal and official fields are in **one file**, `frontmatter/details.tex`.
Names, roll numbers, team number (AB 13) and advisor (Dr. Vinith) are filled in
from the S7 review deck. Still to fill: the advisor's **designation** and the
**submission month/year**.

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
  mythesis.tex            main file (template layout)
  preamble.tex            template packages + additions
  mybib.bib               35 references (the 7 core review papers first)
  frontmatter/            details.tex (names etc.), certificate, declaration,
                          acknowledgements, abbreviations, symbols, abstract
  chapters/               chapter1 ... chapter9
  appendices/             appendixA_reproducibility.tex
  figures/                result figures (PNG); ppt_* are the review-deck figures
  figures/workflows/      workflow diagrams: TikZ sources (*.tex), PDFs used by
                          the thesis, and 300-dpi PNGs in png/ for slides
  scripts/                make_stage_c_figures.py, make_lifecycle_figure.py
  docs/                   decisions/gaps, claim audit, QC report
```

## Workflow images

`figures/workflows/` holds five diagrams in one visual style:
`wf_overview` (two-phase roadmap), `wf_phase1` (Phase 1 pipeline),
`wf_validation` (B1–B4 checks and the three ways to use the graph),
`wf_lifecycle` (drift detection and expert lifecycle) and `wf_phase2`
(India-focused Phase 2). Edit a `.tex` file and rebuild it with:

```bash
cd figures/workflows && pdflatex wf_phase1.tex && pdftoppm -r 300 -png -singlefile wf_phase1.pdf png/wf_phase1
```

## Regenerating generated figures

From the repository root:

```bash
python thesis/scripts/make_stage_c_figures.py code/results/phase1 thesis/figures   # phase1-causal-graph branch
python thesis/scripts/make_lifecycle_figure.py code/results thesis/figures
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
- The list of algorithms was dropped to keep the thesis near 70 pages.
