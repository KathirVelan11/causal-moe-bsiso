# Compilation and QC report

- Build: `latexmk -pdf mythesis.tex` (pdfLaTeX + BibTeX `ieeetr`), exit 0.
- Output: 67 pages (front matter i–xiii, 9 chapters, Appendix A, references, publications page).
- References: 35, all cited; no BibTeX warnings.
- No LaTeX errors, undefined or duplicate labels, missing files, or overfull boxes.
- Every figure and table is referenced in the text.
- All 22 figures inspected after rendering; workflow diagrams checked for
  overlapping labels and arrows.
- Template preserved: class, margins, spacing, title page, certificate,
  declaration, countersignature, page numbering, `ieeetr`.
