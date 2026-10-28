# Final review deck (updated)

`Final_Review_Updated.pptx` (28 slides, 4:3) follows the original S7 final-review
deck's layout and order, updated with the current Phase 1 results and the
thesis workflow figures. `Final_Review_Updated.pdf` is a PDF export.

Rebuild (needs Node with `pptxgenjs`, and Python with `lxml`):

```bash
node build_deck.js ../thesis/figures Final_Review_Updated.pptx
python fix_ppr.py Final_Review_Updated.pptx
```

`build_deck.js` also applies the deck theme via the pptx skill's
`apply_theme.js` (set `APPLY_THEME` to its path). `fix_ppr.py` removes the
duplicate paragraph-property blocks pptxgenjs writes for mixed bold/plain
bullet items.
