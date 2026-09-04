# ICLR workshop LaTeX draft

Anonymous draft using the official **ICLR 2027** style files ([author guidelines](https://iclr.cc/Conferences/2027/AuthorGuidelines)).

## Build

```bash
cd paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Requires a TeX distribution with `iclr2027_conference.sty` companions in this folder (`natbib`, `fancyhdr`, `math_commands.tex`).

## Files

| File | Role |
|------|------|
| `main.tex` | Workshop draft (anonymous) |
| `references.bib` | Bibliography (some placeholders TBD) |
| `figures/progress_board.png` | Mean-metric board (no 9-fold / ETA / queue) |
| `figures/board_summary.json` | Numeric snapshot behind the table |
| `iclr2027/` | Untouched upstream style package copy |

Update Table 1 / figure from a live board:

```bash
python3 ../scripts/export_readme_board.py \
  --board ../vis/iclr2026_aligned_fdg_fs50_f258_board.json \
  --png figures/progress_board.png \
  --out-json figures/board_summary.json
```
