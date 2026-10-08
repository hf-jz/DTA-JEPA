#!/bin/bash
# Post-experiment packaging: report tables/figures -> paper -> PDF -> project site.
set -u
cd "$(dirname "$0")/.." || exit 1
PY=/usr/bin/python3
D=latex-paper/dtajepa-paper

$PY -m dtajepa.report --in results/raw.jsonl --out results
for extra in parity_curves parity_sweep parity_cem; do
  [ -f results/$extra.jsonl ] && $PY -m dtajepa.report --in results/$extra.jsonl --out results/parity
done
[ -f results/parity/figs/fig_success_curves.png ] && mkdir -p results/figs && cp -f results/parity/figs/fig_success_curves.png results/figs/fig_parity_curves.png
$PY -m dtajepa.site

mkdir -p "$D/figs"
cp -f results/tables.tex "$D/results-tables.tex"
cp -f results/figs/*.png "$D/figs/" 2>/dev/null
cd "$D" && pdflatex -interaction=nonstopmode dtajepa-paper-en.tex >/dev/null 2>&1
pdflatex -interaction=nonstopmode dtajepa-paper-en.tex >/dev/null 2>&1
cd - >/dev/null
cp -f "$D/dtajepa-paper-en.pdf" docs/paper.pdf
echo "paper: $(ls -la docs/paper.pdf | awk '{print $5}') bytes, $(pdfinfo docs/paper.pdf 2>/dev/null | grep Pages || echo 'pages=?')"
echo "site:  $(ls -la docs/index.html | awk '{print $5}') bytes"
