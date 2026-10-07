#!/bin/bash
# Post-experiment packaging: report tables/figures -> paper -> PDF -> project site.
set -u
cd "$(dirname "$0")/.." || exit 1
PY=/usr/bin/python3
D=latex-paper/dtajepa-paper

$PY -m dtajepa.report --in results/raw.jsonl --out results
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
