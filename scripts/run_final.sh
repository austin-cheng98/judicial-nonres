#!/bin/bash
# Layer-two analysis, figures, tables, and the compiled paper.
set -e
cd "$(dirname "$0")/.."
unset OUTDIR
echo "$(date +%T) === layer-two ingest and analysis ==="
python3 scripts/analysis/10b_worksheet.py ingest
python3 scripts/analysis/10c_analysis.py
echo "$(date +%T) === human context ablation ==="
python3 scripts/benchmark/08_worksheet.py ablation || echo "(no ablation labels yet)"
echo "$(date +%T) === error analysis ==="
python3 scripts/paper/14_errors.py || echo "(no error cases)"
echo "$(date +%T) === figures and tables ==="
python3 scripts/paper/12_figures.py
python3 scripts/paper/11_tables.py
echo "$(date +%T) === compile ==="
cd paper && ../tectonic -X compile main.tex --outdir build
cd .. && python3 scripts/paper/pagecheck.py
echo "$(date +%T) FINAL_DONE"
