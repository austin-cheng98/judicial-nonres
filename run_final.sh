#!/bin/bash
# Layer-two analysis, figures, tables, and the compiled paper.
set -e
cd "$(dirname "$0")"
unset OUTDIR
echo "$(date +%T) === layer-two ingest and analysis ==="
python3 analysis/10b_worksheet.py ingest
python3 analysis/10c_analysis.py
echo "$(date +%T) === human context ablation ==="
python3 benchmark/08_worksheet.py ablation || echo "(no ablation labels yet)"
echo "$(date +%T) === error analysis ==="
python3 tables/14_errors.py || echo "(no error cases)"
echo "$(date +%T) === figures and tables ==="
python3 tables/12_figures.py
python3 tables/11_tables.py
echo "$(date +%T) === compile ==="
cd paper && ../tectonic -X compile main.tex --outdir build
cd .. && python3 tables/pagecheck.py
echo "$(date +%T) FINAL_DONE"
