#!/bin/bash
# Layer-two analysis and final validation outputs.
set -e
cd "$(dirname "$0")"
unset OUTDIR
echo "$(date +%T) === layer-two ingest and analysis ==="
python3 analysis/10b_worksheet.py ingest
python3 analysis/10c_analysis.py
echo "$(date +%T) === human context ablation ==="
python3 benchmark/08_worksheet.py ablation || echo "(no ablation labels yet)"
echo "$(date +%T) FINAL_DONE"
