#!/bin/bash
set -e
cd "$(dirname "$0")/.."
unset OUTDIR
echo "$(date +%T) === step 3: opinions pass ==="
python3 scripts/corpus/03_opinions.py
echo "$(date +%T) === step 4: citation graph ==="
python3 scripts/corpus/04_citations.py
echo "$(date +%T) === step 5: prevalence ==="
python3 scripts/corpus/05_prevalence.py
echo "$(date +%T) === step 6: item frame ==="
python3 scripts/benchmark/06_frame.py
echo "$(date +%T) === step 7: benchmark sample ==="
python3 scripts/benchmark/07_sample.py
echo "$(date +%T) === step 13: recall audit ==="
python3 scripts/analysis/13_recall.py
echo "$(date +%T) MAIN_PIPELINE_DONE"
