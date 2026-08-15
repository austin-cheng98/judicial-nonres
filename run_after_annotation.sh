#!/bin/bash
# Everything downstream of layer-one annotation. Model steps are ordered cheapest
# first so that a failure late in the run still leaves a reportable paper.
set -e
cd "$(dirname "$0")"
unset OUTDIR

echo "$(date +%T) === ingest layer-one labels ==="
python3 benchmark/08_worksheet.py ingest

echo "$(date +%T) === step 9: sparse models and context test ==="
python3 models/09_models.py

echo "$(date +%T) === step 10a: issue-specific citation chains ==="
python3 analysis/10a_chains.py

echo "$(date +%T) === step 9c: zero-shot language model ==="
python3 models/09c_llm.py || echo "!! 09c failed, continuing"

echo "$(date +%T) === step 9b: fine-tuned encoders ==="
python3 models/09b_encoder.py || echo "!! 09b failed, continuing"

echo "$(date +%T) AFTER_ANNOTATION_DONE"
