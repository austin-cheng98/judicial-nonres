# Need Not Decide — code and artifacts

Pipeline code and released artifacts for *Need Not Decide: A Benchmark for
Judicial Non-Resolution in U.S. Federal Appellate Opinions*.

```
corpus/      01-05  parse the bulk release, apply the dictionary, prevalence
benchmark/   06-08  item frame, stratified sample, annotation worksheets
models/      09     sparse baselines, fine-tuned encoders, zero-shot LM
analysis/    10-21  citation chains, recall audit, escalation, matching
tables/      11-23  LaTeX tables and figures for the paper
lib/                shared modules: common, triggers, issues, textstore
out/                released artifacts: results, prevalence tables, benchmark
run_*.sh            pipeline runners
```

## The benchmark

`out/08_benchmark_labels.csv` holds all 364 annotated items as opinion
identifiers with character offsets, plus stratum, split, and label. It carries
no opinion text, so upstream corrections and removals propagate. The counts the
paper reports read off it directly: 364 annotated, 58 `UNCLEAR`, 306 with a
usable binary label, and 120 / 104 / 36 / 46 across dev, test, temporal, and
court-held-out.

## Results

Every number the paper reports comes from a file in `out/`:

| Claim | File |
|---|---|
| Model scores on the three held-out conditions | `09_results.csv`, `09b_encoder_results.csv`, `09c_llm_results.csv` |
| Accuracy by sampling stratum | `09_by_stratum.csv` |
| Context-width comparison | `09_context_test.csv` |
| Population prevalence, by court, period, type, status | `05_prev_*.csv`, `05_prevalence.json` |
| Learning curve on the adversarial stratum | `20_learning_curve.csv` |
| Within-annotator context ablation | `08_human_ablation.csv` |
| Judicial characterizations and escalation | `15_tight_stats.json`, `19_escalation.json`, `21_validation.json` |
| Search-index cross-check | `api_crosscheck.json` |

## Pipeline

Run in numeric order. Steps 01–05 stream the CourtListener bulk release of
30 June 2026 (58 GB, not included) and are the only expensive ones; everything
after works on derived artifacts. A few large intermediates are also omitted for
size.

| Step | What it does |
|---|---|
| 01–04 | Parse the bulk dockets, clusters, opinions, and citation tables |
| 05 | Apply the frozen trigger dictionary, compute population prevalence |
| 06–07 | Build the item frame, draw the stratified sample |
| 08 | Render annotation worksheets; ingest labels |
| 09 / 09b / 09c | Sparse baselines / fine-tuned encoders / zero-shot LM |
| 10a–10c | Citation chains and analysis |
| 13–14 | Dictionary recall audit, error analysis |
| 15–19, 21 | Judicial characterizations, escalation, proposition matching |

Needs Python 3.11+ with pandas, numpy, scipy, scikit-learn, matplotlib, and
pyarrow.

Support modules sit in `lib/`: `common.py` (bulk CSV streaming — the export escapes with
backslashes, so `csv.reader(fh, escapechar="\\")` is required), `triggers.py`
(the frozen 13-expression dictionary), `issues.py`, `textstore.py`,
`guidelines.py`. Every script adds `lib/` to its path, so run them from
anywhere.

`tables/` emits the LaTeX tables and figures for the paper, whose source is not
distributed here.

## Models

The zero-shot baseline is `microsoft/Phi-3.5-mini-instruct`, revision
`2fe192450127e6a83f7441aef6e3ca586c338b77`, float16 on one Tesla T4. Run
metadata and output hashes are in `out/09c_llm_run_manifest.json`. The model is
scored by comparing the summed log-probability of the ` DECIDED` and
` UNRESOLVED` continuations rather than by generating, so refusals and format
drift cannot vary with prompt length.
