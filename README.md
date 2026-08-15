# Need Not Decide — reproduction package

Code and released artifacts for *Need Not Decide: A Benchmark for Judicial
Non-Resolution in U.S. Federal Appellate Opinions*.

```
scripts/   pipeline, numbered in execution order
out/       released artifacts: results, prevalence tables, benchmark, labels
paper/     LaTeX source plus generated tables, macros, and figures
```

## Rebuild the paper

Needs Python 3.11+ with pandas, numpy, scipy, scikit-learn, matplotlib, and
pyarrow, plus [tectonic](https://tectonic-typesetting.github.io/).

```bash
python3 scripts/11_tables.py        # tables and macros from out/
python3 scripts/14_errors.py        # error-analysis examples
python3 scripts/23_pair_tables.py   # paired supplementary float
python3 scripts/12_figures.py
python3 scripts/22_more_figs.py
cd paper && tectonic -X compile main.tex --outdir build
```

`scripts/pagecheck.py` checks that content ends on page 8.

## The benchmark

`out/08_benchmark_labels.csv` holds all 364 annotated items as opinion
identifiers with character offsets, plus stratum, split, and label. It carries
no opinion text, so upstream corrections and removals propagate. The counts the
paper reports read off it directly: 364 annotated, 58 `UNCLEAR`, 306 with a
usable binary label, and 120 / 104 / 36 / 46 across dev, test, temporal, and
court-held-out.

## Rebuilding out/ from scratch

Steps 01–10 rebuild `out/` from the CourtListener bulk release of 30 June 2026
(58 GB, not included). Steps 05–07 stream that release; everything after works
on derived artifacts. A few large intermediates are also omitted for size, so
`11_tables.py` reports them as missing and skips the handful of appendix macros
that depend on them.

The zero-shot baseline is `microsoft/Phi-3.5-mini-instruct`, revision
`2fe192450127e6a83f7441aef6e3ca586c338b77`, float16 on one Tesla T4. Run
metadata and output hashes are in `out/09c_llm_run_manifest.json`.
