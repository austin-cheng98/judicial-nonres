"""Step 4: citation graph, population only.

Edges kept when the cited opinion is in the population. Whether the citer is too
is recorded rather than filtered, so the out-of-population share is reportable.
"""
import os, sys
import pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "lib"))
from common import DATA, OUT, stream_csv

meta = pd.read_parquet(os.path.join(OUT, "03_opinions_meta.parquet"),
                       columns=["opinion_id"])
pop = set(meta["opinion_id"].tolist())
print(f"population opinions: {len(pop):,}", flush=True)

rows = []
n = 0
for r in stream_csv(os.path.join(DATA, "citation-map.csv.bz2"),
                    want=["depth", "cited_opinion_id", "citing_opinion_id"],
                    progress_every=20_000_000, label="citations"):
    n += 1
    try:
        cited = int(r["cited_opinion_id"])
    except (ValueError, TypeError):
        continue
    if cited not in pop:
        continue
    try:
        citing = int(r["citing_opinion_id"])
    except (ValueError, TypeError):
        continue
    rows.append((citing, cited, int(r["depth"] or 1), citing in pop))

g = pd.DataFrame(rows, columns=["citing_id", "cited_id", "depth", "citing_in_pop"])
print(f"total edges scanned: {n:,}")
print(f"edges into population: {len(g):,}")
print(f"  citing opinion also in population: {g['citing_in_pop'].mean():.3f}")
g.to_parquet(os.path.join(OUT, "04_citations.parquet"), index=False)

indeg = g.groupby("cited_id").size().rename("n_citers")
indeg.to_frame().to_parquet(os.path.join(OUT, "04_indegree.parquet"))
print(indeg.describe())
