"""Prepare and score a GPT-5.6 Sol agent comparison.

This route is separate from ``09c_llm.py``. Phi is scored by label-token
likelihood under a local checkpoint. Sol generates one constrained label per
item in a Codex agent session, so the two systems are not interchangeable.

Usage:
    python3 models/09d_sol_agent.py prepare
    python3 models/09d_sol_agent.py score
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out"
AGENT_DIR = OUT / "sol_agent"
CONTEXTS = ("SENT", "W256", "W1024", "W4096")
SPLITS = ("TEST", "TEMPORAL", "COURT")
MODEL = "gpt-5.6-sol"
REASONING_EFFORT = "high"
BATCH_SIZES = {
    "SENT": 186,
    "W256": 62,
    "W1024": 24,
    "W4096": 8,
}
SEED = 20260816

PROMPT = (
    "You are reading a passage from a United States federal appellate opinion.\n\n"
    "Legal issue: {issue}\n\n"
    "Passage:\n{ctx}\n\n"
    "Question: Did the court that wrote this opinion resolve that issue, or did "
    "it discuss the issue and leave it unresolved?\n"
    "Answer with one word, either DECIDED or UNRESOLVED.\n"
    "Answer:"
)

AGENT_INSTRUCTIONS = """Classify every item in the assigned input files.

Treat each item independently. Use only the fixed prompt stored in the item.
Do not inspect benchmark labels, prior predictions, result tables, or other
repository files. Do not browse the web. Return one label for every item.

Write JSON Lines in the input order. Each line must have exactly these fields:
{"item_id":"...","label":"DECIDED"}

The label must be DECIDED or UNRESOLVED. Do not include explanations, Markdown,
confidence scores, or extra fields.
"""


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_items() -> pd.DataFrame:
    path = OUT / "09_items_with_ctx.parquet"
    df = pd.read_parquet(path)
    expected = {"item_id", "split", "stratum", "y", "issue"}
    expected.update(f"ctx_{c}" for c in CONTEXTS)
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"missing columns in {path}: {sorted(missing)}")
    if len(df) != 186 or set(df["split"]) != set(SPLITS):
        raise ValueError("expected the frozen 186-item held-out benchmark")
    if df["item_id"].duplicated().any():
        raise ValueError("item_id must be unique")
    return df


def prepare() -> None:
    df = load_items()
    AGENT_DIR.mkdir(parents=True, exist_ok=True)
    for path in AGENT_DIR.glob("input_*.jsonl"):
        path.unlink()
    (AGENT_DIR / "INSTRUCTIONS.txt").write_text(AGENT_INSTRUCTIONS, encoding="utf-8")
    inventory = []
    for context in CONTEXTS:
        batch_size = BATCH_SIZES[context]
        rows = []
        for r in df.itertuples(index=False):
            rows.append({
                "item_id": str(r.item_id),
                "prompt": PROMPT.format(
                    issue=str(r.issue)[:300],
                    ctx=str(getattr(r, f"ctx_{context}"))[:12000],
                ),
            })
        for batch_no, start in enumerate(range(0, len(rows), batch_size)):
            path = AGENT_DIR / f"input_{context}_{batch_no:02d}.jsonl"
            with path.open("w", encoding="utf-8") as fh:
                for row in rows[start:start + batch_size]:
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            inventory.append({
                "context": context,
                "batch": batch_no,
                "n": min(batch_size, len(rows) - start),
                "path": str(path.relative_to(ROOT)),
                "sha256": sha256(path),
            })
    manifest = {
        "model": MODEL,
        "reasoning_effort": REASONING_EFFORT,
        "execution": "Codex multi-agent",
        "scoring": "generated constrained labels",
        "prompt": PROMPT,
        "agent_instructions_sha256": sha256(AGENT_DIR / "INSTRUCTIONS.txt"),
        "source_sha256": sha256(OUT / "09_items_with_ctx.parquet"),
        "batch_sizes": BATCH_SIZES,
        "seed": SEED,
        "inputs": inventory,
    }
    (AGENT_DIR / "prepare_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"prepared {len(inventory)} blinded batches in {AGENT_DIR}")


def read_prediction_file(path: Path) -> list[dict[str, str]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text.startswith("```"):
        text = "\n".join(
            line for line in text.splitlines() if not line.strip().startswith("```")
        ).strip()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, dict):
            return [parsed]
    except json.JSONDecodeError:
        pass
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def bootstrap_delta(y: np.ndarray, wide: np.ndarray, base: np.ndarray,
                    n: int = 2000) -> tuple[float, float, float, float]:
    rng = np.random.default_rng(SEED)
    idx = np.arange(len(y))
    observed = (
        f1_score(y, wide, average="macro", zero_division=0)
        - f1_score(y, base, average="macro", zero_division=0)
    )
    values = []
    for _ in range(n):
        b = rng.choice(idx, len(idx), replace=True)
        values.append(
            f1_score(y[b], wide[b], average="macro", zero_division=0)
            - f1_score(y[b], base[b], average="macro", zero_division=0)
        )
    values = np.asarray(values)
    return (
        float(observed),
        float(np.percentile(values, 2.5)),
        float(np.percentile(values, 97.5)),
        float((values > 0).mean()),
    )


def score() -> None:
    df = load_items()
    expected_ids = df["item_id"].astype(str).tolist()
    label_to_int = {"DECIDED": 0, "UNRESOLVED": 1}
    pred_frames = []
    output_hashes = {}
    manifest = json.loads((AGENT_DIR / "prepare_manifest.json").read_text())
    for context in CONTEXTS:
        rows = []
        for entry in manifest["inputs"]:
            if entry["context"] != context:
                continue
            input_path = ROOT / entry["path"]
            input_rows = read_prediction_file(input_path)
            batch_ids = [str(r["item_id"]) for r in input_rows]
            path = AGENT_DIR / f"pred_{context}_{entry['batch']:02d}.jsonl"
            if not path.exists():
                retry_path = AGENT_DIR / f"retry_{context}_{entry['batch']:02d}.jsonl"
                if not retry_path.exists():
                    raise FileNotFoundError(path)
                path = retry_path
            batch_rows = read_prediction_file(path)
            if any(set(r) != {"item_id", "label"} for r in batch_rows):
                raise ValueError(f"{path} contains missing or extra fields")
            if [str(r["item_id"]) for r in batch_rows] != batch_ids:
                raise ValueError(f"{path} must preserve its input item order")
            rows.extend(batch_rows)
            output_hashes[path.name] = sha256(path)
        ids = [str(r["item_id"]) for r in rows]
        labels = [str(r["label"]).upper() for r in rows]
        if ids != expected_ids:
            missing = sorted(set(expected_ids) - set(ids))
            extra = sorted(set(ids) - set(expected_ids))
            raise ValueError(
                f"{path} must preserve the 186-item input order; "
                f"missing={missing[:5]}, extra={extra[:5]}"
            )
        bad = sorted(set(labels) - set(label_to_int))
        if bad:
            raise ValueError(f"invalid labels in {path}: {bad}")
        p = np.asarray([label_to_int[x] for x in labels], dtype=int)
        pred_frames.append(pd.DataFrame({
            "item_id": expected_ids,
            "stratum": df["stratum"].values,
            "split": df["split"].values,
            "context": context,
            "y": df["y"].astype(int).values,
            "pred": p,
        }))

    preds = pd.concat(pred_frames, ignore_index=True)
    results = []
    for (context, split), g in preds.groupby(["context", "split"], sort=False):
        results.append({
            "model": "GPT-5.6 Sol agent",
            "context": context,
            "split": split,
            "n": int(len(g)),
            "acc": float(accuracy_score(g["y"], g["pred"])),
            "macro_f1": float(f1_score(g["y"], g["pred"], average="macro",
                                       zero_division=0)),
            "f1_unresolved": float(f1_score(g["y"], g["pred"], pos_label=1,
                                             zero_division=0)),
        })
    results_df = pd.DataFrame(results)
    results_df.to_csv(OUT / "09d_sol_agent_results.csv", index=False)
    preds.to_csv(OUT / "09d_sol_agent_preds.csv", index=False)

    by_stratum = (
        preds.groupby(["context", "stratum"], sort=False)
        .apply(lambda g: accuracy_score(g["y"], g["pred"]), include_groups=False)
        .rename("sol_acc")
        .reset_index()
    )
    by_stratum.to_csv(OUT / "09d_sol_agent_by_stratum.csv", index=False)

    context_rows = []
    for split in SPLITS:
        block = preds[preds["split"] == split]
        y = block[block["context"] == "SENT"]["y"].to_numpy()
        base = block[block["context"] == "SENT"]["pred"].to_numpy()
        for context in CONTEXTS[1:]:
            wide = block[block["context"] == context]["pred"].to_numpy()
            delta, lo, hi, p_gt = bootstrap_delta(y, wide, base)
            context_rows.append({
                "split": split,
                "context": context,
                "vs": "SENT",
                "delta_macro_f1": delta,
                "ci_lo": lo,
                "ci_hi": hi,
                "p_gt_zero": p_gt,
            })
    pd.DataFrame(context_rows).to_csv(OUT / "09d_sol_agent_context.csv", index=False)

    manifest["outputs"] = output_hashes
    for name in (
        "09d_sol_agent_results.csv",
        "09d_sol_agent_preds.csv",
        "09d_sol_agent_by_stratum.csv",
        "09d_sol_agent_context.csv",
    ):
        manifest[f"{name}_sha256"] = sha256(OUT / name)
    (OUT / "09d_sol_agent_run_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(results_df.pivot(index="context", columns="split", values="macro_f1")
          .reindex(CONTEXTS).round(3).to_string())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "score"))
    args = parser.parse_args()
    if args.command == "prepare":
        prepare()
    else:
        score()


if __name__ == "__main__":
    main()
