"""Plot random-test macro-F1 across context widths for the learned systems."""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out"
PAPER_DIR = Path(os.environ.get("PAPER_DIR", ROOT / "paper"))
GEN = PAPER_DIR / "generated"
GEN.mkdir(parents=True, exist_ok=True)

CONTEXTS = ["SENT", "W256", "W1024", "W4096"]
LABELS = ["sentence", r"$\pm256$", r"$\pm1024$", r"$\pm4096$"]
MODELS = [
    ("tfidf-lr", "TF-IDF", "#6f6f6f", "o", "-"),
    ("distilroberta-base", "DistilRoBERTa", "#009E73", "s", "-"),
    ("legal-bert-base-uncased", "LEGAL-BERT", "#CC79A7", "^", "-"),
    ("Phi-3.5-mini-instruct", "Phi-3.5", "#D55E00", "D", "--"),
    ("GPT-5.6 Sol agent", "GPT-5.6 Sol", "#0072B2", "o", "-"),
]


def load_results() -> pd.DataFrame:
    frames = []
    for name in ("09_results.csv", "09b_encoder_results.csv",
                 "09c_llm_results.csv", "09d_sol_agent_results.csv"):
        path = OUT / name
        if path.exists():
            frames.append(pd.read_csv(path))
    if not frames:
        raise FileNotFoundError("no model result files found")
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    results = load_results()
    test = results[results["split"] == "TEST"]

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "STIXGeneral"],
        "mathtext.fontset": "stix",
        "font.size": 8,
        "axes.labelsize": 8,
        "legend.fontsize": 6.8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "pdf.fonttype": 42,
    })
    fig, ax = plt.subplots(figsize=(3.25, 2.05))
    x = list(range(len(CONTEXTS)))
    for model, label, color, marker, linestyle in MODELS:
        block = test[test["model"] == model].set_index("context")
        missing = [c for c in CONTEXTS if c not in block.index]
        if missing:
            raise ValueError(f"{model} missing contexts: {missing}")
        values = [float(block.loc[c, "macro_f1"]) for c in CONTEXTS]
        is_sol = model == "GPT-5.6 Sol agent"
        ax.plot(
            x,
            values,
            label=label,
            color=color,
            marker=marker,
            linestyle=linestyle,
            linewidth=2.0 if is_sol else 1.15,
            markersize=4.2 if is_sol else 3.4,
            markerfacecolor="white" if is_sol else color,
            markeredgewidth=1.0,
            zorder=5 if is_sol else 3,
        )

    ax.set_xticks(x, LABELS)
    ax.set_ylim(0.30, 1.00)
    ax.set_yticks([0.4, 0.6, 0.8, 1.0])
    ax.set_ylabel(r"random-test macro-$F_1$")
    ax.set_xlabel("context around the anchor (characters)")
    ax.grid(axis="y", color="#d8d8d8", linewidth=0.55)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(
        loc="lower left",
        bbox_to_anchor=(0.0, 1.01),
        ncol=2,
        frameon=False,
        handlelength=2.0,
        columnspacing=1.0,
        borderaxespad=0,
    )
    fig.tight_layout(pad=0.25)
    fig.savefig(GEN / "fig_model_context.pdf", bbox_inches="tight")
    fig.savefig(GEN / "fig_model_context.png", dpi=220, bbox_inches="tight")
    print(f"wrote {GEN / 'fig_model_context.pdf'}")


if __name__ == "__main__":
    main()
