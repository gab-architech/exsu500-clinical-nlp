"""Figure 3: model comparison, macro-F1 with 95% bootstrap CI (final manuscript).

Reads results/summary_{split}.csv written by src/evaluate.py (no note text) and
plots the original-text models, all notes, as points with horizontal CI bars.
One IEEE column wide (3.5 in), grayscale so it prints in black and white.

    python scripts/make_figure3_comparison.py               # validation (draft)
    python scripts/make_figure3_comparison.py --split test  # after the Nov 25 lock

Output: figures/fig3_model_comparison_{split}.png (300 dpi)
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display needed (works in Colab and locally)
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]  # repo root, so paths work from any folder

# Display order (top to bottom) and names. Masked twins are left out: the leakage
# experiment has its own table (results/table_masking_val.md).
MODELS = {
    "tfidf_svm": "TF-IDF + linear SVM",
    "tfidf_lr": "TF-IDF + logistic regression",
    "clinicalbert": "Bio_ClinicalBERT (3 seeds)",
    "keyword_rule": "Keyword rule",
    "majority": "Majority class",
}


def main():
    parser = argparse.ArgumentParser(description="Figure 3: macro-F1 with 95% CI per model.")
    parser.add_argument("--split", choices=["val", "test"], default="val")
    args = parser.parse_args()

    summary_path = ROOT / "results" / f"summary_{args.split}.csv"
    if not summary_path.exists():
        raise SystemExit(f"{summary_path.name} not found; run src/evaluate.py --split {args.split} first")
    summary = pd.read_csv(summary_path)
    rows = summary[(summary["subset"] == "all") & summary["model"].isin(MODELS)]
    rows = rows.set_index("model").loc[[m for m in MODELS if m in set(rows["model"])]]
    if rows.empty:
        raise SystemExit(f"no original-text models in summary_{args.split}.csv; run src/evaluate.py first")

    fig, ax = plt.subplots(figsize=(3.5, 1.9))
    y = list(range(len(rows)))[::-1]  # first model on top
    mean = rows["macro_f1_mean"]
    # Error bars are the distance from the point to each CI bound
    err = [mean - rows["macro_f1_ci_low"], rows["macro_f1_ci_high"] - mean]
    ax.errorbar(mean, y, xerr=err, fmt="o", color="0.2", markersize=3.5, capsize=2.5, linewidth=1)

    # Direct labels: value and CI to the right of each bar (no legend needed)
    for yi, (_, r) in zip(y, rows.iterrows()):
        ax.text(r["macro_f1_ci_high"] + 0.02, yi,
                f"{r['macro_f1_mean']:.3f} ({r['macro_f1_ci_low']:.3f} to {r['macro_f1_ci_high']:.3f})",
                va="center", fontsize=5.5)

    ax.set_yticks(y, [MODELS[m] for m in rows.index], fontsize=6.5)
    ax.set_xlim(0, 1.35)  # room for the labels on the right
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.tick_params(axis="x", labelsize=6.5)
    split_name = "validation" if args.split == "val" else "test"
    ax.set_xlabel(f"Macro-F1, {split_name} split (n = {int(rows['n_notes'].iloc[0])}), 95% bootstrap CI", fontsize=6.5)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)

    out = ROOT / "figures" / f"fig3_model_comparison_{args.split}.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
