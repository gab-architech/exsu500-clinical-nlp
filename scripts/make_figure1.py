"""Figure 1: class distribution of the final dataset (midterm report).

Reads labels/final_labels.csv (text_id, label; no note text) and saves a
horizontal bar chart sized for one IEEE column (3.5 in wide), readable in
black and white:

    python scripts/make_figure1.py

Output: figures/fig1_class_distribution.png (300 dpi)
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display needed (works in Colab and locally)
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]  # repo root, so paths work from any folder
LABELS_PATH = ROOT / "labels" / "final_labels.csv"
FIG_PATH = ROOT / "figures" / "fig1_class_distribution.png"


def main():
    labels = pd.read_csv(LABELS_PATH)
    counts = labels["label"].value_counts().sort_values()  # smallest at the bottom of the list, largest on top
    total = counts.sum()

    # Single series, so one dark gray fill: prints the same in color and black and white
    fig, ax = plt.subplots(figsize=(3.5, 2.9))
    bars = ax.barh(counts.index, counts.values, color="0.35", height=0.7)

    # Direct labels: count and share at the end of each bar (no legend needed)
    for bar, n in zip(bars, counts.values):
        ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height() / 2,
                f"{n} ({n / total:.1%})", va="center", fontsize=6.5)

    ax.set_xlabel(f"Notes (n = {total:,})", fontsize=7.5)
    ax.tick_params(axis="both", labelsize=7)
    ax.set_xlim(0, counts.max() * 1.5)  # room for the direct labels
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    ax.xaxis.grid(True, color="0.88", linewidth=0.5)
    ax.set_axisbelow(True)

    fig.tight_layout()
    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_PATH, dpi=300, bbox_inches="tight")  # keep long labels inside the image
    print(f"Saved {FIG_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
