"""Generate Figure 2: the data cleaning pipeline.

Run from the repo root:  python scripts/make_figure2_cleaning_flow.py
Output: figures/fig2_cleaning_flow.png (300 dpi) and .pdf (vector, for LaTeX).

Every number in the figure comes from docs/midterm_numbers.md, which is
generated from labels/final_labels.csv and the cleaning pipeline.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)

W, H = 4.7, 10.4
fig = plt.figure(figsize=(W, H))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis("off")

INK = "#1d1d1f"
GREY = "#6e6e73"
BLUE = "#0071e3"
GREEN = "#1e7d3a"
FILL = "#ffffff"
FILL_ACCENT = "#eef4fd"
FILL_FINAL = "#e3f5e9"

BOX_X, BOX_W, BOX_H = 0.32, 3.00, 0.72
GAP = 0.26


def box(y, title, body, accent=False, final=False):
    fill = FILL_FINAL if final else (FILL_ACCENT if accent else FILL)
    edge = GREEN if final else (BLUE if accent else "#c7c7cc")
    ax.add_patch(FancyBboxPatch(
        (BOX_X, y), BOX_W, BOX_H,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=0.9, edgecolor=edge, facecolor=fill, zorder=2))
    ax.text(BOX_X + 0.12, y + BOX_H - 0.14, title,
            fontsize=7.0, fontweight="bold", color=INK, va="top", ha="left", zorder=3)
    ax.text(BOX_X + 0.12, y + BOX_H - 0.34, body,
            fontsize=6.4, color=GREY, va="top", ha="left", linespacing=1.45, zorder=3)


def arrow(y_from, y_to):
    ax.add_patch(FancyArrowPatch(
        (BOX_X + BOX_W / 2, y_from), (BOX_X + BOX_W / 2, y_to),
        arrowstyle="-|>", mutation_scale=9, linewidth=0.9, color="#8e8e93", zorder=1))


def note(y, text, color=GREY, size=6.0):
    ax.text(BOX_X + BOX_W + 0.09, y, text, fontsize=size, color=color,
            va="top", ha="left", linespacing=1.35)


steps = [
    ("Raw MTSamples", "4,999 notes, about 40 specialty labels\nStep 1: strip label whitespace (no rows lost)", False, False),
    ("Step 2  Remove notes with no transcription", "-33 rows  ->  4,966", False, False),
    ("Step 3  Remove 6 document-type labels",
     "-938 rows  ->  4,028\n(Consult, SOAP, Discharge Summary, ER, Office, Letters)", False, False),
    ("Step 4  Remove the Surgery label",
     "-1,088 rows  ->  2,940\n1,039 are byte-identical copies of notes filed under a specialty", False, False),
    ("Step 5  Deduplicate to unique normalized texts",
     "2,279 unique texts\n1,701 single-label + 578 conflicting", False, False),
    ("Step 6  Manual review of conflicting notes",
     "606 reviewed (578 conflicting + 28 surgery-only)\n602 assigned, 4 DROP; 3 corrected after audit", False, False),
    ("Step 7  Map to 11 classes and exclude 97",
     "65 no specialty home + 4 DROP\n+ 21 general surgery + 7 thyroid surgery", False, False),
    ("Final dataset", "2,230 notes, 11 classes", True, True),
    ("Frozen split (seed 42, stratified 70/15/15)",
     "train 1,560  |  val 335  |  test 335\ntest read once, in the final evaluation only", False, False),
]

y_top = H - 0.55 - BOX_H
ys = []
for i, (title, body, accent, final) in enumerate(steps):
    y = y_top - i * (BOX_H + GAP)
    box(y, title, body, accent=accent, final=final)
    ys.append(y)

for i in range(len(ys) - 1):
    arrow(ys[i], ys[i + 1] + BOX_H)

# side notes, aligned to the top of their box
note(ys[1] + BOX_H - 0.06, "no text = no signal")
note(ys[2] + BOX_H - 0.06, "document type is not\na routing destination")
note(ys[3] + BOX_H - 0.06, "Surgery is a procedure,\nnot a specialty")
note(ys[4] + BOX_H - 0.06, "95.9% of raw rows\nduplicate another text")
note(ys[4] + BOX_H - 0.60, "+48 surgery-only texts\n= 2,327 candidates", color=BLUE, size=6.3)
note(ys[5] + BOX_H - 0.06, "keyword-rule suggestions\nagreed 48.5% with humans")
note(ys[6] + BOX_H - 0.06, "no routing class = excluded")

# footer: independent validation
fy = 0.06
ax.add_patch(FancyBboxPatch(
    (BOX_X, fy), BOX_W, 0.74,
    boxstyle="round,pad=0.02,rounding_size=0.07",
    linewidth=0.8, edgecolor="#d2d2d7", facecolor="#fafafa", linestyle="--", zorder=2))
ax.text(BOX_X + 0.12, fy + 0.64,
        "Independent validation (not part of the pipeline)\n"
        "Blind audit of 220 notes. Human-reviewed stratum:\n"
        "kappa 0.847 (60 notes). Auto-mapped stratum: error\n"
        "rate 13.8%, or 6.9% excluding the 11 disagreements\n"
        "from labels merged into General Medicine (160 notes).",
        fontsize=5.6, color=GREY, va="top", ha="left", linespacing=1.3, zorder=3)

ax.text(0.06, H - 0.22, "Figure 2. Data cleaning pipeline",
        fontsize=8.8, fontweight="bold", color=INK, va="center", ha="left")

fig.savefig(OUT / "fig2_cleaning_flow.png", dpi=300, facecolor="white")
fig.savefig(OUT / "fig2_cleaning_flow.pdf", facecolor="white")
print(f"saved: {OUT / 'fig2_cleaning_flow.png'}")
print(f"saved: {OUT / 'fig2_cleaning_flow.pdf'}")
