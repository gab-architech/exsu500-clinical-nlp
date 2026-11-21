"""Per-class sensitivity and specificity at a stated threshold.

The evaluation protocol in CLAUDE.md asks for sensitivity and specificity at a stated
threshold. This computes, for every model and every class, one-vs-rest sensitivity (recall
at the threshold) and specificity at a probability threshold of 0.5 on that class's
probability, and writes a long CSV plus a compact markdown table.

    python scripts/class_threshold_metrics.py

Threshold 0.5 is stated here and used everywhere: a note counts as predicted positive for a
class when the model's probability for that class is at least 0.5. This is NOT the argmax
decision rule used for macro-F1, so the numbers differ from the per-class recall that
`src/evaluate.py` reports; the two are kept separate on purpose.

For the two one-hot models (majority class, keyword rule) the probabilities are 0 or 1, so
the threshold is degenerate: specificity is 1 whenever the model ever predicts the class and
0 otherwise, and sensitivity is whatever the hard prediction gives. They are computed and
labelled so the omission is visible rather than silent.
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
THRESHOLD = 0.5

MODELS = [
    ("majority_42", "Majority class"),
    ("keyword_rule_42", "Keyword rule"),
    ("tfidf_lr_42", "TF-IDF + logistic regression"),
    ("tfidf_svm_42", "TF-IDF + linear SVM"),
    ("clinicalbert_42", "Bio_ClinicalBERT seed 42"),
    ("clinicalbert_43", "Bio_ClinicalBERT seed 43"),
    ("clinicalbert_44", "Bio_ClinicalBERT seed 44"),
]
ONE_HOT = {"majority_42", "keyword_rule_42"}


def main():
    classes = sorted(pd.read_csv(ROOT / "labels" / "final_labels.csv")["label"].unique())
    rows = []
    for stem, label in MODELS:
        path = RESULTS / f"preds_{stem}_val.csv"
        if not path.exists():
            raise FileNotFoundError(f"missing prediction file {path}; every model in MODELS must exist")
        df = pd.read_csv(path, dtype={"note_id": str})
        y = df["true_label"].to_numpy()
        for c in classes:
            prob = df[f"prob_{c}"].to_numpy()
            pred_pos = prob >= THRESHOLD
            true_pos = y == c
            tp = int((pred_pos & true_pos).sum())
            fn = int((~pred_pos & true_pos).sum())
            tn = int((~pred_pos & ~true_pos).sum())
            fp = int((pred_pos & ~true_pos).sum())
            sens = tp / (tp + fn) if (tp + fn) else np.nan
            spec = tn / (tn + fp) if (tn + fp) else np.nan
            rows.append({
                "model": stem, "model_label": label, "class": c,
                "support": int(true_pos.sum()),
                "sensitivity": round(float(sens), 4) if sens == sens else None,
                "specificity": round(float(spec), 4) if spec == spec else None,
                "one_hot": stem in ONE_HOT,
            })

    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "metrics" / "threshold_metrics_val.csv", index=False)

    # Compact markdown: macro averages over classes with support, per model.
    lines = ["# Sensitivity and specificity at a 0.5 probability threshold (validation)",
             "",
             "One-vs-rest, per class, probability threshold 0.5 (not the argmax rule used for macro-F1).",
             "Macro averages are over the 11 classes; `n/a` where a class has no positive or no negative note.",
             "",
             "| Model | Macro sensitivity | Macro specificity | Classes with sensitivity defined |",
             "|---|---:|---:|---:|"]
    for stem, label in MODELS:
        sub = out[out.model == stem]
        if sub.empty:
            continue
        sens = sub["sensitivity"].dropna()
        spec = sub["specificity"].dropna()
        tag = " (one-hot, degenerate)" if stem in ONE_HOT else ""
        lines.append(f"| {label}{tag} | {sens.mean():.3f} | {spec.mean():.3f} | {len(sens)}/11 |")
    lines += ["",
              "The two one-hot models predict 0 or 1 for every class, so the 0.5 threshold "
              "coincides with their hard prediction: their sensitivity is the per-class recall "
              "of that hard prediction and their specificity its true-negative rate. Both are "
              "valid numbers, but the notion of a probability threshold is degenerate for them, "
              "and their ECE is not meaningful. "
              "Full per-class values: `results/metrics/threshold_metrics_val.csv`."]
    (RESULTS / "table_threshold_metrics_val.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
