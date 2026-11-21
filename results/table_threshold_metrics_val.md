# Sensitivity and specificity at a 0.5 probability threshold (validation)

One-vs-rest, per class, probability threshold 0.5 (not the argmax rule used for macro-F1).
Macro averages are over the 11 classes; `n/a` where a class has no positive or no negative note.

| Model | Macro sensitivity | Macro specificity | Classes with sensitivity defined |
|---|---:|---:|---:|
| Majority class (one-hot, degenerate) | 0.091 | 0.909 | 11/11 |
| Keyword rule (one-hot, degenerate) | 0.177 | 0.912 | 11/11 |
| TF-IDF + logistic regression | 0.774 | 0.992 | 11/11 |
| TF-IDF + linear SVM | 0.820 | 0.992 | 11/11 |
| Bio_ClinicalBERT seed 42 | 0.706 | 0.991 | 11/11 |
| Bio_ClinicalBERT seed 43 | 0.639 | 0.995 | 11/11 |
| Bio_ClinicalBERT seed 44 | 0.707 | 0.992 | 11/11 |

The two one-hot models predict 0 or 1 for every class, so the 0.5 threshold coincides with their hard prediction: their sensitivity is the per-class recall of that hard prediction and their specificity its true-negative rate. Both are valid numbers, but the notion of a probability threshold is degenerate for them, and their ECE is not meaningful. Full per-class values: `results/metrics/threshold_metrics_val.csv`.
