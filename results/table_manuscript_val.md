# Model comparison, validation split (original text)

| Model | Macro-F1 (95% CI) | Accuracy (95% CI) | AUROC | AUPRC | ECE |
|---|---|---|---:|---:|---:|
| Majority class | 0.025 (0.019-0.030) | 0.161 (0.119-0.200) | degenerate | degenerate | degenerate |
| Keyword rule (MI, 2 per class) | 0.157 (0.101-0.202) | 0.149 (0.113-0.188) | degenerate | degenerate | degenerate |
| TF-IDF + logistic regression | 0.862 (0.816-0.899) | 0.854 (0.818-0.890) | 0.985 | 0.922 | 0.059 |
| TF-IDF + linear SVM (Platt) | 0.875 (0.831-0.910) | 0.875 (0.842-0.907) | 0.986 | 0.924 | 0.102 |

95% bootstrap CIs, 1000 resamples, seed 42, identical draws for every model. "Degenerate" marks models whose probabilities are one-hot, for which AUROC, AUPRC and ECE are not meaningful.

Near-duplicates: 2 of the 335 validation notes have a near-duplicate (cosine >= 0.8) in the training split. The evaluator's "no near-duplicates" subset removes 4 validation notes, because it also drops notes whose only partner is in the test split; those cannot leak into a validation score. Removing them changes macro-F1 by at most 0.0023.

Logistic regression vs linear SVM: the difference in macro-F1 is +0.0127 (paired bootstrap 95% CI -0.0098 to +0.0361); on 2 notes only logistic regression is right and on 9 only the SVM is (exact McNemar p = 0.065). The two are statistically indistinguishable on macro-F1 at this sample size, so the table does not rank them; logistic regression has the lower calibration error.

The logistic-regression regularisation strength C was selected on this same validation split from a pre-declared grid (best C = 500, macro-F1 0.8623); the reported score is therefore a selection maximum and its confidence interval does not carry the selection uncertainty. Evidence: results/lr_c_sweep_42_val.json.
