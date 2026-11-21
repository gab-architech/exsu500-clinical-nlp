# Leakage experiment: original vs label-masked text (validation)

| Model | Macro-F1 original | Macro-F1 masked | Difference |
|---|---:|---:|---:|
| Majority class | 0.025 | 0.025 | +0.0000 |
| Keyword rule (MI, 2 per class) | 0.157 | 0.157 | +0.0000 |
| TF-IDF + logistic regression | 0.862 | 0.862 | -0.0004 |
| TF-IDF + linear SVM (Platt) | 0.875 | 0.874 | -0.0007 |
| Bio_ClinicalBERT (3 seeds) | 0.805 | 0.805 | +0.0003 |

Masking removes 24 specialty names (option C) and touches 10.8% of notes. The keyword rule and the majority baseline are unchanged by construction: none of their 22 keywords is in the mask list, so their identical rows carry no evidence about leakage.
