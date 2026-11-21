# Model comparison, val split

| Model | Notes | Seeds | Macro-F1 (95% CI) | SD | Accuracy (95% CI) | AUROC | AUPRC | ECE |
|---|---:|---:|---|---:|---|---:|---:|---:|
| clinicalbert | 335 | 3 | 0.805 (0.758 to 0.843) | 0.014 | 0.816 (0.776 to 0.854) | 0.980 | 0.889 | 0.208 |
| clinicalbert (no near-duplicates) | 331 | 3 | 0.805 (0.761 to 0.843) | 0.015 | 0.820 (0.783 to 0.855) | 0.980 | 0.888 | 0.209 |
| clinicalbert_masked | 335 | 3 | 0.805 (0.757 to 0.844) | 0.014 | 0.817 (0.778 to 0.856) | 0.979 | 0.887 | 0.237 |
| clinicalbert_masked (no near-duplicates) | 331 | 3 | 0.807 (0.761 to 0.842) | 0.013 | 0.822 (0.784 to 0.858) | 0.979 | 0.886 | 0.240 |
| keyword_rule | 335 | 1 | 0.157 (0.101 to 0.202) |  | 0.149 (0.113 to 0.188) | 0.545 | 0.161 | 0.851 |
| keyword_rule (no near-duplicates) | 331 | 1 | 0.157 (0.103 to 0.200) |  | 0.151 (0.115 to 0.190) | 0.545 | 0.162 | 0.849 |
| keyword_rule_masked | 335 | 1 | 0.157 (0.101 to 0.202) |  | 0.149 (0.113 to 0.188) | 0.545 | 0.161 | 0.851 |
| keyword_rule_masked (no near-duplicates) | 331 | 1 | 0.157 (0.103 to 0.200) |  | 0.151 (0.115 to 0.190) | 0.545 | 0.162 | 0.849 |
| majority | 335 | 1 | 0.025 (0.019 to 0.030) |  | 0.161 (0.119 to 0.200) | 0.500 | 0.091 | 0.839 |
| majority (no near-duplicates) | 331 | 1 | 0.026 (0.020 to 0.031) |  | 0.163 (0.124 to 0.205) | 0.500 | 0.091 | 0.837 |
| majority_masked | 335 | 1 | 0.025 (0.019 to 0.030) |  | 0.161 (0.119 to 0.200) | 0.500 | 0.091 | 0.839 |
| majority_masked (no near-duplicates) | 331 | 1 | 0.026 (0.020 to 0.031) |  | 0.163 (0.124 to 0.205) | 0.500 | 0.091 | 0.837 |
| tfidf_lr | 335 | 1 | 0.862 (0.816 to 0.899) |  | 0.854 (0.818 to 0.890) | 0.985 | 0.922 | 0.059 |
| tfidf_lr (no near-duplicates) | 331 | 1 | 0.860 (0.817 to 0.897) |  | 0.852 (0.813 to 0.891) | 0.984 | 0.918 | 0.060 |
| tfidf_lr_masked | 335 | 1 | 0.862 (0.815 to 0.898) |  | 0.854 (0.818 to 0.890) | 0.984 | 0.920 | 0.071 |
| tfidf_lr_masked (no near-duplicates) | 331 | 1 | 0.860 (0.817 to 0.896) |  | 0.852 (0.813 to 0.891) | 0.984 | 0.915 | 0.072 |
| tfidf_svm | 335 | 1 | 0.875 (0.831 to 0.910) |  | 0.875 (0.842 to 0.907) | 0.986 | 0.924 | 0.102 |
| tfidf_svm (no near-duplicates) | 331 | 1 | 0.873 (0.830 to 0.910) |  | 0.873 (0.837 to 0.909) | 0.986 | 0.921 | 0.101 |
| tfidf_svm_masked | 335 | 1 | 0.874 (0.831 to 0.909) |  | 0.875 (0.839 to 0.907) | 0.986 | 0.925 | 0.102 |
| tfidf_svm_masked (no near-duplicates) | 331 | 1 | 0.872 (0.829 to 0.910) |  | 0.873 (0.837 to 0.909) | 0.986 | 0.921 | 0.102 |
| CLiGNet benchmark* | | | 0.279 | | | | | |

\* Reference only (Barman and Barman 2026): 40 raw MTSamples labels and a different split, so not directly comparable with our 11 classes.
Bootstrap: 1000 resamples, seed 42, same draws for every model. With several seeds, the CI is pooled: the seeds' scores are averaged on each draw. "No near-duplicates" excludes the 4 val notes listed in `results\near_duplicate_pairs.csv`.
