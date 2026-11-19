# Contributions and disclosures

## CRediT contribution statement

Official CRediT roles only ([credit.niso.org](https://credit.niso.org)), one line per author, as required by the course outline. The project lead roles (data, modelling, interface, writing) are listed in the README, not here. Every member must hold at least one role. Update as work happens: this must agree with the commit history and peer evaluations.

- **G. Barrere:** Conceptualization, Data curation, Formal analysis, Project administration, Software, Validation, Visualization, Writing (original draft), Writing (review and editing)
- **J. Oentoro:** Conceptualization, Investigation, Validation, Writing (original draft), Writing (review and editing)
- **Y. Zhang:** Conceptualization, Data curation, Formal analysis, Methodology, Software, Validation, Writing (original draft)
- **K. Kang:** Investigation, Validation, Writing (original draft), Writing (review and editing)

## Use of Generative AI

Every row is a work session where a generative tool made a substantive contribution. Generative AI is never an author and holds no CRediT role.

| Tool + version | Date | Used for | How the output was verified |
|---|---|---|---|
| Claude Code (Anthropic CLI; model backend selected per session) | Nov 19, 2026 | Wrote `src/train_baselines.py` (majority, keyword rule, TF-IDF logistic regression, TF-IDF linear SVM, label-masked variants, validation-only regularisation sweep, model persistence) and `scripts/make_results_tables.py`; ran the baselines on Colab; drafted this section | Three independent subagent audits of the code and of the results; every prediction file passes `src/validate_predictions.py`; code and result transfers verified by SHA-256; all reported numbers recomputed independently from the prediction files. The logistic-regression regularisation strength was tuned at the modelling lead's direction, and the choice is recorded in `results/lr_c_sweep_42_val.json` |
