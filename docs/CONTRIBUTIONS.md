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
| Claude Code (Anthropic CLI; the model backend is configured per session and the identifier is recorded in the session logs available to the instructors) | Nov 6 to Nov 18 | With the data lead, the shared ClinicalBERT pipeline spec; `src/data.py` (data lead); `src/masking.py` and `src/train_clinicalbert.py` (modelling lead); the masked ClinicalBERT runs for seeds 42, 43 and 44 (data lead) | Spec items decided by the modelling lead; masking measured independently on all 2,230 notes by both members (241 changed, 10.8%); the training script adapted to transformers 5.x and verified on a GPU before the Colab runs; every prediction file passes `src/validate_predictions.py` |
| Claude Code (Anthropic CLI; model backend selected per session) | Nov 19, 2026 | Wrote `src/train_baselines.py` (majority, keyword rule, TF-IDF logistic regression, TF-IDF linear SVM, label-masked variants, validation-only regularisation sweep, model persistence) and `scripts/make_results_tables.py`; ran the baselines on Colab; drafted this section | Three independent subagent audits of the code and of the results; every prediction file passes `src/validate_predictions.py`; code and result transfers verified by SHA-256; all reported numbers recomputed independently from the prediction files. The logistic-regression regularisation strength was tuned at the modelling lead's direction, and the choice is recorded in `results/lr_c_sweep_42_val.json` |
| Claude Code (Anthropic CLI; model backend selected per session) | Nov 20, 2026 | Ran the original-text ClinicalBERT seeds 42, 43 and 44 on Colab; wrote `docs/results_validation.md`; added the paired comparisons to `scripts/make_results_tables.py` | One subagent audited the training script before the run, and its five recommendations were applied before training started; a second subagent recomputed every metric, interval and paired test from the prediction files (largest difference 0.00004) and found three factual errors in the draft report, which were corrected; all three checkpoints were reloaded from disk and reproduce their recorded predictions exactly (100% label agreement) |
| Claude Code (Anthropic CLI; model backend selected per session) | Nov 20, 2026 | Final review of the validation report and generated tables; added `src/predict_test.py` (predict-only test path from the frozen models), `docs/model_card.md` (best-model card) and `docs/run_order.md` (numbered run order); corrected two rounding errors and a Windows path-separator bug | `src/predict_test.py --verify` re-checked the two frozen TF-IDF baselines against the committed validation predictions (macro-F1 0.8623 and 0.8750 reproduced exactly) and its ClinicalBERT loading reproduced seed 42 (0.7892) in a separate local check; every number in the report recomputed from the committed metrics JSON, near-duplicate CSV and run logs (per-seed macro-F1, ECE, per-epoch curves, threshold sensitivity/specificity, near-duplicate counts) |
