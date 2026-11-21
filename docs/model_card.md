# Model card: clinical-note specialty classifier (TF-IDF + linear SVM)

- **Model**: TF-IDF + linear SVM, Platt-calibrated (`src/train_baselines.py`)
- **Task**: single-label, multi-class (11 classes) classification of a clinical note's free
  text into its medical specialty
- **Owner**: Yuhan Zhang (Modelling lead), EXSU 500 Group 5, Fall 2026
- **Date**: Nov 20, 2026
- **Licence**: code MIT; data MTSamples, CC0
- **Reference**: `docs/results_validation.md`

## Intended use

Decision support for document routing and referral intake. The model reads a clinical note
and proposes the most likely specialty; a human routing or triage coordinator reviews the
proposal and makes the decision. It is a course prototype for research and educational
purposes only. It is not a medical device and is not for clinical use.

## Out-of-scope and prohibited use

- No autonomous routing: a person must always be in the loop.
- No patient-facing use.
- No use on notes from sources other than the US medical-transcription style represented in
  MTSamples, or in languages other than English.

## Data

- MTSamples (`data/README.md`), CC0 as listed on Kaggle.
- 11 specialty classes; final labels in `labels/final_labels.csv` (manual validation plus a
  blind audit).
- 2,230 unique notes; frozen stratified 70/15/15 split, seed 42 (`splits/split_seed42.csv`).
- Known issues (README, `docs/data_cleaning.md`): filing-label noise, the Surgery overlap
  merged, duplicates kept within one split, rare classes.

## Training

- Features: TF-IDF, 1-2 grams, sublinear tf, min_df 2, no stopword removal, L2 normalised.
- Estimator: class-weighted linear SVM (`LinearSVC`), probabilities from Platt scaling fitted
  on 5-fold cross-validation of the training scores only.
- Fitted on the training split (1,560 notes) only. No test data was used for any choice.

## Evaluation (validation split, 335 notes)

| Metric | Value |
|---|---|
| Macro-F1 (primary) | 0.875 (95% CI 0.831 to 0.910) |
| Accuracy | 0.875 (95% CI 0.842 to 0.907) |
| AUROC (macro, one-vs-rest) | 0.986 |
| AUPRC (macro, one-vs-rest) | 0.924 |
| ECE (15 equal-width bins) | 0.102 |
| Macro sensitivity / specificity at 0.5 | 0.820 / 0.992 |

Per-class values: `results/metrics/tfidf_svm_42_val.json` and
`results/metrics/threshold_metrics_val.csv`.

The logistic-regression baseline (macro-F1 0.862, ECE 0.059) is statistically
indistinguishable from this SVM on macro-F1 (paired bootstrap 95% CI -0.010 to +0.036) and is
better calibrated. The SVM is reported here as "best" on the macro-F1 point estimate only;
the team may prefer logistic regression for its lower calibration error. Test-set numbers are
pending the Nov 25 model lock (single test run, `src/predict_test.py`).

## Limitations

- One dataset, one source, one split; no external validation.
- Imbalanced and small classes: Psychiatry has 8 validation notes, Hematology-Oncology 14, so
  per-class figures there are noisy.
- Labels reflect how mtsamples.com filed each note, not a real routing workflow; the blind
  audit measured 13.8% disagreement (6.9% excluding definitional class merges).
- Under-confident: mean top-class confidence 0.772 against accuracy 0.875.
- Two validation notes have a near-duplicate in the training split; removing the four
  near-duplicate-involved validation notes changes macro-F1 by 0.0020.

## Excluded populations and situations

Non-English clinical text; specialties outside the 11 MTSamples classes; notes from clinical
workflows or populations not represented in MTSamples. The model has not been evaluated for
any of these.

## Use of GenAI

See the Use of GenAI log in `docs/CONTRIBUTIONS.md`.
