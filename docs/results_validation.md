# Validation results: five models, three families

Author: Yuhan Zhang (Modelling lead)
Last updated: Nov 20, 2026
Scope: the validation split only. No test label or test prediction has been used for any
modelling decision, and no test performance has been computed.

How the test rows were touched, stated precisely: `src/data.py` rebuilds every note's text to
match its ID, so the loader reads all 2,230 rows and then discards every row outside the
requested split. Test text was also read for two pre-registered data audits that ran before
any model was fitted: the near-duplicate check (Nov 5) and the masking count (Nov 17). No test
label informed a model, a hyperparameter, a feature or a threshold, and no test prediction
exists.

## 1. Protocol

- Split: `splits/split_seed42.csv`, frozen Nov 3, 70/15/15, stratified, seed 42. Train 1,560,
  validation 335, test 335. One row per unique normalized text.
- Fitting: every model is fitted on the training split only.
- Two model-selection choices were made on the validation split, both permitted by the
  protocol and both disclosed in section 5: the logistic-regression regularisation strength C,
  and the ClinicalBERT epoch. Separately, the pre-registered truncation trigger is evaluated
  on validation (section 8) but did not fire, so it changed nothing.
- Metrics: macro-F1 (primary), accuracy, one-vs-rest AUROC and AUPRC, expected calibration
  error (ECE, 15 equal-width bins, computed on the top-class confidence), and per-class
  sensitivity and specificity at a stated 0.5 threshold (section 8). 95% bootstrap confidence
  intervals, 1,000 resamples, seed 42, the same resamples for every model.
- Multi-seed aggregation: for ClinicalBERT the per-seed metric is computed first and the three
  values are averaged (the reported ECE 0.208 is the mean of the per-seed ECEs 0.181, 0.257 and
  0.187; averaging the seeds' probability vectors first would give 0.218). The interval in
  section 4 is pooled by averaging the seeds' scores on fixed bootstrap draws, so it treats
  the seeds as fixed and does not include training-seed variance; the seed standard deviation
  (0.014) is the separate estimate of that variability.

## 2. Environment

ClinicalBERT runs: Google Colab free tier, one Tesla T4 GPU (16 GB); Torch 2.14.1+cu130,
transformers 5.22.1, tokenizers 0.23.3, datasets 4.8.5, scikit-learn 1.6.1, numpy 2.4.6,
pandas 2.2.3, Python 3.13.16, all recorded in the three run logs.
Baselines: CPU only, same pinned scikit-learn, numpy and pandas, recorded in their metric logs.

## 3. Models

| Family | Model | Notes |
|---|---|---|
| Rule | Majority class | Reference line, not a trained model |
| Rule | Keyword rule | Two keywords per class from mutual information on the training split, vote, majority fallback |
| Linear | TF-IDF + logistic regression | 1-2 grams, sublinear tf, min_df 2, class weighted, C selected on validation |
| Linear | TF-IDF + linear SVM | Same features, class weighted, Platt calibrated (5-fold on training only) |
| Deep | Bio_ClinicalBERT | Fine-tuned 3 epochs, batch 16, lr 2e-5, AdamW, 10% warmup, class weighted, head+tail 255+255, seeds 42/43/44 |

## 4. Results (validation, 335 notes)

| Model | Macro-F1 (95% CI) | Accuracy (95% CI) | AUROC | AUPRC | ECE |
|---|---|---|---|---|---|
| Majority class | 0.025 (0.019 to 0.030) | 0.161 (0.119 to 0.200) | 0.500 (constant ranking) | 0.091 | not meaningful |
| Keyword rule | 0.157 (0.101 to 0.202) | 0.149 (0.113 to 0.188) | 0.545 (coarse, hard labels) | 0.161 | not meaningful |
| TF-IDF + logistic regression | 0.862 (0.816 to 0.899) | 0.854 (0.818 to 0.890) | 0.985 | 0.922 | 0.059 |
| TF-IDF + linear SVM | 0.875 (0.831 to 0.910) | 0.875 (0.842 to 0.907) | 0.986 | 0.924 | 0.102 |
| Bio_ClinicalBERT (3 seeds) | 0.805 (0.758 to 0.843) | 0.816 (0.776 to 0.854) | 0.980 | 0.889 | 0.208 |

ClinicalBERT per seed: 0.789, 0.810, 0.815; standard deviation across seeds 0.014.

The two rule models emit 0 or 1 per class. Their AUROC and AUPRC are computed from those hard
labels: the majority class gives a constant ranking (AUROC 0.500) and the keyword rule gives a
coarse one (AUROC 0.545). ECE is not meaningful for either, because confidence is always 1.0.

The multi-seed interval for ClinicalBERT is pooled over fixed bootstrap draws and therefore
excludes training-seed variance; read it together with the seed standard deviation of 0.014.

Source tables: `results/summary_val.csv`, `results/table_manuscript_val.md`.

## 5. How to read the differences

Three pairwise comparisons are made in this section; no multiple-comparison adjustment was
applied, and the weakest one (ClinicalBERT against logistic regression) is the one most likely
to be a false positive.

1. **The two linear models are not separable.** Paired bootstrap difference 0.0127 in favour
   of the SVM, 95% CI -0.010 to +0.036, which includes zero. A McNemar test gives p = 0.065,
   but it tests the proportion of notes classified correctly (2 notes only logistic regression
   gets right, 9 only the SVM), not the macro-F1 difference; the bootstrap interval is the
   macro-F1 test. The table does not rank the two. Logistic regression is the better calibrated
   of the two (ECE 0.059 against 0.102).
2. **ClinicalBERT is below both linear models.** Paired on the same notes with the three seeds
   averaged on each bootstrap draw: -0.070 against the SVM (95% CI -0.112 to -0.029) and -0.058
   against logistic regression (95% CI -0.100 to -0.015); per-seed differences against the SVM
   are -0.086, -0.065 and -0.060. The pooled interval treats the seeds as fixed and is
   therefore anti-conservative. A one-sample t-test across the three seed differences, which
   does treat the seed as the unit, gives p = 0.013 against the SVM and p = 0.019 against
   logistic regression with n = 3 and is correspondingly underpowered. Without a
   multiple-comparison adjustment, the ClinicalBERT-against-LR result sits just inside the
   unadjusted 5% level and would not survive a Bonferroni correction for three tests.
3. **What this supports.** On this task a linear bag-of-words model outperforms one fixed
   3-epoch fine-tuning recipe. The classes are defined by procedure and anatomy vocabulary,
   which suits lexical models, and the transformer saw no hyperparameter search. This is not a
   general claim that linear models beat transformers on clinical text.
4. **The best-epoch rule changed nothing in the original-text runs.** There all three seeds
   selected epoch 3, the last epoch, and their per-epoch validation macro-F1 rises
   monotonically (0.624, 0.782, 0.789; 0.559, 0.793, 0.810; 0.627, 0.782, 0.815), so no
   optimism was added by early stopping. It did change one masked run: seed 42 selected epoch
   2 (its macro-F1 peaked before its validation loss did, which fell through epoch 3), and
   masked seeds 43 and 44 selected epoch 3. So the rule was not a no-op across the whole
   study, only across the reported original-text numbers. The real caveat is that validation
   was used to watch training and to report, and that no epoch was held out from that use.
5. **The logistic-regression C was selected on validation.** The grid was first swept up to
   C = 100; the optimum landed on the boundary, so the grid was extended once to C = 2,000 and
   re-swept, giving an interior optimum at C = 500 (macro-F1 0.8623). The reported score is a
   selection maximum and its interval does not carry selection uncertainty. The first sweep is
   not saved, so the path is not replayable; the final grid and its scores are in
   `results/lr_c_sweep_42_val.json` and the extension is recorded in the comment above
   `LR_C_GRID`.

## 6. Leakage experiment (original versus label masked)

Masking removes 24 specialty names (option C, `src/masking.py`). It does **not** remove
"cardiovascular" and "pulmonary", two of the three largest classes' name words, nor "general"
and "medicine"; those appear in 31, 26, 152 and 9 validation notes. Orthopedic, the third
largest class, is masked. At the data level the mask touches 241 of the 2,230 notes (10.8%),
which is 194 of the 1,894 train and validation notes (10.2%; 159 train, 35 validation) plus 47
test notes counted as data, not read for a decision.

| Model | Macro-F1 original | Macro-F1 masked | Difference | Predictions changed |
|---|---:|---:|---:|---:|
| Logistic regression | 0.862 | 0.862 | -0.0004 | 2 of 335 |
| Linear SVM | 0.875 | 0.874 | -0.0007 | 2 of 335 |
| Bio_ClinicalBERT (3 seeds) | 0.805 | 0.805 | +0.0003 | 9, 5 and 1 of 335 |

The keyword rule and the majority class are identical between runs for different reasons: the
majority class never reads the text, and none of the keyword rule's 22 keywords is in the mask
list. Neither row is evidence about leakage. Calibration was not part of this experiment, but
for completeness the masked ClinicalBERT ECE is 0.237 against 0.208 unmasked.

Interpretation, scoped to the design: removing these 24 names changes at most 9 of 335
predictions in any model and leaves every macro-F1 within 0.0007. That is consistent with "the
models are not reading the filing label", and equally consistent with "the perturbation is too
small to detect an effect"; 2 changed predictions out of 335 cannot separate the two. A
stronger version would mask the remaining class-name words as well; that variant is not in the
committed pipeline.

## 7. Error analysis

ClinicalBERT seed 42, the weakest seed, has General Medicine at F1 0.413 (precision 0.650,
recall 0.302, support 43). Its errors are concentrated: 10 of those 43 notes are predicted
Orthopedic. This matches the blind audit, where Pain Management notes filed under General
Medicine mostly read as Orthopedic. The next largest confusions are General Medicine to
Cardiovascular / Pulmonary (5 notes) and General Medicine to Neurology (3). Small classes carry
little support, Psychiatry has 8 validation notes, so its F1 of 0.727 (recall 1.000) is one or
two notes away from a different value.

The linear models make fewer General Medicine errors but show the same direction; see the
committed confusion matrices in `figures/` and the counts in `results/metrics/*.json`.

## 8. Sensitivity and specificity at a stated threshold

One-vs-rest, per class, probability threshold 0.5 (not the argmax rule used for macro-F1). The
full per-class table is `results/metrics/threshold_metrics_val.csv` and the summary is
`results/table_threshold_metrics_val.md`.

| Model | Macro sensitivity | Macro specificity |
|---|---:|---:|
| Majority class (one-hot) | 0.091 | 0.909 |
| Keyword rule (one-hot) | 0.177 | 0.912 |
| TF-IDF + logistic regression | 0.774 | 0.992 |
| TF-IDF + linear SVM | 0.820 | 0.992 |
| Bio_ClinicalBERT, seed 42 / 43 / 44 | 0.706 / 0.639 / 0.707 | 0.991 / 0.995 / 0.992 |

For the two one-hot models the 0.5 threshold coincides with their hard prediction, so the
values above are the per-class recall and true-negative rate of that prediction. They are
valid numbers but the notion of a probability threshold is degenerate for them.

## 9. Artifacts

| Path | Content |
|---|---|
| `results/preds_{model}{_masked}_42_val.csv` | Per-note predictions, 335 rows, 11 class probabilities |
| `results/preds_clinicalbert_{42,43,44}_val.csv` | The three ClinicalBERT seeds |
| `results/run_logs/clinicalbert_seed{42,43,44}.json` | Full training logs: GPU, Python and package versions, hyperparameters, per-epoch validation loss and macro-F1, best checkpoint, wall clock, and the scope of that wall clock |
| `results/run_logs/clinicalbert_masked_seed*.json` | Masked half: GPU, commit, versions, hyperparameters, per-epoch validation loss and the final validation macro-F1; no per-epoch macro-F1 and no best-epoch field, so these logs alone cannot show which epoch each masked run kept. The kept checkpoint, reported by the data lead from the checkpoint folders on his Drive (`save_total_limit=1`, 98 steps per epoch, so the step number gives the epoch): seed 42 kept `checkpoint-196`, epoch 2; seeds 43 and 44 kept `checkpoint-294`, epoch 3. Those folders are not in this repository, so that half's selection is recorded, not independently auditable here |
| `results/run_logs/baseline_*_val.json` | Metric records for the eight baseline runs: model, seed, split, sample counts, macro-F1, accuracy, versions, and the LR C. Not training logs |
| `results/metrics/*.json` | Every metric, both subsets, per model and seed, including confusion counts |
| `results/metrics/threshold_metrics_val.csv` | Per-class sensitivity and specificity at the 0.5 threshold |
| `results/keyword_table{_masked}_42_val.json` | The 22 keywords with mutual information and document frequency |
| `results/lr_c_sweep_42_val.json` | The final C grid and its validation scores |
| `results/summary_val.csv`, `summary_val.md` | Full comparison, all models and both subsets |
| `results/table_manuscript_val.md`, `table_masking_val.md`, `table_threshold_metrics_val.md` | Manuscript tables |
| `figures/` | 38 evaluation figures (confusion matrix, reliability diagram and per-class F1 for each of 14 models, minus the four reliability diagrams for the one-hot models), plus the two data figures |
| `models/` (git-ignored) | The two TF-IDF pipelines with and without masking, and the three ClinicalBERT checkpoints (also on Drive) |

Provenance: the original-text runs executed `src/train_clinicalbert.py` at commit `4832af0`;
the masked half executed commit `49789e4`. The diff between them is logging fields, the CUDA
guard and `save_only_model`, with no change to data handling, tokenization or hyperparameters,
so the comparison remains like for like, but this document cannot prove that from artifacts
alone.

The three checkpoints were reloaded from disk and re-predicted as a check: the resulting
macro-F1 matches the recorded predictions exactly (0.7892, 0.8098, 0.8155) with 100% label
agreement.

The four reliability diagrams for the two one-hot models (majority class and keyword rule,
original and masked) are not committed: with one-hot probabilities such a diagram degenerates
to a single point at confidence 1.0 and reads as a calibration curve when it is not one.

## 10. Limitations

- One dataset, one source, one split. Validation has 335 notes. Psychiatry has 8 validation
  notes and Hematology-Oncology 14, so per-class numbers for those classes are noisy.
- ClinicalBERT was fine-tuned with one fixed recipe: no learning-rate search, no layer-wise
  decay, no scheduler search, 3 epochs only. Its deficit is a statement about this recipe.
- Truncation: 211 of 335 validation notes (63.0%) exceed 510 text tokens and are cut to
  head+tail 255+255. The pre-registered chunking trigger did not fire: ClinicalBERT is better on
  notes over 510 text tokens than on notes at or under it (0.796, 0.807, 0.825 against 0.700,
  0.719, 0.707). The two subsets are not class-matched, the short subset has no Psychiatry
  notes and 3 Hematology-Oncology notes, so macro-F1 scores Psychiatry 0 on that side and the
  gap is inflated; the direction is unchanged, so the trigger decision stands while the
  magnitudes should be read as an upper bound.
- Calibration: the three trained probabilistic models are under-confident, with mean top-class
  confidence below their accuracy (ClinicalBERT 0.635, 0.564, 0.642 against 0.806, 0.821,
  0.821; SVM 0.772 against 0.875; logistic regression 0.806 against 0.854), and their
  reliability curves sit mostly above the diagonal. The two one-hot baselines are over-confident
  by construction and their ECE is not meaningful.
- Labels come from how mtsamples.com filed each note, not from a routing workflow. The blind
  audit measured 13.8% disagreement with those labels on auto-mapped notes (6.9% excluding
  definitional class merges); those figures come from the blind-audit table, not from the
  prediction files.

## 11. What remains before the final report

1. A predict-only path for the test split. **Done, and hardened Nov 22** after review by the
   data lead: `src/predict_test.py` now (a) loads every requested model, (b) re-predicts the
   validation split with each and compares the labels against the committed
   `results/preds_*_val.csv`, (c) appends to `results/test_runs.log`, and only then (d) reads
   the test split. Nothing touches test until every model has loaded and reproduced its
   committed validation predictions, so a missing or wrong artifact stops the run while test
   is still closed, and a crash after the read is already logged. The ClinicalBERT loader
   accepts either the checkpoint folder or the parent the Trainer wrote, using its single
   `checkpoint-*` subfolder. `python src/predict_test.py --verify` runs steps (a) and (b) alone
   and never opens test. The run itself is still gated by the Nov 25 model lock, as item 2
   describes.
2. Read the test split once, from those checkpoints, and report the same table with test
   numbers, including macro-F1 with and without the 9 test notes that have a near-duplicate in
   another pile.
3. ~~Decide whether the ClinicalBERT recipe gets a second attempt before the lock.~~
   **Decided (data lead, Nov 21): freeze the recipe as is.** No learning-rate search before the
   lock. The protocol is graded as heavily as the numbers, and "one fixed recipe loses to
   TF-IDF on a vocabulary-driven task" is a clean, honest result. The modelling lead agrees.
4. ~~Re-run the masked half at the final commit if a perfectly matched provenance pair is
   required.~~ **Decided (Nov 21): not re-running.** The numbers would not change, and the
   kept-checkpoint table in section 9 closes the audit gap in the provenance pair.

## 12. Reproduction

```bash
python src/train_baselines.py                      # four baselines, original text, C = 500 by default
python src/train_baselines.py --masked             # the masked twins
python src/train_baselines.py --tune-lr-c          # the C sweep, validation only, writes no predictions
python src/train_clinicalbert.py --seed 42 \
  --output-dir /content/drive/MyDrive/exsu500/checkpoints \
  --results-dir /content/results                   # needs a GPU; repeat for 43, 44, and add --masked
python src/validate_predictions.py results/preds_*_val.csv
python src/evaluate.py --split val
python scripts/make_results_tables.py
python scripts/class_threshold_metrics.py

# The single test run, after the Nov 25 model lock (the only step that reads test):
python src/predict_test.py --verify         # optional dry run: load every model, check it against val
python src/predict_test.py --final          # loads the frozen models, writes preds_*_test.csv
python src/evaluate.py --split test --final # metrics, summary_test.csv/.md, figures
```

`--final` already runs the `--verify` check before it reads test; the separate call is for
running it on its own. Both take `--checkpoint-dir` if the ClinicalBERT folders live outside
`models/` (for example on Drive), and the loader accepts either `clinicalbert_seed{N}/` or the
`clinicalbert_seed{N}/checkpoint-<step>/` layout the Trainer writes.

The ClinicalBERT command needs an explicit `--output-dir`: the default writes under the
repository, which on Colab is the ephemeral `/content`. The numbered step-by-step order
(including which steps run on Colab) is in `docs/run_order.md`.
