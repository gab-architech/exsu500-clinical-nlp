# Prediction file format

> **Test predictions are generated once, after the model lock on Wed Nov 25.** Until then, write and evaluate validation predictions only (`--split val`).

Every model's predictions go through one evaluation script (`src/evaluate.py`). This is the format it expects. Check a file with:

```bash
python src/validate_predictions.py results/preds_tfidf_lr_42_val.csv
```

## File name

`results/preds_{model}_{seed}_{split}.csv`

- `model`: short name, letters, digits and underscores (e.g. `majority`, `keyword_rule`, `tfidf_lr`, `tfidf_svm`, `clinicalbert`). Use the same name for every seed of a model; the summary groups files by it. A masked-text run is a separate model name (e.g. `tfidf_lr_masked`).
- `seed`: an integer (e.g. `42`). Deterministic models still need one; use `42`.
- `split`: `train`, `val` or `test`.

## Columns

One row per note of that split, in any order:

| Column | Content |
|---|---|
| `note_id` | the `text_id` from `labels/final_labels.csv` and `splits/split_seed42.csv` |
| `true_label` | the class from `labels/final_labels.csv`, unchanged |
| `pred_label` | the predicted class, exactly one of the 11 class names |
| `prob_<class>` | 11 columns, one per class, named `prob_` followed by the exact class name |

The 11 probability columns, written exactly like this (class names contain spaces and slashes; pandas handles them as normal column names):

```
prob_Cardiovascular / Pulmonary
prob_ENT / Ophthalmology
prob_Gastroenterology
prob_General Medicine
prob_Hematology - Oncology
prob_Neurology
prob_Obstetrics / Gynecology / Pediatrics
prob_Orthopedic
prob_Psychiatry / Psychology
prob_Radiology
prob_Urology / Nephrology
```

In code: `classes = sorted(pd.read_csv("labels/final_labels.csv")["label"].unique())`, then `f"prob_{c}"` for each class.

## Rules the validator checks

- No other columns, and no note text: IDs, labels and probabilities only (course data rule).
- `note_id` values are unique and match the frozen split exactly: every note of the split, nothing from other splits.
- `true_label` matches `labels/final_labels.csv` for every note.
- Probabilities are between 0 and 1, and each row sums to 1 (tolerance 0.01, for rounding).
- Models without real probabilities (majority class, keyword rule): put 1.0 on the predicted class and 0 elsewhere. AUROC, AUPRC and calibration are then degenerate for that model; report them as such.
- `pred_label` should be the highest-probability class. The validator only warns if it is not, since calibration is computed from the probabilities and the other metrics from `pred_label`.

## Test set

`src/evaluate.py` refuses `--split test` unless `--final` is given, and logs every final run to `results/test_runs.log`. Write test predictions only when every model is frozen, and run the test evaluation once.
