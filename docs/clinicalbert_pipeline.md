# ClinicalBERT pipeline: shared spec for the original and masked runs

Status: **draft for review by Yuhan (Modelling lead)**, Nov 17. Written by Gabriel with Claude Code. Nothing has been trained yet.

Purpose: Yuhan runs ClinicalBERT on the original text (seeds 42, 43, 44) and Gabriel runs it on the masked text (same seeds), on separate Colab accounts. The two sets of runs are only comparable if they use one pipeline, with **one flag (`--masked`) as the only difference**. This document describes that pipeline step by step so it can be agreed before anyone trains.

How to read it:
- **Agreed**: already decided in `docs/model_proposal.md`, `docs/midterm_numbers.md` or Issue #22.
- **TO DECIDE (Yuhan)**: open; a proposal is given where useful, but nothing is fixed until Yuhan confirms.

> **Code status (flag).** As of Nov 17 there is no ClinicalBERT training code in the repository, on any branch. The structure below is therefore a proposal for how Yuhan's code could be organised so the flag works, not a change to existing code.

## 0. Proposed code layout (TO DECIDE, Yuhan)

The smallest structure that makes `--masked` the only difference:

| File | Content | Owner |
|---|---|---|
| `src/data.py` | `load_split_notes(split)`: rebuilds note text from `data/mtsamples.csv` and returns `note_id, text, label` for one split (step 1) | Yuhan |
| `src/masking.py` | `mask_label_strings(text)`: the one masking function used by **every** model family, TF-IDF included (step 2) | Yuhan |
| `src/train_clinicalbert.py` | `--seed N [--masked] --split val`: trains, saves the checkpoint, writes predictions (steps 3 to 6) | Yuhan |

`--masked` would do exactly two things: apply `mask_label_strings` to every note before tokenization, and add `_masked` to the model name used for the checkpoint folder, the prediction file and the run log. Nothing else may depend on it.

## 1. Data loading and the frozen split

**Agreed:**
1. Read `data/mtsamples.csv` (downloaded from Kaggle; never committed). Keep rows that have a transcription.
2. For each row, compute the note ID exactly as the cleaning pipeline does (`notebooks/02_eda.ipynb`, Section 5): lowercase, replace every character that is not a letter, digit or whitespace with a space, collapse whitespace, then take the first 8 characters of the SHA-1 hash. This is the `text_id` in `labels/final_labels.csv` and `splits/split_seed42.csv`.
3. Keep one row per note ID, join the final label (`labels/final_labels.csv`, 2,230 notes) and the split (`splits/split_seed42.csv`).
4. Check the counts before training: **train 1,560, val 335**. Stop if they differ.
5. **The test split is never loaded before Wed Nov 25.** `load_split_notes` should refuse `"test"` unless an explicit final flag is set.

**TO DECIDE (Yuhan):**
- *Which raw text represents a note ID.* A few IDs have several raw strings that differ only in whitespace or punctuation (e.g. the ERCP note filed under Surgery and Gastroenterology). Proposal: keep the first occurrence in file order, so both runs see the identical string.
- *Model input.* Proposal: feed the raw `transcription` (not the normalized lowercase text), so the tokenizer sees the text as written. `keywords` and `description` are never used (label leakage, `docs/data_cleaning.md`).

**Suggestion (flag):** the note ID logic is now needed in three places (the EDA notebook, `scripts/check_near_duplicates.py` and the training code). The team rule is that a function used twice moves to `src/`, so `src/data.py` would become the single copy.

## 2. Masking

Masking removes the specialty name from the note text, so we can measure how much each model relies on it. It defines the leakage experiment, so it must be **one function, applied identically by every model family**.

**Agreed:** the masked variant removes "label strings" from the text (`docs/model_proposal.md`). Only the `transcription` is masked; `keywords` and `description` are already excluded for every run.

**What exactly is masked: TO DECIDE (Yuhan).** The 8.7% figure in the midterm counted rows whose **own raw label**, as a whole string, appears anywhere in the text (case-insensitive, all 4,966 raw rows; `notebooks/02_eda.ipynb`, cell 16). That is a measurement, not a masking rule: a composite label such as "Cardiovascular / Pulmonary" almost never appears word for word, so a rule based on it would mask very little. Two options:

| | Option A: specialty names | Option B: specialty names and related word forms |
|---|---|---|
| What is masked | Each specialty name from the raw labels that feed the 11 classes, as separate terms: e.g. "cardiovascular", "pulmonary", "orthopedic", "gastroenterology", "neurology", "neurosurgery", "urology", "nephrology", "obstetrics", "gynecology", "pediatrics", "radiology", "hematology", "oncology", "psychiatry", "psychology", "ophthalmology", "otolaryngology", "podiatry", "rheumatology", "endocrinology", "pain management", "physical medicine", "sleep medicine" | Option A plus derived forms naming the specialty or specialist: e.g. "urologist", "urologic", "neurologist", "cardiology", "cardiologist", "oncologist", "pediatric", "psychiatric", "orthopaedic" |
| Matching | whole words, case-insensitive, longest term first | same |
| Strength | Simple, auditable, close to "label strings removed" in the midterm | Removes more of the direct cue |
| Weakness | Leaves "urologist" or "cardiology clinic" in place | Starts to remove clinical content ("neurologic exam" is real clinical signal), and the list is a judgment call |

Proposal for the list either way: build it once in code from the raw label names (plus, for B, a short fixed list written down **before** any results), and do not tune it after seeing scores.

**Replacement: TO DECIDE (Yuhan).** Proposal: replace each match with the literal `[MASK]`. The ClinicalBERT tokenizer reads it as its built-in mask token, and TF-IDF sees one constant token, so the replacement carries no class information. Deleting the term instead would also work, but makes sentences ungrammatical and changes note length.

**Applied to:** train and val alike (and test alike at the final run). Masking only one side would be a different experiment.

**Report:** once the list is fixed, count how many of the 2,230 final notes contain at least one masked term, and record it in `docs/midterm_numbers.md`. The 8.7% figure was measured differently and should not be reused for the masked runs.

**Synthetic examples** (made up for illustration; not MTSamples text):

| Original | Masked (option A) |
|---|---|
| "Patient referred to the Urology clinic for recurrent hematuria. Cystoscopy planned." | "Patient referred to the [MASK] clinic for recurrent hematuria. Cystoscopy planned." |
| "PEDIATRICS FOLLOW-UP: 6-year-old seen for asthma; growth on track." | "[MASK] FOLLOW-UP: 6-year-old seen for asthma; growth on track." |
| "Seen in the cardiovascular service; echo shows reduced ejection fraction." | "Seen in the [MASK] service; echo shows reduced ejection fraction." |

Under option B, "Discussed with the urologist" would also become "Discussed with the [MASK]"; under option A it stays unchanged.

## 3. Tokenization and head+tail truncation

**Agreed:** checkpoint `emilyalsentzer/Bio_ClinicalBERT`, its own tokenizer, maximum length 512, head+tail truncation that keeps the beginning and end of long notes (Issue #22).

Steps:
1. Apply masking first if `--masked` (step 2), then tokenize the full note **without** truncation.
2. If the note fits, use all of it. If not, keep the first part and the last part of the token sequence and drop the middle.
3. Add the special tokens (`[CLS]` at the start, `[SEP]` at the end), pad to 512, and build the attention mask.

**TO DECIDE (Yuhan), small but exact:** 512 positions include the 2 special tokens, so only **510** text tokens fit. "256 + 256" is therefore not literally possible. Proposal: first **255** + last **255** text tokens. Whatever is chosen, both runs use the same numbers, and the figure "59.7% of tokens kept" in `docs/midterm_numbers.md` should be recomputed with them.

## 4. Training

**Agreed:**

| Setting | Value |
|---|---|
| Model | `emilyalsentzer/Bio_ClinicalBERT`, sequence classification head, 11 outputs |
| Class order | alphabetical class names (the same order as the `prob_` columns in `docs/predictions_format.md`) |
| Epochs | 3 |
| Batch size | 16 (about 98 steps per epoch on 1,560 notes) |
| Max length | 512, head+tail (step 3) |
| Seeds | 42, 43, 44 (original: Yuhan; masked: Gabriel) |
| Data | train 1,560 for fitting, val 335 for evaluation; never test before Nov 25 |
| Hardware | Colab free tier, one T4 |

**TO DECIDE (Yuhan).** The values below are common defaults for fine-tuning BERT models, listed only as a starting proposal:

| Setting | Proposal |
|---|---|
| Learning rate | 2e-5 |
| Optimizer | AdamW |
| Weight decay | 0.01 |
| Schedule and warmup | linear decay, warmup over the first 10% of steps |
| Class weighting | none, or inverse-frequency weights in the cross-entropy loss (classes are imbalanced 7.2x). Decide one and keep it for both runs |
| Mixed precision | fp16 on the T4 |
| Gradient clipping | max norm 1.0 |

Hyperparameters are **not** tuned separately for the masked runs: both use exactly the same values.

**How the seed is set (proposal):** at the start of each run, before the model is created:
- `transformers.set_seed(seed)`, which seeds Python `random`, NumPy, PyTorch and CUDA;
- the same seed for the training data shuffle (the `Trainer` arguments `seed` and `data_seed`, or a seeded `torch.Generator` passed to the `DataLoader`);
- `torch.backends.cudnn.deterministic = True` and `torch.backends.cudnn.benchmark = False`.

GPU training is still not bit-for-bit reproducible, so the run log (step 6) records versions and hardware.

## 5. Checkpoint rule

**TO DECIDE (Yuhan): final epoch vs best validation epoch.**

- Final epoch: simplest; nothing is selected.
- Best validation epoch: keeps the epoch with the best validation score (state which metric: macro-F1 or loss). Using validation for this is allowed. It must never use the test set.

The same rule applies to every seed and both runs. Checkpoints are saved to Google Drive only (`exsu500/checkpoints/clinicalbert{_masked}_seed{N}/`), never to the repository.

## 6. Predictions and run log

**Agreed:** predictions follow `docs/predictions_format.md` (PR #27).

1. Run the chosen checkpoint on the 335 validation notes; apply softmax to get the 11 class probabilities.
2. Write `results/preds_clinicalbert_{seed}_val.csv` (original) or `results/preds_clinicalbert_masked_{seed}_val.csv` (masked), with columns `note_id, true_label, pred_label, prob_<class>`, where `pred_label` is the highest-probability class. The seed in the file name is a bare number (`42`), as the validator requires.
3. Check with `python src/validate_predictions.py results/preds_clinicalbert*_val.csv`, then `python src/evaluate.py --split val`.
4. Write a run log per seed, `results/run_logs/clinicalbert{_masked}_seed{N}.json`: GPU type, Python and package versions, repository commit, all hyperparameters, the masking option, wall-clock time, and validation loss per epoch. These feed the compute paragraph of the manuscript.

**Test set:** test predictions are written once, after the model lock on Wed Nov 25, from the saved checkpoints, and evaluated with `python src/evaluate.py --split test --final`.

## Also needed before training (flag)

- `requirements.txt` only gives minimum versions (`transformers>=4.40`, `torch>=2.2`). Both Colab accounts need the same versions: pin them from the environment Yuhan trains in.

## Decisions for Yuhan (summary)

1. Code layout and the `--masked` flag (section 0)
2. Which raw text represents a note ID; raw vs normalized input (section 1)
3. Masking: option A or B, and the replacement token (section 2)
4. Head+tail split: 255 + 255 or another exact split (section 3)
5. Learning rate, optimizer, weight decay, warmup, class weighting (section 4)
6. Checkpoint rule: final epoch vs best validation epoch, and which metric (section 5)
7. Pinned package versions
