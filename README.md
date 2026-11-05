# Clinical Note Specialty Classification

EXSU 500 Fundamentals of AI in Medicine, McGill University, Fall 2026
**Team:** Gabriel Barrere, Yuhan Zhang, Jessica Oentoro, Karman Kang

> **Course prototype for research and educational purposes only. Not a medical device. Not for clinical use.**

## At a glance

| | |
|---|---|
| **Problem** | Route clinical documents to the right specialty team |
| **Task** | Predict the medical specialty of a note from its free text |
| **Task type** | Single-label, multi-class text classification (11 classes; final list confirmed by manual validation) |
| **Dataset** | MTSamples medical transcriptions, ~5,000 notes, ~40 raw specialty labels |
| **Licence** | CC0: Public Domain (as listed on Kaggle) |
| **Access** | Public download with a free Kaggle account, no credentialing or data use agreement |
| **Privacy** | Sample transcriptions, not real patient records, no PHI |

## Problem

Clinical documents and referrals must reach the right specialty team. Misrouted documents delay care and create manual rework for referral intake and health information management staff. We classify a note's specialty from its text alone, as decision support for routing teams, with a human always in the loop.

**Research question:** Under a leakage-controlled evaluation, how accurately can a clinical note's medical specialty be predicted from its free text, how do model families compare, and how much does performance change when label strings in the text are masked?

## Dataset

**MTSamples** ([Kaggle: Medical Transcriptions, tboyle10](https://www.kaggle.com/datasets/tboyle10/medicaltranscriptions), file `mtsamples.csv`), scraped from [mtsamples.com](https://www.mtsamples.com).
Columns: `description`, `medical_specialty`, `sample_name`, `transcription`, `keywords`.

The data is **not stored in this repo**. Download steps: [`data/README.md`](data/README.md).

**Known issues we will address before modelling:**

1. Some labels are document types, not specialties (e.g. "Discharge Summary").
2. "Surgery" (~22% of rows) overlaps with surgical subspecialties.
3. Some transcriptions are duplicated under several labels.
4. `keywords`, `description` and, in 8.7% of notes, the transcription itself reveal the label (leakage).
5. Several specialties have very few samples.

The class mapping, deduplication, Surgery handling and leakage control were decided after EDA and are documented in [docs/data_cleaning.md](docs/data_cleaning.md); the final dataset is frozen before any model is trained.

## Data cleaning pipeline

| Step | Action | Tool / rule | Remaining | Change |
|---|---|---|---|---|
| 0 | Raw dataset | — | 4,999 rows | |
| 1 | Strip leading spaces in labels | `str.strip()` | 4,999 rows | 0 |
| 2 | Drop rows with no transcription | `.notna()` filter | 4,966 rows | -33 |
| 3 | Drop document-type labels (6 labels) | label blacklist | 4,028 rows | -938 |
| 4 | Drop the Surgery label | byte-level duplicate matching (1,039 of 1,088 rows are identical to another specialty's note) | 2,940 rows | -1,088 |
| 5 | Deduplicate to unique note texts | `normalize_text` + `groupby` | 2,279 texts (+48 surgery-only) | |
| 6 | Manual review of conflicting notes (Jessica Oentoro, Karman Kang) | offline browser tool; keyword-rule first pass (48.5% agreement with human labels) | 606 texts reviewed | |
| 7 | Map raw labels to 11 final classes | `CLASS_MAP` + thyroid-surgery rule + `text_id` | **2,230 notes** (2,327 candidates - 97 excluded) | |
| | ↳ of which 97 texts are excluded here: 65 no specialty home (incl. 14 Bariatrics surgery notes) + 4 reviewer DROP + 21 general surgery + 7 thyroid surgery | | | |

Pipeline figure: [`figures/fig2_cleaning_flow.png`](figures/fig2_cleaning_flow.png).

Final class distribution:

| Class | Notes | Share |
|---|---:|---:|
| Cardiovascular / Pulmonary | 358 | 16.1% |
| Orthopedic | 345 | 15.5% |
| General Medicine | 286 | 12.8% |
| ENT / Ophthalmology | 221 | 9.9% |
| Gastroenterology | 204 | 9.1% |
| Urology / Nephrology | 198 | 8.9% |
| Neurology | 194 | 8.7% |
| Obstetrics / Gynecology / Pediatrics | 174 | 7.8% |
| Radiology | 105 | 4.7% |
| Hematology - Oncology | 95 | 4.3% |
| Psychiatry / Psychology | 50 | 2.2% |
| **Total** | **2,230** | 100% |

Full rationale, tools and keyword tables: [docs/data_cleaning.md](docs/data_cleaning.md).

## Planned approach

**Models:** majority-class reference line + a keyword-rule baseline (mutual-information-selected keywords, voting, majority fallback); TF-IDF + logistic regression and TF-IDF + linear SVM; fine-tuned ClinicalBERT. Full proposal with literature: [docs/model_proposal.md](docs/model_proposal.md).

**Evaluation protocol**
- Stratified train / validation / test split, with duplicates kept within one split
- Test set used **once**, at the end. All tuning on validation.
- Primary metric: **macro-F1**. Secondary: per-class precision, recall, F1; one-vs-rest AUROC and AUPRC.
- Uncertainty: multiple random seeds and bootstrap 95% confidence intervals
- Confusion matrix with error analysis, plus a reliability diagram for calibration

## Repository

```
README.md
LICENSE                          MIT
requirements.txt                 pinned before final submission
data/README.md                   download instructions (data itself is gitignored)
notebooks/
  01_data_check.ipynb            dataset loads, class balance, lengths, duplicates
  02_eda.ipynb                   (planned)
  03_ onward                     (planned) one notebook per model, set by Modelling lead
  NN_test_evaluation.ipynb       (planned) last notebook, only one that touches the test set
docs/
  CONTRIBUTIONS.md               CRediT + Use of Generative AI statements
  model_card.md                  (planned)
```

## Team roles

| Role | Member |
|---|---|
| Data lead, team coordinator | Gabriel Barrere |
| Modelling lead | Yuhan Zhang |
| Writing lead | Jessica Oentoro |
| Interface lead, writing support | Karman Kang |

All members contribute code and writing and commit under their own accounts.

## Status

**October 2, 2026:** topic and dataset selected, repo set up. Reproduction steps and results will be added as the project progresses.

**October 26, 2026:** EDA and label cleaning complete: final dataset of 2,230 notes across 11 classes. The pipeline and every decision are documented in [docs/data_cleaning.md](docs/data_cleaning.md).

## Use of Generative AI statement

| Tool + version | Date | Used for | How output was verified |
|---|---|---|---|
| Claude Code, Claude Opus 5.5 | Sep 27 to Oct 9 | repo setup, README, data README, CLAUDE.md conventions, 01_data_check configuration | verified by review and a Colab run |
| Claude Code (Anthropic CLI; model backend selected per session: claude-sonnet-5.5 and kimi-k3) | Oct 15 to Nov 5 | EDA and the data cleaning pipeline, the review tools, the model proposal, draft literature notes, and a targeted audit of the first-pass labels (3 suspicious labels flagged, each correction confirmed by the modelling lead). No AI assigned a final label. | pipeline re-run top to bottom (2,230 notes, 11 classes); session logs available to the instructors on request |
