# Clinical Note Specialty Classification

EXSU 500 Fundamentals of AI in Medicine, McGill University, Fall 2026
**Team:** Gabriel Barrere, Emilia Mastromatteo, Jessica Oentoro, Yuhan Zhang

> **Course prototype for research and educational purposes only. Not a medical device. Not for clinical use.**

## At a glance

| | |
|---|---|
| **Problem** | Route clinical documents to the right specialty team |
| **Task** | Predict the medical specialty of a note from its free text |
| **Task type** | Single-label, multi-class text classification (6 to 10 classes, fixed after EDA) |
| **Dataset** | MTSamples medical transcriptions, ~5,000 notes, ~40 raw specialty labels |
| **Licence** | CC0: Public Domain (as listed on Kaggle) |
| **Access** | Public download with a free Kaggle account, no credentialing or data use agreement |
| **Privacy** | Sample transcriptions, not real patient records, no PHI |

## Problem

Clinical documents and referrals must reach the right specialty team. Misrouted documents delay care and create manual rework for referral intake and health information management staff. We classify a note's specialty from its text alone, as decision support for routing teams, with a human always in the loop.

**Research question:** How much of the reported performance on this task comes from duplicates and label leakage, and does a domain-pretrained transformer still beat TF-IDF once these are controlled?

## Dataset

**MTSamples** ([Kaggle: Medical Transcriptions, tboyle10](https://www.kaggle.com/datasets/tboyle10/medicaltranscriptions), file `mtsamples.csv`), scraped from [mtsamples.com](https://www.mtsamples.com).
Columns: `description`, `medical_specialty`, `sample_name`, `transcription`, `keywords`.

The data is **not stored in this repo**. Download steps: [`data/README.md`](data/README.md).

**Known issues we will address before modelling:**

1. Some labels are document types, not specialties (e.g. "Discharge Summary").
2. "Surgery" (~22% of rows) overlaps with surgical subspecialties.
3. Some transcriptions are duplicated under several labels.
4. `keywords`, `description` and some note headers reveal the label (leakage).
5. Several specialties have very few samples.

The class mapping, deduplication, Surgery handling and leakage masking will be decided after EDA, documented here, and frozen before any model is trained.

## Planned approach

**Models:** TBD (decided by Modelling lead).

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
| Interface lead, writing support | Emilia Mastromatteo |

All members contribute code and writing and commit under their own accounts.

## Status

**October 2, 2026:** topic and dataset selected, repo set up. Reproduction steps and results will be added as the project progresses.
