# [Project title: e.g. Classifying X in clinical text]

EXSU 500 Fundamentals of AI in Medicine, McGill University, Fall 2026
Team: Gabriel Barrere, Jessica Oentoro, Yuhan Zhang

> Course prototype for research and educational purposes only. Not a medical device. Not for clinical use.

## Problem

**Task statement:** We predict **[what]** from **[which input text]** for **[who: clinicians, patients, researchers]**, because **[why it matters clinically]**.

**Task type:** [binary / multi-class / multi-label classification / summarisation]

## Dataset

| Field | Value |
|---|---|
| Name | [dataset name] |
| Source | [URL] |
| Licence | [e.g. CC BY 4.0] |
| Size | [n documents, n classes] |
| Access | Public, no credentialing or data use agreement |

The dataset is **not** committed to this repo. See [`data/README.md`](data/README.md) for download instructions.

## Team roles

| Role | Member |
|---|---|
| Data lead | [name] |
| Modelling lead | [name] |
| Interface lead | [name] |
| Writing lead | [name] |

Every member contributes to code and manuscript; roles define accountability.

## Repository structure

```
notebooks/   Run in numbered order (01_, 02_, ...)
data/        Download instructions only; raw data is git-ignored
docs/        CRediT, Use of GenAI statement, model card (later)
```

## How to reproduce

1. `pip install -r requirements.txt` (or open the notebooks in Google Colab)
2. Follow `data/README.md` to download the data
3. Run `notebooks/01_data_check.ipynb`

*Headline results table and interface link will be added before Dec 11.*

## Licence

Code released under the MIT licence (see `LICENSE`). Dataset remains under its own licence.
