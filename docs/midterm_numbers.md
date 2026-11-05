# Midterm numbers

Single source of truth for every number in the midterm report (Nov 9). If a number is not here, it does not go in the report. Each section names its source file; update the source first, then this sheet.

Last checked: Nov 5, 2026 (literature counts; data numbers Nov 3), against `labels/final_labels.csv` (unchanged since 47d04d8) and `splits/split_seed42.csv` (c029afe).

## 1. Dataset

Source: `data/README.md`, `docs/data_cleaning.md`

- MTSamples (Kaggle `tboyle10/medicaltranscriptions`, `mtsamples.csv`), CC0 as listed on Kaggle
- Raw: 4,999 rows, 6 columns, about 40 raw specialty labels
- Leakage measured in EDA: `keywords` contains the label in 97.9% of rows that have keywords (3,850 of 3,931); the label string appears in the transcription itself in 8.7% of rows with text (431 of 4,966). Only `transcription` is used as model input.
- Duplication: 95.9% of raw rows duplicate another row's text

## 2. Cleaning steps (Table 1)

Source: `docs/data_cleaning.md` (Steps 0 to 7, Step 7 exclusions)

| Step | What was removed | Removed | Left |
|---|---|---:|---:|
| 0 | Raw dataset | | 4,999 rows |
| 1 | Rows with no transcription | 33 | 4,966 rows |
| 2 | Document-type labels (6, e.g. Consult, Discharge Summary) | 938 | 4,028 rows |
| 3 | Surgery label (1,039 copies of notes already filed under a specialty; 49 surgery-only notes set aside) | 1,088 | 2,940 rows |
| 4 | Duplicate texts collapsed (one row per unique normalized text) | | 2,279 texts (+48 surgery-only) |
| 5 | Manual review of conflicting notes (578 cross-label + 28 surgery-only) | | 606 reviewed |
| 6 | Excluded at mapping: no specialty home (65, incl. 14 Bariatrics), general surgery (21), thyroid surgery (7), reviewer DROP (4) | 97 | **2,230 notes** |

Resolved (Nov 5): the surgery-only rows add 48 unique texts, not 49. One ERCP note exists twice with a whitespace/punctuation difference, once under Surgery and once under Gastroenterology; after normalization it is the same text and is already counted among the 2,279. Arithmetic: 2,279 + 48 = 2,327, minus 97 = 2,230. Explained in `docs/data_cleaning.md` Step 5.

Manual review detail: 2 reviewers (Jessica Oentoro, Karman Kang), 303 notes each; 602 assigned to a class, 4 DROP; 3 labels corrected in a second-review audit (Appendix A.6). A keyword-rule suggestion (not an AI model) was shown with each note; it agreed with the final human label 48.5% of the time (294/606).

## 3. Final classes (Figure 1)

Source: `labels/final_labels.csv` (counts), `docs/data_cleaning.md` Appendix A.2 (merges)

| Class | Raw labels merged | Notes | % |
|---|---|---:|---:|
| Cardiovascular / Pulmonary | Cardiovascular / Pulmonary | 358 | 16.1% |
| Orthopedic | Orthopedic, Podiatry | 345 | 15.5% |
| General Medicine | General Medicine, Pain Management, Physical Medicine - Rehab, Sleep Medicine, Endocrinology, Rheumatology, Diets and Nutritions, Allergy / Immunology | 286 | 12.8% |
| ENT / Ophthalmology | ENT - Otolaryngology, Ophthalmology, Dentistry, Cosmetic / Plastic Surgery | 221 | 9.9% |
| Gastroenterology | Gastroenterology | 204 | 9.1% |
| Urology / Nephrology | Urology, Nephrology | 198 | 8.9% |
| Neurology | Neurology, Neurosurgery | 194 | 8.7% |
| Obstetrics / Gynecology / Pediatrics | Obstetrics / Gynecology, Pediatrics - Neonatal | 174 | 7.8% |
| Radiology | Radiology | 105 | 4.7% |
| Hematology - Oncology | Hematology - Oncology, Lab Medicine - Pathology | 95 | 4.3% |
| Psychiatry / Psychology | Psychiatry / Psychology | 50 | 2.2% |
| **Total** | | **2,230** | 100% |

Imbalance: 7.2x (358 vs 50). Excluded labels with no specialty home: Autopsy, Bariatrics, Chiropractic, Dermatology, Hospice - Palliative Care, IME-QME-Work Comp etc., Speech - Language.

## 4. Split

Source: `splits/split_seed42.csv`, `scripts/make_split.py`

70/15/15, stratified by class, seed 42; one row per unique normalized text, so all copies of a duplicated note are in the same pile.

| Class | Train | Val | Test | Total |
|---|---:|---:|---:|---:|
| Cardiovascular / Pulmonary | 250 | 54 | 54 | 358 |
| Orthopedic | 241 | 52 | 52 | 345 |
| General Medicine | 200 | 43 | 43 | 286 |
| ENT / Ophthalmology | 155 | 33 | 33 | 221 |
| Gastroenterology | 143 | 30 | 31 | 204 |
| Urology / Nephrology | 138 | 30 | 30 | 198 |
| Neurology | 136 | 29 | 29 | 194 |
| Obstetrics / Gynecology / Pediatrics | 122 | 26 | 26 | 174 |
| Radiology | 73 | 16 | 16 | 105 |
| Hematology - Oncology | 67 | 14 | 14 | 95 |
| Psychiatry / Psychology | 35 | 8 | 7 | 50 |
| **Total** | **1,560** | **335** | **335** | **2,230** |

## 4b. Near-duplicates across piles

Source: `results/near_duplicate_pairs.csv`, `scripts/check_near_duplicates.py`

- TF-IDF cosine at or above 0.8: 11 cross-pile pairs, all same label; 9 test and 4 validation notes affected
- At or above 0.9: 4 pairs, 4 test notes. At or above 0.95: 2 pairs, 2 test notes
- Decision (Nov 5): keep the split; report test macro-F1 with and without the 9 test notes

## 5. Blind review

Source: `results/blind_review_stats.md`

- Sample: 220 notes, stratified: 60 human-reviewed, 160 auto-mapped
- Human-reviewed: 86.7% agreement, Cohen's kappa 0.847 (95% bootstrap CI 0.742 to 0.941)
- Auto-mapped: error rate 13.8%, 22/160 (95% Wilson CI 9.3% to 19.9%)
- Auto-mapped, sensitivity (excluding the 11 disagreements from labels merged into General Medicine): 6.9%, 11/160 (3.9% to 11.9%)
- Pooled: 86.4% (190/220), kappa 0.848 (0.791 to 0.894)
- Finding: 14 of the 30 disagreements are auto-mapped notes filed under General Medicine that the blind reviewer placed elsewhere (8 Orthopedic, 4 Cardiovascular / Pulmonary, 1 Neurology, 1 Obstetrics / Gynecology / Pediatrics). The class merges 8 raw labels by design, confirmed by the modelling lead: reported as a definitional difference. **No labels were corrected.** Raw labels checked Nov 5: 11 of the 14 come from merged labels (Pain Management 8, Physical Medicine - Rehab 1, Sleep Medicine 1, Diets and Nutritions 1) and 3 from raw General Medicine. Only the 11 are set aside in the sensitivity estimate. All 8 sampled Pain Management notes disagreed (7 Orthopedic, 1 Neurology); 45 of the 286 General Medicine notes carry that label.
- 6 of the 8 human-reviewed disagreements involve Radiology (imaging reports).

## 6. Literature

Source: `docs/literature_notes/` ("Read by" lines)

- 21 notes, all with a named reader: Karman 9, Yuhan 5, Gabriel 4, Jessica 3 (Li 2023 still has no read date)
- Cited in the current drafts: 16 papers plus the dataset (17 references). Available but not cited yet: Joachims 1998, Joulin 2017, Peluso 2024, Dahl 2025, Gaber 2025
- All DOIs and arXiv IDs checked against Crossref / arXiv
- Course minimum: 15 references, mostly primary research from the last 5 years
- Page limit: 3 pages, references included (CRediT and GenAI statements excluded)

## 7. Models and compute

Source: `docs/model_proposal.md`, Issue #3

- Training scope: Standard (#26)
- 5 models, 3 families: majority class (reference line) and keyword-rule baseline; TF-IDF + logistic regression and TF-IDF + linear SVM; fine-tuned Bio_ClinicalBERT
- Note length: median 415 words; 810 notes (36.3%) exceed 512 words, a floor for the share over 512 BERT tokens (exact tokenizer count pending, Yuhan)
- Compute: Colab free tier, single T4; ClinicalBERT about 20 to 30 min per run (3 epochs), a few seeds in about 2 hours
- External reference: CLiGNet best leakage-free macro-F1 0.279 on 40 raw classes (Barman 2026); not directly comparable with our 11 classes

## Captions

**Table 1.** Data cleaning steps from the raw MTSamples dataset to the final 2,230 notes. Rows until step 3, unique normalized texts from step 4. Counts from `docs/data_cleaning.md`.

**Figure 1.** Class distribution of the final dataset (2,230 notes, 11 classes). Imbalance between the largest and smallest class is 7.2x. The stratified 70/15/15 split keeps these proportions in each pile.

**Figure 2.** Data cleaning pipeline from the 4,999 raw notes to the final 2,230 notes in 11 classes, with the 97 exclusions broken down. The dashed box reports the independent blind audit, which is not part of the pipeline. Source: `docs/data_cleaning.md`; generated by `scripts/make_figure2_cleaning_flow.py`.

## Table 1 for the IEEE template (LaTeX)

```latex
\begin{table}[t]
\caption{Data cleaning steps from raw MTSamples to the final dataset}
\label{tab:cleaning}
\centering
\footnotesize
\begin{tabular}{@{}clrr@{}}
\toprule
Step & Removed & $n$ removed & $n$ left \\
\midrule
0 & Raw dataset (rows) & & 4,999 \\
1 & No transcription & 33 & 4,966 \\
2 & Document-type labels (6) & 938 & 4,028 \\
3 & Surgery label$^a$ & 1,088 & 2,940 \\
4 & Duplicates collapsed (unique texts) & & 2,279$^b$ \\
5 & Manual review of conflicts & & 606 reviewed \\
6 & No specialty home, surgery-only, reviewer drop & 97 & \textbf{2,230} \\
\bottomrule
\multicolumn{4}{@{}p{0.95\columnwidth}@{}}{\scriptsize $^a$1,039 copies of notes filed under a specialty; 49 surgery-only notes set aside. $^b$Plus the 48 surgery-only unique texts (49 raw rows; one normalizes to the same text as a Gastroenterology note).}
\end{tabular}
\end{table}
```
(Needs `\usepackage{booktabs}`.)
