# Blind review statistics (Issue #14)

Gabriel Barrere blind-labelled a stratified sample of 220 notes from the final dataset (2,230 notes) without seeing the existing labels. IDs and labels only, no note text.

- Blind labels: `labels/blind_review_gabriel.csv`
- Answer key: labels from `labels/final_labels.csv`; stratum = `reviewed` if the ID is in `labels/reviewed_ids.csv` (the 606 notes reviewed by Jessica Oentoro and Karman Kang), `auto` otherwise
- None of the 3 second-review corrections (`docs/data_cleaning.md`, Appendix A.6) is in the sample, so the reviewed stratum compares directly with the two reviewers' labels
- Script: `scripts/blind_review_stats.py` (bootstrap: 2,000 resamples, seed 42; Wilson score interval for proportions)

## Results

| Stratum | n | Agreement | Statistic (95% CI) |
|---|---:|---:|---|
| Human-reviewed | 60 | 86.7% | Cohen's kappa 0.847 (0.742 to 0.941) |
| Auto-mapped | 160 | 86.3% | error rate 13.8%, 22 notes (9.3% to 19.9%) |
| Auto-mapped, sensitivity | 160 | 95.0% | error rate 5.0%, 8 notes (2.6% to 9.6%), excluding the 14 General Medicine disagreements |
| Pooled | 220 | 86.4% (190/220) | Cohen's kappa 0.848 (0.791 to 0.894) |

## Disagreements (30 of 220)

| Stratum | Answer key | Blind label | n |
|---|---|---|---:|
| auto | General Medicine | Orthopedic | 8 |
| auto | General Medicine | Cardiovascular / Pulmonary | 4 |
| auto | General Medicine | Neurology | 1 |
| auto | General Medicine | Obstetrics / Gynecology / Pediatrics | 1 |
| auto | Cardiovascular / Pulmonary | Hematology - Oncology | 1 |
| auto | Cardiovascular / Pulmonary | Obstetrics / Gynecology / Pediatrics | 1 |
| auto | ENT / Ophthalmology | General Medicine | 1 |
| auto | Gastroenterology | Cardiovascular / Pulmonary | 1 |
| auto | Gastroenterology | General Medicine | 1 |
| auto | Hematology - Oncology | Neurology | 1 |
| auto | Orthopedic | General Medicine | 1 |
| auto | Urology / Nephrology | Obstetrics / Gynecology / Pediatrics | 1 |
| reviewed | Neurology | Radiology | 4 |
| reviewed | Orthopedic | Radiology | 2 |
| reviewed | Neurology | Cardiovascular / Pulmonary | 1 |
| reviewed | Orthopedic | General Medicine | 1 |

## Interpretation

- **General Medicine (14 notes).** Auto-mapped notes filed under General Medicine were placed elsewhere by the blind reviewer. The class merges 8 raw labels by design (`CLASS_MAP`: General Medicine, Pain Management, Physical Medicine - Rehab, Sleep Medicine, Endocrinology, Rheumatology, Diets and Nutritions, Allergy / Immunology); the modelling lead confirmed the merge is deliberate (small labels with no better home). These are reported as a definitional difference; no labels were changed and the blind labels were kept as recorded. Which raw label each of the 14 came from was not checked (needs the raw dataset), so some may be raw General Medicine notes rather than merged ones.
- **Radiology (6 of 8 reviewed-stratum disagreements).** Imaging reports can reasonably be filed under the imaging specialty or the clinical one. A limitation of the class definition.
- **Decision:** no label fixes; the split (#2) will be made from `labels/final_labels.csv` unchanged since 47d04d8.

## Reproduce (from the repo root)

```python
import pandas as pd
f = pd.read_csv("labels/final_labels.csv"); r = pd.read_csv("labels/reviewed_ids.csv")
f["stratum"] = f.text_id.isin(r.text_id).map({True: "reviewed", False: "auto"})
f.to_csv("answer_key.csv", index=False)
```

then `python scripts/blind_review_stats.py --blind labels/blind_review_gabriel.csv --key answer_key.csv`. Pooled kappa: `cohen_kappa_score` and `bootstrap_kappa_ci` on all 220 notes.
