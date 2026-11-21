# Final manuscript (due Fri Dec 11; team target Fri Dec 4)

IEEE two-column, **6 to 8 pages**. Check on myCourses whether references count toward the limit (they did for the midterm). CRediT and GenAI statements are outside the count. One file per section, each with a named owner. Assembled in the IEEE Word template at the end (like `docs/midterm/`).

## Sections

| File | Section | Budget | Owner | Status |
|---|---|---|---|---|
| `00_title_authors.md` | Title, authors | | TBD | skeleton |
| `01_abstract.md` | Abstract (max 250 words) | 0.25 p | TBD | skeleton |
| `02_introduction.md` | I. Introduction and clinical motivation | 0.75 p | TBD | skeleton |
| `03_related_work.md` | II. Related work | 0.75 p | TBD | skeleton |
| `04_data.md` | III. Data | 1 p | TBD | skeleton |
| `05_methods.md` | IV. Methods | 1.25 p | TBD | skeleton |
| `06_results.md` | V. Results | 1.5 p | TBD | skeleton |
| `07_discussion_limitations.md` | VI. Discussion and limitations | 1 p | TBD | skeleton |
| `08_conclusion.md` | VII. Conclusion | 0.25 p | TBD | skeleton |
| `09_credit_genai.md` | CRediT and Use of GenAI | outside | TBD | skeleton |
| `10_references.md` | References | 1 p | TBD | skeleton |

## Figures and tables

| Item | Content | Source | Status |
|---|---|---|---|
| Fig 1 | Class distribution | `scripts/make_figure1.py` | ready |
| Fig 2 | Cleaning flow | `scripts/make_figure2_cleaning_flow.py` | ready |
| Fig 3 | Model comparison, macro-F1 with 95% CI | `scripts/make_figure3_comparison.py --split test` | script ready; test numbers Nov 25 |
| Fig 4 | Confusion matrix of the best model (error analysis) | `src/evaluate.py --split test --final` | Nov 25 |
| Table 1 | Cleaning steps | `docs/midterm_numbers.md` | ready |
| Table 2 | Model comparison with CIs | `scripts/make_results_tables.py` | val ready; test Nov 25 |
| Table 3 | Leakage experiment | `results/table_masking_val.md` | ready |

## Rules

- Every number comes from a committed file (`results/`, `docs/midterm_numbers.md`, `docs/results_validation.md`); cite the file in a comment while drafting.
- Headline results are **test** numbers, read once after the Nov 25 lock. Validation numbers are labelled as validation.
- Every reference has a literature note with a named reader.
- No em dashes. No note text from MTSamples.
- The checklist in the repo `CLAUDE.md` ("Final manuscript checklist") is the definition of done.
