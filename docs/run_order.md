# Run order

The repository is a numbered pipeline. Each step consumes the previous step's committed
outputs and nothing else. Steps 03 and 04 are the only ones that fit models; everything after
them only reads predictions or metrics. The test split is read exactly once, by step 10,
after the Nov 25 model lock.

| Step | Command | Where | Produces |
|---|---|---|---|
| 01 | `python scripts/make_split.py` | local | `splits/split_seed42.csv` (frozen; already run) |
| 02 | `python scripts/check_near_duplicates.py` | local | `results/near_duplicate_pairs.csv` (needs the dataset) |
| 03 | `python src/train_baselines.py` | local (CPU) | baseline val predictions, logs, `models/*_42.joblib` |
| 04 | `python src/train_clinicalbert.py --seed 42 --output-dir <drive> --results-dir <repo>/results` (repeat 43, 44, and `--masked`) | Colab (GPU) | ClinicalBERT checkpoints, val predictions, run logs |
| 05 | `python src/evaluate.py --split val` | local | `results/metrics/*.json`, `summary_val.csv/.md`, figures |
| 06 | `python scripts/make_results_tables.py` | local | `results/table_manuscript_val.md`, `table_masking_val.md` |
| 07 | `python scripts/class_threshold_metrics.py` | local | `results/metrics/threshold_metrics_val.csv`, `table_threshold_metrics_val.md` |
| 08 | `python scripts/make_figure1.py` and `make_figure2_cleaning_flow.py` | local | `figures/fig1_*`, `fig2_*` |
| 09 | `python scripts/blind_review_stats.py --blind <file> --key <file>` | local | blind-review agreement stats |
| 10 | `python src/predict_test.py --verify`, then `python src/predict_test.py --final`, then `python src/evaluate.py --split test --final` | Colab (pinned env) | test predictions, test metrics and summary (single run) |

Steps 01 and 02 read all 2,230 notes for data checks; step 10 is the only step that reads
the test split for a modelling output. `src/data.py` refuses the test split unless
`final=True`, and the two scripts that accept `--final` (`src/predict_test.py` and
`src/evaluate.py`) each log the run to `results/test_runs.log`. Step 10 must run where the
frozen artifacts are: the `models/*_42.joblib` baselines and the
`models/clinicalbert_seed{42,43,44}/` checkpoints (on Colab, pass `--checkpoint-dir` at the
Drive copy saved by step 04, or copy them into `models/` first). The ClinicalBERT loader
accepts either `clinicalbert_seed{N}/` or the `clinicalbert_seed{N}/checkpoint-<step>/`
layout the Trainer writes, so a folder copied straight from Drive works as is. These model
files are git-ignored by design (the course rule keeps model weights out of the repository):
a fresh clone regenerates them by re-running steps 03 and 04, which write these exact names.

`src/` holds the pipeline entry points plus the shared library (`data.py`, `masking.py`,
`validate_predictions.py`); `scripts/` holds one-off analysis and figure scripts. The
notebooks (`notebooks/01_data_check.ipynb`, `02_eda.ipynb`, `05_clinicalbert_masked.ipynb`)
cover data exploration and the masked ClinicalBERT run.
