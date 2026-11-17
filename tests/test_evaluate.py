"""Sanity tests for src/evaluate.py on dummy predictions (written to a temporary folder, never committed).

    python tests/test_evaluate.py          # or: pytest tests/

Checks:
  1. a perfect model scores macro-F1 = 1
  2. a random model (3 seeds) scores about 1/11
  3. the pooled multi-seed CI resamples note IDs and averages macro-F1 across seeds within each
     resample (recomputed here independently with scikit-learn and compared with the summary)
  4. the test split is refused without --final, and a --final run is logged
"""

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
EVALUATE = ROOT / "src" / "evaluate.py"
SEEDS = (1, 2, 3)


def make_dummy(folder, split):
    """A perfect model (seed 1) and a random model (seeds 1, 2, 3) for one split, from the frozen labels."""
    labels = pd.read_csv(ROOT / "labels" / "final_labels.csv")
    notes = labels.merge(pd.read_csv(ROOT / "splits" / "split_seed42.csv"), on="text_id")
    notes = notes[notes["split"] == split].reset_index(drop=True)
    classes = sorted(labels["label"].unique())

    perfect = pd.DataFrame({"note_id": notes["text_id"], "true_label": notes["label"], "pred_label": notes["label"]})
    for c in classes:
        perfect[f"prob_{c}"] = (notes["label"] == c).astype(float)
    perfect.to_csv(folder / f"preds_perfect_1_{split}.csv", index=False)

    for seed in SEEDS:
        probs = np.random.default_rng(seed).dirichlet(np.ones(len(classes)), size=len(notes))
        rand = pd.DataFrame({"note_id": notes["text_id"], "true_label": notes["label"],
                             "pred_label": np.array(classes)[probs.argmax(axis=1)]})
        for i, c in enumerate(classes):
            rand[f"prob_{c}"] = probs[:, i]
        # shuffle rows, so the test also checks that files are aligned by note_id, not by row order
        rand.sample(frac=1, random_state=seed).to_csv(folder / f"preds_random_{seed}_{split}.csv", index=False)
    return classes


def run(folder, *extra):
    return subprocess.run([sys.executable, str(EVALUATE), "--pred-dir", str(folder), "--out-dir", str(folder),
                           "--fig-dir", str(folder / "figs"), *extra], capture_output=True, text=True, check=False)


def expected_pooled_ci(folder, classes):
    """Independent recomputation: resample note IDs, macro-F1 per seed on that resample, average over seeds."""
    files = [pd.read_csv(folder / f"preds_random_{s}_val.csv").set_index("note_id") for s in SEEDS]
    ids = np.array(sorted(files[0].index))
    draws = np.random.default_rng(42).integers(0, len(ids), size=(1000, len(ids)))  # same draws as evaluate.py
    pooled = []
    for d in draws:
        sample = ids[d]  # resampled note IDs (with repeats)
        per_seed = [f1_score(f.loc[sample, "true_label"], f.loc[sample, "pred_label"], labels=classes,
                             average="macro", zero_division=0) for f in files]
        pooled.append(np.mean(per_seed))
    return np.percentile(pooled, [2.5, 97.5])


def test_evaluate_on_dummy_predictions():
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        classes = make_dummy(folder, "val")
        make_dummy(folder, "test")

        result = run(folder, "--split", "val")
        assert result.returncode == 0, result.stderr + result.stdout
        summary = pd.read_csv(folder / "summary_val.csv")
        row = lambda model: summary[(summary["model"] == model) & (summary["subset"] == "all")].iloc[0]  # noqa: E731

        # 1. perfect model
        assert row("perfect")["macro_f1_mean"] == 1.0
        assert row("perfect")["accuracy_mean"] == 1.0

        # 2. random model, 3 seeds: about 1/11 (slightly below, because classes are imbalanced)
        rand = row("random")
        assert rand["n_seeds"] == 3
        assert abs(rand["macro_f1_mean"] - 1 / 11) < 0.03, rand["macro_f1_mean"]

        # 3. pooled CI matches the independent recomputation (summary is rounded to 4 decimals)
        low, high = expected_pooled_ci(folder, classes)
        assert abs(rand["macro_f1_ci_low"] - low) < 1e-4, (rand["macro_f1_ci_low"], low)
        assert abs(rand["macro_f1_ci_high"] - high) < 1e-4, (rand["macro_f1_ci_high"], high)

        # 4. test-set guard
        refused = run(folder, "--split", "test")
        assert refused.returncode != 0 and "Refusing" in refused.stderr
        assert not (folder / "summary_test.csv").exists()
        final = run(folder, "--split", "test", "--final")
        assert final.returncode == 0, final.stderr + final.stdout
        assert len((folder / "test_runs.log").read_text().splitlines()) == 1


if __name__ == "__main__":
    test_evaluate_on_dummy_predictions()
    print("All evaluation sanity tests passed.")
