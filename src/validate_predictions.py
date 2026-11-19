"""Check a prediction file against the team's format (docs/predictions_format.md).

    python src/validate_predictions.py results/preds_tfidf_lr_42_val.csv [more files...]

A file passes when:
  - its name is preds_{model}_{seed}_{split}.csv (seed is an integer, split is train / val / test)
  - it has exactly the columns note_id, true_label, pred_label and one prob_<class> per class
  - note_ids are unique and match the frozen split for that split exactly (none missing, none extra)
  - true_label agrees with labels/final_labels.csv for every note
  - pred_label is one of the 11 classes
  - each row's probabilities are in [0, 1] and sum to about 1

Exits with code 1 if any file fails. Used by src/evaluate.py before any metric is computed.
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]  # repo root, so paths work from any folder
LABELS_PATH = ROOT / "labels" / "final_labels.csv"
SPLIT_PATH = ROOT / "splits" / "split_seed42.csv"
PROB_TOLERANCE = 0.01  # rows may sum to 0.99 to 1.01 (rounding when saving)
FILENAME = re.compile(r"^preds_(?P<model>.+)_(?P<seed>\d+)_(?P<split>train|val|test)\.csv$")


def load_reference():
    """Class list (alphabetical), true labels and frozen split, all keyed by note ID."""
    labels = pd.read_csv(LABELS_PATH)
    split = pd.read_csv(SPLIT_PATH)
    classes = sorted(labels["label"].unique())
    truth = dict(zip(labels["text_id"], labels["label"]))
    pile = dict(zip(split["text_id"], split["split"]))
    return classes, truth, pile


def parse_name(path):
    """Return (model, seed, split) from the file name, or None if the name is wrong."""
    m = FILENAME.match(Path(path).name)
    if not m:
        return None
    return m["model"], int(m["seed"]), m["split"]


def check_file(path, classes, truth, pile):
    """Return a list of problems; an empty list means the file passes."""
    problems = []
    parsed = parse_name(path)
    if parsed is None:
        return [f"file name must be preds_{{model}}_{{seed}}_{{split}}.csv, got {Path(path).name}"]
    _, _, split = parsed

    df = pd.read_csv(path)
    prob_cols = [f"prob_{c}" for c in classes]
    expected = ["note_id", "true_label", "pred_label"] + prob_cols
    missing = [c for c in expected if c not in df.columns]
    extra = [c for c in df.columns if c not in expected]
    if missing:
        problems.append(f"missing columns: {missing}")
    if extra:
        problems.append(f"unexpected columns: {extra}")
    if missing:
        return problems  # the checks below need every column

    # Note IDs: unique, and exactly the notes of this split
    if df["note_id"].duplicated().any():
        problems.append(f"{df['note_id'].duplicated().sum()} duplicated note_id values")
    expected_ids = {i for i, s in pile.items() if s == split}
    got_ids = set(df["note_id"])
    if got_ids - expected_ids:
        problems.append(f"{len(got_ids - expected_ids)} note_ids are not in the {split} split")
    if expected_ids - got_ids:
        problems.append(f"{len(expected_ids - got_ids)} {split} notes are missing")

    # Labels
    wrong_truth = (df["true_label"] != df["note_id"].map(truth)).sum()
    if wrong_truth:
        problems.append(f"{wrong_truth} rows have a true_label that differs from labels/final_labels.csv")
    bad_pred = (~df["pred_label"].isin(classes)).sum()
    if bad_pred:
        problems.append(f"{bad_pred} rows have a pred_label that is not one of the 11 classes")

    # Probabilities
    probs = df[prob_cols].to_numpy(dtype=float)
    if np.isnan(probs).any():
        problems.append("probabilities contain empty values")
    elif (probs < 0).any() or (probs > 1).any():
        problems.append("probabilities outside [0, 1]")
    else:
        off = np.abs(probs.sum(axis=1) - 1) > PROB_TOLERANCE
        if off.any():
            problems.append(f"{off.sum()} rows have probabilities that do not sum to 1 (tolerance {PROB_TOLERANCE})")
    return problems


def argmax_disagreements(df, classes):
    """Rows where pred_label is not the highest-probability class (allowed, but worth a warning)."""
    top = np.array(classes)[df[[f"prob_{c}" for c in classes]].to_numpy().argmax(axis=1)]
    return int((top != df["pred_label"].to_numpy()).sum())


def main():
    parser = argparse.ArgumentParser(description="Validate prediction files against the team format.")
    parser.add_argument("files", nargs="+")
    args = parser.parse_args()

    classes, truth, pile = load_reference()
    failed = False
    for path in args.files:
        problems = check_file(path, classes, truth, pile)
        if problems:
            failed = True
            print(f"FAIL {path}")
            for p in problems:
                print(f"  - {p}")
        else:
            n = argmax_disagreements(pd.read_csv(path), classes)
            note = f" (warning: pred_label is not the top-probability class in {n} rows)" if n else ""
            print(f"OK   {path}{note}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
