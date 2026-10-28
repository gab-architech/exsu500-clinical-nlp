"""Freeze the train / validation / test split (Issue #2).

Reads the final labels (one row per unique note text, no note text) and writes
a stratified split with a fixed seed. Run once, from anywhere:

    python scripts/make_split.py

Input:  labels/final_labels.csv   columns: text_id, label
Output: splits/split_seed42.csv   columns: text_id, split  (train / val / test)

Duplicates: final_labels.csv already has one row per unique normalized text
(text_id = SHA-1 prefix of the normalized text, see docs/data_cleaning.md
Step 5). Every copy of a duplicated note shares one text_id, so all copies
automatically land in the same pile. No separate duplicate-group column is
needed or available.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]  # repo root, so paths work from any folder
LABELS_PATH = ROOT / "labels" / "final_labels.csv"
SPLIT_PATH = ROOT / "splits" / "split_seed42.csv"
RANDOM_SEED = 42


def main():
    parser = argparse.ArgumentParser(description="Make the frozen stratified split.")
    parser.add_argument("--val", type=float, default=0.15, help="validation share (default 0.15)")
    parser.add_argument("--test", type=float, default=0.15, help="test share (default 0.15)")
    parser.add_argument("--force", action="store_true", help="overwrite an existing split file")
    args = parser.parse_args()

    # The split is frozen: never overwrite it by accident
    if SPLIT_PATH.exists() and not args.force:
        sys.exit(f"{SPLIT_PATH} already exists. The split is frozen; use --force only if the team agreed.")

    labels = pd.read_csv(LABELS_PATH)

    # Sanity checks on the input file
    assert list(labels.columns) == ["text_id", "label"], f"Unexpected columns: {list(labels.columns)}"
    assert labels["text_id"].is_unique, "Duplicate text_id: duplicates would leak across piles"
    assert labels["label"].notna().all(), "Missing labels"

    # Step 1: carve out the test set, stratified by class
    rest, test = train_test_split(
        labels, test_size=args.test, stratify=labels["label"], random_state=RANDOM_SEED
    )
    # Step 2: carve the validation set out of what is left.
    # val share of the remainder = val / (1 - test), e.g. 0.15 / 0.85
    train, val = train_test_split(
        rest, test_size=args.val / (1 - args.test), stratify=rest["label"], random_state=RANDOM_SEED
    )

    split = pd.concat([
        train.assign(split="train"),
        val.assign(split="val"),
        test.assign(split="test"),
    ])

    # Save IDs + pile only (no labels needed here, no note text ever)
    SPLIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    split[["text_id", "split"]].sort_values("text_id").to_csv(SPLIT_PATH, index=False)

    # Per-class count table per pile, for the README and the midterm Data section
    table = pd.crosstab(split["label"], split["split"], margins=True, margins_name="total")
    table = table[["train", "val", "test", "total"]]
    print(f"Saved {len(split):,} rows to {SPLIT_PATH.relative_to(ROOT)}\n")
    print(table.to_string())


if __name__ == "__main__":
    main()
