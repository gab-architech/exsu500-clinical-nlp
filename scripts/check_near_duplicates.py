"""Check for near-duplicate notes across the frozen piles (Issue #2).

Exact duplicates cannot cross piles (one row per unique normalized text), but
near-duplicates can. This compares every pair of final notes with TF-IDF cosine
similarity and saves the cross-pile pairs at or above the threshold:

    python scripts/check_near_duplicates.py

Input:  data/mtsamples.csv (not committed, see data/README.md),
        labels/final_labels.csv, splits/split_seed42.csv
Output: results/near_duplicate_pairs.csv (IDs and labels only, no note text)

Team decision (Nov 5): keep the split, and report test macro-F1 both with and
without the test notes listed in this file. The threshold is fixed before any
modelling.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parents[1]  # repo root, so paths work from any folder
sys.path.insert(0, str(ROOT / "src"))
from data import load_notes, normalize_text  # noqa: E402  (shared note loading, src/data.py)

THRESHOLD = 0.8


def main():
    # All 2,230 final notes with their split (a data check, so every split is read)
    notes = load_notes().rename(columns={"note_id": "text_id"})
    notes["norm"] = notes["text"].map(normalize_text)
    assert len(notes) == 2230, f"Expected 2,230 notes, found {len(notes)}"

    # Cosine similarity of every pair; TF-IDF vectors are already unit length
    vectors = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True).fit_transform(notes["norm"])
    similarity = (vectors @ vectors.T).toarray()
    first, second = np.triu_indices_from(similarity, k=1)  # each pair once

    pairs = pd.DataFrame({
        "id_a": notes["text_id"].values[first],
        "id_b": notes["text_id"].values[second],
        "cosine": similarity[first, second].round(3),
        "split_a": notes["split"].values[first],
        "split_b": notes["split"].values[second],
        "label_a": notes["label"].values[first],
        "label_b": notes["label"].values[second],
    })
    pairs = pairs[(pairs["cosine"] >= THRESHOLD) & (pairs["split_a"] != pairs["split_b"])]
    pairs = pairs.sort_values("cosine", ascending=False)

    out = ROOT / "results" / "near_duplicate_pairs.csv"
    pairs.to_csv(out, index=False)

    # Which notes are affected in each held-out pile
    for pile in ["val", "test"]:
        affected = set(pairs.loc[pairs["split_a"] == pile, "id_a"]) | set(pairs.loc[pairs["split_b"] == pile, "id_b"])
        print(f"{pile}: {len(affected)} notes affected")
    print(f"{len(pairs)} cross-pile pairs at cosine >= {THRESHOLD}; saved to {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
