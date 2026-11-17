"""Load the final notes (text, label, split) from the raw MTSamples file. Shared by every model.

    from data import load_split_notes
    train = load_split_notes("train")   # columns: note_id, text, label
    val = load_split_notes("val")

Rebuilds the note text for each ID in labels/final_labels.csv, because the repo stores IDs
and labels only (no note text, course rule). Decisions (docs/clinicalbert_pipeline.md, section 1):
  - note ID = first 8 characters of the SHA-1 of the normalized text, exactly as in the cleaning
    pipeline (notebooks/02_eda.ipynb, Section 5)
  - when several raw rows share an ID, the first one in file order is used
  - the text is returned as written (not lowercased or normalized); keywords and description are never used

The test split is refused unless final=True, which is only for the single run after the model lock (Nov 25).
"""

import hashlib
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]  # repo root, so paths work from any folder
RAW_PATH = ROOT / "data" / "mtsamples.csv"  # downloaded from Kaggle, never committed (data/README.md)
LABELS_PATH = ROOT / "labels" / "final_labels.csv"
SPLIT_PATH = ROOT / "splits" / "split_seed42.csv"
EXPECTED = {"train": 1560, "val": 335, "test": 335}


def normalize_text(s):
    """Lowercase, strip punctuation, collapse whitespace (same as notebooks/02_eda.ipynb, Section 5)."""
    if not isinstance(s, str):
        return ""
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def text_id(text):
    """Stable note ID: first 8 characters of the SHA-1 of the normalized text."""
    return hashlib.sha1(normalize_text(text).encode()).hexdigest()[:8]


def load_notes(raw_path=RAW_PATH):
    """All 2,230 final notes with note_id, text, label and split.

    For data checks only (e.g. the near-duplicate check). Model code must use load_split_notes,
    which keeps the test split closed.
    """
    raw = pd.read_csv(raw_path)
    raw = raw[raw["transcription"].notna()].copy()
    raw["note_id"] = raw["transcription"].map(text_id)
    first = raw.drop_duplicates("note_id", keep="first")[["note_id", "transcription"]]

    labels = pd.read_csv(LABELS_PATH).rename(columns={"text_id": "note_id"})
    split = pd.read_csv(SPLIT_PATH).rename(columns={"text_id": "note_id"})
    notes = labels.merge(split, on="note_id").merge(first, on="note_id", how="left")

    missing = notes["transcription"].isna().sum()
    assert missing == 0, f"{missing} final notes not found in {raw_path}; is it the Kaggle mtsamples.csv?"
    return notes.rename(columns={"transcription": "text"})[["note_id", "text", "label", "split"]]


def load_split_notes(split, final=False, raw_path=RAW_PATH):
    """Notes of one split (note_id, text, label), with a count check. The test split needs final=True."""
    if split not in EXPECTED:
        raise ValueError(f"split must be train, val or test, got {split!r}")
    if split == "test" and not final:
        raise PermissionError("The test split is read once, after the model lock (Nov 25). Pass final=True only for that run.")
    notes = load_notes(raw_path)
    part = notes[notes["split"] == split].drop(columns="split").reset_index(drop=True)
    assert len(part) == EXPECTED[split], f"expected {EXPECTED[split]} {split} notes, found {len(part)}"
    return part


if __name__ == "__main__":
    # Quick check: counts per split (prints no note text)
    for s in ["train", "val"]:
        print(s, len(load_split_notes(s)))
