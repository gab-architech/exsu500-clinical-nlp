"""Blind review statistics (Issue #14).

Compares Gabriel's blind labels with the answer key and reports:
  - Cohen's kappa with a 95% bootstrap CI, on the human-reviewed stratum (~60 notes)
  - error rate with a 95% Wilson CI, on the auto-mapped stratum (~160 notes)
  - a table of every disagreement (answer key label vs blind label)

    python scripts/blind_review_stats.py --blind my_blind_labels.csv --key answer_key.csv

Inputs (ID + label files only, no note text):
  --blind  columns: text_id, label  (or final_label, as exported by the review tool)
  --key    columns: text_id, label, stratum   (stratum = "reviewed" or "auto")

The answer key's label is the label currently in labels/final_labels.csv;
stratum says how that label was produced.
"""

import argparse
import math

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

RANDOM_SEED = 42
N_BOOTSTRAP = 2000


def bootstrap_kappa_ci(key_labels, blind_labels, n_boot=N_BOOTSTRAP, seed=RANDOM_SEED):
    """95% CI for Cohen's kappa: resample notes with replacement, take the 2.5th and 97.5th percentiles."""
    rng = np.random.default_rng(seed)
    key_labels, blind_labels = np.asarray(key_labels), np.asarray(blind_labels)
    n = len(key_labels)
    kappas = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        k = cohen_kappa_score(key_labels[idx], blind_labels[idx])
        if not math.isnan(k):  # a resample with a single class has undefined kappa
            kappas.append(k)
    return np.percentile(kappas, 2.5), np.percentile(kappas, 97.5)


def wilson_ci(errors, n, z=1.96):
    """95% Wilson score interval for a proportion; behaves well with few or zero errors."""
    p = errors / n
    centre = (p + z**2 / (2 * n)) / (1 + z**2 / n)
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def main():
    parser = argparse.ArgumentParser(description="Kappa, error rate and disagreements for the blind review.")
    parser.add_argument("--blind", required=True, help="CSV with text_id, label (Gabriel's blind labels)")
    parser.add_argument("--key", required=True, help="CSV with text_id, label, stratum (answer key)")
    args = parser.parse_args()

    blind = pd.read_csv(args.blind)
    # The review tool exports the label column as "final_label"; accept either name
    blind = blind.rename(columns={"final_label": "label"})[["text_id", "label"]].rename(columns={"label": "blind"})
    key = pd.read_csv(args.key)[["text_id", "label", "stratum"]].rename(columns={"label": "key"})

    # Labels compared as plain strings, trimmed (DROP counts as its own label)
    for df, col in [(blind, "blind"), (key, "key")]:
        df[col] = df[col].astype(str).str.strip()

    merged = key.merge(blind, on="text_id", how="left")
    missing = merged["blind"].isna().sum()
    if missing:
        print(f"WARNING: {missing} answer-key notes have no blind label; they are left out.\n")
    merged = merged.dropna(subset=["blind"])
    merged["agree"] = merged["key"] == merged["blind"]

    # 1. Inter-rater agreement on the human-reviewed stratum
    rev = merged[merged["stratum"] == "reviewed"]
    if len(rev):
        kappa = cohen_kappa_score(rev["key"], rev["blind"])
        lo, hi = bootstrap_kappa_ci(rev["key"], rev["blind"])
        print(f"Human-reviewed notes (n={len(rev)})")
        print(f"  Raw agreement: {rev['agree'].mean():.1%}")
        print(f"  Cohen's kappa: {kappa:.3f} (95% CI {lo:.3f} to {hi:.3f}, {N_BOOTSTRAP} bootstrap resamples)\n")

    # 2. Error rate of the auto-mapped labels, taking the blind label as reference
    auto = merged[merged["stratum"] == "auto"]
    if len(auto):
        errors = int((~auto["agree"]).sum())
        lo, hi = wilson_ci(errors, len(auto))
        print(f"Auto-mapped notes (n={len(auto)})")
        print(f"  Disagreements: {errors}")
        print(f"  Error rate: {errors / len(auto):.1%} (95% Wilson CI {lo:.1%} to {hi:.1%})\n")

    # 3. Every disagreement, grouped by stratum and label pair
    dis = merged[~merged["agree"]]
    print(f"Disagreements: {len(dis)} of {len(merged)}")
    if len(dis):
        table = dis.groupby(["stratum", "key", "blind"]).size().rename("n").reset_index()
        table = table.rename(columns={"key": "answer key", "blind": "blind label"})
        print(table.sort_values(["stratum", "n"], ascending=[True, False]).to_string(index=False))


if __name__ == "__main__":
    main()
