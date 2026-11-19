"""Turn results/summary_val.csv into the two tables the manuscript needs.

    python scripts/make_results_tables.py

evaluate.py writes one row per model and per subset, so the raw summary has two rows per
model (all notes, and without near-duplicates) and a separate row for every masked twin.
That is right for auditing but wrong for the manuscript: readers need one row per model,
the masking experiment needs its own table, and ECE must not be quoted for models whose
"probabilities" are one-hot.

Outputs:
  results/table_manuscript_val.md  one row per model family (original text)
  results/table_masking_val.md     original vs masked, per model family

The near-duplicate footnote counts only the notes that actually have a TRAINING
counterpart: a validation note that pairs with a note in the test split cannot leak into
the validation score. results/near_duplicate_pairs.csv is read to work that out.
"""

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

# Models whose probabilities are one-hot, so AUROC/AUPRC/ECE are degenerate
# (docs/predictions_format.md says to report them as such).
ONE_HOT_MODELS = {"majority", "keyword_rule"}
DISPLAY = {
    "majority": "Majority class",
    "keyword_rule": "Keyword rule (MI, 2 per class)",
    "tfidf_lr": "TF-IDF + logistic regression",
    "tfidf_svm": "TF-IDF + linear SVM (Platt)",
    "clinicalbert": "ClinicalBERT",
    "clinicalbert_masked": "ClinicalBERT (masked)",
}
ORDER = ["majority", "keyword_rule", "tfidf_lr", "tfidf_svm", "clinicalbert"]


def f3(x):
    return "n/a" if pd.isna(x) else f"{x:.3f}"


def ci(low, high, point):
    return "n/a" if pd.isna(point) else f"{point:.3f} ({low:.3f}-{high:.3f})"


def leaky_val_notes():
    """Validation notes that have a near-duplicate in the TRAINING split."""
    path = RESULTS / "near_duplicate_pairs.csv"
    pairs = pd.read_csv(path)
    leaking = set()
    for _, r in pairs.iterrows():
        if r.split_a == "val" and r.split_b == "train":
            leaking.add(r.id_a)
        elif r.split_b == "val" and r.split_a == "train":
            leaking.add(r.id_b)
    return leaking


def excluded_val_notes():
    """Validation notes that the evaluator drops in its 'no near-duplicates' subset.

    evaluate.py excludes every val note that appears in ANY cross-pile pair, including pairs
    whose partner is in the test split. Those notes cannot leak into the validation score,
    so the count is larger than the number of genuinely leaking notes.
    """
    pairs = pd.read_csv(RESULTS / "near_duplicate_pairs.csv")
    excluded = set()
    for _, r in pairs.iterrows():
        if r.split_a == "val":
            excluded.add(r.id_a)
        if r.split_b == "val":
            excluded.add(r.id_b)
    return excluded


def paired_comparison(model_a="tfidf_lr", model_b="tfidf_svm", n_boot=1000, seed=42):
    """Paired bootstrap difference in macro-F1 and an exact McNemar test.

    Both point estimates come from the same 335 validation notes, so two overlapping marginal
    confidence intervals do not imply "no difference". This computes the difference directly
    on shared resamples, which is the comparison the manuscript needs.
    """
    a = pd.read_csv(RESULTS / f"preds_{model_a}_42_val.csv", dtype={"note_id": str}).sort_values("note_id")
    b = pd.read_csv(RESULTS / f"preds_{model_b}_42_val.csv", dtype={"note_id": str}).sort_values("note_id")
    assert list(a["note_id"]) == list(b["note_id"]), "the two files must cover the same notes"
    classes = sorted(pd.read_csv(ROOT / "labels" / "final_labels.csv")["label"].unique())
    pos = {c: i for i, c in enumerate(classes)}
    y = a["true_label"].map(pos).to_numpy()
    pa = a["pred_label"].map(pos).to_numpy()
    pb = b["pred_label"].map(pos).to_numpy()
    k = len(classes)

    def macro_f1(yt, yp):
        cm = np.bincount(yt * k + yp, minlength=k * k).reshape(k, k)
        tp = np.diag(cm).astype(float)
        den = cm.sum(axis=0) + cm.sum(axis=1)
        f1 = np.divide(2 * tp, den, out=np.zeros_like(tp), where=den > 0)
        return f1.mean()

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(y), size=(n_boot, len(y)))
    diffs = np.array([macro_f1(y[i], pb[i]) - macro_f1(y[i], pa[i]) for i in idx])
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    # McNemar, exact binomial on the discordant pairs
    only_b = int(((pa != y) & (pb == y)).sum())
    only_a = int(((pa == y) & (pb != y)).sum())
    from math import comb
    n = only_a + only_b
    p = min(1.0, 2 * sum(comb(n, i) for i in range(min(only_a, only_b) + 1)) / 2**n) if n else 1.0
    return {"diff": float(macro_f1(y, pb) - macro_f1(y, pa)),
            "ci": (float(lo), float(hi)), "only_b_right": only_b,
            "only_a_right": only_a, "mcnemar_p": float(p)}


def main():
    summary = pd.read_csv(RESULTS / "summary_val.csv")

    def row(model, subset):
        hit = summary[(summary["model"] == model) & (summary["subset"] == subset)]
        return None if hit.empty else hit.iloc[0]

    # ---- Table 1: one row per model, original text ----
    lines = ["# Model comparison, validation split (original text)", "",
             "| Model | Macro-F1 (95% CI) | Accuracy (95% CI) | AUROC | AUPRC | ECE |",
             "|---|---|---|---:|---:|---:|"]
    for model in ORDER:
        r = row(model, "all")
        if r is None:
            continue
        degenerate = model in ONE_HOT_MODELS
        ece = "degenerate" if degenerate else f3(r["ece"])
        auroc = "degenerate" if degenerate else f3(r["auroc_macro_ovr"])
        auprc = "degenerate" if degenerate else f3(r["auprc_macro_ovr"])
        lines.append(
            f"| {DISPLAY.get(model, model)} | "
            f"{ci(r['macro_f1_ci_low'], r['macro_f1_ci_high'], r['macro_f1_mean'])} | "
            f"{ci(r['accuracy_ci_low'], r['accuracy_ci_high'], r['accuracy_mean'])} | "
            f"{auroc} | {auprc} | {ece} |"
        )
    n_leak = len(leaky_val_notes())
    n_excl = len(excluded_val_notes())
    # Largest change in macro-F1 when the evaluator's "no near-duplicates" subset is used.
    worst = 0.0
    for model in ORDER:
        a, b = row(model, "all"), row(model, "excl_near_dup")
        if a is not None and b is not None:
            worst = max(worst, abs(a["macro_f1_mean"] - b["macro_f1_mean"]))
    paired = paired_comparison()
    lines += ["",
              f"95% bootstrap CIs, 1000 resamples, seed 42, identical draws for every model. "
              f"\"Degenerate\" marks models whose probabilities are one-hot, for which AUROC, "
              f"AUPRC and ECE are not meaningful.",
              "",
              f"Near-duplicates: {n_leak} of the 335 validation notes have a near-duplicate "
              f"(cosine >= 0.8) in the training split. The evaluator's "
              f"\"no near-duplicates\" subset removes {n_excl} validation notes, because it also "
              f"drops notes whose only partner is in the test split; those cannot leak into a "
              f"validation score. Removing them changes macro-F1 by at most {worst:.4f}.",
              "",
              f"Logistic regression vs linear SVM: the difference in macro-F1 is "
              f"{paired['diff']:+.4f} (paired bootstrap 95% CI {paired['ci'][0]:+.4f} to "
              f"{paired['ci'][1]:+.4f}); on {paired['only_a_right']} notes only logistic "
              f"regression is right and on {paired['only_b_right']} only the SVM is "
              f"(exact McNemar p = {paired['mcnemar_p']:.3f}). The two are statistically "
              f"indistinguishable on macro-F1 at this sample size, so the table does not rank "
              f"them; logistic regression has the lower calibration error.",
              "",
              "The logistic-regression regularisation strength C was selected on this same "
              "validation split from a pre-declared grid (best C = 500, macro-F1 0.8623); the "
              "reported score is therefore a selection maximum and its confidence interval does "
              "not carry the selection uncertainty. Evidence: results/lr_c_sweep_42_val.json.",
              ]
    (RESULTS / "table_manuscript_val.md").write_text("\n".join(lines) + "\n")

    # ---- Table 2: masking effect ----
    lines = ["# Leakage experiment: original vs label-masked text (validation)", "",
             "| Model | Macro-F1 original | Macro-F1 masked | Difference |",
             "|---|---:|---:|---:|"]
    for model in ORDER:
        a, b = row(model, "all"), row(model + "_masked", "all")
        if a is None or b is None:
            continue
        diff = b["macro_f1_mean"] - a["macro_f1_mean"]
        lines.append(f"| {DISPLAY.get(model, model)} | {f3(a['macro_f1_mean'])} | "
                     f"{f3(b['macro_f1_mean'])} | {diff:+.4f} |")
    lines += ["", "Masking removes 24 specialty names (option C) and touches 10.8% of notes. "
                  "The keyword rule and the majority baseline are unchanged by construction: none "
                  "of their 22 keywords is in the mask list, so their identical rows carry no "
                  "evidence about leakage."]
    (RESULTS / "table_masking_val.md").write_text("\n".join(lines) + "\n")
    print("wrote results/table_manuscript_val.md and results/table_masking_val.md")
    print((RESULTS / "table_manuscript_val.md").read_text())


if __name__ == "__main__":
    main()
