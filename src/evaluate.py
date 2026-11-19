"""Evaluate every model's predictions for one split (the team's single evaluation path).

    python src/evaluate.py --split val
    python src/evaluate.py --split test --final      # the one final test-set run

Reads every results/preds_{model}_{seed}_{split}.csv (format: docs/predictions_format.md),
validates them all first, then writes:
  - results/metrics/{model}_{seed}_{split}.json   every metric, on all notes and without near-duplicates
  - results/summary_{split}.csv and .md           one row per model and subset, plus the CLiGNet reference row
  - figures/{model}_{seed}_{split}_{confusion,reliability,per_class_f1}.png  (300 dpi, IEEE column width)

Test-set guard: --split test is refused unless --final is given, and every --final run is
appended to results/test_runs.log with a timestamp. We read the test set once.
"""

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display needed (works in Colab and locally)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from validate_predictions import check_file, load_reference, parse_name  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]  # repo root, so paths work from any folder
NEAR_DUP_PATH = ROOT / "results" / "near_duplicate_pairs.csv"  # from scripts/check_near_duplicates.py
RANDOM_SEED = 42
N_BOOT = 1000
N_BINS = 15  # equal-width confidence bins for ECE and the reliability diagram
FIG_WIDTH = 3.5  # inches, one IEEE column
CLIGNET = {"macro_f1": 0.279, "note": "Reference only (Barman and Barman 2026): 40 raw MTSamples labels and a different split, so not directly comparable with our 11 classes."}

# Short class names so 11 labels fit on a one-column figure
SHORT = {
    "Cardiovascular / Pulmonary": "Cardio/Pulm",
    "ENT / Ophthalmology": "ENT/Ophth",
    "Gastroenterology": "GI",
    "General Medicine": "Gen Med",
    "Hematology - Oncology": "Heme-Onc",
    "Neurology": "Neuro",
    "Obstetrics / Gynecology / Pediatrics": "OB/GYN/Peds",
    "Orthopedic": "Ortho",
    "Psychiatry / Psychology": "Psych",
    "Radiology": "Radiology",
    "Urology / Nephrology": "Uro/Neph",
}


# ---------- metrics ----------

def confusion(y_true, y_pred, k):
    """k x k count matrix, rows = true class, columns = predicted class."""
    return np.bincount(y_true * k + y_pred, minlength=k * k).reshape(k, k)


def macro_f1_from_cm(cm):
    """Macro-F1 over all k classes; a class with no true and no predicted notes scores 0 (sklearn zero_division=0)."""
    tp = np.diag(cm).astype(float)
    denom = cm.sum(axis=0) + cm.sum(axis=1)  # (tp + fp) + (tp + fn)
    f1 = np.divide(2 * tp, denom, out=np.zeros_like(tp), where=denom > 0)
    return f1.mean()


def ece_and_bins(confidence, correct, n_bins=N_BINS):
    """Expected calibration error on the top-class confidence, plus the per-bin values for the plot."""
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(confidence, edges[1:-1], right=True), 0, n_bins - 1)
    bins, ece = [], 0.0
    for b in range(n_bins):
        mask = idx == b
        if mask.any():
            acc, conf = correct[mask].mean(), confidence[mask].mean()
            ece += mask.mean() * abs(acc - conf)
            bins.append({"low": edges[b], "high": edges[b + 1], "n": int(mask.sum()), "accuracy": acc, "confidence": conf})
    return ece, bins


def one_vs_rest(y_true, probs, k):
    """Per-class AUROC and AUPRC; a class without both positive and negative notes is skipped (None)."""
    auroc, auprc = [], []
    for c in range(k):
        positive = (y_true == c).astype(int)
        if 0 < positive.sum() < len(positive):
            auroc.append(roc_auc_score(positive, probs[:, c]))
            auprc.append(average_precision_score(positive, probs[:, c]))
        else:
            auroc.append(None)
            auprc.append(None)
    return auroc, auprc


def bootstrap_indices(n, n_boot=N_BOOT, seed=RANDOM_SEED):
    """The same resamples for every file of a split, so models are compared on identical draws (paired)."""
    return np.random.default_rng(seed).integers(0, n, size=(n_boot, n))


def bootstrap_scores(y_true, y_pred, k, idx):
    """Macro-F1 and accuracy on each bootstrap resample."""
    f1 = np.array([macro_f1_from_cm(confusion(y_true[i], y_pred[i], k)) for i in idx])
    acc = (y_true[idx] == y_pred[idx]).mean(axis=1)
    return f1, acc


def ci(values):
    """95% percentile interval."""
    lo, hi = np.percentile(values, [2.5, 97.5])
    return [float(lo), float(hi)]


def evaluate(df, classes, idx):
    """Every metric for one set of notes. df is sorted by note_id. Returns (metrics, bootstrap arrays)."""
    k = len(classes)
    pos = {c: i for i, c in enumerate(classes)}
    y_true = df["true_label"].map(pos).to_numpy()
    y_pred = df["pred_label"].map(pos).to_numpy()
    probs = df[[f"prob_{c}" for c in classes]].to_numpy(dtype=float)

    cm = confusion(y_true, y_pred, k)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, row_sums, out=np.zeros(cm.shape), where=row_sums > 0)
    p, r, f, support = precision_recall_fscore_support(y_true, y_pred, labels=range(k), zero_division=0)
    auroc, auprc = one_vs_rest(y_true, probs, k)
    confidence = probs.max(axis=1)
    ece, bins = ece_and_bins(confidence, (probs.argmax(axis=1) == y_true).astype(float))
    boot_f1, boot_acc = bootstrap_scores(y_true, y_pred, k, idx)

    defined = [a for a in auroc if a is not None]
    metrics = {
        "n_notes": len(df),
        "macro_f1": float(macro_f1_from_cm(cm)),
        "macro_f1_ci95": ci(boot_f1),
        "accuracy": float((y_true == y_pred).mean()),
        "accuracy_ci95": ci(boot_acc),
        "auroc_macro_ovr": float(np.mean(defined)) if defined else None,
        "auprc_macro_ovr": float(np.mean([a for a in auprc if a is not None])) if defined else None,
        "n_classes_in_auc": len(defined),
        "ece": float(ece),
        "per_class": {
            c: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(support[i]),
                "auroc": auroc[i], "auprc": auprc[i]}
            for i, c in enumerate(classes)
        },
        "confusion_counts": cm.tolist(),
        "confusion_row_normalized": cm_norm.round(4).tolist(),
        "reliability_bins": bins,
    }
    return metrics, (boot_f1, boot_acc)


# ---------- figures ----------

def save_figures(m, classes, stem, fig_dir):
    """Confusion matrix (row-normalized), reliability diagram and per-class F1, one IEEE column wide."""
    names = [SHORT.get(c, c) for c in classes]
    plt.rcParams.update({"font.size": 6, "axes.titlesize": 7, "axes.labelsize": 6.5})

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, 3.3))
    cm = np.array(m["confusion_row_normalized"])
    ax.imshow(cm, cmap="Greys", vmin=0, vmax=1)
    for i in range(len(classes)):
        for j in range(len(classes)):
            if cm[i, j] >= 0.005:
                ax.text(j, i, f"{cm[i, j] * 100:.0f}", ha="center", va="center", fontsize=4.5,
                        color="white" if cm[i, j] > 0.5 else "black")
    ax.set_xticks(range(len(classes)), names, rotation=60, ha="right", fontsize=5)
    ax.set_yticks(range(len(classes)), names, fontsize=5)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title("Confusion matrix (% of each true class)")
    fig.tight_layout()
    fig.savefig(fig_dir / f"{stem}_confusion.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, 2.6))
    bins = m["reliability_bins"]
    ax.plot([0, 1], [0, 1], linestyle="--", color="0.6", linewidth=0.8, label="Perfect calibration")
    ax.plot([b["confidence"] for b in bins], [b["accuracy"] for b in bins], marker="o", markersize=3,
            color="0.15", linewidth=1, label="Model")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Mean predicted confidence (top class)")
    ax.set_ylabel("Observed accuracy")
    ax.set_title(f"Reliability diagram (ECE = {m['ece']:.3f}, {N_BINS} bins)")
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(fig_dir / f"{stem}_reliability.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, 2.6))
    f1 = [m["per_class"][c]["f1"] for c in classes]
    ax.barh(names, f1, color="0.35", height=0.7)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("F1")
    ax.set_title(f"Per-class F1 (macro-F1 = {m['macro_f1']:.3f})")
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(fig_dir / f"{stem}_per_class_f1.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------- test-set guard ----------

def log_final_run(log_path, files):
    """Append one line per --final run: UTC time, git commit and the files evaluated."""
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False).stdout.strip()
    except OSError:
        commit = "unknown"
    previous = log_path.read_text().count("\n") if log_path.exists() else 0
    if previous:
        print(f"WARNING: the test set has already been evaluated {previous} time(s); see {log_path}")
    with open(log_path, "a") as fh:
        fh.write(f"{datetime.now(UTC).isoformat(timespec='seconds')}\tcommit={commit}\t{','.join(f.name for f in files)}\n")


# ---------- main ----------

def main():
    parser = argparse.ArgumentParser(description="Evaluate all prediction files for one split.")
    parser.add_argument("--split", required=True, choices=["train", "val", "test"])
    parser.add_argument("--final", action="store_true", help="required to evaluate the test split (logged)")
    parser.add_argument("--pred-dir", type=Path, default=ROOT / "results", help="where the preds_*.csv files are")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results", help="where metrics, summaries and the test log go")
    parser.add_argument("--fig-dir", type=Path, default=ROOT / "figures")
    args = parser.parse_args()

    # 1. Test-set guard, before anything is read
    if args.split == "test" and not args.final:
        sys.exit("Refusing to evaluate the test split without --final. The test set is read once, at the end.")
    if args.final and args.split != "test":
        sys.exit("--final is only for the test split.")

    files = sorted(args.pred_dir.glob(f"preds_*_{args.split}.csv"))
    if not files:
        sys.exit(f"No preds_*_{args.split}.csv files in {args.pred_dir}")

    # 2. Validate every file first; stop if any fails
    classes, truth, pile = load_reference()
    bad = {f: check_file(f, classes, truth, pile) for f in files}
    bad = {f: p for f, p in bad.items() if p}
    if bad:
        for f, problems in bad.items():
            print(f"FAIL {f.name}: " + "; ".join(problems))
        sys.exit("Fix the files above (see docs/predictions_format.md); nothing was evaluated.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.final:
        log_final_run(args.out_dir / "test_runs.log", files)

    # 3. Near-duplicate notes of this split: any note in a cross-pile pair at cosine >= 0.8
    pairs = pd.read_csv(NEAR_DUP_PATH)
    near_dup = set(pairs.loc[pairs["split_a"] == args.split, "id_a"]) | set(pairs.loc[pairs["split_b"] == args.split, "id_b"])

    # The same bootstrap draws for every file of this split (files have identical note sets)
    split_ids = sorted(i for i, s in pile.items() if s == args.split)
    kept_ids = [i for i in split_ids if i not in near_dup]
    idx = {"all": bootstrap_indices(len(split_ids)), "excl_near_dup": bootstrap_indices(len(kept_ids))}

    metrics_dir = args.out_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    args.fig_dir.mkdir(parents=True, exist_ok=True)

    # 4. Metrics per file
    per_model = {}  # model -> list of (seed, {subset: (metrics, bootstrap arrays)})
    for f in files:
        model, seed, split = parse_name(f)
        df = pd.read_csv(f).sort_values("note_id").reset_index(drop=True)
        results = {
            "all": evaluate(df, classes, idx["all"]),
            "excl_near_dup": evaluate(df[~df["note_id"].isin(near_dup)].reset_index(drop=True), classes, idx["excl_near_dup"]),
        }
        out = {"model": model, "seed": seed, "split": split, "file": f.name, "classes": classes,
               "near_duplicate_source": str(NEAR_DUP_PATH.relative_to(ROOT)), "n_near_dup_excluded": len(near_dup),
               "bootstrap": {"resamples": N_BOOT, "seed": RANDOM_SEED, "interval": "95% percentile"},
               "subsets": {name: r[0] for name, r in results.items()}}
        (metrics_dir / f"{model}_{seed}_{split}.json").write_text(json.dumps(out, indent=2))
        save_figures(results["all"][0], classes, f"{model}_{seed}_{split}", args.fig_dir)
        per_model.setdefault(model, []).append((seed, results))
        print(f"{f.name}: macro-F1 {results['all'][0]['macro_f1']:.3f} (all), {results['excl_near_dup'][0]['macro_f1']:.3f} (without near-duplicates)")

    # 5. Summary: one row per model and subset. Several seeds -> mean +- SD of the point estimates,
    #    and a pooled CI: average the seeds' scores on each shared bootstrap draw, then take percentiles.
    rows = []
    for model, runs in sorted(per_model.items()):
        for subset in ["all", "excl_near_dup"]:
            ms = [r[subset][0] for _, r in runs]
            f1s, accs = np.array([m["macro_f1"] for m in ms]), np.array([m["accuracy"] for m in ms])
            pooled_f1 = np.mean([r[subset][1][0] for _, r in runs], axis=0)
            pooled_acc = np.mean([r[subset][1][1] for _, r in runs], axis=0)
            several = len(runs) > 1
            rows.append({
                "model": model, "subset": subset, "n_seeds": len(runs), "seeds": " ".join(str(s) for s, _ in sorted(runs)),
                "n_notes": ms[0]["n_notes"],
                "macro_f1_mean": f1s.mean(), "macro_f1_sd": f1s.std(ddof=1) if several else np.nan,
                "macro_f1_ci_low": ci(pooled_f1)[0], "macro_f1_ci_high": ci(pooled_f1)[1],
                "accuracy_mean": accs.mean(), "accuracy_sd": accs.std(ddof=1) if several else np.nan,
                "accuracy_ci_low": ci(pooled_acc)[0], "accuracy_ci_high": ci(pooled_acc)[1],
                "auroc_macro_ovr": np.mean([m["auroc_macro_ovr"] for m in ms]),
                "auprc_macro_ovr": np.mean([m["auprc_macro_ovr"] for m in ms]),
                "ece": np.mean([m["ece"] for m in ms]),
                "note": "CI pooled over seeds" if several else "single seed",
            })
    rows.append({"model": "CLiGNet (reference)", "subset": "all", "macro_f1_mean": CLIGNET["macro_f1"], "note": CLIGNET["note"]})
    summary = pd.DataFrame(rows)
    summary.to_csv(args.out_dir / f"summary_{args.split}.csv", index=False, float_format="%.4f")

    # Readable version of the comparison table, with the footnote
    lines = [f"# Model comparison, {args.split} split", "",
             "| Model | Notes | Seeds | Macro-F1 (95% CI) | SD | Accuracy (95% CI) | AUROC | AUPRC | ECE |",
             "|---|---:|---:|---|---:|---|---:|---:|---:|"]
    for r in rows:
        if r["model"].startswith("CLiGNet"):
            lines.append(f"| CLiGNet benchmark* | | | {r['macro_f1_mean']:.3f} | | | | | |")
            continue
        label = r["model"] if r["subset"] == "all" else f"{r['model']} (no near-duplicates)"
        sd = "" if np.isnan(r["macro_f1_sd"]) else f"{r['macro_f1_sd']:.3f}"
        lines.append(f"| {label} | {r['n_notes']} | {r['n_seeds']} | {r['macro_f1_mean']:.3f} ({r['macro_f1_ci_low']:.3f} to {r['macro_f1_ci_high']:.3f}) | {sd} "
                     f"| {r['accuracy_mean']:.3f} ({r['accuracy_ci_low']:.3f} to {r['accuracy_ci_high']:.3f}) | {r['auroc_macro_ovr']:.3f} | {r['auprc_macro_ovr']:.3f} | {r['ece']:.3f} |")
    footnote = (f"Bootstrap: {N_BOOT} resamples, seed {RANDOM_SEED}, same draws for every model. With several seeds, the CI is pooled: "
                "the seeds' scores are averaged on each draw. "
                f"\"No near-duplicates\" excludes the {len(near_dup)} {args.split} notes listed in `{NEAR_DUP_PATH.relative_to(ROOT)}`.")
    lines += ["", f"\\* {CLIGNET['note']}", footnote]
    (args.out_dir / f"summary_{args.split}.md").write_text("\n".join(lines) + "\n")
    print(f"\nWrote {len(files)} metrics files, {args.out_dir / f'summary_{args.split}.csv'} and figures in {args.fig_dir}")


if __name__ == "__main__":
    main()
