"""Predict the test split once from the frozen, saved models (the model lock).

    python src/predict_test.py --final            # the single test run, after the Nov 25 lock
    python src/predict_test.py --verify           # check the frozen baselines against val, no test read

This is the only script that reads the test split for a modelling output. It loads the
already-fitted models saved by the training scripts and writes the test predictions; it
never fits anything, so the locked models are not touched. The flow from here is:

    python src/predict_test.py --final            # writes results/preds_*_test.csv
    python src/evaluate.py --split test --final   # reads those files, writes metrics + summary

What each model loads:

  - majority       the most frequent training class (recomputed from the frozen labels)
  - keyword_rule   the keyword table saved by src/train_baselines.py (results/keyword_table_42_val.json)
                   and the same majority fallback
  - tfidf_lr       models/tfidf_lr_42.joblib  (vectorizer + fitted LogisticRegression)
  - tfidf_svm      models/tfidf_svm_42.joblib (vectorizer + fitted CalibratedClassifierCV)
  - clinicalbert   models/clinicalbert_seed{42,43,44}/ (weights + config + tokenizer), one
                   seed per file

The joblib files and checkpoints are git-ignored (course rule: no model weights in the repo);
they are the same artifacts the validation numbers were produced from. The TF-IDF baselines
are the only frozen artifacts where "the wrong file on disk" is a real risk, so by default
this script first reloads them, re-predicts the VALIDATION split, and checks the result
against the committed results/preds_*_val.csv. If they do not match, it stops before reading
test. (majority and keyword_rule are deterministic rules, so they are not re-checked;
ClinicalBERT was verified at save time, see docs/results_validation.md section 9.)

The masked twins are deliberately not predicted: the leakage experiment is a validation-only
comparison, and the final test table reports the original-text models.

Requires --final and appends one line to results/test_runs.log, the same guard as
src/evaluate.py, so the "test set read once" rule is enforced by both scripts.
"""

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import load_split_notes  # noqa: E402
from train_baselines import (  # noqa: E402
    check_class_order,
    class_list,
    one_hot,
    prediction_frame,
    run_keyword_rule,
    run_majority,
)

ROOT = Path(__file__).resolve().parents[1]
RANDOM_SEED = 42
CLINICALBERT_SEEDS = (42, 43, 44)
BASELINE_MODELS = ("majority", "keyword_rule", "tfidf_lr", "tfidf_svm")


# ---------- loaders (no fitting) ----------

def load_keyword_table(path):
    """Reconstruct {class: [term, ...]} from the JSON the training script wrote."""
    payload = json.loads(path.read_text())
    return {c: [entry["term"] for entry in entries] for c, entries in payload["keywords"].items()}


def predict_tfidf(joblib_path, df, classes, name):
    """Load a saved vectorizer + estimator and predict. Returns (pred_ids, probs)."""
    import joblib

    bundle = joblib.load(joblib_path)
    vectorizer, model = bundle["vectorizer"], bundle["model"]
    check_class_order(model, len(classes), name)
    X = vectorizer.transform(df["text"])
    probs = model.predict_proba(X)
    return probs.argmax(axis=1), probs


def predict_clinicalbert(checkpoint_dir, df, classes):
    """Load a saved checkpoint + tokenizer and predict. Returns (pred_ids, probs).

    Heavy imports are local so the baseline path runs without torch/transformers.
    """
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from train_clinicalbert import softmax, tokenize_head_tail

    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(checkpoint_dir), num_labels=len(classes))
    model.eval()
    input_ids, attention_mask = tokenize_head_tail(df["text"], tokenizer)
    with torch.no_grad():
        logits = model(
            input_ids=torch.tensor(input_ids),
            attention_mask=torch.tensor(attention_mask),
        ).logits
    probs = softmax(logits.numpy())
    return probs.argmax(axis=1), probs


# ---------- verification (validation only, before any test read) ----------

def verify_tfidf_baselines(joblib_dir):
    """Reload the two frozen TF-IDF pipelines and check they reproduce the committed
    validation predictions. Returns a list of problem strings (empty = all good)."""
    classes = class_list()
    label2id = {c: i for i, c in enumerate(classes)}
    # Sort by note_id so the recomputed predictions line up row-for-row with the committed
    # file, which was written in the same order the validator reads (not necessarily sorted).
    val_df = load_split_notes("val").sort_values("note_id").reset_index(drop=True)
    problems = []
    for name in ("tfidf_lr", "tfidf_svm"):
        frozen = joblib_dir / f"{name}_{RANDOM_SEED}.joblib"
        if not frozen.exists():
            problems.append(f"{name}: missing {frozen}")
            continue
        pred_ids, probs = predict_tfidf(frozen, val_df, classes, name)
        pred = np.array([classes[i] for i in pred_ids])
        committed = pd.read_csv(ROOT / "results" / f"preds_{name}_{RANDOM_SEED}_val.csv", dtype={"note_id": str})
        committed = committed.sort_values("note_id")["pred_label"].to_numpy()
        if not (pred == committed).all():
            n = int((pred != committed).sum())
            problems.append(f"{name}: frozen model differs from the committed validation predictions on {n} notes")
        else:
            f1 = f1_score(val_df["label"].map(label2id).to_numpy(), pred_ids, average="macro", zero_division=0)
            print(f"  verify {name}: frozen model reproduces the committed val predictions exactly (macro-F1 {f1:.4f})")
    return problems


# ---------- test-set guard ----------

def log_test_run(log_path, models):
    """Append one line per --final run: UTC time, commit and the models predicted."""
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                                capture_output=True, text=True, check=False).stdout.strip()
    except OSError:
        commit = "unknown"
    previous = log_path.read_text().count("\n") if log_path.exists() else 0
    if previous:
        print(f"WARNING: the test set has already been predicted {previous} time(s); see {log_path}")
    with open(log_path, "a") as fh:
        fh.write(f"{datetime.now(UTC).isoformat(timespec='seconds')}\tcommit={commit}\t{','.join(models)}\n")


# ---------- main ----------

def main():
    parser = argparse.ArgumentParser(description="Predict the test split from the frozen, saved models.")
    parser.add_argument("--final", action="store_true", help="required to read and predict the test split")
    parser.add_argument("--verify", action="store_true",
                        help="re-check the frozen TF-IDF baselines against val, then stop (no test read)")
    parser.add_argument("--skip-verify", action="store_true",
                        help="do not re-check the frozen baselines before predicting test")
    parser.add_argument("--models", nargs="+", default=list(BASELINE_MODELS) + ["clinicalbert"],
                        help=f"models to predict (default: {', '.join(BASELINE_MODELS)}, clinicalbert)")
    parser.add_argument("--joblib-dir", type=Path, default=ROOT / "models",
                        help="where the baseline .joblib files are (git-ignored)")
    parser.add_argument("--checkpoint-dir", type=Path, default=ROOT / "models",
                        help="where the clinicalbert_seed{42,43,44} checkpoints are")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results")
    args = parser.parse_args()

    unknown = set(args.models) - set(BASELINE_MODELS) - {"clinicalbert"}
    if unknown:
        sys.exit(f"unknown models: {sorted(unknown)}; choose from {BASELINE_MODELS} and clinicalbert")

    if args.verify:
        problems = verify_tfidf_baselines(args.joblib_dir)
        sys.exit(0 if not problems else "\n".join("FAIL " + p for p in problems))

    # Everything below reads the test split: it needs the explicit, logged --final.
    if not args.final:
        sys.exit("Refusing to predict the test split without --final. The test set is read once, after the Nov 25 lock.")

    # Verify only the frozen TF-IDF baselines this run will actually use.
    need_tfidf = bool({"tfidf_lr", "tfidf_svm"} & set(args.models))
    if not args.skip_verify and need_tfidf:
        problems = verify_tfidf_baselines(args.joblib_dir)
        if problems:
            sys.exit("\n".join("FAIL " + p for p in problems) + "\nFix the frozen artifacts before reading test.")

    classes = class_list()
    label2id = {c: i for i, c in enumerate(classes)}
    train_df = load_split_notes("train")
    test_df = load_split_notes("test", final=True)

    # Keyword rule and majority need the training majority class and keyword table.
    majority_id = None
    keyword_table = None
    if {"majority", "keyword_rule"} & set(args.models):
        majority_id = run_majority(train_df, classes, label2id)
    if "keyword_rule" in args.models:
        keyword_table = load_keyword_table(ROOT / "results" / f"keyword_table_{RANDOM_SEED}_val.json")

    written = []
    for name in args.models:
        print(f"[{name}]")
        if name == "majority":
            pred_ids = np.full(len(test_df), majority_id)
            probs = one_hot(pred_ids, len(classes))
        elif name == "keyword_rule":
            pred_ids, probs, _ = run_keyword_rule(train_df, test_df, classes, label2id, keyword_table)
        elif name in ("tfidf_lr", "tfidf_svm"):
            joblib_path = args.joblib_dir / f"{name}_{RANDOM_SEED}.joblib"
            if not joblib_path.exists():
                sys.exit(f"missing {joblib_path}; train the baseline first or point --joblib-dir at the frozen models")
            pred_ids, probs = predict_tfidf(joblib_path, test_df, classes, name)
        else:  # clinicalbert
            for seed in CLINICALBERT_SEEDS:
                ckpt = args.checkpoint_dir / f"clinicalbert_seed{seed}"
                if not ckpt.exists():
                    sys.exit(f"missing checkpoint {ckpt}; point --checkpoint-dir at the frozen checkpoints")
                pred_ids, probs = predict_clinicalbert(ckpt, test_df, classes)
                frame = prediction_frame(test_df, classes, pred_ids, probs)
                path = args.out_dir / f"preds_clinicalbert_{seed}_test.csv"
                path.parent.mkdir(parents=True, exist_ok=True)
                frame.to_csv(path, index=False)
                written.append(f"clinicalbert_seed{seed}")
                print(f"  wrote {path.name} ({len(frame)} rows)")
            continue

        frame = prediction_frame(test_df, classes, pred_ids, probs)
        path = args.out_dir / f"preds_{name}_{RANDOM_SEED}_test.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)
        written.append(name)
        print(f"  wrote {path.name} ({len(frame)} rows)")

    log_test_run(args.out_dir / "test_runs.log", written)
    print(f"\nWrote {len(written)} test prediction files. Next: python src/evaluate.py --split test --final")


if __name__ == "__main__":
    main()
