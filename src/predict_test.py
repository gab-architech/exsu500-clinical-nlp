"""Predict the test split once from the frozen, saved models (the model lock).

    python src/predict_test.py --verify           # load every model and check it against val; no test read
    python src/predict_test.py --final            # the single test run, after the Nov 25 lock

This is the only script that reads the test split for a modelling output. It loads the
already-fitted models saved by the training scripts and writes the test predictions; it
never fits anything, so the locked models are not touched. The flow from here is:

    python src/predict_test.py --final            # writes results/preds_*_test.csv
    python src/evaluate.py --split test --final   # reads those files, writes metrics + summary

Order of operations (this order is part of the protocol, not a convenience):

  1. load every requested model from disk, and re-predict the VALIDATION split with each,
     comparing the labels against the committed results/preds_*_val.csv;
  2. append one line to results/test_runs.log;
  3. only then read the test split and predict.

So nothing reads test until every model has loaded and been shown to reproduce the committed
validation predictions. A missing or wrong artifact stops the run while the test split is
still closed, and a crash after the test read is already logged.

What each model loads:

  - majority       the most frequent training class (recomputed from the frozen labels)
  - keyword_rule   the keyword table saved by src/train_baselines.py (results/keyword_table_42_val.json)
                   and the same majority fallback (deterministic, so not re-checked)
  - tfidf_lr       models/tfidf_lr_42.joblib  (vectorizer + fitted LogisticRegression)
  - tfidf_svm      models/tfidf_svm_42.joblib (vectorizer + fitted CalibratedClassifierCV)
  - clinicalbert   models/clinicalbert_seed{42,43,44}/, one seed per file

The ClinicalBERT checkpoint folder may be either the folder itself (config.json inside it) or
the parent the Trainer wrote, in which case the single checkpoint-* subfolder is used. The
trainer saves under <output-dir>/clinicalbert_seed{N}/checkpoint-<step>/, so a checkpoint
copied straight from Drive usually needs that resolution.

The joblib files and checkpoints are git-ignored (course rule: no model weights in the repo);
they are the same artifacts the validation numbers were produced from.

The masked twins are deliberately not predicted: the leakage experiment is a validation-only
comparison, and the final test table reports the original-text models.
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

def resolve_checkpoint(path):
    """Return the folder that holds config.json / model weights.

    Accepts either that folder directly, or the folder the Trainer created, which holds a
    single checkpoint-* subfolder (save_total_limit=1 keeps only the best one).
    """
    if (path / "config.json").exists():
        return path
    subs = sorted(p for p in path.glob("checkpoint-*") if (p / "config.json").exists())
    if len(subs) == 1:
        return subs[0]
    if not subs:
        raise FileNotFoundError(f"{path} has no config.json and no checkpoint-* subfolder with one")
    raise RuntimeError(
        f"{path} holds {len(subs)} checkpoint-* folders ({', '.join(p.name for p in subs)}); "
        "keep only the best one, or point --checkpoint-dir at the folder you want"
    )


def load_keyword_table(path):
    """Reconstruct {class: [term, ...]} from the JSON the training script wrote."""
    payload = json.loads(path.read_text())
    return {c: [entry["term"] for entry in entries] for c, entries in payload["keywords"].items()}


def load_tfidf(joblib_path, classes, name):
    """Load a saved vectorizer + estimator. Returns (vectorizer, model)."""
    import joblib

    if not joblib_path.exists():
        raise FileNotFoundError(f"missing {joblib_path}; train the baseline first or point --joblib-dir")
    bundle = joblib.load(joblib_path)
    model = bundle["model"]
    check_class_order(model, len(classes), name)
    return bundle["vectorizer"], model


def load_clinicalbert(path, classes):
    """Load a checkpoint + tokenizer. Returns (tokenizer, model) with the model in eval mode."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    ckpt = resolve_checkpoint(path)
    tokenizer = AutoTokenizer.from_pretrained(str(ckpt))
    model = AutoModelForSequenceClassification.from_pretrained(str(ckpt), num_labels=len(classes))
    model.eval()
    return tokenizer, model


def predict_tfidf(loaded, df, name):
    vectorizer, model = loaded[name]
    probs = model.predict_proba(vectorizer.transform(df["text"]))
    return probs.argmax(axis=1), probs


def predict_clinicalbert(loaded, df, key):
    """Predict with an already-loaded (tokenizer, model). No disk access."""
    import torch

    from train_clinicalbert import softmax, tokenize_head_tail

    tokenizer, model = loaded[key]
    input_ids, attention_mask = tokenize_head_tail(df["text"], tokenizer)
    with torch.no_grad():
        logits = model(
            input_ids=torch.tensor(input_ids),
            attention_mask=torch.tensor(attention_mask),
        ).logits
    probs = softmax(logits.numpy())
    return probs.argmax(axis=1), probs


# ---------- preflight: load everything, then check it against the committed val predictions ----------

def load_models(models, classes, joblib_dir, checkpoint_dir):
    """Load every requested model artifact into memory. Raises before any test read if one fails."""
    loaded = {}
    for name in models:
        if name in ("majority", "keyword_rule"):
            continue  # deterministic rules, rebuilt from the training split at run time
        print(f"  loading {name} ...")
        if name in ("tfidf_lr", "tfidf_svm"):
            loaded[name] = load_tfidf(joblib_dir / f"{name}_{RANDOM_SEED}.joblib", classes, name)
        else:
            for seed in CLINICALBERT_SEEDS:
                key = f"clinicalbert_seed{seed}"
                loaded[key] = load_clinicalbert(checkpoint_dir / key, classes)
                print(f"    {key} loaded")
    return loaded


def verify_models(loaded, classes):
    """Re-predict the validation split with each loaded model and compare with the committed
    prediction files. Returns a list of problem strings (empty = every model reproduces)."""
    label2id = {c: i for i, c in enumerate(classes)}
    # Sorted by note_id so the recomputed labels line up row-for-row with the committed file.
    val_df = load_split_notes("val").sort_values("note_id").reset_index(drop=True)
    y = val_df["label"].map(label2id).to_numpy()
    problems = []

    for key in loaded:
        if key.startswith("clinicalbert_"):
            seed = key.rsplit("seed", 1)[1]
            pred_ids, _ = predict_clinicalbert(loaded, val_df, key)
            pred_file = f"preds_clinicalbert_{seed}_val.csv"
        else:
            pred_ids, _ = predict_tfidf(loaded, val_df, key)
            pred_file = f"preds_{key}_{RANDOM_SEED}_val.csv"
        pred = np.array([classes[i] for i in pred_ids])
        committed = pd.read_csv(ROOT / "results" / pred_file, dtype={"note_id": str})
        committed = committed.sort_values("note_id")["pred_label"].to_numpy()
        disagree = int((pred != committed).sum())
        f1 = f1_score(y, pred_ids, average="macro", zero_division=0)
        if disagree:
            problems.append(f"{key}: {disagree} of {len(y)} validation labels differ from the committed predictions")
        else:
            print(f"  verify {key}: reproduces the committed validation predictions exactly (macro-F1 {f1:.4f})")
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
                        help="load every model and check it against the committed val predictions, then stop")
    parser.add_argument("--skip-verify", action="store_true",
                        help="skip the validation reproduction check before predicting test")
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
    if args.verify and args.final:
        sys.exit("--verify does not read the test split; run it without --final.")
    # Refuse a bare run before loading 1.3 GB of checkpoints for nothing.
    if not args.final and not args.verify:
        sys.exit("Pass --verify (loading and validation check only) or --final (the one test run).")

    classes = class_list()

    # 1. Load every model. This happens before the test split is opened, so a missing or
    #    broken artifact stops the run while the test set is still closed.
    print("loading models ...")
    try:
        loaded = load_models(args.models, classes, args.joblib_dir, args.checkpoint_dir)
    except (FileNotFoundError, RuntimeError, OSError) as e:
        sys.exit(f"Could not load the frozen models; the test split was not read.\n  {e}")

    # 2. Reproduce the committed validation predictions (unless asked to skip).
    if args.verify or not args.skip_verify:
        print("checking each model against the committed validation predictions ...")
        problems = verify_models(loaded, classes)
        if problems:
            sys.exit("\n".join("FAIL " + p for p in problems) + "\nFix the frozen artifacts before reading test.")
    if args.verify:
        sys.exit(0)

    # 3. Everything below reads the test split, which only a --final run may do.
    args.out_dir.mkdir(parents=True, exist_ok=True)
    log_test_run(args.out_dir / "test_runs.log", args.models)

    # 4. Read test and predict with the models already in memory.
    print("reading the test split ...")
    label2id = {c: i for i, c in enumerate(classes)}
    train_df = load_split_notes("train")
    test_df = load_split_notes("test", final=True)

    majority_id = run_majority(train_df, classes, label2id) if "majority" in args.models else None
    keyword_table = None
    if "keyword_rule" in args.models:
        keyword_table = load_keyword_table(ROOT / "results" / f"keyword_table_{RANDOM_SEED}_val.json")

    written = []
    for name in args.models:
        print(f"[{name}]")
        if name == "majority":
            pred_ids = np.full(len(test_df), majority_id)
            probs = one_hot(pred_ids, len(classes))
            frame = prediction_frame(test_df, classes, pred_ids, probs)
            path = args.out_dir / f"preds_majority_{RANDOM_SEED}_test.csv"
            frame.to_csv(path, index=False)
            written.append("majority")
            print(f"  wrote {path.name} ({len(frame)} rows)")
        elif name == "keyword_rule":
            pred_ids, probs, _ = run_keyword_rule(train_df, test_df, classes, label2id, keyword_table)
            frame = prediction_frame(test_df, classes, pred_ids, probs)
            path = args.out_dir / f"preds_keyword_rule_{RANDOM_SEED}_test.csv"
            frame.to_csv(path, index=False)
            written.append("keyword_rule")
            print(f"  wrote {path.name} ({len(frame)} rows)")
        elif name in ("tfidf_lr", "tfidf_svm"):
            pred_ids, probs = predict_tfidf(loaded, test_df, name)
            frame = prediction_frame(test_df, classes, pred_ids, probs)
            path = args.out_dir / f"preds_{name}_{RANDOM_SEED}_test.csv"
            frame.to_csv(path, index=False)
            written.append(name)
            print(f"  wrote {path.name} ({len(frame)} rows)")
        else:  # clinicalbert: one file per seed
            for seed in CLINICALBERT_SEEDS:
                key = f"clinicalbert_seed{seed}"
                pred_ids, probs = predict_clinicalbert(loaded, test_df, key)
                frame = prediction_frame(test_df, classes, pred_ids, probs)
                path = args.out_dir / f"preds_clinicalbert_{seed}_test.csv"
                frame.to_csv(path, index=False)
                written.append(key)
                print(f"  wrote {path.name} ({len(frame)} rows)")

    print(f"\nWrote {len(written)} test prediction files. Next: python src/evaluate.py --split test --final")


if __name__ == "__main__":
    main()
