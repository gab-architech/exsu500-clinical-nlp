"""Train the four non-transformer baselines and write their predictions.

    python src/train_baselines.py                        # all four, validation split, original text
    python src/train_baselines.py --only tfidf_lr        # a single model
    python src/train_baselines.py --masked               # the label-masked variant (leakage experiment)
    python src/train_baselines.py --split test --final    # the single test run, after the Nov 25 lock

The four models, in the order the manuscript reports them:
  1. majority      - always predicts the most frequent training class (reference line)
  2. keyword_rule  - two keywords per class chosen by mutual information on the training
                     split only, each keyword assigned to its strongest class, predict by vote
  3. tfidf_lr      - TF-IDF (1-2 grams) with class-weighted logistic regression
  4. tfidf_svm     - TF-IDF (1-2 grams) with a class-weighted linear SVM; probabilities from
                     Platt scaling fitted on cross-validated training scores

Fitting discipline: every model is fitted on the training split only, and so are the
TF-IDF vocabulary and idf, the mutual-information table, the class weights and the Platt
calibrator. The validation split is used to report results and to select one
hyperparameter, the logistic-regression regularisation strength C (--tune-lr-c), which the
protocol allows; the selected value is stated in the outputs. Nothing is fitted on
validation or test. The rows of the other splits are loaded with the rest (src/data.py
rebuilds every note to match its ID) and then discarded; load_split_notes refuses the test
split unless final=True, and this script requires an explicit --final as well.

--masked applies src/masking.mask_label_strings to the training and validation text before
any fitting, so the mutual-information table and the TF-IDF vocabulary are recomputed on the
masked text. It is the only difference from the original run.

Output: results/preds_{model}{_masked}_42_{split}.csv in the format of
docs/predictions_format.md (one row per note of that split: note_id, true_label, pred_label,
prob_<class> x 11). Check them with:
    python src/validate_predictions.py results/preds_*_42_val.csv
The keyword table and the vote statistics are written to
results/keyword_table{_masked}_42_{split}.json.

Determinism: every model is seeded with RANDOM_SEED and the file name carries seed 42, so a
re-run in the same environment reproduces the same predictions.
"""

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import LinearSVC

sys.path.insert(0, str(Path(__file__).resolve().parent))  # allow `from data import ...`
from data import load_split_notes  # noqa: E402
from masking import mask_label_strings  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RANDOM_SEED = 42
MODEL_NAMES = ["majority", "keyword_rule", "tfidf_lr", "tfidf_svm"]
# Two keywords per class, fixed in advance (docs/model_proposal.md, section 1).
KEYWORDS_PER_CLASS = 2
N_CALIBRATION_FOLDS = 5

# Regularisation grid for the logistic-regression baseline, fixed before the final run.
# A value is selected on the VALIDATION split only (the protocol allows tuning on
# validation) and the chosen value is written into the outputs so the manuscript can state
# it. --tune-lr-c runs the sweep and writes nothing. LR_C_DEFAULT is the a-priori value used
# when no tuning is done.
#
# The grid was extended once (from a maximum of 100 to 2000) after the first sweep put the
# optimum on the boundary; the second sweep has an interior optimum, so this is the final
# grid. Recorded here so the extension is visible rather than silent.
LR_C_GRID = (0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0)
# Selected on the validation split by --tune-lr-c (best of the grid: macro-F1 0.8623 at
# C=500) and then fixed for every later run, so that plain
# `python src/train_baselines.py` reproduces the reported number without extra flags.
# Evidence: results/lr_c_sweep_42_val.json.
LR_C_DEFAULT = 500.0


# ---------- shared helpers ----------

def class_list():
    """The 11 class names, alphabetical, matching docs/predictions_format.md."""
    return sorted(pd.read_csv(ROOT / "labels" / "final_labels.csv")["label"].unique())


def prediction_frame(df: pd.DataFrame, classes: list, pred: np.ndarray, probs: np.ndarray) -> pd.DataFrame:
    """Build the prediction file frame in the exact team format.

    note_id is written as text: four validation IDs are all digits and must keep their
    leading zeros when the file is read back by the validator.
    """
    out = pd.DataFrame({
        "note_id": df["note_id"].astype(str).to_numpy(),
        "true_label": df["label"].to_numpy(),
        "pred_label": [classes[i] for i in pred],
    })
    for i, c in enumerate(classes):
        out[f"prob_{c}"] = probs[:, i]
    return out


def one_hot(pred: np.ndarray, k: int) -> np.ndarray:
    """Deterministic models have no real probabilities: 1.0 on the prediction, 0 elsewhere
    (allowed by docs/predictions_format.md; their AUROC/AUPRC/calibration are reported as
    degenerate)."""
    probs = np.zeros((len(pred), k))
    probs[np.arange(len(pred)), pred] = 1.0
    return probs


def check_class_order(model, k: int, name: str) -> None:
    """Guard the positional use of predict_proba columns against a class-order surprise."""
    got = list(getattr(model, "classes_", range(k)))
    if got != list(range(k)):
        raise AssertionError(f"{name}: model.classes_ is {got}, expected 0..{k - 1}")


def report(name: str, y_true: np.ndarray, pred: np.ndarray) -> float:
    f1 = f1_score(y_true, pred, average="macro", zero_division=0)
    acc = float((y_true == pred).mean())
    print(f"  {name:13s} macro-F1 {f1:.4f}   accuracy {acc:.4f}")
    return f1


# ---------- model 1: majority class ----------

def run_majority(train_df: pd.DataFrame, classes: list, label2id: dict):
    """Always predict the most frequent training class. No fitting beyond a count."""
    counts = train_df["label"].value_counts()
    top = counts.index[0]
    print(f"  majority class = {top} ({counts.iloc[0]}/{len(train_df)} training notes)")
    return label2id[top]


# ---------- model 2: keyword rule (mutual information) ----------

def mutual_information_tables(train_df: pd.DataFrame, classes: list, label2id: dict):
    """Pick two keywords per class using mutual information computed on the training split.

    For every vocabulary term t and class c we compute the mutual information between two
    binary variables over the training notes: (t is present) and (label == c). It is the sum
    of the four cells of the 2x2 table, each written as p * log(p / (p_row * p_col)). Each
    term is then assigned to the class where its MI is largest (its "home"), so a term
    belongs to at most one class (no duplicates). Within each class the top-2 terms by MI
    become that class's keywords.

    Returns (table, info) where table maps class name -> list of terms and info maps
    (class name, term) -> {"mi", "df"} for the appendix.
    """
    counts = CountVectorizer(binary=True, lowercase=True, token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b")
    X = counts.fit_transform(train_df["text"]).toarray().astype(bool)  # notes x terms
    terms = np.array(counts.get_feature_names_out())
    y = train_df["label"].map(label2id).to_numpy()
    n = len(y)

    best_class = np.full(len(terms), -1, dtype=int)  # class each term is assigned to
    best_mi = np.zeros(len(terms))                   # its MI with that class
    p_t = X.mean(axis=0)                             # P(term), document frequency
    one_minus_p_t = 1.0 - p_t

    for c in range(len(classes)):
        in_class = y == c
        p_c = in_class.sum() / n
        if p_c == 0:
            continue
        p_tc = (X & in_class[:, None]).mean(axis=0)  # P(term and class)
        ones = np.ones_like(p_t)

        def kl(p_joint, row, col):
            """p_joint * log(p_joint / (row * col)), with 0 where the joint is 0."""
            with np.errstate(divide="ignore", invalid="ignore"):
                val = np.where(p_joint > 0, p_joint * np.log(p_joint / (row * col)), 0.0)
            return np.nan_to_num(val, nan=0.0, posinf=0.0, neginf=0.0)

        mi = (
            kl(p_tc, p_t, p_c * ones)
            + kl(p_t - p_tc, p_t, (1.0 - p_c) * ones)
            + kl(p_c - p_tc, one_minus_p_t, p_c * ones)
            + kl(1.0 - p_t - p_c + p_tc, one_minus_p_t, (1.0 - p_c) * ones)
        )
        better = mi > best_mi
        best_mi[better] = mi[better]
        best_class[better] = c

    table, info = {}, {}
    for c, name in enumerate(classes):
        owned = np.flatnonzero(best_class == c)
        if len(owned) == 0:
            table[name] = []
            continue
        # Stable sort with the term string as the secondary key, so an exact MI tie is
        # broken by the term itself rather than by the sort implementation.
        order = sorted(owned, key=lambda i: (-best_mi[i], terms[i]))
        chosen = order[:KEYWORDS_PER_CLASS]
        table[name] = [str(terms[i]) for i in chosen]
        for i in chosen:
            info[(name, str(terms[i]))] = {"mi": float(best_mi[i]), "df": int(X[:, i].sum())}
    return table, info


def run_keyword_rule(train_df, val_df, classes, label2id, table):
    """Vote by keyword hits; ties (and notes with no hit) fall back to the majority class.

    The fallback is part of the pre-specified rule in docs/model_proposal.md, not a patch:
    over a third of the validation notes fall back to the majority class (vote ties plus
    notes with no hit), so the counts are returned for the manuscript and must not be
    re-tuned after seeing validation.
    """
    majority = label2id[train_df["label"].value_counts().index[0]]
    lowered = val_df["text"].str.lower()

    votes = np.zeros((len(val_df), len(classes)), dtype=int)
    for c, name in enumerate(classes):
        for kw in table[name]:
            votes[:, c] += lowered.str.contains(rf"\b{kw}\b", regex=True).to_numpy()

    no_hit = votes.sum(axis=1) == 0
    at_max = votes == votes.max(axis=1)[:, None]
    tied = (~no_hit) & (at_max.sum(axis=1) > 1)

    pred_rule = np.where(no_hit, majority, votes.argmax(axis=1))
    pred = pred_rule.copy()
    pred[tied] = majority
    # Only the tied notes whose keyword vote was not already the majority class actually
    # change prediction; the other ties were predicted the majority class anyway.
    changed = int((tied & (pred_rule != majority)).sum())

    stats = {
        "notes": int(len(val_df)),
        "no_hit": int(no_hit.sum()),
        "tie": int(tied.sum()),
        "changed_by_tie_rule": changed,
        "predicted_per_class": {name: int((pred == c).sum()) for c, name in enumerate(classes)},
    }
    print(f"  votes: {stats['no_hit']} no-hit, {stats['tie']} tied, "
          f"{changed} of the ties changed the prediction -> majority "
          f"({100 * (stats['no_hit'] + stats['tie']) / len(val_df):.1f}% of notes)")
    return pred, one_hot(pred, len(classes)), stats


# ---------- models 3 and 4: TF-IDF + linear ----------

def tfidf_matrices(train_df: pd.DataFrame, val_df: pd.DataFrame):
    """TF-IDF weights fitted on the training text only, then applied to the other split.

    1-2 grams, sublinear term frequency, terms seen in at least 2 training notes
    (single-occurrence terms are noise on 1,560 notes), L2-normalised, no stopword removal,
    default token pattern. The input text keeps its original capitalisation
    (src/data.py returns it as written), but TfidfVectorizer lower-cases by default, so the
    features themselves are case-insensitive.

    Returns the fitted vectorizer together with both matrices, so the vectorizer can be
    saved alongside the model.
    """
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2)
    X_train = vectorizer.fit_transform(train_df["text"])
    X_val = vectorizer.transform(val_df["text"])
    print(f"  TF-IDF: {X_train.shape[1]} features, {X_train.nnz} non-zero training entries")
    return vectorizer, X_train, X_val


def run_tfidf_lr(X_train, X_val, y_train, classes, C):
    """Class-weighted multinomial logistic regression with regularisation strength C.

    C is either the a-priori default or a value chosen on validation by --tune-lr-c; the
    value in use is printed and saved, and the manuscript must state which it is.

    Note for the manuscript: calibration depends strongly on C. At the validation-selected
    C=500 this model is well calibrated (ECE 0.059), while at the a-priori default C=1 it is
    strongly under-confident (ECE 0.452). Report whichever regime was actually run; no
    calibrator is ever fitted on validation.
    """
    model = LogisticRegression(
        C=C, class_weight="balanced", max_iter=2000, random_state=RANDOM_SEED
    )
    model.fit(X_train, y_train)
    check_class_order(model, len(classes), "logistic regression")
    probs = model.predict_proba(X_val)
    return probs.argmax(axis=1), probs, model


def tune_lr_c(X_train, y_train, X_val, y_val) -> list:
    """Macro-F1 on the validation split for every C in the pre-declared grid.

    Used to pick the logistic-regression regularisation strength. Only validation rows are
    involved, never test. Returns [(C, macro_f1), ...].
    """
    out = []
    for C in LR_C_GRID:
        m = LogisticRegression(C=C, class_weight="balanced", max_iter=2000,
                               random_state=RANDOM_SEED).fit(X_train, y_train)
        f1 = f1_score(y_val, m.predict(X_val), average="macro", zero_division=0)
        out.append((C, float(f1)))
        print(f"  C={C:<5} validation macro-F1 {f1:.4f}")
    return out


def run_tfidf_svm(X_train, X_val, y_train, classes):
    """Class-weighted linear SVM with Platt scaling.

    A plain LinearSVC gives no probabilities, and inventing them from the decision function
    would make ECE meaningless. Platt scaling fits a sigmoid on cross-validated training
    scores, which never touches validation or test data.

    ensemble=False matters: with the sklearn default (True) the reported prediction is the
    average of five LinearSVCs each trained on 80% of the training notes, which is not the
    single linear SVM the manuscript describes. With ensemble=False the base SVM is refitted
    on all training notes and the calibrator is fitted on its out-of-fold scores.
    """
    folds = StratifiedKFold(n_splits=N_CALIBRATION_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    base = LinearSVC(class_weight="balanced", random_state=RANDOM_SEED, max_iter=5000)
    model = CalibratedClassifierCV(base, method="sigmoid", cv=folds, ensemble=False)
    model.fit(X_train, y_train)
    check_class_order(model, len(classes), "calibrated SVM")
    probs = model.predict_proba(X_val)
    return probs.argmax(axis=1), probs, model


# ---------- main ----------

def save_model(model_dir, name, vectorizer, model, meta, split) -> None:
    """Persist a fitted model (vectorizer + estimator) with its metadata.

    Models are written to models/, which is git-ignored: the course rules keep model files
    out of the repository, and the interface needs the fitted TF-IDF pipeline anyway. The
    split is part of the file name so a later test run cannot overwrite the validation model.
    """
    model_dir.mkdir(parents=True, exist_ok=True)
    path = model_dir / f"{name}_{RANDOM_SEED}_{split}.joblib"
    meta = {**meta, "split": split, "scikit_learn": __import__("sklearn").__version__}
    joblib.dump({"vectorizer": vectorizer, "model": model, "meta": meta}, path, compress=3)
    print(f"  saved model {path.name} ({path.stat().st_size / 1e6:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(description="Train the baselines and write predictions.")
    parser.add_argument("--only", choices=MODEL_NAMES, default=None, help="run a single model")
    parser.add_argument("--masked", action="store_true",
                        help="mask specialty names before fitting (leakage experiment)")
    parser.add_argument("--split", choices=["val", "test"], default="val",
                        help="evaluate this split; test needs --final")
    parser.add_argument("--final", action="store_true",
                        help="required for --split test (the one run after the Nov 25 model lock)")
    parser.add_argument("--lr-c", type=float, default=LR_C_DEFAULT,
                        help=f"logistic-regression regularisation strength (default {LR_C_DEFAULT})")
    parser.add_argument("--tune-lr-c", action="store_true",
                        help="sweep LR_C_GRID on validation and exit without writing predictions")
    parser.add_argument("--model-dir", type=Path, default=ROOT / "models",
                        help="where fitted models are saved (git-ignored)")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results")
    args = parser.parse_args()

    if args.split == "test" and not args.final:
        sys.exit("Refusing to evaluate the test split without --final. The test set is read once, after the Nov 25 lock.")
    if args.final and args.split != "test":
        sys.exit("--final is only for the test split.")

    # Argument checks that must happen before any data is read, so an invalid combination
    # can never consume the one-time test read.
    requested = [args.only] if args.only else MODEL_NAMES
    if args.tune_lr_c:
        if args.split != "val":
            sys.exit("--tune-lr-c tunes on validation only; do not pass --split test.")
        if "tfidf_lr" not in requested:
            sys.exit("--tune-lr-c needs the tfidf_lr model; drop --only or select tfidf_lr.")

    suffix = "_masked" if args.masked else ""
    classes = class_list()
    label2id = {c: i for i, c in enumerate(classes)}

    train_df = load_split_notes("train")
    target_df = load_split_notes(args.split, final=args.final)
    if args.masked:
        train_df = train_df.assign(text=train_df["text"].map(mask_label_strings))
        target_df = target_df.assign(text=target_df["text"].map(mask_label_strings))

    y_true = target_df["label"].map(label2id).to_numpy()
    y_train = train_df["label"].map(label2id).to_numpy()
    print(f"train {len(train_df)} notes, {args.split} {len(target_df)} notes, "
          f"{len(classes)} classes, masked={args.masked}\n")

    models = requested
    need_tfidf = any(m in models for m in ("tfidf_lr", "tfidf_svm"))
    vectorizer, X_train, X_target = tfidf_matrices(train_df, target_df) if need_tfidf else (None, None, None)

    # Tuning mode: sweep the pre-declared C grid on validation and stop. Nothing is written
    # except the sweep record, so a tuning run can never be confused with the run that
    # produces predictions.
    if args.tune_lr_c:
        sweep = tune_lr_c(X_train, y_train, X_target, y_true)
        best = max(sweep, key=lambda t: t[1])
        print(f"\nbest C on validation: {best[0]} (macro-F1 {best[1]:.4f})")
        out = args.out_dir / f"lr_c_sweep{suffix}_{RANDOM_SEED}_val.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(
            {"grid": list(LR_C_GRID), "sweep": [{"C": c, "macro_f1": f} for c, f in sweep],
             "best_C": best[0], "best_macro_f1": best[1], "masked": args.masked}, indent=2))
        print(f"wrote {out}")
        return

    table, mi_info = mutual_information_tables(train_df, classes, label2id) if "keyword_rule" in models else (None, {})

    summary = []
    keyword_stats = None
    for name in models:
        print(f"[{name}]")
        if name == "majority":
            top = run_majority(train_df, classes, label2id)
            pred = np.full(len(target_df), top)
            probs = one_hot(pred, len(classes))
        elif name == "keyword_rule":
            print("  keywords: " + "; ".join(f"{c}: {', '.join(v) or '-'}" for c, v in table.items()))
            pred, probs, keyword_stats = run_keyword_rule(train_df, target_df, classes, label2id, table)
        elif name == "tfidf_lr":
            pred, probs, fitted = run_tfidf_lr(X_train, X_target, y_train, classes, args.lr_c)
            print(f"  C = {args.lr_c}")
            save_model(args.model_dir, f"tfidf_lr{suffix}", vectorizer, fitted,
                       {"C": args.lr_c, "masked": args.masked, "seed": RANDOM_SEED}, args.split)
        else:
            pred, probs, fitted = run_tfidf_svm(X_train, X_target, y_train, classes)
            save_model(args.model_dir, f"tfidf_svm{suffix}", vectorizer, fitted,
                       {"masked": args.masked, "seed": RANDOM_SEED}, args.split)

        f1 = report(name, y_true, pred)
        summary.append({"model": name, "macro_f1": f1, "accuracy": float((y_true == pred).mean())})

        frame = prediction_frame(target_df, classes, pred, probs)
        path = args.out_dir / f"preds_{name}{suffix}_{RANDOM_SEED}_{args.split}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)
        try:
            shown = path.relative_to(ROOT)
        except ValueError:  # --out-dir outside the repo
            shown = path
        print(f"  wrote {shown}")

        # Provenance: one log per run, so the environment and the settings that produced a
        # number are in the repository and not only in a terminal scrollback.
        log_dir = args.out_dir / "run_logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log = {
            "model": name, "masked": args.masked, "split": args.split, "seed": RANDOM_SEED,
            "n_train": int(len(train_df)), "n_eval": int(len(target_df)),
            "macro_f1": f1, "accuracy": float((y_true == pred).mean()),
            "hyperparameters": {"lr_C": args.lr_c if name == "tfidf_lr" else None},
            "versions": {p: __import__(p).__version__ for p in ("sklearn", "pandas", "numpy")},
        }
        (log_dir / f"baseline_{name}{suffix}_{RANDOM_SEED}_{args.split}.json").write_text(
            json.dumps(log, indent=2))
        print()

    # Keep the keyword evidence with the predictions so the appendix and the vote counts
    # can be rebuilt without re-running the script.
    if table is not None:
        payload = {
            "model": "keyword_rule", "masked": args.masked, "split": args.split,
            "seed": RANDOM_SEED, "n_train": int(len(train_df)),
            "n_keyword_entries": int(len(mi_info)),
            "keywords": {c: [{"term": t, **mi_info[(c, t)]} for t in kws] for c, kws in table.items()},
            "vote_stats": keyword_stats,
        }
        kw_path = args.out_dir / f"keyword_table{suffix}_{RANDOM_SEED}_{args.split}.json"
        kw_path.write_text(json.dumps(payload, indent=2))
        try:
            shown = kw_path.relative_to(ROOT)
        except ValueError:  # --out-dir outside the repo
            shown = kw_path
        print(f"wrote {shown}")

    print("\nsummary")
    for row in summary:
        print(f"  {row['model']:13s} macro-F1 {row['macro_f1']:.4f}   accuracy {row['accuracy']:.4f}")


if __name__ == "__main__":
    main()
