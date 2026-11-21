"""Fine-tune Bio_ClinicalBERT for the specialty-classification task.

One shared script for the original and masked runs: ``--masked`` is the only
difference between them. It masks specialty names before tokenization and adds
``_masked`` to the output names; nothing else may depend on it. See
docs/clinicalbert_pipeline.md (PR #28).

    python src/train_clinicalbert.py --seed 42            # original text
    python src/train_clinicalbert.py --seed 42 --masked   # masked text

Loads data with src/data.load_split_notes, masks with
src/masking.mask_label_strings, trains 3 epochs with the settings agreed on
PR #28, keeps the checkpoint with the best validation macro-F1, writes the
validation predictions in docs/predictions_format.md format, and writes a run
log to results/run_logs/. The test split is never loaded here (data.py refuses
it); test predictions come from the saved checkpoints after the model lock.

Hyperparameters are fixed module-level constants on purpose: they must stay
identical for the original and masked runs, so they are not command-line flags.
"""

import argparse
import json
import platform
import subprocess
import sys
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    get_linear_schedule_with_warmup,
    set_seed,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))  # allow `from data import ...`
from data import load_split_notes  # noqa: E402
from masking import MASK_TOKEN, mask_label_strings  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

# --- Fixed hyperparameters (agreed on PR #28; identical for both runs) ---
MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"
MAX_LENGTH = 512
HEAD_TAIL = (MAX_LENGTH - 2) // 2  # 255 text tokens each side; 2 slots hold [CLS]/[SEP]
EPOCHS = 3
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1  # linear warmup over the first 10% of steps
GRADIENT_CLIP = 1.0


# --- helpers ---

def softmax(logits: np.ndarray) -> np.ndarray:
    """Row-wise softmax, numerically stable."""
    shifted = logits - logits.max(axis=1, keepdims=True)
    exps = np.exp(shifted)
    return exps / exps.sum(axis=1, keepdims=True)


def tokenize_head_tail(texts, tokenizer):
    """Tokenize each text and truncate to head+tail (255 + 255) with the special
    tokens, padded to MAX_LENGTH. Returns (input_ids, attention_mask) lists."""
    enc = tokenizer(list(texts), add_special_tokens=False, truncation=False)
    input_ids, attention_mask = [], []
    for ids in enc["input_ids"]:
        if len(ids) > MAX_LENGTH - 2:
            ids = ids[:HEAD_TAIL] + ids[-HEAD_TAIL:]
        ids = [tokenizer.cls_token_id] + ids + [tokenizer.sep_token_id]
        pad = MAX_LENGTH - len(ids)
        input_ids.append(ids + [tokenizer.pad_token_id] * pad)
        attention_mask.append([1] * len(ids) + [0] * pad)
    return input_ids, attention_mask


def build_dataset(df: pd.DataFrame, tokenizer, label2id: dict) -> Dataset:
    """Return a Dataset with input_ids, attention_mask and integer labels."""
    input_ids, attention_mask = tokenize_head_tail(df["text"], tokenizer)
    return Dataset.from_dict({
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": [label2id[label] for label in df["label"]],
    })


def inverse_frequency_weights(train_labels: pd.Series, classes: list) -> torch.Tensor:
    """Standard balanced class weights, w_c = n / (k * n_c), so every class
    contributes equal total weight (n / k) to the loss. Rare classes get larger
    per-example weights so they are not drowned out by common ones."""
    counts = train_labels.value_counts()
    n, k = len(train_labels), len(classes)
    return torch.tensor([n / (k * counts[c]) for c in classes], dtype=torch.float)


def compute_metrics(eval_pred):
    """Validation macro-F1, the metric used to select the best checkpoint."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"macro_f1": f1_score(labels, preds, average="macro", zero_division=0)}


class WeightedTrainer(Trainer):
    """Trainer with inverse-frequency class weights in the cross-entropy loss."""

    def __init__(self, class_weights: torch.Tensor, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss = torch.nn.functional.cross_entropy(
            outputs.logits, labels, weight=self.class_weights.to(outputs.logits.device)
        )
        return (loss, outputs) if return_outputs else loss


# --- main ---

def main():
    parser = argparse.ArgumentParser(description="Fine-tune Bio_ClinicalBERT (original or masked).")
    parser.add_argument("--seed", type=int, required=True, help="random seed (42, 43, 44)")
    parser.add_argument("--masked", action="store_true", help="mask specialty names before tokenization")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="checkpoint root; defaults to <repo>/exsu500/checkpoints, pass a Drive path on Colab")
    parser.add_argument("--results-dir", type=Path, default=None,
                        help="where predictions and run logs go; defaults to <repo>/results")
    args = parser.parse_args()

    # Fail loudly instead of training on a CPU: on Colab a CPU run takes hours, and the
    # default checkpoint path would silently write to the ephemeral /content anyway.
    if not torch.cuda.is_available():
        sys.exit("No CUDA device. Switch the Colab runtime to a T4 GPU before training.")

    suffix = "_masked" if args.masked else ""
    model_key = f"clinicalbert{suffix}"
    results_dir = args.results_dir or ROOT / "results"
    checkpoint_dir = (args.output_dir or ROOT / "exsu500" / "checkpoints") / f"clinicalbert{suffix}_seed{args.seed}"

    # Seed everything, before the model and the data shuffle.
    set_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Data: train for fitting, val for checkpoint selection. Never test here.
    train_df = load_split_notes("train")
    val_df = load_split_notes("val")
    if args.masked:
        train_df["text"] = train_df["text"].map(mask_label_strings)
        val_df["text"] = val_df["text"].map(mask_label_strings)

    # Class order: alphabetical, matching docs/predictions_format.md and the
    # probability columns of the validator.
    classes = sorted(pd.read_csv(ROOT / "labels" / "final_labels.csv")["label"].unique())
    label2id = {c: i for i, c in enumerate(classes)}

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=len(classes))

    train_dataset = build_dataset(train_df, tokenizer, label2id)
    val_dataset = build_dataset(val_df, tokenizer, label2id)

    class_weights = inverse_frequency_weights(train_df["label"], classes)

    # Optimizer + linear schedule with 10% warmup. Ceil so a trailing partial
    # batch still counts as a step (Trainer keeps the last batch by default).
    steps_per_epoch = -(-len(train_dataset) // BATCH_SIZE)
    total_steps = steps_per_epoch * EPOCHS
    warmup_steps = int(WARMUP_RATIO * total_steps)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    training_args = TrainingArguments(
        output_dir=str(checkpoint_dir),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        max_grad_norm=GRADIENT_CLIP,
        fp16=torch.cuda.is_available(),
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_macro_f1",
        greater_is_better=True,
        save_total_limit=1,  # keep only the best checkpoint (Drive space)
        save_only_model=True,  # weights and config only: optimizer state would triple the size
        seed=args.seed,
        data_seed=args.seed,
        report_to=[],  # no wandb/tensorboard
        logging_strategy="epoch",
    )

    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
        optimizers=(optimizer, scheduler),
    )

    start = time.time()
    trainer.train()
    wall_clock = time.time() - start

    # The checkpoint directory a best model came from carries its global step, not an epoch
    # number; convert it so the run log does not mislabel a step as an epoch.
    best_ckpt = trainer.state.best_model_checkpoint
    best_step = int(best_ckpt.split("checkpoint-")[-1]) if best_ckpt else None
    best_epoch = round(best_step / steps_per_epoch) if best_step else None

    # Predict the validation split with the best checkpoint and write the
    # prediction file in the exact team format.
    logits = trainer.predict(val_dataset).predictions
    probs = softmax(logits)
    pred_idx = probs.argmax(axis=1)

    pred = pd.DataFrame({
        "note_id": val_df["note_id"].tolist(),
        "true_label": val_df["label"].tolist(),
        "pred_label": [classes[i] for i in pred_idx],
    })
    for i, c in enumerate(classes):
        pred[f"prob_{c}"] = probs[:, i]
    pred_path = results_dir / f"preds_{model_key}_{args.seed}_val.csv"
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    pred.to_csv(pred_path, index=False)

    val_macro_f1 = f1_score(
        val_df["label"].map(label2id).to_numpy(), pred_idx, average="macro", zero_division=0
    )
    print(f"validation macro-F1 (seed {args.seed}{suffix}): {val_macro_f1:.4f}")
    print(f"wrote {pred_path}")

    # Run log: versions, hardware, hyperparameters, masking option, per-epoch loss.
    log_dir = results_dir / "run_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
            capture_output=True, text=True, check=False,
        ).stdout.strip()
    except OSError:
        commit = "unknown"
    run_log = {
        "model": model_key,
        "seed": args.seed,
        "masked": args.masked,
        "masking_option": "C (see src/masking.py)" if args.masked else "none",
        "mask_token": MASK_TOKEN if args.masked else None,
        "commit": commit,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "python": platform.python_version(),
        "versions": {pkg: version(pkg) for pkg in
                     ["torch", "transformers", "tokenizers", "datasets", "scikit-learn", "numpy", "pandas"]},
        "hyperparameters": {
            "model": MODEL_NAME, "epochs": EPOCHS, "batch_size": BATCH_SIZE,
            "max_length": MAX_LENGTH, "head_tail": f"{HEAD_TAIL}+{HEAD_TAIL}",
            "learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY,
            "warmup_ratio": WARMUP_RATIO, "gradient_clip": GRADIENT_CLIP,
            "class_weighting": "inverse-frequency", "fp16": torch.cuda.is_available(),
        },
        "validation_loss_per_epoch": [h["eval_loss"] for h in trainer.state.log_history if "eval_loss" in h],
        "validation_macro_f1_per_epoch": [h["eval_macro_f1"] for h in trainer.state.log_history if "eval_macro_f1" in h],
        "best_checkpoint": str(trainer.state.best_model_checkpoint),
        "best_checkpoint_step": best_step,
        "best_epoch": best_epoch,
        "n_train": int(len(train_dataset)),
        "n_val": int(len(val_dataset)),
        "validation_macro_f1": val_macro_f1,
        "wall_clock_seconds": round(wall_clock, 1),
        "wall_clock_scope": "trainer.train() only, excludes tokenization, prediction and logging",
    }
    (log_dir / f"{model_key}_seed{args.seed}.json").write_text(json.dumps(run_log, indent=2))
    print(f"wrote {log_dir / f'{model_key}_seed{args.seed}.json'}")


if __name__ == "__main__":
    main()
