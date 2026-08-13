"""Fine-tune a RoBERTa-family encoder as a multi-label CRADLEBench crisis-type
classifier (Layer 1 of the pipeline in project_framework.md.txt).

Trains on CRADLEBench's fixed train/dev/test splits (no re-splitting), with a
sigmoid output over the 7 crisis types in labels.CRISIS_TYPES. The
flagged/not-flagged binary view is derived at eval time as "any crisis type
predicted", not trained as a separate head.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    Trainer,
    TrainingArguments,
)

from src.classifier.labels import CRISIS_TYPES
from src.classifier.metrics import compute_multilabel_metrics

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = PROJECT_ROOT / "data/processed/cradlebench_clf"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/cradlebench-roberta-classifier"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default="roberta-base")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--epochs", type=float, default=5.0)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke-test", action="store_true", help="Train for two steps")
    parser.add_argument(
        "--resume-from-checkpoint", type=Path, default=None, help="Resume training from a saved checkpoint dir"
    )
    parser.add_argument("--metric-for-best-model", default="f1_macro")
    parser.add_argument(
        "--class-weighted-loss",
        action="store_true",
        help="Weight BCE loss per class by inverse train-set frequency (see --pos-weight-power)",
    )
    parser.add_argument(
        "--pos-weight-power",
        type=float,
        default=0.5,
        help="Dampens class weights: pos_weight = ((n_train - n_pos) / n_pos) ** power. "
        "1.0 = full inverse frequency, 0.5 = sqrt-dampened (gentler), 0.0 = unweighted.",
    )
    return parser.parse_args()


class MultiLabelDataCollator:
    """Pads like DataCollatorWithPadding, then forces labels to float32.

    datasets.map() keeps the original int64 Arrow dtype for a column when a
    map function overwrites it under the same name (here, "labels" started as
    0/1 ints from the JSONL), even though the values returned are Python
    floats. BCEWithLogitsLoss requires a float target, so cast explicitly
    after padding rather than fighting the schema inference.
    """

    def __init__(self, tokenizer):
        self._pad = DataCollatorWithPadding(tokenizer)

    def __call__(self, features):
        batch = self._pad(features)
        batch["labels"] = batch["labels"].float()
        return batch


class WeightedBCETrainer(Trainer):
    """Trainer variant that applies a per-class pos_weight to BCEWithLogitsLoss.

    AutoModelForSequenceClassification's built-in multi_label_classification
    loss is an unweighted BCEWithLogitsLoss computed inside model.forward(), so
    weighting requires recomputing the loss here from logits rather than using
    the model's returned loss.
    """

    def __init__(self, *args, pos_weight: torch.Tensor, **kwargs):
        super().__init__(*args, **kwargs)
        self.pos_weight = pos_weight

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss_fct = torch.nn.BCEWithLogitsLoss(pos_weight=self.pos_weight.to(outputs.logits.device))
        loss = loss_fct(outputs.logits, labels)
        return (loss, outputs) if return_outputs else loss


def compute_pos_weight(train_dataset, power: float) -> torch.Tensor:
    labels = np.array(train_dataset["labels"], dtype=np.float64)
    n_total = labels.shape[0]
    n_pos = labels.sum(axis=0)
    weight = ((n_total - n_pos) / n_pos) ** power
    return torch.tensor(weight, dtype=torch.float32)


def make_compute_metrics(threshold: float):
    def compute_metrics(eval_pred: EvalPrediction) -> dict[str, float]:
        logits = eval_pred.predictions
        labels = eval_pred.label_ids
        probs = 1 / (1 + np.exp(-logits))
        preds = (probs >= threshold).astype(int)
        return compute_multilabel_metrics(labels, preds)

    return compute_metrics


def main() -> None:
    args = parse_args()

    files = {
        "train": str(args.data_dir / "train.jsonl"),
        "validation": str(args.data_dir / "validation.jsonl"),
        "test": str(args.data_dir / "test.jsonl"),
    }
    missing = [path for path in files.values() if not Path(path).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing processed data: {missing}. Run python -m src.classifier.preprocess first."
        )
    dataset = load_dataset("json", data_files=files)

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)

    def tokenize(batch: dict) -> dict:
        encoded = tokenizer(batch["text"], truncation=True, max_length=args.max_length)
        encoded["labels"] = [[float(x) for x in row] for row in batch["labels"]]
        return encoded

    dataset = dataset.map(tokenize, batched=True, remove_columns=dataset["train"].column_names)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_id,
        num_labels=len(CRISIS_TYPES),
        problem_type="multi_label_classification",
        id2label={i: t for i, t in enumerate(CRISIS_TYPES)},
        label2id={t: i for i, t in enumerate(CRISIS_TYPES)},
    )

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        max_steps=2 if args.smoke_test else -1,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_ratio=0.06,
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        logging_steps=20,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model=args.metric_for_best_model,
        greater_is_better=True,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        report_to="none",
        seed=args.seed,
    )

    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        processing_class=tokenizer,
        data_collator=MultiLabelDataCollator(tokenizer),
        compute_metrics=make_compute_metrics(args.threshold),
    )
    if args.class_weighted_loss:
        pos_weight = compute_pos_weight(dataset["train"], args.pos_weight_power)
        print(f"Using class-weighted BCE loss, pos_weight={dict(zip(CRISIS_TYPES, pos_weight.tolist()))}")
        trainer = WeightedBCETrainer(pos_weight=pos_weight, **trainer_kwargs)
    else:
        trainer = Trainer(**trainer_kwargs)
    trainer.train(resume_from_checkpoint=str(args.resume_from_checkpoint) if args.resume_from_checkpoint else None)

    test_metrics = trainer.evaluate(dataset["test"], metric_key_prefix="test")
    print("Test metrics:", test_metrics)

    final_model = args.output_dir / "final_model"
    trainer.save_model(str(final_model))
    tokenizer.save_pretrained(str(final_model))
    (final_model / "test_metrics.json").write_text(json.dumps(test_metrics, indent=2), encoding="utf-8")
    print(f"Saved classifier to {final_model}")


if __name__ == "__main__":
    main()
