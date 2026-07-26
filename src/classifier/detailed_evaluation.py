"""Detailed evaluation for the CRADLEBench crisis-type classifier.

This script adds stronger evaluation beyond basic F1:
1. Dataset sanity checks
2. Model prediction probabilities
3. Basic overall metrics
4. Per-class precision/recall/F1/support
5. TP/FP/FN/TN confusion breakdown
6. Miss rate and false alarm rate
7. False positive and false negative examples for qualitative analysis
8. Threshold sweep using validation data
9. Per-class threshold tuning using validation data
10. Bootstrap confidence intervals for test metrics
11. Calibration check for predicted probabilities

Usage:
    python3 -m src.classifier.detailed_evaluation \
        --model /path/to/final_model
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.classifier.labels import CRISIS_TYPES


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data/processed/cradlebench_clf"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results/classifier_detailed"


def print_eval_explanation(title: str, line_1: str, line_2: str) -> None:
    """Print a short explanation before each evaluation section."""
    print(f"\n{title}")
    print("=" * 40)
    print(line_1)
    print(line_2)


def get_device() -> str:
    """Use CUDA if available, otherwise Apple Silicon MPS, otherwise CPU."""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_split(path: Path) -> tuple[list[str], np.ndarray]:
    """Load one JSONL split and return texts and multi-hot labels."""
    texts: list[str] = []
    labels: list[list[int]] = []

    with path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            texts.append(record["text"])
            labels.append(record["labels"])

    return texts, np.array(labels, dtype=int)


def print_dataset_summary(split_name: str, texts: list[str], labels: np.ndarray) -> None:
    """Print number of examples and label counts for one split."""
    print(f"\n{split_name.upper()} SET")
    print("-" * 40)
    print(f"Examples: {len(texts)}")
    print(f"Label matrix shape: {labels.shape}")
    print(f"Flagged examples: {int(labels.any(axis=1).sum())}")

    print("\nPer-class support:")
    for i, crisis_type in enumerate(CRISIS_TYPES):
        support = int(labels[:, i].sum())
        print(f"  {crisis_type:28s} {support}")


def load_model_and_tokenizer(model_dir: Path, device: str):
    """Load the saved Hugging Face model and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)

    model.to(device)
    model.eval()

    return model, tokenizer


def predict_probabilities(
    model,
    tokenizer,
    texts: list[str],
    device: str,
    batch_size: int,
    max_length: int,
) -> np.ndarray:
    """Run model inference and return sigmoid probabilities."""
    all_probs: list[np.ndarray] = []

    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start : start + batch_size]

            inputs = tokenizer(
                batch_texts,
                truncation=True,
                padding=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(device)

            logits = model(**inputs).logits
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)

            end = min(start + batch_size, len(texts))
            if end % 100 == 0 or end == len(texts):
                print(f"  scored {end}/{len(texts)} examples", flush=True)

    return np.concatenate(all_probs, axis=0)


def compute_basic_metrics(labels: np.ndarray, preds: np.ndarray) -> dict[str, float]:
    """Compute overall multi-label and flagged/unflagged metrics."""
    flagged_true = labels.any(axis=1)
    flagged_pred = preds.any(axis=1)

    return {
        "micro_precision": float(precision_score(labels, preds, average="micro", zero_division=0)),
        "micro_recall": float(recall_score(labels, preds, average="micro", zero_division=0)),
        "micro_f1": float(f1_score(labels, preds, average="micro", zero_division=0)),
        "macro_precision": float(precision_score(labels, preds, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(labels, preds, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(labels, preds, average="weighted", zero_division=0)),
        "exact_match_accuracy": float(accuracy_score(labels, preds)),
        "flagged_accuracy": float((flagged_true == flagged_pred).mean()),
        "flagged_f1": float(f1_score(flagged_true, flagged_pred, zero_division=0)),
    }


def compute_per_class_breakdown(labels: np.ndarray, preds: np.ndarray) -> pd.DataFrame:
    """Compute per-class metrics and confusion-matrix counts."""
    precision, recall, f1, support = precision_recall_fscore_support(
        labels,
        preds,
        average=None,
        zero_division=0,
    )

    rows: list[dict[str, float | int | str]] = []

    for i, crisis_type in enumerate(CRISIS_TYPES):
        col_true = labels[:, i]
        col_pred = preds[:, i]

        tp = int(((col_true == 1) & (col_pred == 1)).sum())
        fp = int(((col_true == 0) & (col_pred == 1)).sum())
        fn = int(((col_true == 1) & (col_pred == 0)).sum())
        tn = int(((col_true == 0) & (col_pred == 0)).sum())

        miss_rate = fn / (fn + tp) if (fn + tp) > 0 else np.nan
        false_alarm_rate = fp / (fp + tn) if (fp + tn) > 0 else np.nan

        rows.append(
            {
                "crisis_type": crisis_type,
                "support": int(support[i]),
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "miss_rate": float(miss_rate),
                "false_alarm_rate": float(false_alarm_rate),
            }
        )

    return pd.DataFrame(rows)


def label_names_from_vector(vector: np.ndarray) -> list[str]:
    """Convert a multi-hot label vector into readable label names."""
    return [
        crisis_type
        for crisis_type, present in zip(CRISIS_TYPES, vector)
        if int(present) == 1
    ]


def save_error_examples(
    texts: list[str],
    labels: np.ndarray,
    preds: np.ndarray,
    probs: np.ndarray,
    output_dir: Path,
    prefix: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Save false negatives and false positives for qualitative error analysis."""
    false_negative_rows = []
    false_positive_rows = []

    for example_id, text in enumerate(texts):
        true_vector = labels[example_id]
        pred_vector = preds[example_id]
        prob_vector = probs[example_id]

        true_labels = label_names_from_vector(true_vector)
        predicted_labels = label_names_from_vector(pred_vector)

        for i, crisis_type in enumerate(CRISIS_TYPES):
            true_value = int(true_vector[i])
            predicted_value = int(pred_vector[i])
            probability = float(prob_vector[i])

            row = {
                "example_id": example_id,
                "crisis_type": crisis_type,
                "text": text,
                "true_labels": ";".join(true_labels),
                "predicted_labels": ";".join(predicted_labels),
                "probability_for_crisis_type": probability,
            }

            if true_value == 1 and predicted_value == 0:
                false_negative_rows.append(row)

            if true_value == 0 and predicted_value == 1:
                false_positive_rows.append(row)

    false_negatives_df = pd.DataFrame(false_negative_rows)
    false_positives_df = pd.DataFrame(false_positive_rows)

    false_negatives_path = output_dir / f"{prefix}_false_negatives.csv"
    false_positives_path = output_dir / f"{prefix}_false_positives.csv"

    false_negatives_df.to_csv(false_negatives_path, index=False)
    false_positives_df.to_csv(false_positives_path, index=False)

    return false_negatives_df, false_positives_df


def run_threshold_sweep(
    labels: np.ndarray,
    probs: np.ndarray,
    thresholds: list[float],
) -> pd.DataFrame:
    """Evaluate model performance across possible single thresholds."""
    rows = []

    for threshold in thresholds:
        preds = (probs >= threshold).astype(int)
        metrics = compute_basic_metrics(labels, preds)

        rows.append(
            {
                "threshold": threshold,
                "micro_precision": metrics["micro_precision"],
                "micro_recall": metrics["micro_recall"],
                "micro_f1": metrics["micro_f1"],
                "macro_precision": metrics["macro_precision"],
                "macro_recall": metrics["macro_recall"],
                "macro_f1": metrics["macro_f1"],
                "flagged_f1": metrics["flagged_f1"],
                "exact_match_accuracy": metrics["exact_match_accuracy"],
            }
        )

    return pd.DataFrame(rows)


def tune_per_class_thresholds(
    labels: np.ndarray,
    probs: np.ndarray,
    thresholds: list[float],
) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    """Choose the best threshold for each crisis type using validation F1."""
    sweep_rows = []
    selected_rows = []
    selected_thresholds = []

    for i, crisis_type in enumerate(CRISIS_TYPES):
        best_row = None

        for threshold in thresholds:
            col_true = labels[:, i]
            col_pred = (probs[:, i] >= threshold).astype(int)

            precision, recall, f1, support = precision_recall_fscore_support(
                col_true,
                col_pred,
                average="binary",
                zero_division=0,
            )

            tp = int(((col_true == 1) & (col_pred == 1)).sum())
            fp = int(((col_true == 0) & (col_pred == 1)).sum())
            fn = int(((col_true == 1) & (col_pred == 0)).sum())
            tn = int(((col_true == 0) & (col_pred == 0)).sum())

            row = {
                "crisis_type": crisis_type,
                "threshold": threshold,
                "support": int(labels[:, i].sum()),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
            }
            sweep_rows.append(row)

            if best_row is None:
                best_row = row
            else:
                current_key = (row["f1"], row["recall"], row["precision"])
                best_key = (best_row["f1"], best_row["recall"], best_row["precision"])
                if current_key > best_key:
                    best_row = row

        assert best_row is not None
        selected_rows.append(best_row)
        selected_thresholds.append(float(best_row["threshold"]))

    return (
        np.array(selected_thresholds, dtype=float),
        pd.DataFrame(selected_rows),
        pd.DataFrame(sweep_rows),
    )


def apply_per_class_thresholds(probs: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """Apply one threshold per class to probability outputs."""
    return (probs >= thresholds.reshape(1, -1)).astype(int)


def bootstrap_confidence_intervals(
    labels: np.ndarray,
    preds: np.ndarray,
    n_bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    """Bootstrap test examples to estimate 95% confidence intervals for metrics."""
    rng = np.random.default_rng(seed)
    n_examples = labels.shape[0]

    metric_names = [
        "micro_f1",
        "macro_f1",
        "weighted_f1",
        "exact_match_accuracy",
        "flagged_f1",
    ]

    values: dict[str, list[float]] = {metric: [] for metric in metric_names}

    for _ in range(n_bootstrap):
        sample_indices = rng.integers(0, n_examples, size=n_examples)
        sample_labels = labels[sample_indices]
        sample_preds = preds[sample_indices]
        sample_metrics = compute_basic_metrics(sample_labels, sample_preds)

        for metric in metric_names:
            values[metric].append(sample_metrics[metric])

    rows = []
    point_metrics = compute_basic_metrics(labels, preds)

    for metric in metric_names:
        boot_values = np.array(values[metric], dtype=float)

        rows.append(
            {
                "metric": metric,
                "point_estimate": point_metrics[metric],
                "ci_lower_95": float(np.percentile(boot_values, 2.5)),
                "ci_upper_95": float(np.percentile(boot_values, 97.5)),
                "bootstrap_mean": float(boot_values.mean()),
                "bootstrap_std": float(boot_values.std(ddof=1)),
                "n_bootstrap": n_bootstrap,
            }
        )

    return pd.DataFrame(rows)


def bootstrap_per_class_f1_intervals(
    labels: np.ndarray,
    preds: np.ndarray,
    n_bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    """Bootstrap test examples to estimate 95% CIs for per-class F1."""
    rng = np.random.default_rng(seed)
    n_examples = labels.shape[0]

    values: dict[str, list[float]] = {crisis_type: [] for crisis_type in CRISIS_TYPES}

    for _ in range(n_bootstrap):
        sample_indices = rng.integers(0, n_examples, size=n_examples)
        sample_labels = labels[sample_indices]
        sample_preds = preds[sample_indices]

        _, _, f1, support = precision_recall_fscore_support(
            sample_labels,
            sample_preds,
            average=None,
            zero_division=0,
        )

        for crisis_type, f1_value in zip(CRISIS_TYPES, f1):
            values[crisis_type].append(float(f1_value))

    point_breakdown = compute_per_class_breakdown(labels, preds)

    rows = []
    for crisis_type in CRISIS_TYPES:
        boot_values = np.array(values[crisis_type], dtype=float)
        point_f1 = float(point_breakdown.loc[point_breakdown["crisis_type"] == crisis_type, "f1"].iloc[0])
        support = int(point_breakdown.loc[point_breakdown["crisis_type"] == crisis_type, "support"].iloc[0])

        rows.append(
            {
                "crisis_type": crisis_type,
                "support": support,
                "point_f1": point_f1,
                "ci_lower_95": float(np.percentile(boot_values, 2.5)),
                "ci_upper_95": float(np.percentile(boot_values, 97.5)),
                "bootstrap_mean": float(boot_values.mean()),
                "bootstrap_std": float(boot_values.std(ddof=1)),
                "n_bootstrap": n_bootstrap,
            }
        )

    return pd.DataFrame(rows)


def compute_calibration_bins(
    labels_1d: np.ndarray,
    probs_1d: np.ndarray,
    n_bins: int,
    group_name: str,
) -> tuple[pd.DataFrame, dict[str, float | str | int]]:
    """Compute reliability-bin rows and summary calibration metrics."""
    labels_1d = labels_1d.astype(int)
    probs_1d = probs_1d.astype(float)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []

    total_count = len(labels_1d)
    ece = 0.0

    for bin_id in range(n_bins):
        left = bin_edges[bin_id]
        right = bin_edges[bin_id + 1]

        if bin_id == n_bins - 1:
            mask = (probs_1d >= left) & (probs_1d <= right)
        else:
            mask = (probs_1d >= left) & (probs_1d < right)

        count = int(mask.sum())

        if count == 0:
            avg_confidence = np.nan
            empirical_accuracy = np.nan
            abs_gap = np.nan
        else:
            avg_confidence = float(probs_1d[mask].mean())
            empirical_accuracy = float(labels_1d[mask].mean())
            abs_gap = abs(avg_confidence - empirical_accuracy)
            ece += (count / total_count) * abs_gap

        rows.append(
            {
                "group": group_name,
                "bin_id": bin_id,
                "bin_left": float(left),
                "bin_right": float(right),
                "count": count,
                "avg_confidence": avg_confidence,
                "empirical_accuracy": empirical_accuracy,
                "abs_gap": abs_gap,
            }
        )

    brier_score = float(np.mean((probs_1d - labels_1d) ** 2))

    summary = {
        "group": group_name,
        "n_predictions": total_count,
        "n_bins": n_bins,
        "brier_score": brier_score,
        "expected_calibration_error": float(ece),
    }

    return pd.DataFrame(rows), summary


def compute_calibration_report(
    labels: np.ndarray,
    probs: np.ndarray,
    n_bins: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute overall and per-class calibration metrics."""
    all_bin_dfs = []
    summary_rows = []

    overall_bins_df, overall_summary = compute_calibration_bins(
        labels_1d=labels.reshape(-1),
        probs_1d=probs.reshape(-1),
        n_bins=n_bins,
        group_name="overall",
    )

    all_bin_dfs.append(overall_bins_df)
    summary_rows.append(overall_summary)

    for i, crisis_type in enumerate(CRISIS_TYPES):
        class_bins_df, class_summary = compute_calibration_bins(
            labels_1d=labels[:, i],
            probs_1d=probs[:, i],
            n_bins=n_bins,
            group_name=crisis_type,
        )

        all_bin_dfs.append(class_bins_df)
        summary_rows.append(class_summary)

    calibration_bins_df = pd.concat(all_bin_dfs, ignore_index=True)
    calibration_summary_df = pd.DataFrame(summary_rows)

    return calibration_summary_df, calibration_bins_df, overall_bins_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Path to processed CRADLEBench classifier data.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to the saved final_model directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Where to save detailed evaluation results.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for model inference.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Maximum sequence length for truncation.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Decision threshold for multi-label prediction.",
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=1000,
        help="Number of bootstrap samples for confidence intervals.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=13,
        help="Random seed for bootstrap sampling.",
    )
    parser.add_argument(
        "--calibration-bins",
        type=int,
        default=10,
        help="Number of probability bins for calibration evaluation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = get_device()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Detailed Classifier Evaluation")
    print("=" * 40)
    print(f"Device: {device}")
    print(f"Model path: {args.model}")
    print(f"Threshold: {args.threshold}")
    print(f"Output directory: {args.output_dir}")
    print(f"Number of crisis types: {len(CRISIS_TYPES)}")
    print(f"Crisis types: {list(CRISIS_TYPES)}")

    print_eval_explanation(
        "EVALUATION 1: DATASET SANITY CHECK",
        "This checks whether the train, validation, and test files loaded correctly.",
        "It also shows class imbalance, which matters because rare classes can have unstable F1 scores.",
    )

    train_texts, train_labels = load_split(args.data_dir / "train.jsonl")
    val_texts, val_labels = load_split(args.data_dir / "validation.jsonl")
    test_texts, test_labels = load_split(args.data_dir / "test.jsonl")

    print_dataset_summary("train", train_texts, train_labels)
    print_dataset_summary("validation", val_texts, val_labels)
    print_dataset_summary("test", test_texts, test_labels)

    print_eval_explanation(
        "EVALUATION 2: MODEL PREDICTION PROBABILITIES",
        "This runs the trained RoBERTa classifier and gets sigmoid probabilities.",
        "These probabilities are later converted into labels using a decision threshold.",
    )

    print("\nLoading model...")
    model, tokenizer = load_model_and_tokenizer(args.model, device)
    print("Model loaded successfully.")

    print("\nScoring validation set...")
    val_probs = predict_probabilities(
        model=model,
        tokenizer=tokenizer,
        texts=val_texts,
        device=device,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )

    print("\nScoring test set...")
    test_probs = predict_probabilities(
        model=model,
        tokenizer=tokenizer,
        texts=test_texts,
        device=device,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )

    print("\nPrediction check")
    print("-" * 40)
    print(f"Validation probabilities shape: {val_probs.shape}")
    print(f"Test probabilities shape: {test_probs.shape}")
    print(f"Validation probabilities range: {val_probs.min():.4f} to {val_probs.max():.4f}")
    print(f"Test probabilities range: {test_probs.min():.4f} to {test_probs.max():.4f}")

    test_preds = (test_probs >= args.threshold).astype(int)

    print_eval_explanation(
        "EVALUATION 3: BASIC OVERALL METRICS",
        "This reports overall performance using micro F1, macro F1, weighted F1, and exact-match accuracy.",
        "For crisis detection, flagged F1 is also useful because it checks whether the model detects any crisis at all.",
    )

    basic_metrics = compute_basic_metrics(test_labels, test_preds)
    basic_metrics_path = args.output_dir / "basic_metrics.json"
    basic_metrics_path.write_text(json.dumps(basic_metrics, indent=2), encoding="utf-8")

    print("\nBASIC TEST METRICS")
    print("-" * 40)
    for key, value in basic_metrics.items():
        print(f"{key:24s} {value:.4f}")

    print_eval_explanation(
        "EVALUATION 4: PER-CLASS PRECISION, RECALL, F1, AND SUPPORT",
        "This shows how well the model performs on each crisis category instead of hiding everything in one score.",
        "Recall is especially important here because low recall means the model is missing real crisis examples.",
    )

    per_class_df = compute_per_class_breakdown(test_labels, test_preds)
    per_class_path = args.output_dir / "per_class_report.csv"
    per_class_df[
        ["crisis_type", "support", "precision", "recall", "f1"]
    ].to_csv(per_class_path, index=False)

    print("\nPER-CLASS REPORT")
    print("-" * 40)
    print(
        per_class_df[
            ["crisis_type", "support", "precision", "recall", "f1"]
        ].to_string(index=False)
    )

    print_eval_explanation(
        "EVALUATION 5: CONFUSION-MATRIX-LEVEL ERROR BREAKDOWN",
        "This gives TP, FP, FN, and TN for each crisis type so we can see the exact kind of mistakes.",
        "Miss rate is critical for safety because it measures how many real crisis cases were not detected.",
    )

    confusion_path = args.output_dir / "confusion_breakdown.csv"
    per_class_df[
        ["crisis_type", "tp", "fp", "fn", "tn", "miss_rate", "false_alarm_rate"]
    ].to_csv(confusion_path, index=False)

    print("\nCONFUSION BREAKDOWN")
    print("-" * 40)
    print(
        per_class_df[
            ["crisis_type", "tp", "fp", "fn", "tn", "miss_rate", "false_alarm_rate"]
        ].to_string(index=False)
    )

    print_eval_explanation(
        "EVALUATION 6: FALSE POSITIVE AND FALSE NEGATIVE EXAMPLES",
        "This saves the actual examples where the model made mistakes so we can inspect them manually.",
        "False negatives are the most important for this project because they are real crisis examples the model missed.",
    )

    false_negatives_df, false_positives_df = save_error_examples(
        texts=test_texts,
        labels=test_labels,
        preds=test_preds,
        probs=test_probs,
        output_dir=args.output_dir,
        prefix="threshold_0_5",
    )

    false_negatives_path = args.output_dir / "threshold_0_5_false_negatives.csv"
    false_positives_path = args.output_dir / "threshold_0_5_false_positives.csv"

    print("\nERROR EXAMPLES")
    print("-" * 40)
    print(f"False negative rows: {len(false_negatives_df)}")
    print(f"False positive rows: {len(false_positives_df)}")

    print_eval_explanation(
        "EVALUATION 7: THRESHOLD SWEEP",
        "This tests several thresholds instead of assuming 0.5 is the best cutoff.",
        "The best threshold is selected on validation data, then evaluated once on the test set.",
    )

    thresholds = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

    val_threshold_df = run_threshold_sweep(
        labels=val_labels,
        probs=val_probs,
        thresholds=thresholds,
    )

    threshold_sweep_path = args.output_dir / "threshold_sweep_validation.csv"
    val_threshold_df.to_csv(threshold_sweep_path, index=False)

    best_row = val_threshold_df.sort_values(
        by=["macro_f1", "micro_f1"],
        ascending=False,
    ).iloc[0]

    best_threshold = float(best_row["threshold"])

    test_preds_best_threshold = (test_probs >= best_threshold).astype(int)
    test_metrics_best_threshold = compute_basic_metrics(
        test_labels,
        test_preds_best_threshold,
    )

    best_threshold_test_path = args.output_dir / "test_metrics_best_single_threshold.json"
    best_threshold_test_path.write_text(
        json.dumps(
            {
                "selected_on_validation": {
                    "best_threshold": best_threshold,
                    "selection_metric": "macro_f1",
                    "validation_metrics": best_row.to_dict(),
                },
                "test_metrics_at_best_threshold": test_metrics_best_threshold,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\nVALIDATION THRESHOLD SWEEP")
    print("-" * 40)
    print(val_threshold_df.to_string(index=False))

    print("\nBEST SINGLE THRESHOLD")
    print("-" * 40)
    print(f"Best threshold selected on validation macro F1: {best_threshold}")

    print("\nTEST METRICS AT BEST SINGLE THRESHOLD")
    print("-" * 40)
    for key, value in test_metrics_best_threshold.items():
        print(f"{key:24s} {value:.4f}")

    print_eval_explanation(
        "EVALUATION 8: PER-CLASS THRESHOLD TUNING",
        "This chooses a separate threshold for each crisis type using validation F1.",
        "It can improve performance when different labels have different confidence patterns or class imbalance.",
    )

    per_class_thresholds, selected_thresholds_df, per_class_threshold_sweep_df = tune_per_class_thresholds(
        labels=val_labels,
        probs=val_probs,
        thresholds=thresholds,
    )

    per_class_thresholds_path = args.output_dir / "per_class_thresholds_selected.csv"
    per_class_threshold_sweep_path = args.output_dir / "per_class_threshold_sweep_validation.csv"

    selected_thresholds_df.to_csv(per_class_thresholds_path, index=False)
    per_class_threshold_sweep_df.to_csv(per_class_threshold_sweep_path, index=False)

    test_preds_per_class_thresholds = apply_per_class_thresholds(
        probs=test_probs,
        thresholds=per_class_thresholds,
    )

    test_metrics_per_class_thresholds = compute_basic_metrics(
        test_labels,
        test_preds_per_class_thresholds,
    )

    test_per_class_threshold_metrics_path = args.output_dir / "test_metrics_per_class_thresholds.json"
    test_per_class_threshold_metrics_path.write_text(
        json.dumps(
            {
                "selected_on_validation": selected_thresholds_df.to_dict(orient="records"),
                "test_metrics_at_per_class_thresholds": test_metrics_per_class_thresholds,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    per_class_threshold_test_breakdown = compute_per_class_breakdown(
        test_labels,
        test_preds_per_class_thresholds,
    )

    per_class_threshold_test_breakdown_path = args.output_dir / "test_breakdown_per_class_thresholds.csv"
    per_class_threshold_test_breakdown.to_csv(per_class_threshold_test_breakdown_path, index=False)

    print("\nSELECTED PER-CLASS THRESHOLDS")
    print("-" * 40)
    print(
        selected_thresholds_df[
            ["crisis_type", "threshold", "support", "precision", "recall", "f1"]
        ].to_string(index=False)
    )

    print("\nTEST METRICS AT PER-CLASS THRESHOLDS")
    print("-" * 40)
    for key, value in test_metrics_per_class_thresholds.items():
        print(f"{key:24s} {value:.4f}")

    print("\nTEST PER-CLASS BREAKDOWN AT PER-CLASS THRESHOLDS")
    print("-" * 40)
    print(
        per_class_threshold_test_breakdown[
            ["crisis_type", "support", "precision", "recall", "f1", "tp", "fp", "fn", "tn", "miss_rate"]
        ].to_string(index=False)
    )

    print_eval_explanation(
        "EVALUATION 9: BOOTSTRAP CONFIDENCE INTERVALS",
        "This resamples the test set many times to estimate how stable the reported metrics are.",
        "Wide intervals mean the result is noisy, especially for smaller or harder crisis categories.",
    )

    bootstrap_overall_df = bootstrap_confidence_intervals(
        labels=test_labels,
        preds=test_preds,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
    )

    bootstrap_per_class_df = bootstrap_per_class_f1_intervals(
        labels=test_labels,
        preds=test_preds,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
    )

    bootstrap_overall_path = args.output_dir / "bootstrap_overall_metrics.csv"
    bootstrap_per_class_path = args.output_dir / "bootstrap_per_class_f1.csv"

    bootstrap_overall_df.to_csv(bootstrap_overall_path, index=False)
    bootstrap_per_class_df.to_csv(bootstrap_per_class_path, index=False)

    print("\nBOOTSTRAP OVERALL METRICS")
    print("-" * 40)
    print(bootstrap_overall_df.to_string(index=False))

    print("\nBOOTSTRAP PER-CLASS F1")
    print("-" * 40)
    print(bootstrap_per_class_df.to_string(index=False))

    print_eval_explanation(
        "EVALUATION 10: CALIBRATION CHECK",
        "This checks whether the model confidence scores behave like meaningful probabilities.",
        "Brier score and expected calibration error help show whether probability outputs can be trusted.",
    )

    calibration_summary_df, calibration_bins_df, overall_bins_df = compute_calibration_report(
        labels=test_labels,
        probs=test_probs,
        n_bins=args.calibration_bins,
    )

    calibration_summary_path = args.output_dir / "calibration_summary.csv"
    calibration_bins_path = args.output_dir / "calibration_bins.csv"
    calibration_overall_bins_path = args.output_dir / "calibration_overall_bins.csv"

    calibration_summary_df.to_csv(calibration_summary_path, index=False)
    calibration_bins_df.to_csv(calibration_bins_path, index=False)
    overall_bins_df.to_csv(calibration_overall_bins_path, index=False)

    print("\nCALIBRATION SUMMARY")
    print("-" * 40)
    print(calibration_summary_df.to_string(index=False))

    print("\nOVERALL CALIBRATION BINS")
    print("-" * 40)
    print(overall_bins_df.to_string(index=False))

    print("\nSaved files:")
    print(f"  {basic_metrics_path}")
    print(f"  {per_class_path}")
    print(f"  {confusion_path}")
    print(f"  {false_negatives_path}")
    print(f"  {false_positives_path}")
    print(f"  {threshold_sweep_path}")
    print(f"  {best_threshold_test_path}")
    print(f"  {per_class_thresholds_path}")
    print(f"  {per_class_threshold_sweep_path}")
    print(f"  {test_per_class_threshold_metrics_path}")
    print(f"  {per_class_threshold_test_breakdown_path}")
    print(f"  {bootstrap_overall_path}")
    print(f"  {bootstrap_per_class_path}")
    print(f"  {calibration_summary_path}")
    print(f"  {calibration_bins_path}")
    print(f"  {calibration_overall_bins_path}")


if __name__ == "__main__":
    main()