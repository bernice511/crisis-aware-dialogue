"""Compare plain truncated inference against the sliding-window wrapper on
the CRADLEBench test split, to measure whether windowing recovers signal
lost to truncation (see src/classifier/sliding_window.py).

Usage:
    python -m src.classifier.evaluate_sliding_window \\
        --model outputs/cradlebench-roberta-classifier-ml512-weighted/final_model
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.classifier.labels import CRISIS_TYPES
from src.classifier.metrics import compute_multilabel_metrics
from src.classifier.sliding_window import DEFAULT_MAX_LENGTH, DEFAULT_STRIDE, predict_long_text

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = PROJECT_ROOT / "data/processed/cradlebench_clf"


def load_split(path: Path) -> tuple[list[str], np.ndarray]:
    texts: list[str] = []
    labels: list[list[int]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            texts.append(record["text"])
            labels.append(record["labels"])
    return texts, np.array(labels, dtype=int)


def score_truncated(model, tokenizer, texts: list[str], max_length: int, batch_size: int, device: str) -> np.ndarray:
    all_probs = []
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            inputs = tokenizer(
                batch, truncation=True, max_length=max_length, padding=True, return_tensors="pt"
            ).to(device)
            logits = model(**inputs).logits
            all_probs.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(all_probs, axis=0)


def score_sliding_window(
    model, tokenizer, texts: list[str], max_length: int, stride: int, device: str
) -> tuple[np.ndarray, list[int]]:
    all_probs = []
    num_windows = []
    for i, text in enumerate(texts):
        result = predict_long_text(model, tokenizer, text, max_length=max_length, stride=stride, device=device)
        all_probs.append([result["scores"][t] for t in CRISIS_TYPES])
        num_windows.append(result["num_windows"])
        if (i + 1) % 50 == 0 or (i + 1) == len(texts):
            print(f"  windowed scoring: {i + 1}/{len(texts)}", flush=True)
    return np.array(all_probs), num_windows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="Path to a saved final_model directory")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--stride", type=int, default=DEFAULT_STRIDE)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output", type=Path, default=None, help="Where to write the comparison JSON")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model)
    model.to(device)
    model.eval()

    texts, labels = load_split(args.data_dir / "test.jsonl")
    print(f"Loaded {len(texts)} test examples. Scoring on device={device}.", flush=True)

    print("Scoring truncated baseline (batched)...", flush=True)
    truncated_probs = score_truncated(model, tokenizer, texts, args.max_length, args.batch_size, device)
    print("Scoring sliding-window wrapper (per-example)...", flush=True)
    windowed_probs, num_windows = score_sliding_window(model, tokenizer, texts, args.max_length, args.stride, device)

    truncated_preds = (truncated_probs >= args.threshold).astype(int)
    windowed_preds = (windowed_probs >= args.threshold).astype(int)

    truncated_metrics = compute_multilabel_metrics(labels, truncated_preds)
    windowed_metrics = compute_multilabel_metrics(labels, windowed_preds)

    multi_window_count = sum(1 for n in num_windows if n > 1)
    print(f"{multi_window_count}/{len(texts)} test examples needed more than one window\n")

    print(f"{'crisis_type':28s} {'f1_truncated':>13s} {'f1_windowed':>12s} {'delta':>8s}")
    for crisis_type in CRISIS_TYPES:
        t = truncated_metrics[f"f1_{crisis_type}"]
        w = windowed_metrics[f"f1_{crisis_type}"]
        print(f"{crisis_type:28s} {t:13.4f} {w:12.4f} {w - t:+8.4f}")

    print(
        f"\n{'micro/macro F1':28s} "
        f"{truncated_metrics['f1_micro']:.4f}/{truncated_metrics['f1_macro']:.4f}"
        f"    {windowed_metrics['f1_micro']:.4f}/{windowed_metrics['f1_macro']:.4f}"
    )
    print(
        f"{'flagged (acc/f1)':28s} "
        f"{truncated_metrics['flagged_accuracy']:.4f}/{truncated_metrics['flagged_f1']:.4f}"
        f"    {windowed_metrics['flagged_accuracy']:.4f}/{windowed_metrics['flagged_f1']:.4f}"
    )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(
                {
                    "multi_window_count": multi_window_count,
                    "num_test_examples": len(texts),
                    "truncated_metrics": truncated_metrics,
                    "windowed_metrics": windowed_metrics,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\nSaved results to {args.output}")


if __name__ == "__main__":
    main()
