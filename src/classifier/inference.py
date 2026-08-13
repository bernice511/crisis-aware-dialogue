"""Run one crisis-type prediction with a trained CRADLEBench classifier."""

from __future__ import annotations

import argparse
from pathlib import Path

from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.classifier.sliding_window import DEFAULT_MAX_LENGTH, DEFAULT_STRIDE, predict_long_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="Path to a saved final_model directory")
    parser.add_argument("--text", required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument(
        "--stride",
        type=int,
        default=DEFAULT_STRIDE,
        help="Step between window starts for inputs longer than --max-length",
    )
    return parser.parse_args()


def predict(
    model_dir: Path,
    text: str,
    threshold: float = 0.5,
    max_length: int = DEFAULT_MAX_LENGTH,
    stride: int = DEFAULT_STRIDE,
) -> dict:
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    return predict_long_text(model, tokenizer, text, max_length=max_length, stride=stride, threshold=threshold)


def main() -> None:
    args = parse_args()
    result = predict(args.model, args.text, threshold=args.threshold, max_length=args.max_length, stride=args.stride)
    print(f"flagged: {result['flagged']}")
    print(f"predicted_types: {result['predicted_types']}")
    print(f"num_windows: {result['num_windows']}")
    for crisis_type, score in result["scores"].items():
        print(f"  {crisis_type:28s} {score:.3f}")


if __name__ == "__main__":
    main()
