"""Sliding-window inference for classifier inputs longer than max_length.

roberta-base's position embeddings cap sequence length well below what some
CRADLEBench posts need -- childabuse_endangerment posts run long enough that
even at max_length=512 (the trained model's ceiling), truncation drops
content for a meaningful share of them. This module splits a long input into
overlapping windows, runs one batched forward pass over all windows, and
max-pools per-class probabilities across windows so a positive signal
anywhere in the text survives. Short inputs (the common case) take a single
window identical to a plain forward pass -- this is a superset of the old
behavior, not a separate code path.
"""

from __future__ import annotations

import torch

from src.classifier.labels import CRISIS_TYPES

DEFAULT_MAX_LENGTH = 512
DEFAULT_STRIDE = 384  # max_length - stride = 128-token overlap between windows


def chunk_token_ids(
    content_ids: list[int],
    max_length: int = DEFAULT_MAX_LENGTH,
    stride: int = DEFAULT_STRIDE,
) -> list[list[int]]:
    """Split content token ids (no special tokens) into overlapping windows.

    Each window holds up to max_length - 2 tokens, leaving room for the
    <s>/</s> special tokens added later. `stride` is the step between window
    starts, so it must be smaller than the window size for windows to
    overlap (a crisis statement split across a boundary should still land
    fully inside at least one window).
    """
    window_size = max_length - 2
    if window_size <= 0:
        raise ValueError("max_length must be greater than 2 to leave room for special tokens")
    if not content_ids:
        return [[]]
    if len(content_ids) <= window_size:
        return [content_ids]
    if stride <= 0 or stride >= window_size:
        raise ValueError("stride must be a positive number smaller than max_length - 2")

    windows = []
    start = 0
    while True:
        end = start + window_size
        windows.append(content_ids[start:end])
        if end >= len(content_ids):
            break
        start += stride
    return windows


def predict_long_text(
    model,
    tokenizer,
    text: str,
    *,
    max_length: int = DEFAULT_MAX_LENGTH,
    stride: int = DEFAULT_STRIDE,
    threshold: float = 0.5,
    device: str | torch.device | None = None,
) -> dict:
    """Predict crisis types for `text`, windowing and max-pooling if it's long.

    Returns the same shape as a plain single-window prediction: {"flagged",
    "predicted_types", "scores"}, plus "num_windows" for observability.
    """
    if device is not None:
        model = model.to(device)

    content_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    windows = chunk_token_ids(content_ids, max_length=max_length, stride=stride)
    input_ids_list = [[tokenizer.bos_token_id, *window, tokenizer.eos_token_id] for window in windows]

    encoded = tokenizer.pad({"input_ids": input_ids_list}, padding=True, return_tensors="pt")
    if device is not None:
        encoded = {k: v.to(device) for k, v in encoded.items()}

    model.eval()
    with torch.inference_mode():
        logits = model(**encoded).logits
    pooled = torch.sigmoid(logits).max(dim=0).values.tolist()

    scores = dict(zip(CRISIS_TYPES, pooled))
    predicted = [crisis_type for crisis_type, prob in scores.items() if prob >= threshold]
    return {
        "flagged": len(predicted) > 0,
        "predicted_types": predicted,
        "scores": scores,
        "num_windows": len(windows),
    }
