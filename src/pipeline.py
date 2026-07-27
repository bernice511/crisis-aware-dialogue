"""Glue between the Layer 1 crisis classifier and the Layer 3 Llama response model.

Given raw user text plus the running conversation state, this module runs the
classifier, folds its prediction into the risk-context schema the response
model was trained on, and generates the next Listener reply.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch
from peft import AutoPeftModelForCausalLM, PeftModel
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer

from src.classifier.sliding_window import DEFAULT_MAX_LENGTH, DEFAULT_STRIDE, predict_long_text
from src.cradle_response.inference import build_messages, build_risk_context
from src.cradle_response.preprocess import redact_pii

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CLASSIFIER_DIR = PROJECT_ROOT / "outputs/cradlebench-roberta-classifier-ml512-weighted/final_model"
DEFAULT_ADAPTER_DIR = PROJECT_ROOT / "outputs/llama-3.2-1b-cradle-response-qlora-full/final_adapter"

# adapter_config.json points at the gated meta-llama/Llama-3.2-1B-Instruct repo.
# Set this env var to an ungated mirror (e.g. "unsloth/Llama-3.2-1B-Instruct",
# same weights/config, just re-uploaded without Meta's manual-review gate) to
# unblock local testing while a gated-access request is pending review.
BASE_MODEL_OVERRIDE = os.environ.get("CRADLE_BASE_MODEL_ID")


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_classifier(model_dir: Path = DEFAULT_CLASSIFIER_DIR):
    """Load the trained RoBERTa crisis-type classifier."""
    device = get_device()
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device).eval()
    return model, tokenizer


def load_response_model(adapter_dir: Path = DEFAULT_ADAPTER_DIR, base_model_id: str | None = BASE_MODEL_OVERRIDE):
    """Load the Llama-3.2-1B-Instruct base model with the CRADLE-response LoRA adapter.

    No quantization here (unlike training) so this runs on CPU/MPS without CUDA
    or bitsandbytes. By default the base model downloads from the gated
    meta-llama/Llama-3.2-1B-Instruct repo per adapter_config.json (requires
    accepted HF access); pass base_model_id (or set CRADLE_BASE_MODEL_ID) to
    load the adapter onto a different, ungated copy of the same weights
    instead.
    """
    device = get_device()
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
    if base_model_id:
        base_model = AutoModelForCausalLM.from_pretrained(base_model_id, dtype=torch.float32)
        model = PeftModel.from_pretrained(base_model, adapter_dir)
    else:
        model = AutoPeftModelForCausalLM.from_pretrained(adapter_dir, dtype=torch.float32)
    model = model.to(device).eval()
    return model, tokenizer


def classify(model, tokenizer, text: str, threshold: float = 0.5) -> dict[str, Any]:
    """Run the sliding-window classifier and return its scores/labels."""
    return predict_long_text(
        model,
        tokenizer,
        text,
        max_length=DEFAULT_MAX_LENGTH,
        stride=DEFAULT_STRIDE,
        threshold=threshold,
        device=model.device,
    )


def update_known_events(known_events: list[str], predicted_types: list[str]) -> list[str]:
    """Accumulate newly seen crisis labels across the conversation, de-duplicated."""
    updated = list(known_events)
    for label in predicted_types:
        if label not in updated:
            updated.append(label)
    return updated


def build_context(
    classifier_result: dict[str, Any], known_events: list[str]
) -> tuple[dict[str, Any], list[str]]:
    """Map a classifier prediction into the trained risk-context schema.

    Returns the context dict for this turn and the updated known_events list.
    """
    predicted_types = classifier_result.get("predicted_types", [])
    updated_known_events = update_known_events(known_events, predicted_types)
    context = build_risk_context(predicted_types, updated_known_events)
    return context, updated_known_events


def append_turn(history: list[dict[str, str]], role: str, content: str) -> dict[str, str]:
    """Redact and append one turn to the running history, returning what was stored."""
    turn = {"role": role, "content": redact_pii(content)}
    history.append(turn)
    return turn


def generate_reply(
    model,
    tokenizer,
    history: list[dict[str, str]],
    text: str,
    context: dict[str, Any],
    *,
    max_history_turns: int = 14,
    max_new_tokens: int = 180,
) -> str:
    """Generate the next Listener reply given history, the current turn, and risk context."""
    messages = build_messages(history, text, context, max_history_turns)
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
    ).to(model.device)
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = output[0, inputs["input_ids"].shape[1] :]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()
