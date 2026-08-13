"""Glue between the Layer 1 crisis classifier and the Layer 3 Llama response model.

Given raw user text plus the running conversation state, this module runs the
classifier, folds its prediction into the risk-context schema the response
model was trained on, and generates the next Listener reply.
"""

from __future__ import annotations

import os
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
from peft import AutoPeftModelForCausalLM, PeftModel
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer

from src.classifier.sliding_window import DEFAULT_MAX_LENGTH, DEFAULT_STRIDE, predict_long_text
from src.cradle_response.inference import build_risk_context
from src.cradle_response.preprocess import format_system_prompt, redact_pii

# Deliberately has no crisis/Listener framing and no risk-context JSON. The
# fine-tuned SYSTEM_PROMPT primes even a disabled-adapter base model into a
# probing, therapeutic register regardless of the model weights behind it --
# disabling the adapter alone did not fix "hi" getting a therapy-toned reply,
# because the prompt itself was still the crisis-Listener one.
CASUAL_SYSTEM_PROMPT = (
    "You are a friendly conversational assistant having ordinary small talk. "
    "Reply the way a person would: briefly and naturally. Do not ask about "
    "feelings, do not use therapy language, and do not mention support "
    "resources unless the user brings up something serious themselves."
)

# Deliberately not the Listener persona -- summarizing under the adapter would
# produce another in-character reply instead of a neutral recap, so
# summarize_turns always runs with the adapter disabled regardless of context.
# Asks for exactly one sentence about only the turns given (not "merge with
# what came before") -- update_summary appends each sentence in Python, so
# retention of earlier content never depends on the model re-stating it.
SUMMARY_SYSTEM_PROMPT = (
    "Summarize the conversation turns below in exactly one concise sentence. "
    "State only concrete facts about the user's situation from these turns. "
    "Do not add commentary, advice, or opinions, and do not omit anything "
    "related to safety, risk, or harm."
)

# Shared by generate_reply's history window and update_summary's summarization
# trigger so the two stay in lockstep: a turn is either still visible in raw
# form, or already folded into the summary -- never both, never neither.
#
# Token-budgeted rather than turn-count-capped: a fixed turn count treats a
# handful of one-line turns the same as a handful of paragraph-long ones, so
# verbose conversations could still blow past what the adapter was actually
# fine-tuned on (max_length=2048 during training) well before hitting a turn
# limit. 1200 tokens leaves headroom in that 2048 budget for the system
# prompt (crisis framing + risk-context JSON + summary), the current user
# turn, and generation, while still holding a meaningful amount of history.
DEFAULT_MAX_HISTORY_TOKENS = 1200

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


def build_classifier_input(
    history: list[dict[str, str]], current_text: str, max_context_turns: int | None = None
) -> str:
    """Concatenate every prior user turn with the current message for classification.

    CRADLEBench trains the classifier on single first-person posts, not
    role-tagged dialogue, so only prior *user* turns are folded in (no
    Listener turns, no role prefixes) -- plain concatenated prose stays closer
    to the training distribution than a chat-formatted transcript would.
    This is the classifier's own documented "mechanically supported but
    unvalidated" multi-turn path (see README), so treat flags produced from
    concatenated context as a useful signal to sanity-check, not a fully
    calibrated probability the way single-post scores are.

    No window/cap by default: a disclosure earlier in the conversation
    shouldn't scroll out of view just because later turns followed it --
    that's exactly the bug an earlier, capped version of this function had
    (a real crisis disclosure fell out of a 3-turn window a few turns later
    and the badge went back to "clear" mid-conversation). known_events
    already never expires for the session, so a bounded classifier window
    was inconsistent with that. The sliding-window classifier handles
    arbitrarily long concatenated input mechanically; pass max_context_turns
    if a pathologically long session ever makes that too slow in practice.
    """
    user_turns = [turn["content"] for turn in history if turn["role"] == "user"]
    if max_context_turns is not None:
        user_turns = user_turns[-max_context_turns:]
    return " ".join([*user_turns, current_text])


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


def visible_window(
    tokenizer, history: list[dict[str, str]], max_tokens: int
) -> tuple[list[dict[str, str]], int]:
    """Return the newest suffix of history that fits in max_tokens, and how many turns it drops.

    Walks backward from the most recent turn accumulating token counts (not
    turn counts), so a handful of long messages don't get to keep the same
    window a handful of short ones would. Always keeps at least the single
    most recent turn, even if it alone exceeds max_tokens.
    """
    kept: list[dict[str, str]] = []
    total = 0
    cutoff = len(history)
    for turn in reversed(history):
        turn_tokens = len(tokenizer.encode(turn["content"]))
        if kept and total + turn_tokens > max_tokens:
            break
        kept.append(turn)
        total += turn_tokens
        cutoff -= 1
    kept.reverse()
    # Don't start a prompt on an assistant turn -- drop it and count it as
    # gone (available to be folded into the summary) rather than visible.
    while kept and kept[0]["role"] == "assistant":
        kept.pop(0)
        cutoff += 1
    return kept, cutoff


def append_turn(history: list[dict[str, str]], role: str, content: str) -> dict[str, str]:
    """Redact and append one turn to the running history, returning what was stored."""
    turn = {"role": role, "content": redact_pii(content)}
    history.append(turn)
    return turn


def summarize_turns(model, tokenizer, turns: list[dict[str, str]], *, max_new_tokens: int = 60) -> str:
    """Condense one batch of turns into a single sentence, with the adapter disabled.

    Deliberately does not ask the model to also re-incorporate the existing
    summary: a small model asked to "merge old summary + new turns" shows
    strong recency bias and tends to just respond to the newest content,
    silently dropping the old summary instead of preserving it. Keeping this
    function's job to "condense only what's new" and letting update_summary
    concatenate in Python instead guarantees retention isn't the model's job.
    """
    transcript = "\n".join(f"{turn['role']}: {turn['content']}" for turn in turns)
    messages = [
        {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
        {"role": "user", "content": transcript},
    ]
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
    ).to(model.device)
    disable = model.disable_adapter() if hasattr(model, "disable_adapter") else nullcontext()
    with disable, torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = output[0, inputs["input_ids"].shape[1] :]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def update_summary(
    model,
    tokenizer,
    summary: str,
    history: list[dict[str, str]],
    summarized_through: int,
    *,
    keep_recent_tokens: int = DEFAULT_MAX_HISTORY_TOKENS,
) -> tuple[str, int]:
    """Fold turns about to scroll out of generate_reply's visible window into a running summary.

    Only turns not already covered by summarized_through are folded in, so a
    long conversation never re-summarizes the same turns on every reply. Uses
    the same token-budgeted visible_window as generate_reply so the two never
    disagree about where the visible window ends.

    This is purely a fallback for generic content once raw history is trimmed
    -- risk history is never at stake here. known_events is tracked
    separately in build_context/update_known_events and is always re-injected
    into the risk-context JSON in full, regardless of what this summary says
    or whether it's stale. Summarization must never become the mechanism that
    preserves a crisis disclosure; that job stays with known_events.
    """
    _, cutoff = visible_window(tokenizer, history, keep_recent_tokens)
    new_turns = history[summarized_through:cutoff]
    if not new_turns:
        return summary, summarized_through
    new_sentence = summarize_turns(model, tokenizer, new_turns)
    updated_summary = f"{summary} {new_sentence}".strip() if summary else new_sentence
    return updated_summary, cutoff


def has_no_risk_history(context: dict[str, Any]) -> bool:
    """True when neither this turn nor any prior turn in the conversation carries a risk signal.

    When true, generate_reply switches to CASUAL_SYSTEM_PROMPT and disables
    the adapter, since both the crisis-Listener system prompt and the
    CRADLE-response fine-tuning bias replies toward a therapeutic register
    even for plain small talk.
    """
    return context.get("current_signals") in (None, ["none"]) and not context.get("known_events")


def generate_reply(
    model,
    tokenizer,
    history: list[dict[str, str]],
    text: str,
    context: dict[str, Any],
    *,
    summary: str = "",
    max_history_tokens: int = DEFAULT_MAX_HISTORY_TOKENS,
    max_new_tokens: int = 180,
) -> str:
    """Generate the next Listener reply given history, the current turn, and risk context.

    summary covers turns already trimmed out of the visible window (see
    update_summary) -- pass "" for short conversations that don't need it.
    """
    no_risk = has_no_risk_history(context)
    visible, _ = visible_window(tokenizer, history, max_history_tokens)
    system_prompt = CASUAL_SYSTEM_PROMPT if no_risk else format_system_prompt(context)
    messages = [
        {"role": "system", "content": system_prompt},
        *visible,
        {"role": "user", "content": redact_pii(text)},
    ]

    if summary:
        messages[0] = {
            "role": "system",
            "content": messages[0]["content"] + "\n\nSummary of earlier conversation:\n" + summary,
        }

    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
    ).to(model.device)

    def run() -> str:
        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated = output[0, inputs["input_ids"].shape[1] :]
        return tokenizer.decode(generated, skip_special_tokens=True).strip()

    if no_risk and hasattr(model, "disable_adapter"):
        with model.disable_adapter():
            return run()
    return run()
