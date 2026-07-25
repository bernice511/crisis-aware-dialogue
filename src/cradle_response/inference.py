"""Generate the next Listener reply from history and classifier risk context."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .preprocess import format_system_prompt, redact_pii, trim_history


def build_risk_context(
    current_signals: list[str], known_events: list[str], *, source: str = "classifier_prediction"
) -> dict[str, Any]:
    return {
        "current_signals": current_signals or ["none"],
        "known_events": list(dict.fromkeys(known_events)),
        "source": source,
    }


def load_history(path: Path | None) -> list[dict[str, str]]:
    if path is None:
        return []
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("History file must contain a JSON list")
    history: list[dict[str, str]] = []
    for item in raw:
        if not isinstance(item, dict) or item.get("role") not in {"user", "assistant"}:
            raise ValueError("Every history item must have role user/assistant and content")
        history.append({"role": item["role"], "content": redact_pii(item.get("content", ""))})
    return history


def build_messages(
    history: list[dict[str, str]], current_text: str, context: dict[str, Any], max_history_turns: int
) -> list[dict[str, str]]:
    visible = trim_history(history, max_history_turns or None)
    return [
        {"role": "system", "content": format_system_prompt(context)},
        *visible,
        {"role": "user", "content": redact_pii(current_text)},
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--text", required=True, help="Current User turn")
    parser.add_argument("--history-file", type=Path, help="JSON list of earlier user/assistant messages")
    parser.add_argument("--current-signal", action="append", default=[])
    parser.add_argument("--known-event", action="append", default=[])
    parser.add_argument("--max-history-turns", type=int, default=14)
    parser.add_argument("--max-new-tokens", type=int, default=180)
    return parser.parse_args()


def main() -> None:
    import torch
    from peft import AutoPeftModelForCausalLM
    from transformers import AutoTokenizer

    args = parse_args()
    context = build_risk_context(args.current_signal, args.known_event)
    messages = build_messages(
        load_history(args.history_file), args.text, context, args.max_history_turns
    )
    tokenizer = AutoTokenizer.from_pretrained(args.adapter)
    model = AutoPeftModelForCausalLM.from_pretrained(
        args.adapter,
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
    ).to(model.device)
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = output[0, inputs["input_ids"].shape[1] :]
    print(tokenizer.decode(generated, skip_special_tokens=True).strip())


if __name__ == "__main__":
    main()
