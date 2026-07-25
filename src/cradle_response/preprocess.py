"""Build multi-turn next-response SFT data from CRADLE-Dialogue.

Each Listener turn becomes a target completion.  Its prompt contains the
preceding conversation and a normalized risk context.  During training the
risk context is derived from CRADLE gold labels; at deployment the same schema
can be populated by the upstream classifier.

Usage:
    python -m src.cradle_response.preprocess
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = PROJECT_ROOT / "data/processed/cradle_response"
DATASET_ID = "SungJoo/Cradle-Dialogue"

SYSTEM_PROMPT = (
    "You are a supportive, crisis-aware Listener in a mental-health conversation. "
    "Use the complete conversation history and the upstream risk context when "
    "writing the next response. Acknowledge the user's feelings, stay relevant, "
    "avoid diagnosis and judgment, do not provide harmful instructions, and ask "
    "clear safety questions or encourage appropriate real-world support when the "
    "risk context indicates that it is needed. Respond naturally and concisely."
)

EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)
URL_RE = re.compile(r"https?://\S+|www\.\S+", re.I)
PHONE_RE = re.compile(r"(?<!\w)(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)\d{3}[-.\s]?\d{4}(?!\w)")
REDDIT_USER_RE = re.compile(r"(?<!\w)(?:u/|/u/)[A-Za-z0-9_-]+", re.I)


def redact_pii(text: str) -> str:
    """Redact common contact identifiers before they enter an SFT example."""
    text = EMAIL_RE.sub("[EMAIL]", str(text))
    text = URL_RE.sub("[URL]", text)
    text = PHONE_RE.sub("[PHONE]", text)
    text = REDDIT_USER_RE.sub("[USER]", text)
    return " ".join(text.split()).strip()


def parse_turn(text: str) -> dict[str, str]:
    """Convert a prefixed CRADLE turn into a chat-template message."""
    cleaned = redact_pii(text)
    if cleaned.startswith("User:"):
        role, content = "user", cleaned[len("User:") :].strip()
    elif cleaned.startswith("Listener:"):
        role, content = "assistant", cleaned[len("Listener:") :].strip()
    else:
        raise ValueError(f"Turn does not start with User: or Listener:: {cleaned[:80]!r}")
    if not content:
        raise ValueError("Turn content is empty")
    return {"role": role, "content": content}


def collapse_continuations(dialogue: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge unprefixed bullet/continuation rows into the preceding message."""
    collapsed: list[dict[str, Any]] = []
    for row in dialogue:
        text = str(row["text"]).strip()
        if text.startswith(("User:", "Listener:")):
            collapsed.append(dict(row))
            continue
        if not collapsed:
            raise ValueError("Dialogue begins with an unprefixed continuation row")
        collapsed[-1]["text"] = str(collapsed[-1]["text"]).rstrip() + "\n" + text
        extra_labels = parse_labels(row.get("labels"))
        if extra_labels:
            existing = parse_labels(collapsed[-1].get("labels"))
            collapsed[-1]["labels"] = "; ".join(dict.fromkeys(existing + extra_labels))
    return collapsed


def parse_labels(value: Any) -> list[str]:
    """Return sorted non-empty labels; CRADLE uses '; ' for multi-label turns."""
    if value is None:
        return []
    raw = str(value).strip()
    if not raw or raw.lower() == "none":
        return []
    return sorted({label.strip() for label in raw.split(";") if label.strip()})


def risk_context(current_signals: list[str], known_events: list[str]) -> dict[str, Any]:
    """Stable interface shared with the future classifier integration."""
    return {
        "current_signals": current_signals or ["none"],
        "known_events": known_events,
        "source": "gold_cradle_labels",
    }


def format_system_prompt(context: dict[str, Any]) -> str:
    payload = json.dumps(context, ensure_ascii=False, sort_keys=True)
    return f"{SYSTEM_PROMPT}\n\nUpstream risk context:\n{payload}"


def ensure_dialogue_ids(
    rows: Iterable[dict[str, Any]], *, split_name: str = "unknown"
) -> list[dict[str, Any]]:
    """Restore dialogue IDs when the Hub parquet omits its documented column.

    CRADLE rows are stored in dialogue order and every new dialogue resets
    ``turn_id`` to zero.  Preserve supplied IDs when present; otherwise create
    deterministic split-local IDs from those boundaries.
    """
    materialized = [dict(row) for row in rows]
    if not materialized:
        return materialized
    has_ids = ["dialogue_id" in row and row["dialogue_id"] not in (None, "") for row in materialized]
    if all(has_ids):
        return materialized
    if any(has_ids):
        raise ValueError("dialogue_id is present for only some rows")

    dialogue_index = -1
    for index, row in enumerate(materialized):
        if "turn_id" not in row:
            raise ValueError(f"Row {index} is missing required field: turn_id")
        turn_id = int(row["turn_id"])
        if turn_id == 0:
            dialogue_index += 1
        elif dialogue_index < 0:
            raise ValueError("Cannot reconstruct dialogue IDs: first turn_id is not zero")
        row["dialogue_id"] = f"{split_name}-{dialogue_index:06d}"
    return materialized


def trim_history(history: list[dict[str, str]], max_turns: int | None) -> list[dict[str, str]]:
    """Keep recent context without starting a prompt on an assistant turn."""
    visible = history[-max_turns:] if max_turns else history[:]
    while visible and visible[0]["role"] == "assistant":
        visible = visible[1:]
    return visible


def _ordered_dialogues(rows: Iterable[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for index, row in enumerate(rows):
        missing = [key for key in ("dialogue_id", "turn_id", "text", "labels") if key not in row]
        if missing:
            raise ValueError(f"Row {index} is missing required fields: {missing}")
        grouped[str(row["dialogue_id"])].append(dict(row))

    dialogues: list[list[dict[str, Any]]] = []
    for dialogue_id in sorted(grouped):
        ordered = sorted(grouped[dialogue_id], key=lambda item: int(item["turn_id"]))
        turn_ids = [int(item["turn_id"]) for item in ordered]
        if len(turn_ids) != len(set(turn_ids)):
            raise ValueError(f"Duplicate turn_id in dialogue {dialogue_id}")
        dialogues.append(ordered)
    return dialogues


def build_response_examples(
    rows: Iterable[dict[str, Any]], *, max_history_turns: int | None = 14
) -> list[dict[str, Any]]:
    """Create one prompt-completion example for every Listener turn."""
    if max_history_turns is not None and max_history_turns < 1:
        raise ValueError("max_history_turns must be positive or None")

    examples: list[dict[str, Any]] = []
    for raw_dialogue in _ordered_dialogues(rows):
        dialogue = collapse_continuations(raw_dialogue)
        history: list[dict[str, str]] = []
        known_events: list[str] = []
        latest_user_signals: list[str] = []
        dialogue_id = str(dialogue[0]["dialogue_id"])

        for row in dialogue:
            turn = parse_turn(row["text"])
            labels = parse_labels(row["labels"])

            if turn["role"] == "user":
                latest_user_signals = labels
                for label in labels:
                    if label not in known_events:
                        known_events.append(label)
                history.append(turn)
                continue

            if not history:
                # Some held-out dialogues begin with a Listener greeting. It has
                # no preceding User input, so it cannot be a next-response target.
                continue
            if history[-1]["role"] != "user":
                raise ValueError(
                    f"Listener turn {row['turn_id']} in dialogue {dialogue_id} "
                    "does not follow a User turn"
                )

            visible_history = trim_history(history, max_history_turns)
            context = risk_context(latest_user_signals, known_events[:])
            examples.append(
                {
                    "prompt": [
                        {"role": "system", "content": format_system_prompt(context)},
                        *visible_history,
                    ],
                    "completion": [{"role": "assistant", "content": turn["content"]}],
                    "metadata": {
                        "dialogue_id": dialogue_id,
                        "target_turn_id": int(row["turn_id"]),
                        "risk_context": context,
                    },
                }
            )
            history.append(turn)

    if not examples:
        raise ValueError("No Listener response examples were produced")
    return examples


def write_jsonl(rows: Iterable[dict[str, Any]], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def preprocess_dataset(output_dir: Path, *, max_history_turns: int | None = 14) -> dict[str, int]:
    from datasets import load_dataset

    dataset = load_dataset(DATASET_ID)
    counts: dict[str, int] = {}
    dialogue_counts: dict[str, int] = {}
    for split_name in ("train", "validation", "test"):
        rows = ensure_dialogue_ids(dataset[split_name], split_name=split_name)
        dialogue_counts[split_name] = len({str(row["dialogue_id"]) for row in rows})
        examples = build_response_examples(rows, max_history_turns=max_history_turns)
        counts[split_name] = write_jsonl(examples, output_dir / f"{split_name}.jsonl")

    manifest = {
        "dataset": DATASET_ID,
        "task": "risk_conditioned_multi_turn_response_generation",
        "counts": counts,
        "dialogue_counts": dialogue_counts,
        "max_history_turns": max_history_turns,
        "risk_interface": {
            "current_signals": "labels emitted for the latest User turn, or ['none']",
            "known_events": "unique crisis events detected earlier in this dialogue",
            "source": "gold_cradle_labels during training; classifier predictions in deployment",
        },
        "response_provenance": "GPT-5-generated Listener turns; not clinician-authored gold responses",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--max-history-turns",
        type=int,
        default=14,
        help="Keep the most recent N turns; use 0 to retain the full dialogue",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    max_history_turns = args.max_history_turns or None
    counts = preprocess_dataset(args.output_dir, max_history_turns=max_history_turns)
    print(f"Wrote CRADLE response SFT data to {args.output_dir}: {counts}")


if __name__ == "__main__":
    main()
