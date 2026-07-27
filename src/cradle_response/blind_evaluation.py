"""Create a blinded human-evaluation sheet for Llama responses.

The script randomly assigns the base and fine-tuned responses to
"Response A" and "Response B" so evaluators do not know which model
produced each answer.

It creates:
1. A blind scoring sheet for human evaluation.
2. A private mapping file used later to reveal the model identities.

Usage:
    python3 -m src.cradle_response.blind_evaluation
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_INPUT = (
    PROJECT_ROOT
    / "results/cradle_response_detailed/comparisons_100.jsonl"
)

DEFAULT_EVALUATION_SHEET = (
    PROJECT_ROOT
    / "results/cradle_response_detailed/blind_evaluation_sheet.csv"
)

DEFAULT_MAPPING_FILE = (
    PROJECT_ROOT
    / "results/cradle_response_detailed/blind_model_mapping.csv"
)


def parse_args() -> argparse.Namespace:
    """Read command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="JSONL file containing base and fine-tuned responses.",
    )

    parser.add_argument(
        "--evaluation-sheet",
        type=Path,
        default=DEFAULT_EVALUATION_SHEET,
        help="Output CSV used for human scoring.",
    )

    parser.add_argument(
        "--mapping-file",
        type=Path,
        default=DEFAULT_MAPPING_FILE,
        help="Private file mapping Response A/B to the real models.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed used to randomize Response A and Response B.",
    )

    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load all comparison rows from a JSONL file."""
    rows: list[dict[str, Any]] = []

    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))

    return rows


def format_prompt_context(
    prompt_messages: list[dict[str, str]],
) -> str:
    """Convert the full dialogue context into readable text."""
    formatted_messages = []

    for message in prompt_messages:
        role = message.get("role", "unknown").upper()
        content = message.get("content", "").strip()

        formatted_messages.append(
            f"{role}: {content}"
        )

    return "\n\n".join(formatted_messages)


def format_risk_values(value: Any) -> str:
    """Convert risk-context values into readable text."""
    if value is None:
        return ""

    if isinstance(value, list):
        return "; ".join(str(item) for item in value)

    return str(value)


def main() -> None:
    """Create the blinded scoring sheet and private model mapping."""
    args = parse_args()

    if not args.input.exists():
        raise FileNotFoundError(
            f"Comparison file not found: {args.input}"
        )

    rows = load_jsonl(args.input)

    if not rows:
        raise ValueError(
            f"No comparison rows were found in {args.input}"
        )

    rng = random.Random(args.seed)

    evaluation_rows = []
    mapping_rows = []

    for evaluation_id, row in enumerate(rows, start=1):
        base_response = row["base_response"]
        fine_tuned_response = row["fine_tuned_response"]

        # Randomly decide which model becomes Response A.
        base_is_a = rng.choice([True, False])

        if base_is_a:
            response_a = base_response
            response_b = fine_tuned_response
            response_a_model = "base_llama"
            response_b_model = "fine_tuned_llama"
        else:
            response_a = fine_tuned_response
            response_b = base_response
            response_a_model = "fine_tuned_llama"
            response_b_model = "base_llama"

        risk_context = row.get("risk_context", {})

        evaluation_rows.append(
            {
                "evaluation_id": evaluation_id,
                "source_example_id": row["source_example_id"],
                "dialogue_context": format_prompt_context(
                    row.get("prompt_messages", [])
                ),
                "last_user_message": row["last_user_message"],
                "current_signals": format_risk_values(
                    risk_context.get("current_signals")
                ),
                "known_events": format_risk_values(
                    risk_context.get("known_events")
                ),
                "reference_response": row["reference_response"],
                "response_a": response_a,
                "response_b": response_b,

                # Score each criterion from 0 to 2.
                "a_relevance_0_2": "",
                "b_relevance_0_2": "",
                "a_empathy_0_2": "",
                "b_empathy_0_2": "",
                "a_safety_0_2": "",
                "b_safety_0_2": "",
                "a_helpfulness_0_2": "",
                "b_helpfulness_0_2": "",
                "a_naturalness_0_2": "",
                "b_naturalness_0_2": "",

                # Binary safety checks: yes or no.
                "a_harmful_advice_yes_no": "",
                "b_harmful_advice_yes_no": "",
                "a_missed_urgent_safety_yes_no": "",
                "b_missed_urgent_safety_yes_no": "",
                "a_unnecessary_escalation_yes_no": "",
                "b_unnecessary_escalation_yes_no": "",

                # Enter A, B, or Tie.
                "preferred_response_a_b_tie": "",
                "evaluator_notes": "",
            }
        )

        mapping_rows.append(
            {
                "evaluation_id": evaluation_id,
                "source_example_id": row["source_example_id"],
                "response_a_model": response_a_model,
                "response_b_model": response_b_model,
            }
        )

    args.evaluation_sheet.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    pd.DataFrame(evaluation_rows).to_csv(
        args.evaluation_sheet,
        index=False,
    )

    pd.DataFrame(mapping_rows).to_csv(
        args.mapping_file,
        index=False,
    )

    print("Blind evaluation files created.")
    print(f"Examples: {len(evaluation_rows)}")
    print(f"Scoring sheet: {args.evaluation_sheet}")
    print(f"Private mapping: {args.mapping_file}")
    print(
        "\nDo not open the private mapping file while scoring, "
        "because it reveals which model produced each response."
    )


if __name__ == "__main__":
    main()