"""Generate responses from the base and fine-tuned Llama models.

This script evaluates actual response generation rather than only
completion loss, perplexity, and token accuracy.

For every selected held-out test prompt:
1. Generate a response from the original base Llama with the adapter disabled.
2. Generate a response from the fine-tuned Llama with the adapter enabled.
3. Save both responses alongside the reference response and risk context.

This script performs inference only. It does not train either model.

Example:
    python3 -m src.cradle_response.cradle_response_eval \
        --adapter outputs/llama-3.2-1b-cradle-response-qlora-full/final_adapter \
        --samples 200
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from datasets import load_dataset
from peft import AutoPeftModelForCausalLM
from tqdm.auto import tqdm
from transformers import AutoTokenizer, BitsAndBytesConfig


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_TEST_FILE = (
    PROJECT_ROOT / "data/processed/cradle_response/test.jsonl"
)

DEFAULT_OUTPUT_FILE = (
    PROJECT_ROOT
    / "results/cradle_response_detailed/generated_comparisons.jsonl"
)


def parse_args() -> argparse.Namespace:
    """Read command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--adapter",
        type=Path,
        required=True,
        help="Path to Alice's trained final_adapter directory.",
    )

    parser.add_argument(
        "--test-file",
        type=Path,
        default=DEFAULT_TEST_FILE,
        help="Held-out CRADLE response test JSONL file.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_FILE,
        help="Where generated comparison rows will be saved.",
    )

    parser.add_argument(
        "--samples",
        type=int,
        default=200,
        help="Number of held-out examples to evaluate. Use 0 for all examples.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used when selecting test examples.",
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=180,
        help="Maximum number of tokens generated for each response.",
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue from an existing partially completed output file.",
    )

    return parser.parse_args()


def get_device() -> str:
    """Select CUDA, Apple MPS, or CPU."""
    if torch.cuda.is_available():
        return "cuda"

    if (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        return "mps"

    return "cpu"


def load_model_and_tokenizer(
    adapter_path: Path,
    device: str,
):
    """Load the base Llama model together with the trained LoRA adapter."""
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    if device == "cuda":
        dtype = (
            torch.bfloat16
            if torch.cuda.is_bf16_supported()
            else torch.float16
        )

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
        )

        model = AutoPeftModelForCausalLM.from_pretrained(
            adapter_path,
            quantization_config=quantization_config,
            torch_dtype=dtype,
            device_map="auto",
        )

    else:
        dtype = torch.float16 if device == "mps" else torch.float32

        model = AutoPeftModelForCausalLM.from_pretrained(
            adapter_path,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )

        model.to(device)

    model.eval()

    return model, tokenizer


def generate_response(
    model,
    tokenizer,
    messages: list[dict[str, str]],
    max_new_tokens: int,
) -> str:
    """Generate one deterministic model response."""
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )

    inputs = {
        name: tensor.to(model.device)
        for name, tensor in inputs.items()
    }

    prompt_length = inputs["input_ids"].shape[1]

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated_tokens = output[0, prompt_length:]

    return tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    ).strip()


def get_last_user_message(
    messages: list[dict[str, str]],
) -> str:
    """Return the latest user message from a prompt."""
    for message in reversed(messages):
        if message["role"] == "user":
            return message["content"]

    return ""


def load_existing_rows(
    output_path: Path,
) -> tuple[list[dict[str, Any]], set[int]]:
    """Load completed rows when resuming an interrupted run."""
    rows: list[dict[str, Any]] = []
    completed_ids: set[int] = set()

    if not output_path.exists():
        return rows, completed_ids

    with output_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue

            row = json.loads(line)
            rows.append(row)
            completed_ids.add(int(row["source_example_id"]))

    return rows, completed_ids


def save_csv(
    rows: list[dict[str, Any]],
    output_path: Path,
) -> Path:
    """Save a readable CSV copy of the generated comparisons."""
    csv_path = output_path.with_suffix(".csv")

    csv_rows = []

    for row in rows:
        csv_rows.append(
            {
                "source_example_id": row["source_example_id"],
                "dialogue_id": row["dialogue_id"],
                "target_turn_id": row["target_turn_id"],
                "last_user_message": row["last_user_message"],
                "current_signals": "; ".join(
                    row["risk_context"].get(
                        "current_signals",
                        [],
                    )
                ),
                "known_events": "; ".join(
                    row["risk_context"].get(
                        "known_events",
                        [],
                    )
                ),
                "reference_response": row["reference_response"],
                "base_response": row["base_response"],
                "fine_tuned_response": row["fine_tuned_response"],
                "base_generation_seconds": row[
                    "base_generation_seconds"
                ],
                "fine_tuned_generation_seconds": row[
                    "fine_tuned_generation_seconds"
                ],
            }
        )

    pd.DataFrame(csv_rows).to_csv(csv_path, index=False)

    return csv_path


def main() -> None:
    """Run base-versus-fine-tuned response generation."""
    args = parse_args()
    device = get_device()

    print("CRADLE Response Generation Evaluation")
    print("=" * 50)
    print(f"Device: {device}")
    print(f"Adapter: {args.adapter}")
    print(f"Test file: {args.test_file}")
    print(f"Requested samples: {args.samples}")
    print(f"Output: {args.output}")
    print("Training: disabled — this script performs inference only.")

    if not args.adapter.exists():
        raise FileNotFoundError(
            f"Adapter directory was not found: {args.adapter}"
        )

    if not args.test_file.exists():
        raise FileNotFoundError(
            f"Test file was not found: {args.test_file}"
        )

    dataset = load_dataset(
        "json",
        data_files={"test": str(args.test_file)},
    )["test"]

    dataset = dataset.add_column(
        "source_example_id",
        list(range(len(dataset))),
    )

    dataset = dataset.shuffle(seed=args.seed)

    if args.samples > 0:
        sample_count = min(args.samples, len(dataset))
        dataset = dataset.select(range(sample_count))

    print(f"Selected examples: {len(dataset)}")

    print("\nLoading base Llama and fine-tuned adapter...")
    model, tokenizer = load_model_and_tokenizer(
        adapter_path=args.adapter,
        device=device,
    )
    print("Model loaded successfully.")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    existing_rows: list[dict[str, Any]] = []
    completed_ids: set[int] = set()

    if args.resume:
        existing_rows, completed_ids = load_existing_rows(
            args.output
        )
        print(
            f"Resuming with {len(completed_ids)} "
            "already completed examples."
        )
    else:
        args.output.write_text("", encoding="utf-8")

    all_rows = existing_rows[:]

    with args.output.open("a", encoding="utf-8") as handle:
        for example in tqdm(
            dataset,
            desc="Generating base and fine-tuned responses",
        ):
            source_example_id = int(
                example["source_example_id"]
            )

            if source_example_id in completed_ids:
                continue

            prompt_messages = [
                {
                    "role": message["role"],
                    "content": message["content"],
                }
                for message in example["prompt"]
            ]

            # Adapter disabled: original base Llama.
            base_start = time.perf_counter()

            with model.disable_adapter():
                base_response = generate_response(
                    model=model,
                    tokenizer=tokenizer,
                    messages=prompt_messages,
                    max_new_tokens=args.max_new_tokens,
                )

            base_seconds = time.perf_counter() - base_start

            # Adapter enabled: Alice's fine-tuned Llama.
            fine_tuned_start = time.perf_counter()

            fine_tuned_response = generate_response(
                model=model,
                tokenizer=tokenizer,
                messages=prompt_messages,
                max_new_tokens=args.max_new_tokens,
            )

            fine_tuned_seconds = (
                time.perf_counter() - fine_tuned_start
            )

            metadata = dict(example["metadata"])
            risk_context = dict(
                metadata.get("risk_context", {})
            )

            row = {
                "source_example_id": source_example_id,
                "dialogue_id": metadata.get("dialogue_id"),
                "target_turn_id": metadata.get(
                    "target_turn_id"
                ),
                "risk_context": risk_context,
                "prompt_messages": prompt_messages,
                "last_user_message": get_last_user_message(
                    prompt_messages
                ),
                "reference_response": example[
                    "completion"
                ][0]["content"],
                "base_response": base_response,
                "fine_tuned_response": fine_tuned_response,
                "base_generation_seconds": base_seconds,
                "fine_tuned_generation_seconds":
                    fine_tuned_seconds,
            }

            handle.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )
            handle.flush()

            all_rows.append(row)
            completed_ids.add(source_example_id)

    csv_path = save_csv(
        rows=all_rows,
        output_path=args.output,
    )

    print("\nGeneration completed.")
    print(f"Comparison rows: {len(all_rows)}")
    print(f"JSONL saved to: {args.output}")
    print(f"CSV saved to: {csv_path}")


if __name__ == "__main__":
    main()