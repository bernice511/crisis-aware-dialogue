"""Compare base and response-adapted Llama on held-out completion likelihood."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from datasets import load_dataset
from peft import AutoPeftModelForCausalLM
from torch.nn.utils.rnn import pad_sequence
from tqdm.auto import tqdm
from transformers import AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TEST = PROJECT_ROOT / "data/processed/cradle_response/test.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--test-file", type=Path, default=DEFAULT_TEST)
    parser.add_argument("--output", type=Path, default=Path("test_comparison_metrics.json"))
    parser.add_argument("--samples", type=int, default=0, help="0 evaluates the complete test set")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def encode_example(tokenizer, example, max_length: int) -> dict[str, list[int]]:
    def ids(messages, add_generation_prompt: bool) -> list[int]:
        encoded = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
            return_tensors="pt",
            return_dict=True,
        )
        return encoded["input_ids"][0].tolist()

    prompt_ids = ids(example["prompt"], True)
    full_ids = ids(example["prompt"] + example["completion"], False)
    common = 0
    for prompt_token, full_token in zip(prompt_ids, full_ids):
        if prompt_token != full_token:
            break
        common += 1
    if common < len(prompt_ids) - 8:
        raise ValueError(
            f"Large chat-template mismatch: prompt={len(prompt_ids)}, common={common}"
        )
    labels = [-100] * common + full_ids[common:]
    return {"input_ids": full_ids[-max_length:], "labels": labels[-max_length:]}


def evaluate(model, tokenizer, rows, batch_size: int, name: str) -> dict[str, float | int | str]:
    model.eval()
    total_nll = total_tokens = total_correct = 0
    for start in tqdm(range(0, len(rows), batch_size), desc=name):
        batch = rows[start : start + batch_size]
        ids = pad_sequence(
            [torch.tensor(row["input_ids"]) for row in batch],
            batch_first=True,
            padding_value=tokenizer.pad_token_id,
        ).to(model.device)
        labels = pad_sequence(
            [torch.tensor(row["labels"]) for row in batch],
            batch_first=True,
            padding_value=-100,
        ).to(model.device)
        attention = pad_sequence(
            [torch.ones(len(row["input_ids"]), dtype=torch.long) for row in batch],
            batch_first=True,
            padding_value=0,
        ).to(model.device)
        with torch.inference_mode():
            output = model(input_ids=ids, attention_mask=attention, labels=labels)
        shifted = labels[:, 1:]
        mask = shifted.ne(-100)
        token_count = int(mask.sum())
        predictions = output.logits[:, :-1].argmax(-1)
        total_correct += int(((predictions == shifted) & mask).sum())
        total_nll += float(output.loss) * token_count
        total_tokens += token_count
    loss = total_nll / total_tokens
    return {
        "model": name,
        "examples": len(rows),
        "completion_tokens": total_tokens,
        "loss": loss,
        "perplexity": math.exp(loss),
        "token_accuracy": total_correct / total_tokens,
    }


def main() -> None:
    args = parse_args()
    dataset = load_dataset("json", data_files={"test": str(args.test_file)})["test"]
    if args.samples:
        dataset = dataset.shuffle(seed=args.seed).select(range(min(args.samples, len(dataset))))

    tokenizer = AutoTokenizer.from_pretrained(args.adapter)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.float16 if device == "mps" else torch.float32

    print(f"Loading base Llama and fine-tuned adapter on {device}...")

    model = AutoPeftModelForCausalLM.from_pretrained(
        args.adapter,
        dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model.to(device)
    encoded = [encode_example(tokenizer, row, args.max_length) for row in dataset]
    with model.disable_adapter():
        base = evaluate(model, tokenizer, encoded, args.batch_size, "base_llama")
    tuned = evaluate(model, tokenizer, encoded, args.batch_size, "fine_tuned_llama")
    comparison = {
        "dataset": "CRADLE-Dialogue held-out test",
        "base": base,
        "fine_tuned": tuned,
        "loss_reduction_percent": (base["loss"] - tuned["loss"]) / base["loss"] * 100,
        "perplexity_reduction_percent":
            (base["perplexity"] - tuned["perplexity"]) / base["perplexity"] * 100,
        "token_accuracy_improvement": tuned["token_accuracy"] - base["token_accuracy"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
