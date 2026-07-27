"""QLoRA SFT for risk-conditioned, multi-turn Listener response generation."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = PROJECT_ROOT / "data/processed/cradle_response"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/llama-3.2-1b-cradle-response-qlora"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-eval-samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument("--smoke-test", action="store_true", help="Use small splits and train two steps")
    return parser.parse_args()


def _limit_split(split, limit: int):
    return split.select(range(min(limit, len(split)))) if limit else split


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU not found. Run QLoRA training on Colab or another CUDA machine.")

    files = {name: str(args.data_dir / f"{name}.jsonl") for name in ("train", "validation")}
    missing = [path for path in files.values() if not Path(path).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing processed data: {missing}. Run python -m src.cradle_response.preprocess first."
        )
    dataset = load_dataset("json", data_files=files)
    train_limit = 64 if args.smoke_test else args.max_train_samples
    eval_limit = 16 if args.smoke_test else args.max_eval_samples
    train_data = _limit_split(dataset["train"], train_limit)
    eval_data = _limit_split(dataset["validation"], eval_limit)

    use_4bit = not args.no_4bit
    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    quantization_config = None
    if use_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        quantization_config=quantization_config,
        dtype=compute_dtype,
        device_map="auto",
    )
    model.config.use_cache = False
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )
    training_args = SFTConfig(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        max_steps=2 if args.smoke_test else -1,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        max_length=args.max_length,
        truncation_mode="keep_end",
        completion_only_loss=True,
        gradient_checkpointing=True,
        bf16=compute_dtype == torch.bfloat16,
        fp16=compute_dtype == torch.float16,
        optim="paged_adamw_8bit" if use_4bit else "adamw_torch_fused",
        report_to="none",
        seed=args.seed,
    )

    print(
        "Training on CRADLE Listener turns. These are synthetic GPT-5 responses, "
        "not clinician-authored ground truth."
    )
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=eval_data,
        processing_class=tokenizer,
        peft_config=lora_config,
    )
    trainer.train()
    final_adapter = args.output_dir / "final_adapter"
    trainer.save_model(str(final_adapter))
    tokenizer.save_pretrained(str(final_adapter))
    print(f"Saved LoRA response adapter to {final_adapter}")


if __name__ == "__main__":
    main()
