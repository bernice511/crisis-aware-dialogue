# Crisis-Aware Dialogue

Crisis-Aware Adaptation of Small Language Models for Harmful Dialogue Understanding.

This project investigates model-adaptation strategies that transform a general-purpose
small language model (Llama 3.2) into a **crisis-aware dialogue model** able to recognize
escalation and urgency in harmful conversations, while preserving the base model's language
understanding. Training and evaluation use two public datasets: **CRADLEBench** and **DeepSuiMind**.

## Datasets

| Dataset | Source | Unit | Task | Size |
|---|---|---|---|---|
| **CRADLEBench** | [`SungJoo/Cradle-Bench`](https://huggingface.co/datasets/SungJoo/Cradle-Bench) (arXiv [2510.23845](https://arxiv.org/abs/2510.23845)) | social-media post | multi-label crisis type + temporal (ongoing/past) | 8,259 rows (train 7,239 / dev 420 / test 600) |
| **DeepSuiMind** | [`babytreecc/Implicit-suicide-detection`](https://huggingface.co/datasets/babytreecc/Implicit-suicide-detection) (arXiv [2502.17899](https://arxiv.org/abs/2502.17899)) | synthetic narrative | categorical: scenario / core belief / intention | 1,605 rows |

> The DeepSuiMind link is the HuggingFace repository cited in the paper's footnote. Both datasets
> are public. They contain sensitive mental-health content (self-harm, suicide, abuse) and are used
> here strictly for academic research; the raw data is gitignored and not committed.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate   # optional
pip install -r requirements.txt
python src/download_data.py     # downloads both datasets into data/raw/
```

## EDA

The exploratory analysis lives in [`notebooks/eda.ipynb`](notebooks/eda.ipynb). Run it with:

```bash
jupyter lab notebooks/eda.ipynb
# or headless:
jupyter nbconvert --to notebook --execute --inplace notebooks/eda.ipynb
```

It covers, per dataset: split sizes, missingness/duplicates, label/class distributions,
CRADLEBench crisis-type frequency + temporal (ongoing/past) + multi-label co-occurrence,
DeepSuiMind scenario × core-belief structure, text-length distributions, a cross-dataset
comparison, and a **data-quality note** normalizing inconsistent raw label spellings
(e.g. `No crisis` → `no_crisis`, `suicideideation(passive)` → `suicideideation_passive`).
Figures are saved to `reports/figures/`.

## Llama 3.2 supervised fine-tuning

The DeepSuiMind source has one narrative (`synthetic_text`) and three labels
(`Scenario`, `Negative Core Belief`, and `Intention Category`). It does **not**
contain gold-standard assistant replies. The current SFT task therefore trains
Llama to produce structured crisis-understanding labels, not therapeutic or
crisis-response text.

### 1. Prepare instruction data

```bash
python -m src.llama.preprocess
```

This validates and deduplicates examples, creates an intention-stratified
80/10/10 split, and writes conversational prompt-completion JSONL files under
`data/processed/deepsuimind_sft/`. Each assistant completion is a JSON object:

```json
{
  "intention_category": "Death-Me",
  "negative_core_belief": "Overgeneralization",
  "scenario": "Hopelessness"
}
```

TRL applies the Llama tokenizer's chat template at training time. Keeping the
prompt and completion separate also makes the SFT loss apply only to the target
assistant tokens.

### 2. Fine-tune with QLoRA

Training requires a CUDA GPU. The default model is gated: request access to
[`meta-llama/Llama-3.2-1B-Instruct`](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct),
accept the license, and authenticate before running the command.

First verify the full pipeline with two training steps:

```bash
python -m src.llama.train_sft --smoke-test
```

Then run the default three-epoch experiment:

```bash
python -m src.llama.train_sft
```

The default configuration uses 4-bit NF4 quantization, LoRA on all linear
layers, gradient checkpointing, completion-only loss, and saves the final
adapter under `outputs/llama-3.2-1b-deepsuimind-qlora/final_adapter/`.

### 3. Inference

```bash
python -m src.llama.inference \
  --adapter outputs/llama-3.2-1b-deepsuimind-qlora/final_adapter \
  --text "I feel trapped and cannot imagine things getting better."
```

### Tests

The preprocessing and formatting stages run without downloading a model:

```bash
python -m unittest discover -s tests -v
```

### Important scope limitation

To fine-tune the final chatbot's *reply behavior*, the project still needs a
reviewed dataset containing user narratives paired with safe target responses
(for example, a `response` field). Synthetic or model-generated responses
should not be treated as clinical ground truth. The current structured output
can instead be passed into the later prompt-engineering/response layer.

## Layout

```
data/raw/            # downloaded datasets (gitignored)
notebooks/eda.ipynb  # exploratory data analysis
src/download_data.py # dataset downloader
src/llama/            # preprocessing, QLoRA SFT, and inference
tests/                # pipeline unit tests
reports/figures/     # generated EDA figures
```
