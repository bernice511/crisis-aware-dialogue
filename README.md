# Crisis-Aware Dialogue

Crisis-Aware Adaptation of Small Language Models for Harmful Dialogue Understanding.

This project investigates model-adaptation strategies that transform a general-purpose
small language model (Llama 3.2) into a **crisis-aware dialogue model** that uses
multi-turn context and upstream crisis signals to generate an appropriate Listener reply.
The active pipeline uses **CRADLEBench** and **CRADLE-Dialogue**.

## Datasets

| Dataset | Source | Unit | Task | Size |
|---|---|---|---|---|
| **CRADLEBench** | [`SungJoo/Cradle-Bench`](https://huggingface.co/datasets/SungJoo/Cradle-Bench) (arXiv [2510.23845](https://arxiv.org/abs/2510.23845)) | social-media post | multi-label crisis type + temporal (ongoing/past) | 8,259 rows (train 7,239 / dev 420 / test 600) |
| **CRADLE-Dialogue** | [`SungJoo/Cradle-Dialogue`](https://huggingface.co/datasets/SungJoo/Cradle-Dialogue) | multi-turn conversation | risk-conditioned next-Listener response | 64k+ turns |

> These datasets contain sensitive mental-health content (self-harm, suicide, abuse) and are used
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

It covers split sizes, missingness/duplicates, label distributions,
crisis-type frequency, temporal labels, and multi-label co-occurrence. Figures
are saved to `reports/figures/`.

## CRADLE multi-turn response generation

This is a separate task from the RoBERTa crisis classifier. The classifier
predicts risk signals; the response model receives those signals together with
recent dialogue history and generates the next Listener reply. Both components
can therefore be developed in parallel. Training uses CRADLE's gold labels in
the same normalized risk schema; deployment replaces them with classifier
predictions.

Prepare the official dataset splits without re-splitting dialogues:

```bash
python -m src.cradle_response.preprocess --max-history-turns 14
```

Every Listener turn becomes a completion target. The prompt contains a system
instruction, cumulative risk context, and preceding User/Listener turns. Common
contact identifiers are redacted. The loader also reconstructs dialogue IDs
omitted by the current Hub parquet, merges unprefixed bullet rows into the
preceding message, and skips opening Listener greetings that have no preceding
User input. Test the GPU pipeline first, then train:

```bash
python -m src.cradle_response.train_response_sft --smoke-test
python -m src.cradle_response.train_response_sft
```

The one-epoch A100 experiment used all 22,830 training response targets. On all
4,327 held-out test targets, fine-tuning reduced completion loss from 3.1089 to
2.0117 (35.29%), reduced perplexity from 22.40 to 7.48 (66.62%), and increased
token accuracy from 38.58% to 51.95%. Reproduce the comparison with:

```bash
python -m src.cradle_response.evaluate \
  --adapter outputs/llama-3.2-1b-cradle-response-qlora-full/final_adapter \
  --output reports/cradle_response_test_metrics.json
```

Example deployment-style inference with classifier output:

```bash
python -m src.cradle_response.inference \
  --adapter outputs/llama-3.2-1b-cradle-response-qlora/final_adapter \
  --history-file history.json \
  --text "I don't feel safe tonight." \
  --current-signal alert_ongoing \
  --known-event alert_ongoing
```

CRADLE Listener replies are GPT-5-generated synthetic responses, not
clinician-authored ground truth. They are useful for an academic response-model
baseline, but safety evaluation and expert review are required before any
real-world claim or use.

Run the active preprocessing tests without downloading a model:

```bash
python -m unittest discover -s tests -v
```


## Layout

```
data/raw/            # downloaded datasets (gitignored)
notebooks/eda.ipynb  # exploratory data analysis
src/download_data.py # dataset downloader
src/cradle_response/  # multi-turn response data, training, and inference
tests/                # pipeline unit tests
reports/figures/     # generated EDA figures
```
