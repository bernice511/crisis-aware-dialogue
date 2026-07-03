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

## Layout

```
data/raw/            # downloaded datasets (gitignored)
notebooks/eda.ipynb  # exploratory data analysis
src/download_data.py # dataset downloader
reports/figures/     # generated EDA figures
```
