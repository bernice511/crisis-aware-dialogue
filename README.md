# Crisis-Aware Dialogue

Crisis-Aware Adaptation of Small Language Models for Harmful Dialogue Understanding.

This project investigates whether a **routed, prompt-specialized pipeline** for handling
distress-signal user input (self-harm, suicide ideation, abuse disclosure, etc.) beats direct
baseline queries to the same underlying model. The target system is a three-layer pipeline (see
`project_framework.md.txt`):

1. **Classifier (this checkout)** — a multi-label crisis-type classifier + derived flagged/not-flagged
   detector, trained on CRADLEBench.
2. **Prompt routing** (planned) — rule-based mapping from predicted crisis category to a
   category-specific system-prompt template.
3. **Llama 3.2 QLoRA fine-tune** (`src/llama/`, on the `alice-llama-sft` branch, not in this
   checkout) — receives the routed prompt + user input and generates a response.

This branch is Layer 1 only. It does not yet integrate with Layers 2/3.

## Datasets

| Dataset | HF repo | Role |
|---|---|---|
| **CRADLEBench** | [`SungJoo/Cradle-Bench`](https://huggingface.co/datasets/SungJoo/Cradle-Bench) ([paper](https://arxiv.org/abs/2510.23845)) | Classifier training/eval (Layer 1, this checkout) |
| **CRADLE-Dialogue** | [`SungJoo/Cradle-Dialogue`](https://huggingface.co/datasets/SungJoo/Cradle-Dialogue) | Llama QLoRA SFT + eval source (Layer 3) — see `alice-llama-sft` branch |

Both are public but contain sensitive mental-health content (self-harm, suicide, abuse) and are
used strictly for academic research. Raw/processed data directories are gitignored — never commit
dataset files. CRADLEBench and CRADLE-Dialogue are companion benchmarks from the same
clinician-annotated taxonomy (dialogue-level vs. single-post) and share crisis-category labels.

## Setup

```bash
pip install -r requirements.txt
python src/download_data.py                # downloads CRADLEBench into data/raw/
python -m src.classifier.preprocess         # builds data/processed/cradlebench_clf/{train,validation,test}.jsonl
```

## Layer 1: Crisis-Type Classifier

A `roberta-base` encoder fine-tuned as a **7-way multi-label** classifier over CRADLEBench's
clinician-annotated crisis categories (`src/classifier/labels.py:CRISIS_TYPES`):
`selfharm`, `suicideideation_passive`, `suicideideation_active`, `domesticviolence`, `rape`,
`sexualharassment`, `childabuse_endangerment`. A binary flagged/not-flagged view is derived at
eval time as "any crisis type predicted," not trained as a separate head. Temporal
(ongoing/past) labels are parsed and retained per-record but are not currently a model output.

Trained on CRADLEBench's fixed splits (no re-splitting): 4,181 train / 420 validation / 600 test,
using the **Consensus** training subset (≥2 of 3 LLM annotators — GPT-5, Claude-4-Sonnet,
Gemini-2.5-Pro — agree), not the stricter Unanimous subset, for more training data at the cost of
some non-unanimous labels.

### Commands

```bash
# Train (best config below; see --help for all options)
python -m src.classifier.train_classifier \
    --max-length 512 --class-weighted-loss --pos-weight-power 0.5

# Single-text inference (handles inputs >512 tokens via the sliding-window wrapper automatically)
python -m src.classifier.inference \
    --model outputs/cradlebench-roberta-classifier-ml512-weighted/final_model \
    --text "..."

# Compare plain-truncated vs. sliding-window inference on the test set
python -m src.classifier.evaluate_sliding_window \
    --model outputs/cradlebench-roberta-classifier-ml512-weighted/final_model

# Per-class threshold tuning experiment (rejected — see Known limitations)
python -m src.classifier.tune_thresholds \
    --model outputs/cradlebench-roberta-classifier-ml512-weighted/final_model

# Tests
python -m unittest discover -s tests -v
```

Trained models are **not committed to git** (`outputs/` is gitignored — each run is ~3GB+ with
checkpoints). Regenerate via the training command above (roughly 30-45 min on a 6GB GPU at
batch_size=16). Ask the team for the artifact directly if you need the exact trained weights
rather than a fresh run.

### Current best model and results

`outputs/cradlebench-roberta-classifier-ml512-weighted/final_model` — `--max-length 512
--class-weighted-loss --pos-weight-power 0.5 --metric-for-best-model f1_macro`.

| Class | F1 | Precision | Recall |
|---|---|---|---|
| selfharm | 0.893 | 0.833 | 0.962 |
| domesticviolence | 0.844 | 0.771 | 0.931 |
| rape | 0.855 | 0.823 | 0.890 |
| sexualharassment | 0.833 | 0.797 | 0.873 |
| suicideideation_active | 0.769 | 0.769 | 0.769 |
| suicideideation_passive | 0.749 | 0.684 | 0.827 |
| childabuse_endangerment | 0.554 | 0.459 | 0.700 |
| **macro F1** | **0.785** | | |
| micro F1 | 0.801 | | |
| flagged (any crisis) F1 | 0.920 | accuracy 0.883 | |

**How this compares to the CRADLEBench paper.** The paper (`Proposal/CRADLEBench.pdf`) evaluates
15 zero/few-shot-prompted LLMs plus a 3-model (GPT-5 + Claude-4-Sonnet + Gemini-2.5-Pro)
majority-vote ensemble, and separately fine-tunes 14B-72B LLMs (aggregate metrics only, no
per-class breakdown for those). Comparing our per-class F1 against their per-label tables
(averaging their `_ongoing`/`_past` split per class as a rough proxy, since our schema collapses
that axis):

| Class | Ours | GPT-5 alone | Their 3-model Ensemble |
|---|---|---|---|
| selfharm | **0.893** | 0.801 | 0.815 |
| domesticviolence | **0.844** | 0.782 | 0.831 |
| sexualharassment | **0.833** | 0.816 | 0.866 |
| rape | 0.855 | 0.869 | 0.899 |
| suicideideation_active | 0.769 | 0.873 | 0.889 |
| suicideideation_passive | 0.749 | 0.878 | 0.842 |
| childabuse_endangerment | 0.554 | 0.668 | 0.731 |
| macro avg | 0.785 | 0.812 | 0.839 |

A 125M-parameter fine-tuned classifier is within ~2.7 points of GPT-5 alone on average and beats
it outright on 3/7 classes, despite being ~100-500x smaller than the paper's reference models. The
gap is concentrated almost entirely in suicide ideation and childabuse, not spread evenly.

### Known limitation: childabuse_endangerment

This is CRADLEBench's smallest class (182/4,181 train, ~4.4%) and its posts run ~50-70% longer
than average, so it was disproportionately hurt by input truncation. What was tried, in order of
impact:

1. **`--max-length 512`** (roberta-base's ceiling): the dominant fix. Childabuse F1
   0.157→0.525, macro F1 0.702→0.776, all other classes flat-to-improved.
2. **Class-weighted BCE loss** (`--class-weighted-loss --pos-weight-power 0.5`, sqrt-dampened
   inverse train frequency): childabuse F1 0.525→0.554 (recall 0.525→0.700, trading precision for
   recall — the right direction for crisis detection). Macro F1 0.776→0.785.
3. **Rejected: per-class threshold tuning.** Helps the unweighted models modestly, hurts the
   weighted model — validation has only 8 childabuse positives, so the "optimal" threshold found
   there is noise.
4. **Rejected: sliding-window inference wrapper** (`src/classifier/sliding_window.py`). Built to
   recover the ~42.5% of childabuse test posts still truncated at 512 tokens: splits long inputs
   into overlapping windows, batches them through the existing model with no retraining, and
   max-pools per-class probabilities across windows. Verified the mechanism works in isolation
   (a synthetic long post with a disclosure only in the final sentence: naive truncation scored it
   0.009, windowing scored it 0.992) — but on the real test set it netted out to ~zero on
   childabuse (F1 0.5545→0.5577) while *hurting* domesticviolence (0.844→0.794) and macro F1
   (0.785→0.777). Diagnosis: the model was only ever trained on start-of-post text, so mid/tail
   windows are out-of-distribution for it at inference time — max-pooling then adds a little
   recall but more false positives across the board. A real fix would need retraining on windowed
   segments, which is riskier than it sounds: CRADLEBench gives post-level labels only (no
   evidence spans), so naively propagating a post's label to every chunk would inject label noise.

**Decision: scoped and handed off in this state**, rather than continuing to chase childabuse.
Estimated ceiling on further engineering here (head+tail truncation, windowed retraining) is
small — plausibly +0.03 to +0.10 F1 at best, and hard to even confirm as real given only 40 test
positives. The paper's own analysis (Section 6) identifies childabuse as the hardest category for
*every* model they test, including 70B+ fine-tuned LLMs — this looks like a genuinely hard class
given the available data, not a fixable engineering gap. Ideas for whoever picks this back up:
data augmentation for the rare class, cross-validation/pooled splits for a less noisy childabuse
metric, or a longer-context backbone (Longformer/BigBird) if truncation is revisited.

### Other open items

- No non-distress control inputs exist yet in eval data, so classifier false-positive rate against
  clearly benign input is currently unmeasurable.
- Multi-turn use (concatenating several dialogue turns into one input for Layer 1) is mechanically
  supported — preprocessing/training treat input as one flat string — but unvalidated: training
  data is single first-person posts, not role-tagged dialogue, and no multi-turn eval set exists
  yet to confirm calibration transfers.

## Testing

```bash
python -m unittest discover -s tests -v
```

## Layout

```
data/raw/                    # downloaded CRADLEBench (gitignored)
data/processed/cradlebench_clf/  # preprocessed JSONL splits (gitignored, regenerate via preprocess.py)
outputs/                     # trained model checkpoints (gitignored, regenerate via train_classifier.py)
reports/                     # small result artifacts tracked in git (e.g. sliding-window eval comparison)
src/download_data.py         # dataset downloader
src/classifier/
    labels.py                 # CRISIS_TYPES, label normalization/multi-hot encoding
    preprocess.py              # CRADLEBench CSV -> JSONL splits
    train_classifier.py        # fine-tuning entry point (Trainer + optional weighted-BCE)
    inference.py                # single-text prediction CLI
    sliding_window.py           # long-input windowing + max-pooling (see Known limitation above)
    evaluate_sliding_window.py  # truncated-vs-windowed test-set comparison
    tune_thresholds.py          # per-class threshold sweep (rejected, kept for reference)
    metrics.py                   # shared multi-label eval metrics
    baseline.py                  # bag-of-words baseline for comparison
tests/                        # unittest suite for the above (excluding model-dependent code)
```

## Repo layout note

Team members currently keep separate local checkouts of this repo rather than branching within
one shared clone — see the top-level `Project/CLAUDE.md` (one directory up) for details on the
other checkouts (`alice-git-branch/` has the Llama SFT pipeline on `alice-llama-sft`, not yet
merged to `main`) before assuming this checkout reflects the full project state.
