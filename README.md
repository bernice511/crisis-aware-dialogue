# Crisis-Aware Dialogue

Crisis-Aware Adaptation of Small Language Models for Harmful Dialogue Understanding.

This project investigates whether a **routed, prompt-specialized pipeline** for handling
distress-signal user input (self-harm, suicide ideation, abuse disclosure, etc.) beats direct
baseline queries to the same underlying model. The target system is a three-layer pipeline (see
`project_framework.md.txt`):

1. **Classifier (this checkout)** — a multi-label crisis-type classifier + derived flagged/not-flagged
   detector, trained on CRADLEBench.
2. **Prompt routing** — the classifier's predicted crisis types are folded directly into a
   structured risk-context object (`src/cradle_response/inference.py:build_risk_context`) rather
   than a separate rule-based routing layer.
3. **Llama 3.2 QLoRA fine-tune** (`src/cradle_response/`, ported from the `alice-llama-sft` /
   `alice-cradle-response` branches) — receives the risk context + user input and generates a
   response.

This checkout now wires Layers 1 and 3 together end-to-end via `src/pipeline.py` and the
`app.py` Streamlit demo — see [Interactive demo](#interactive-demo-classifier--llama-response)
below.

## Datasets

| Dataset | HF repo | Role |
|---|---|---|
| **CRADLEBench** | [`SungJoo/Cradle-Bench`](https://huggingface.co/datasets/SungJoo/Cradle-Bench) ([paper](https://arxiv.org/abs/2510.23845)) | Classifier training/eval (Layer 1, this checkout) |
| **CRADLE-Dialogue** | [`SungJoo/Cradle-Dialogue`](https://huggingface.co/datasets/SungJoo/Cradle-Dialogue) | Llama QLoRA SFT + eval source (Layer 3, `src/cradle_response/`) |

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

## Layer 3: Llama Response Model

`src/cradle_response/` fine-tunes `meta-llama/Llama-3.2-1B-Instruct` with QLoRA on
CRADLE-Dialogue to generate the next Listener reply given conversation history and a risk
context (crisis types + accumulated known events). The trained adapter lives at
`outputs/llama-3.2-1b-cradle-response-qlora-full/final_adapter` (gitignored like the
classifier's `outputs/`; ask the team for the artifact directly).

```bash
# Prepare the official dataset splits
python -m src.cradle_response.preprocess --max-history-turns 14

# Train (requires a CUDA GPU; raises immediately otherwise — trains on Colab/cloud)
python -m src.cradle_response.train_response_sft

# Compare base vs. fine-tuned completion likelihood on held-out data
python -m src.cradle_response.evaluate \
    --adapter outputs/llama-3.2-1b-cradle-response-qlora-full/final_adapter

# CLI inference with classifier-style risk context
python -m src.cradle_response.inference \
    --adapter outputs/llama-3.2-1b-cradle-response-qlora-full/final_adapter \
    --text "I don't feel safe tonight." \
    --current-signal alert_ongoing
```

One-epoch A100 results (`reports/cradle_response_results.json`): fine-tuning reduced test
completion loss from 3.11 to 2.01 (-35%), perplexity from 22.4 to 7.5 (-67%), and improved token
accuracy from 38.6% to 52.0%. CRADLE Listener replies are GPT-5-generated synthetic responses,
not clinician-authored ground truth — useful as an academic baseline, not a real-world claim.

### Known limitation: over-therapizing some benign, non-crisis input

`python -m src.cradle_response.evaluate_benign_register` (results in
`reports/cradle_response_benign_register_eval.json`) runs 12 ordinary small-talk openers through
the unmodified classifier -> Llama pipeline. Most come back naturally (`'hey' -> 'hey'`,
`'whats up' -> 'not much. how's your day going?'`, `'tell me a joke' -> an actual joke`), but a
few — most clearly bare greetings like `'hi' -> "I'm glad you reached out. What's going on
today?"` — get an unwarranted supportive-Listener register.

This is not a classifier or prompt-wiring bug (the classifier correctly reports `flagged: False`
for all 12); it reproduces with the base model too, given the same system prompt. Root cause,
confirmed by inspecting raw `SungJoo/Cradle-Dialogue` rows directly: **every training dialogue
originates from a support-seeking context, even turns with no crisis label at all** (e.g. an
unlabeled opening turn: `"Hi—hope it's okay to post here. I'm looking to hear from anyone who's
experienced sexual harassment..."` → `"Thanks for checking in. That sounds like an important
piece..."`). The model never saw a genuinely casual chit-chat exchange during fine-tuning, so it
has no learned counterexample to draw a plain register from — it's a training-data-scope gap, not
something a system-prompt tweak actually fixes.

A real fix means augmenting the SFT training set with genuinely benign, non-support dialogue
turns (`current_signals: ["none"]`, `known_events: []`, paired with an ordinary casual reply) so
the model learns that a `"none"` risk context can mean *actually nothing going on*, not just *no
new disclosure this turn within an ongoing support conversation* — then retraining
(`train_response_sft.py`, requires CUDA). Out of scope for this checkout; flagged here for
whoever picks up Layer 3 training next, the same way the classifier's `childabuse_endangerment`
gap is documented above.

## Interactive demo (classifier -> Llama response)

`app.py` is a local Streamlit chat UI: user text is classified by Layer 1
(`src/classifier`), the predicted crisis label(s) are shown, and both the text and the
resulting risk context are passed to the Layer 3 Llama adapter to generate a reply
(`src/pipeline.py` does the wiring).

**Prerequisites:**
1. `pip install -r requirements.txt` (adds `peft` and `streamlit` on top of the classifier deps).
2. Accept the license for the gated base model
   [`meta-llama/Llama-3.2-1B-Instruct`](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct)
   on huggingface.co, then authenticate locally: `hf auth login`.
3. Both the classifier (`outputs/cradlebench-roberta-classifier-ml512-weighted/final_model`) and
   the response adapter (`outputs/llama-3.2-1b-cradle-response-qlora-full/final_adapter`) must be
   present locally (see above for how to obtain each).

```bash
streamlit run app.py
```

This is a research prototype, not a real crisis service — the app displays that disclaimer and a
988 (Suicide & Crisis Lifeline) pointer on every load. No CUDA/bitsandbytes is required to run
it: inference loads both models in plain float32 and runs on MPS (Apple Silicon) or CPU.

**If your gated-access request is still "awaiting review":** Meta's approval isn't always
instant. In the meantime you can point the adapter at an ungated re-upload of the identical
weights/config (adapter merging is architecture-based, not tied to which repo the base came
from):

```bash
CRADLE_BASE_MODEL_ID=unsloth/Llama-3.2-1B-Instruct streamlit run app.py
```

Drop the env var once your official `meta-llama/Llama-3.2-1B-Instruct` access is approved.

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
src/cradle_response/
    preprocess.py              # CRADLE-Dialogue -> risk-conditioned response SFT examples
    train_response_sft.py      # Llama 3.2 QLoRA fine-tuning (requires CUDA)
    evaluate.py                 # base-vs-fine-tuned held-out likelihood comparison
    evaluate_benign_register.py  # documents the over-therapizing gap (see Known limitation above)
    inference.py                 # CLI inference; build_risk_context/build_messages helpers
src/pipeline.py               # classifier -> risk context -> Llama response glue, used by app.py
app.py                        # Streamlit interactive demo (see above)
tests/                        # unittest suite for the above (excluding model-dependent code)
```

## Repo layout note

Team members currently keep separate local checkouts of this repo rather than branching within
one shared clone — see the top-level `Project/CLAUDE.md` (one directory up) for details on the
other checkouts (`alice-git-branch/` has the Llama SFT pipeline on `alice-llama-sft`). The
`src/cradle_response/` code and trained adapter from that branch have been copied into this
checkout (see [Layer 3](#layer-3-llama-response-model) above) to build the end-to-end demo, but
`train_response_sft.py`/`evaluate.py` are still best run from a CUDA-capable checkout — this one
is fine for the classifier and for demo inference only.
