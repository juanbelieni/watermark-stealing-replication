# Watermark Stealing: Replication

> Capstone project developed during the last week of the [AI Security Bootcamp](https://aisb.dev).

This repository contains a small-scale replication of the paper *Watermark Stealing Attacks on Large Language Models* (2024), which demonstrates that statistical text watermarking schemes can be extracted and circumvented by low-budget adversaries.


## Overview

**Text watermarking** aims to mark model outputs so that detectors can later attribute authorship to an LLM. A common family of schemes (e.g. **KGW / SelfHash**) works by partitioning the vocabulary at each decoding step into “green” and “red” sets, boosting the probability of green tokens. Detection then applies a hypothesis test (e.g. z-test) over the fraction of green tokens in a text.

**Watermark stealing** shows that an adversary, with black-box access to a watermarked model, can learn a surrogate predictor for the watermark’s hidden partitioning function. Once the surrogate is trained, the attacker can:

* **Spoof**: Generate text with an unwatermarked model, but bias token choice using the surrogate, so the detector falsely flags it as watermarked.
* **Scrub**: Take genuine watermarked text and paraphrase it under surrogate guidance, reducing detector confidence so it is falsely judged as non-watermarked.

These attacks undermine the core security goals of watermarking: reliable attribution and provenance.

## Technical Components

* **Victim model**: Any open LLM (3B–7B) wrapped with a KGW-style watermark layer.
* **Detector**: Statistical test (z-test) comparing green-token rate against expected baseline.
* **Attacker surrogate**: Lightweight classifier (logistic regression / small MLP) trained on a small set of watermarked samples to approximate “greenness” of tokens in context.
* **Spoofing attack**: Use the surrogate to bias generation from an unwatermarked model; measure fraction of texts that pass detection.
* **Metrics**:

  * **Detection scores** (z-scores).
  * **Spoofing FPR**: % of spoofed outputs flagged as watermarked.

## What’s in this repo

- `src/lm.py`: Lightweight wrapper around Hugging Face causal LMs with an optional logits processor hook and a convenience `generate` method.
- `src/kgw.py`: KGW-style watermark implementation (logits processor) and a `WatermarkedLM` helper with a built-in detector.
- `scripts/generate_samples.py`: Generate JSON datasets of base vs watermarked completions on UltraFeedback prompts.
- `scripts/mark_green_red_tokens.py`: Recreate green/red classification per token for sample pairs.
- `scripts/spoof_attack.py`: Minimal spoofing replication using a clipped likelihood-ratio surrogate.
- `scripts/test_watermark.py`: Quick sanity test for watermark generation and detection.
- `web/app.py`: Streamlit app that explains watermarking, the stealing threat model, and shows toy/annotated examples.
- `justfile`: Handy `uv run` recipes for common workflows.

## Requirements

- Python 3.11+
- GPU recommended for model inference (flash-attn enabled); falls back to CPU if needed.
- Hugging Face token with access to `meta-llama/Llama-3.2-3B-Instruct` if gated.
- Dependencies defined in `pyproject.toml` and pinned via `uv.lock`.

Install options:

- With `uv` (recommended): `uv sync`
- With pip: `pip install -e .`

## Usage


### Quick sanity test

Verify watermarked outputs are detected and base outputs are not:

```
uv run python -m scripts.test_watermark
```

### Generate datasets

This script now uses the UltraFeedback train split and records metadata in the output JSON.

- Base: `uv run python -m scripts.generate_samples --mode base`
- Watermarked: `uv run python -m scripts.generate_samples --mode watermarked`
- Both: `uv run python -m scripts.generate_samples --mode both`

Key flags: `--limit`, `--per_prompt`, `--max_new_tokens`, `--temperature`, `--self_hash`, `--device`.

Outputs (under `data/`) include:

- `mode`: `base` or `watermarked`
- `split`: `train`
- `generation_params`: run settings
- `watermark_params`: present for watermarked runs (e.g., `gamma`, `delta`, `hash_key`, `window_size`, `self_hash`, `vocab_size`)

### Spoof attack replication

Run the spoofing script (logs optional metrics to Weights & Biases):

```
uv run python -m scripts.spoof_attack \
  --delta_att 10 \
  --limit_samples 10000 \
  --seed 0 \
  --wandb_project watermark-spoof
```

### Streamlit app

- Run: `uv run streamlit run web/app.py`
- The “Replication” tab can visualize annotated examples if you provide a JSON file with structure `{ prompt, base: { tokens, green }, watermarked: { tokens, green } }`.

### Annotate examples for the app

Produce token-level green/red for a subset of prompts and completions:

```
uv run python -m scripts.mark_green_red_tokens \
  --base data/<base.json> \
  --watermarked data/<watermarked.json> \
  --out data/green_red_examples.json \
  --num 5 --completion-index 0 --display text
```

The Streamlit app loads `web/examples.json`. Copy or symlink your output:

```
cp data/green_red_examples.json web/examples.json
```



