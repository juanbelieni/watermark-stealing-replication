# Watermark Stealing: Replication

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
* **Scrubbing attack**: Feed watermarked text into a paraphraser, guided by the surrogate; measure false negative rate (FNR) while preserving semantics.
* **Metrics**:

  * **Detection scores** (z-scores).
  * **Spoofing success rate**: % of unwatermarked outputs flagged as watermarked.
  * **Scrubbing success rate**: FNR at a fixed false positive rate (e.g., 10⁻³).
  * **Semantic fidelity**: BLEU/BERTScore to confirm meaning retention.
