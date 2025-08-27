#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Watermark-spoof (full 3-token set contexts only)

- Loads base and watermarked corpora of completions.
- Builds P_w and P_b over targets conditioned on the unordered set of the last 3 tokens.
- Computes s(T, ctx) = (1/c) * min( p_w / p_b, c ) for ratios >= 1 else 0, with c=2 by default.
- During generation, adds delta_att * s(T, ctx) to logits for targets T in the current ctx
  ONLY when the set of the last 3 tokens has size exactly 3.
- No Tmin heuristic; no partial contexts; no fallbacks.

Assumes your local `src.lm.LM` accepts a `LogitsProcessorList` via `self.logits_processor`.
"""


import os
import sys
from pathlib import Path
sys.path.append(str(Path(os.path.abspath("")).parent))

import json
import math
import torch
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizer, LogitsProcessor, LogitsProcessorList

from src.lm import LM
from src.kgw import WatermarkedLM

# --- Config (can be overridden by CLI if desired) ----------------------------------------------

MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"
BASE_PATH = "data/samples_meta_llama_llama_3.2_3b_instruct_base_20250826T165913Z.json"
WATERMARKED_PATH = "data/samples_meta_llama_llama_3.2_3b_instruct_watermarked_20250827T094752Z.json"

WINDOW_SIZE = 2           # we only use full 3-token set contexts
CLIP_C = 2.0              # paper uses c = 2
DELTA_ATT = 5.0           # attack logit multiplier (independent of CLIP_C)
LIMIT_SAMPLES = 1000     # limit to first N samples in each corpus for faster testing

NUM_PROMPTS = 100
MAX_NEW_TOKENS = 500


# -------------------------------- Data ---------------------------------------------------------

def load_samples(json_path: str) -> list[str]:
    with open(json_path, "r") as f:
        data = json.load(f)
    return [c for s in data["samples"][:LIMIT_SAMPLES] for c in s["completions"]]
    # return [s["completions"][0] for s in data["samples"]]  # only first completion per prompt


def batch_tokenize(tokenizer: PreTrainedTokenizer, samples: list[str], batch_size: int = 16) -> list[list[int]]:
    dl = torch.utils.data.DataLoader(samples, batch_size=batch_size)
    all_ids: list[list[int]] = []
    for batch in tqdm(dl, desc="Tokenizing"):
        ids = tokenizer(batch, add_special_tokens=False).input_ids
        all_ids.extend(ids)
    return all_ids


# ------------------------------ Counting (full 3-sets only) ------------------------------------

conditional_id = tuple[int, frozenset[int]]  # (target_token_id, context_set_of_size_3)


def powerset(n: int):
    s = range(0, n)
    return [set(c) for r in range(len(s) + 1) for c in combinations(s, r)]

# def conditional_count_full3(input_ids: list[list[int]], window_size: int = 3) -> dict[conditional_id, int]:
#     """
#     Count occurrences of target token given the unordered set of the last 3 tokens.
#     Ignores positions where the 3-token window has <3 unique tokens.
#     """
#     assert window_size == 3, "This script is specialized to WINDOW_SIZE == 3."
#     counts = defaultdict(int)
#     for seq in tqdm(input_ids, desc="Counting (full 3-sets)"):
#         n = len(seq)
#         if n <= window_size:
#             continue
#         for i in range(window_size, n):
#             ctx_window = seq[i - window_size:i]
#             ctx_set = frozenset(ctx_window)
#             if len(ctx_set) != window_size:
#                 # duplicates in the window; skip (no partial contexts)
#                 continue
#             target = seq[i]
#             counts[(target, ctx_set)] += 1
#     return counts


def conditional_count(
    input_ids: list[list[int]], window_size: int = 3
) -> dict[conditional_id, int]:
    # the keys will be type tuple[int, set[int]]
    counts = defaultdict(int)

    for seq in tqdm(input_ids, "Counting (all subsets)"):
        for i in range(len(seq)):
            h = min(i, window_size)
            context = tuple(seq[i - h : i])
            target = seq[i]

            # for subset in powerset(h):
            #     context_subset = frozenset(context[j] for j in subset)
            #     counts[(target, context_subset)] += 1
            counts[(target, frozenset(context))] += 1

    return counts


# def compute_probabilities(counts: dict[conditional_id, int]) -> dict[conditional_id, float]:
#     """
#     Compute P(target | ctx_set) for ctx_set with exactly 3 unique tokens.
#     """
#     totals = defaultdict(int)
#     for (_, ctx), cnt in counts.items():
#         totals[ctx] += cnt

#     probs: dict[conditional_id, float] = {}
#     for (t, ctx), cnt in counts.items():
#         denom = totals[ctx]
#         probs[(t, ctx)] = cnt / denom if denom > 0 else 0.0
#     return probs


# def compute_scores(
#     base_probs: dict[conditional_id, float],
#     watermarked_probs: dict[conditional_id, float],
#     c: float = 2.0,
# ) -> dict[conditional_id, float]:
#     """
#     s(T, ctx) = (1/c) * min( pw/pb, c ) if ratio >= 1 else 0
#     Only keys present in watermarked_probs are considered.
#     """
#     out = {}
#     for key, pw in tqdm(watermarked_probs.items(), desc="Scoring (ratio+clip)"):
#         pb = base_probs.get(key, 0.0)

#         # No evidence in watermarked corpus: no boost.
#         if pw <= 0.0:
#             out[key] = 0.0
#             continue

#         # If base probability is zero but watermarked > 0, treat ratio as +inf.
#         # After clipping at c and dividing by c, this yields s = 1.0.
#         if pb <= 0.0:
#             out[key] = 1.0
#             continue

#         ratio = pw / pb
#         out[key] = (1.0 / c) * min(ratio, c) if ratio >= 1.0 else 0.0
#     return out

def compute_boosts(
    base_counts: dict[conditional_id, int],
    wm_counts: dict[conditional_id, int],
    clip_c: float = CLIP_C,
    min_wm_count_nonempty: int = 2,
) -> dict[frozenset[int], dict[int, float]]:
    """
    Compute P(target | ctx_set) for ctx_set with exactly 3 unique tokens.
    s(T, ctx) = (1/c) * min( pw/pb, c ) if ratio >= 1 else 0
    """

    base_totals = defaultdict(int)
    for (_, ctx), cnt in base_counts.items():
        base_totals[ctx] += cnt

    wm_totals = defaultdict(int)
    for (_, ctx), cnt in wm_counts.items():
        wm_totals[ctx] += cnt

    base_probs: dict[conditional_id, float] = defaultdict(float)
    for (t, ctx), cnt in base_counts.items():
        denom = base_totals[ctx]
        base_probs[(t, ctx)] = cnt / denom if denom > 0 else 0.0

    wm_probs: dict[conditional_id, float] = defaultdict(float)
    for (t, ctx), cnt in wm_counts.items():
        denom = wm_totals[ctx]
        wm_probs[(t, ctx)] = cnt / denom if denom > 0 else 0.0

    boosts_by_ctx: dict[frozenset[int], dict[int, float]] = defaultdict(dict)

    for key, pw in tqdm(wm_probs.items(), desc="Calculating boosts"):
        pb = base_probs[key]

        # No evidence in watermarked corpus: no boost.
        if pw <= 0.0:
            ratio = 0.0

        # If base probability is zero but watermarked > 0, treat ratio as +inf.
        # After clipping at c and dividing by c, this yields s = 1.0.
        elif pb <= 0.0:
            ratio = 1.0
        else:
            ratio = pw / pb
            ratio = (1.0 / clip_c) * min(ratio, clip_c) if ratio >= 1.0 else 0.0

        target_id, ctx_set = key
        boosts_by_ctx[ctx_set][target_id] = ratio + 1e3

    # normalize
    for ctx, tv in tqdm(boosts_by_ctx.items(), desc="Normalizing boosts"):
        max_tv = max(tv.values())
        if max_tv > 0.0:
            for t in tv:
                tv[t] /= max_tv

    return boosts_by_ctx




def group_scores_by_ctx(scores: dict[conditional_id, float]) -> dict[frozenset[int], dict[int, float]]:
    """
    {ctx_set -> {target_id -> s(T, ctx_set)}} with only positive scores kept.
    """
    by_ctx: dict[frozenset[int], dict[int, float]] = defaultdict(dict)
    for (t, ctx), v in scores.items():
        if v > 0.0:
            by_ctx[ctx][t] = v
    return dict(by_ctx)


# ------------------------------ Attack runtime --------------------------------------------------

@dataclass
class AttackConfig:
    delta: float         # attack logit multiplier
    window_size: int     # must be 3 for this script
    vocab_size: int


class AttackLogitsProcessor(LogitsProcessor):
    """
    Adds delta * s(T, ctx_set) to logits for the current full 3-set context.
    No fallbacks, no partial contexts.
    """
    def __init__(self, config: AttackConfig, ctx_to_idx: dict[frozenset[int], torch.LongTensor],
                 ctx_to_val: dict[frozenset[int], torch.FloatTensor]) -> None:
        super().__init__()
        # assert config.window_size == 3, "This script is specialized to window_size == 3."
        self.config = config
        self.ctx_to_idx = ctx_to_idx
        self.ctx_to_val = ctx_to_val

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        bsz, seqlen = input_ids.size(0), input_ids.size(1)
        if seqlen < self.config.window_size:
            return scores
        scores = scores.clone()

        for i in range(bsz):
            ctx_set = frozenset(input_ids[i, -self.config.window_size:].tolist())
            idx = self.ctx_to_idx.get(ctx_set)
            if idx is None or idx.numel() == 0:
                # print(f"Context {ctx_set} not found or empty; skipping.")
                continue
            idx = idx.cuda()
            vals = self.ctx_to_val[ctx_set].cuda()  # same length as idx
            scores[i, idx] += self.config.delta * vals

        return scores


class AttackLM(LM):
    def __init__(self, model_name_or_path: str, *, delta: float = 2.5, window_size: int = 3,
                 ctx_to_idx=None, ctx_to_val=None) -> None:
        super().__init__(model_name_or_path)
        cfg = AttackConfig(delta=delta, window_size=window_size, vocab_size=self.tokenizer.vocab_size)
        self.attack = AttackLogitsProcessor(cfg, ctx_to_idx, ctx_to_val)
        self.logits_processor = LogitsProcessorList([self.attack])


# ------------------------------ Prompting -------------------------------------------------------

def build_prompts(n: int) -> list[str]:
    from datasets import load_dataset
    ds = load_dataset("trl-lib/ultrafeedback-prompt", split="train").select(range(n))
    return [d[0]["content"] for d in ds["prompt"]]


def apply_chat_template(tokenizer: PreTrainedTokenizer, prompts: list[str]) -> list[str]:
    out = []
    for p in prompts:
        out.append(
            tokenizer.apply_chat_template([{"role": "user", "content": p}],
                                          add_generation_prompt=True,
                                          tokenize=False)
        )
    return out


# ------------------------------ Main ------------------------------------------------------------

if __name__ == "__main__":
    # Tokenizer
    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token

    # Load and tokenize corpora
    base_samples = load_samples(BASE_PATH)
    wm_samples = load_samples(WATERMARKED_PATH)

    base_ids = batch_tokenize(tokenizer, base_samples)
    wm_ids = batch_tokenize(tokenizer, wm_samples)

    # print the ammount of tokens in each corpus
    base_total_len = sum(len(ids) for ids in base_ids)
    base_mean_len = base_total_len / len(base_ids)
    wm_total_len = sum(len(ids) for ids in wm_ids)
    wm_mean_len = wm_total_len / len(wm_ids)
    print(f"Base corpus: {base_total_len} tokens, {len(base_ids)} samples, mean length {base_mean_len:.2f}")
    print(f"Watermarked corpus: {wm_total_len} tokens, {len(wm_ids)} samples, mean length {wm_mean_len:.2f}")

    # Count / P(target | full 3-set) / Scores
    base_counts = conditional_count(base_ids, window_size=WINDOW_SIZE)
    wm_counts = conditional_count(wm_ids, window_size=WINDOW_SIZE)

    # base_probs = compute_probabilities(base_counts)
    # wm_probs = compute_probabilities(wm_counts)

    # scores = compute_scores(base_probs, wm_probs, c=CLIP_C)  # s(T, {T1,T2,T3})
    # scores_by_ctx = group_scores_by_ctx(scores)
    boosts_by_ctx = compute_boosts(base_counts, wm_counts)

    # Prebuild tensors per context for fast biasing
    ctx_to_idx: dict[frozenset[int], torch.LongTensor] = {}
    ctx_to_val: dict[frozenset[int], torch.FloatTensor] = {}
    for ctx, tv in boosts_by_ctx.items():
        if not tv:
            continue
        ids, vals = zip(*tv.items())  # token ids, s-values
        ctx_to_idx[ctx] = torch.tensor(ids, dtype=torch.long)
        ctx_to_val[ctx] = torch.tensor(vals, dtype=torch.float32)

    # Build attack LM
    alm = AttackLM(MODEL_ID, delta=DELTA_ATT, window_size=WINDOW_SIZE,
                   ctx_to_idx=ctx_to_idx, ctx_to_val=ctx_to_val)

    # Generate a few samples
    raw_prompts = build_prompts(NUM_PROMPTS)
    templated_prompts = apply_chat_template(alm.tokenizer, raw_prompts)
    samples = alm.generate(templated_prompts, max_new_tokens=MAX_NEW_TOKENS)

    # Print outputs
    # for i, text in enumerate(samples):
    #     print(f"\n--- SAMPLE {i:02d} ---\n{text}\n")

    # If you want to run detection here, instantiate your WatermarkedLM and call detect_watermark.
    del alm

    wlm = WatermarkedLM(MODEL_ID, self_hash=False)

    success_count = 0
    z_sum = 0.0

    for i, text in enumerate(samples):
        is_wm, z = wlm.detect_watermark(text)
        print(f"[WATERMARKED {i:02d}] z={z:.2f} flagged={is_wm}")

        success_count += int(is_wm)
        z_sum += z

    success_rate = success_count / len(samples)
    z_avg = z_sum / len(samples) if samples else 0.0
    print(f"\nAttack success rate: {success_rate*100:.2f}% ({success_count}/{len(samples)})")
    print(f"Average z-score: {z_avg:.2f}")
