#!/usr/bin/env python3

import wandb
import argparse
import json
import random
import torch
import numpy as np
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    PreTrainedTokenizer,
    LogitsProcessor,
    LogitsProcessorList,
)

from src.lm import LM
from src.kgw import WatermarkedLM


MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"
BASE_PATH = "data/samples_meta_llama_llama_3.2_3b_instruct_base_20250826T165913Z.json"
WATERMARKED_PATH = (
    "data/samples_meta_llama_llama_3.2_3b_instruct_watermarked_20250827T133902Z.json"
)

WINDOW_SIZE = 2
CLIP_C = 2.0  # paper uses c = 2
DELTA_ATT = 10  # attack logit multiplier
LIMIT_SAMPLES = 10000  # limit to first N samples in each corpus for faster testing

# Default sweep / logging
WANDB_PROJECT = "watermark-spoof"

NUM_PROMPTS = 100
MAX_NEW_TOKENS = 500


# -------------------------------- Data ---------------------------------------------------------


def load_samples(json_path: str) -> list[str]:
    with open(json_path, "r") as f:
        data = json.load(f)
    return [c for s in data["samples"][:LIMIT_SAMPLES] for c in s["completions"]]
    # return [s["completions"][0] for s in data["samples"]]  # only first completion per prompt


def batch_tokenize(
    tokenizer: PreTrainedTokenizer, samples: list[str], batch_size: int = 16
) -> list[list[int]]:
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
            counts[(target, frozenset({}))] += 1

    return counts


def compute_probabilities(
    counts: dict[conditional_id, int],
) -> dict[conditional_id, float]:
    """
    Compute P(target | ctx_set) for ctx_set with exactly 3 unique tokens.
    """
    totals = defaultdict(int)
    for (_, ctx), cnt in counts.items():
        totals[ctx] += cnt

    probs = defaultdict(float)
    for (t, ctx), cnt in counts.items():
        denom = totals[ctx]
        probs[(t, ctx)] = cnt / denom if denom > 0 else 0.0

    return probs


def compute_boosts(
    base_counts: dict[conditional_id, int],
    wm_counts: dict[conditional_id, int],
    clip_c: float = CLIP_C,
) -> dict[frozenset[int], dict[int, float]]:
    base_probs = compute_probabilities(base_counts)
    wm_probs = compute_probabilities(wm_counts)

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
    for tv in tqdm(boosts_by_ctx.values(), desc="Normalizing boosts"):
        max_tv = max(tv.values())
        if max_tv > 0.0:
            for t in tv:
                tv[t] /= max_tv

    return boosts_by_ctx


def get_boost(
    ctx: frozenset[int],
    target: int,
    boosts_by_ctx: dict[frozenset[int], dict[int, float]],
    w2: float = 0.5,
) -> float:
    full_ctx_boost = boosts_by_ctx[ctx][target]
    empty_ctx_boost = boosts_by_ctx[frozenset()][target]

    z = 1 + w2

    boost = (full_ctx_boost + w2 * empty_ctx_boost) / z
    return boost


@dataclass
class AttackConfig:
    delta: float  # attack logit multiplier
    window_size: int
    vocab_size: int


class AttackLogitsProcessor(LogitsProcessor):
    def __init__(
        self,
        config: AttackConfig,
        ctx_to_idx: dict[frozenset[int], torch.LongTensor],
        ctx_to_val: dict[frozenset[int], torch.FloatTensor],
    ) -> None:
        super().__init__()
        # assert config.window_size == 3, "This script is specialized to window_size == 3."
        self.config = config
        self.ctx_to_idx = ctx_to_idx
        self.ctx_to_val = ctx_to_val

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        bsz, seqlen = input_ids.size(0), input_ids.size(1)
        if seqlen < self.config.window_size:
            return scores
        scores = scores.clone()

        for i in range(bsz):
            ctx_set = frozenset(input_ids[i, -self.config.window_size :].tolist())
            idx = self.ctx_to_idx.get(ctx_set)
            if idx is None or idx.numel() == 0:
                # print(f"Context {ctx_set} not found or empty; skipping.")
                continue
            idx = idx.cuda()
            vals = self.ctx_to_val[ctx_set].cuda()  # same length as idx
            scores[i, idx] += self.config.delta * vals

        return scores


class AttackLM(LM):
    def __init__(
        self,
        model_name_or_path: str,
        *,
        delta: float = 2.5,
        window_size: int = 3,
        ctx_to_idx=None,
        ctx_to_val=None,
    ) -> None:
        super().__init__(model_name_or_path)
        cfg = AttackConfig(
            delta=delta, window_size=window_size, vocab_size=self.tokenizer.vocab_size
        )
        self.attack = AttackLogitsProcessor(cfg, ctx_to_idx, ctx_to_val)
        self.logits_processor = LogitsProcessorList([self.attack])


def build_prompts(n: int) -> list[str]:
    from datasets import load_dataset

    ds = load_dataset("trl-lib/ultrafeedback-prompt", split="train").select(range(n))
    return [d[0]["content"] for d in ds["prompt"]]


def apply_chat_template(
    tokenizer: PreTrainedTokenizer, prompts: list[str]
) -> list[str]:
    out = []
    for p in prompts:
        out.append(
            tokenizer.apply_chat_template(
                [{"role": "user", "content": p}],
                add_generation_prompt=True,
                tokenize=False,
            )
        )
    return out


if __name__ == "__main__":
    # Args: expose only delta_att, limit_samples, seed (others remain globals)
    ap = argparse.ArgumentParser(description="Spoof attack generation and evaluation")

    ap.add_argument(
        "--delta_att",
        type=float,
        default=DELTA_ATT,
        help="Attack delta multiplier for biased logits.",
    )
    ap.add_argument(
        "--limit_samples",
        type=int,
        default=LIMIT_SAMPLES,
        help="Limit of samples from each corpus used to estimate boosts.",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for generation and dataloaders.",
    )
    ap.add_argument(
        "--wandb_project",
        type=str,
        default=WANDB_PROJECT,
        help="Weights & Biases project name.",
    )
    args = ap.parse_args()

    # Override globals from args to keep the rest as global constants
    DELTA_ATT = args.delta_att
    LIMIT_SAMPLES = args.limit_samples

    # Seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    wandb.init(
        project=args.wandb_project,
        config={
            "delta_att": DELTA_ATT,
            "limit_samples": LIMIT_SAMPLES,
            "seed": args.seed,
            "model_id": MODEL_ID,
            "window_size": WINDOW_SIZE,
            "clip_c": CLIP_C,
            "num_prompts": NUM_PROMPTS,
            "max_new_tokens": MAX_NEW_TOKENS,
        },
    )

    # Tokenizer
    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, use_fast=True
    )
    tokenizer.pad_token = tokenizer.eos_token

    # Load and tokenize corpora
    base_samples = load_samples(BASE_PATH)
    wm_samples = load_samples(WATERMARKED_PATH)

    base_ids = batch_tokenize(tokenizer, base_samples)
    wm_ids = batch_tokenize(tokenizer, wm_samples)

    # print the amount of tokens in each corpus
    base_total_len = sum(len(ids) for ids in base_ids)
    base_mean_len = base_total_len / len(base_ids)
    wm_total_len = sum(len(ids) for ids in wm_ids)
    wm_mean_len = wm_total_len / len(wm_ids)
    print(
        f"Base corpus: {base_total_len} tokens, {len(base_ids)} samples, mean length {base_mean_len:.2f}"
    )
    print(
        f"Watermarked corpus: {wm_total_len} tokens, {len(wm_ids)} samples, mean length {wm_mean_len:.2f}"
    )

    base_counts = conditional_count(base_ids, window_size=WINDOW_SIZE)
    wm_counts = conditional_count(wm_ids, window_size=WINDOW_SIZE)

    boosts_by_ctx = compute_boosts(base_counts, wm_counts)

    # Prebuild tensors per context for fast biasing
    ctx_to_idx: dict[frozenset[int], torch.LongTensor] = {}
    ctx_to_val: dict[frozenset[int], torch.FloatTensor] = {}
    for ctx, tv in tqdm(boosts_by_ctx.items()):
        targets = list(tv.keys())
        boosts = [get_boost(ctx, target, boosts_by_ctx) for target in targets]
        ctx_to_idx[ctx] = torch.tensor(targets, dtype=torch.long)
        ctx_to_val[ctx] = torch.tensor(boosts, dtype=torch.float32)

    # Build attack LM
    alm = AttackLM(
        MODEL_ID,
        delta=DELTA_ATT,
        window_size=WINDOW_SIZE,
        ctx_to_idx=ctx_to_idx,
        ctx_to_val=ctx_to_val,
    )

    # Generate a few samples
    raw_prompts = build_prompts(NUM_PROMPTS)
    templated_prompts = apply_chat_template(alm.tokenizer, raw_prompts)
    samples = alm.generate(templated_prompts, max_new_tokens=MAX_NEW_TOKENS)

    del alm

    wlm = WatermarkedLM(MODEL_ID, self_hash=True)

    success_count = 0
    z_sum = 0.0

    for i, text in enumerate(samples):
        is_wm, z = wlm.detect_watermark(text)
        print(f"[WATERMARKED {i:02d}] z={z:.2f} flagged={is_wm}")

        success_count += int(is_wm)
        z_sum += z

    success_rate = success_count / len(samples)
    z_avg = z_sum / len(samples) if samples else 0.0
    print(
        f"\nAttack success rate: {success_rate * 100:.2f}% ({success_count}/{len(samples)})"
    )
    print(f"Average z-score: {z_avg:.2f}")


    wandb.log({
        "success_rate": success_rate,
        "z_avg": z_avg,
    })
    wandb.finish()
