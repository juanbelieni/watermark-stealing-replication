#!/usr/bin/env python3
"""
Generate a dataset of samples on the TRL UltraFeedback prompts (test split),
both watermarked and not-watermarked (plain).

Features
- Selects the first 1000 prompts from `trl-lib/ultrafeedback-prompt` test split
- Generates 10 sampled completions per prompt
- Saves CSV rows as: prompt,completion
- Resumable: if stopped, re-run continues from where it left off by reading the
  existing CSV and only generating the missing completions per prompt.

Usage examples
- python scripts/generate_samples.py --mode both
- python scripts/generate_samples.py --mode plain --limit 200

Environment
- This script expects access to a GPU by default (see src/lm.py). Ensure your
  environment can load the model on CUDA and that you've accepted HF terms for
  the selected model. If gated, set HUGGINGFACE_HUB_TOKEN.
"""

import argparse
import pandas as pd
from collections import Counter
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm

from src.lm import LM
from src.kgw import WatermarkedLM

DATASET_ID = "trl-lib/ultrafeedback-prompt"
DEFAULT_MODEL = "meta-llama/Llama-3.2-3B-Instruct"

def generate_for_mode(
    mode: str,
    model_name: str,
    prompts: list[str],
    out_csv: Path,
    per_prompt: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
) -> None:
    print(f"Generating {mode} samples to {out_csv}...")

    if out_csv.exists():
        df = pd.read_csv(out_csv)
        counts = Counter(df["prompt"].to_list())
    else:
        counts = Counter()

    if mode == "watermarked":
        lm = WatermarkedLM(model_name)
    elif mode == "plain":
        lm = LM(model_name)
    else:
        raise ValueError("mode must be 'watermarked' or 'plain'")


    for prompt in tqdm(prompts):
        done = counts.get(prompt, 0)
        remaining = per_prompt - done
        if remaining <= 0:
            continue

        # Generate remaining completions for this prompt
        templated_prompt = lm.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            tokenize=False,
        )
        batch_prompts = [templated_prompt] * remaining
        completions = lm.generate(
            batch_prompts,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
        )

        # Append to CSV
        new_rows = [(prompt, comp) for comp in completions]
        new_df = pd.DataFrame(new_rows, columns=["prompt", "completion"])
        if out_csv.exists():
            new_df.to_csv(out_csv, mode="a", header=False, index=False)
        else:
            new_df.to_csv(out_csv, index=False)

    del lm


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help="HF model id or path.",
    )
    ap.add_argument(
        "--mode",
        type=str,
        choices=["plain", "watermarked", "both"],
        default="both",
        help="Generation mode: plain, watermarked, or both",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=1000,
        help="Number of prompts from test split to use.",
    )
    ap.add_argument(
        "--per-prompt",
        type=int,
        default=10,
        help="Number of completions per prompt.",
    )
    ap.add_argument(
        "--max-new-tokens",
        type=int,
        default=200,
        help="Max new tokens per completion.",
    )
    ap.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature.",
    )
    ap.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Nucleus sampling p.",
    )
    ap.add_argument(
        "--top-k",
        type=int,
        default=0,
        help="Top-k sampling (0 to disable)",
    )
    ap.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.0,
        help="Repetition penalty",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data"),
        help="Directory to write CSVs",
    )
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset: {DATASET_ID} [test split]")
    ds = load_dataset(DATASET_ID, split="test").select(range(args.limit))
    prompts = [prompt[0]["content"] for prompt in ds["prompt"]]
    print(f"Loaded {len(prompts)} prompts.")

    # Filenames include mode for clarity
    model_name_sanitized = args.model.replace("/", "_").replace("-", "_").lower()
    base_name = f"samples_{model_name_sanitized}"
    plain_csv = args.out_dir / f"{base_name}_plain.csv"
    wm_csv = args.out_dir / f"{base_name}_watermarked.csv"

    if args.mode in ("plain", "both"):
        generate_for_mode(
            mode="plain",
            model_name=args.model,
            prompts=prompts,
            out_csv=plain_csv,
            per_prompt=args.per_prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            repetition_penalty=args.repetition_penalty,
        )

    if args.mode in ("watermarked", "both"):
        generate_for_mode(
            mode="watermarked",
            model_name=args.model,
            prompts=prompts,
            out_csv=wm_csv,
            per_prompt=args.per_prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            repetition_penalty=args.repetition_penalty,
        )

    print("Done.")
