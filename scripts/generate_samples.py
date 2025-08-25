#!/usr/bin/env python3
"""
Generate a dataset of samples on the TRL UltraFeedback prompts (test split),
both watermarked and not-watermarked (base), and emit a JSON file per mode
with all generated samples alongside the generation and watermark
configuration. Filenames include a timestamp.

Features
- Selects the first `args.limit` prompts from `trl-lib/ultrafeedback-prompt` test split
- Generates `args.per_prompt` sampled completions per prompt
- Outputs one JSON per mode with:
  - run metadata (dataset, model, timestamp)
  - generation params
  - watermark params (if applicable)
  - samples: for each prompt, a list of completions

Usage examples
- python scripts/generate_samples.py --mode both
- python scripts/generate_samples.py --mode base --limit 200

Environment
- This script expects access to a GPU by default (see src/lm.py). Ensure your
  environment can load the model on CUDA and that you've accepted HF terms for
  the selected model. If gated, set HUGGINGFACE_HUB_TOKEN.
"""

import argparse
import json
from dataclasses import asdict
from datetime import datetime, timezone
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
    per_prompt: int,
    max_new_tokens: int,
    temperature: float,
) -> tuple[list, dict | None]:
    print(f"Generating {mode} samples...")

    if mode == "watermarked":
        lm = WatermarkedLM(model_name)
        watermark_config = asdict(lm.kgw.config)
    elif mode == "base":
        lm = LM(model_name)
        watermark_config = None
    else:
        raise ValueError("mode must be 'watermarked' or 'base'")

    samples: list[dict] = []

    for prompt in tqdm(prompts):
        remaining = per_prompt

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
        )

        samples.append(
            {
                "prompt": prompt,
                "completions": completions,
            }
        )

    return samples, watermark_config


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
        choices=["base", "watermarked", "both"],
        default="both",
        help="Generation mode: base, watermarked, or both",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=1500,
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
        default=800,
        help="Max new tokens per completion.",
    )
    ap.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature.",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data"),
        help="Directory to write output JSON files.",
    )
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset: {DATASET_ID} [test split]")
    ds = load_dataset(DATASET_ID, split="test").select(range(args.limit))
    prompts = [prompt[0]["content"] for prompt in ds["prompt"]]
    print(f"Loaded {len(prompts)} prompts.")

    # Filenames include model for clarity
    model_name_sanitized = args.model.replace("/", "_").replace("-", "_").lower()
    base_name = f"samples_{model_name_sanitized}"
    # Timestamp for filenames and metadata
    ts_file = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    ts_iso = datetime.now(timezone.utc).isoformat() + "Z"

    if args.mode in ("base", "both"):
        samples, _ = generate_for_mode(
            mode="base",
            model_name=args.model,
            prompts=prompts,
            per_prompt=args.per_prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )

        output = {
            "dataset_id": DATASET_ID,
            "split": "test",
            "limit": args.limit,
            "model": args.model,
            "timestamp": ts_iso,
            "mode": "base",
            "generation_params": {
                "per_prompt": args.per_prompt,
                "max_new_tokens": args.max_new_tokens,
                "temperature": args.temperature,
            },
            "samples": samples,
        }

        base_path = args.out_dir / f"{base_name}_base_{ts_file}.json"

        with open(base_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

        print(f"Wrote base samples to {base_path}")

    if args.mode in ("watermarked", "both"):
        samples, watermark_config = generate_for_mode(
            mode="watermarked",
            model_name=args.model,
            prompts=prompts,
            per_prompt=args.per_prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )

        output = {
            "dataset_id": DATASET_ID,
            "split": "test",
            "limit": args.limit,
            "model": args.model,
            "timestamp": ts_iso,
            "mode": "watermarked",
            "generation_params": {
                "per_prompt": args.per_prompt,
                "max_new_tokens": args.max_new_tokens,
                "temperature": args.temperature,
            },
            "samples": samples,
        }

        output["watermark_params"] = watermark_config

        wm_path = args.out_dir / f"{base_name}_watermarked_{ts_file}.json"

        with open(wm_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

        print(f"Wrote watermarked samples to {wm_path}")

    print("Done.")
