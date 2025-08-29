#!/usr/bin/env python3
"""
Produce green/red token annotations for base and watermarked completions.

Given a pair of data files produced by scripts/generate_samples.py
(`mode=base` and `mode=watermarked`), this script:

- Loads the watermarked config (gamma/hash_key/window_size/self_hash)
- Reconstructs the chat-formatted prompt used for generation
- For a subset of prompts, selects one completion index (default 0)
- Tokenizes using the model tokenizer and, step-by-step, computes whether
  each generated token falls in the watermark green-list for that step
  (using the same KGW green-list definition as used during generation)
- Emits a JSON array with the requested structure

Example:

python scripts/mark_green_red_tokens.py \
  --base data/samples_meta_llama_llama_3.2_3b_instruct_base_20250826T165913Z.json \
  --watermarked data/samples_meta_llama_llama_3.2_3b_instruct_watermarked_20250827T133902Z.json \
  --num 5 --completion-index 0 --out data/green_red_examples.json

"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from transformers import AutoTokenizer, PreTrainedTokenizer

from src.kgw import KGWConfig, KGWLogitsProcessor


def load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def apply_chat_template(tokenizer: PreTrainedTokenizer, prompt: str) -> str:
    """Rebuild the exact chat-formatted prompt used in generation."""
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        tokenize=False,
    )


@torch.inference_mode()
def classify_completion_ids(
    tokenizer: PreTrainedTokenizer,
    model,  # AutoModelForCausalLM or compatible
    kgw: KGWLogitsProcessor,
    prompt_text: str,
    completion_text: str,
) -> Tuple[List[int], List[bool]]:
    """
    Return (tokens, green_flags) for the generated portion only.

    The green/red classification follows the KGW green-list for each step using
    the full context (prompt + previously generated tokens). This matches the
    generation-time setup of the logits processor.
    """
    # Encode full sequence and prompt alone (no special tokens)
    enc_prompt = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)
    enc_full = tokenizer(
        prompt_text + completion_text, return_tensors="pt", add_special_tokens=False
    )

    input_ids_prompt = enc_prompt["input_ids"][0].to(model.device)
    input_ids_full = enc_full["input_ids"][0].to(model.device)

    # Generated token ids
    gen_ids = input_ids_full[len(input_ids_prompt) :]
    if gen_ids.numel() == 0:
        return [], []

    # One forward pass to get logits for every position
    logits = model(input_ids_full.unsqueeze(0)).logits[0]  # [T, V]

    green_flags: List[bool] = []
    prompt_len = input_ids_prompt.size(0)

    # For j-th generated token at absolute pos (prompt_len + j)
    # use scores at previous position (prompt_len + j - 1)
    for j in range(gen_ids.size(0)):
        step_pos = prompt_len + j
        prev_scores = logits[step_pos - 1]
        prefix_ids = input_ids_full[:step_pos]
        green_ids = kgw.get_green_ids(prefix_ids, prev_scores)
        is_green = int(gen_ids[j].item()) in set(green_ids.tolist())
        green_flags.append(bool(is_green))

    return [int(t) for t in gen_ids.tolist()], green_flags


def ids_to_tokens(
    tokenizer: PreTrainedTokenizer, ids: List[int], display: str = "text"
) -> List[str]:
    """Convert token ids to strings.

    - display='internal': return tokenizer vocabulary pieces (may include markers like Ġ/Ċ)
    - display='text': decode each id to its textual piece with spaces/newlines rendered normally
    """
    if not ids:
        return []
    if display == "internal":
        return tokenizer.convert_ids_to_tokens(ids)
    # display == 'text'
    return [
        tokenizer.decode([tid], skip_special_tokens=False, clean_up_tokenization_spaces=False)
        for tid in ids
    ]


def build_kgw_from_params(tokenizer: PreTrainedTokenizer, wm_params: Dict[str, Any]) -> KGWLogitsProcessor:
    cfg = KGWConfig(
        gamma=float(wm_params.get("gamma", 0.5)),
        delta=float(wm_params.get("delta", 2.0)),
        hash_key=int(wm_params.get("hash_key", 42)),
        window_size=int(wm_params.get("window_size", 3)),
        vocab_size=int(wm_params.get("vocab_size", tokenizer.vocab_size)),
        self_hash=bool(wm_params.get("self_hash", True)),
    )
    return KGWLogitsProcessor(cfg)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", type=Path, required=True, help="Path to base JSON file")
    ap.add_argument(
        "--watermarked", type=Path, required=True, help="Path to watermarked JSON file"
    )
    ap.add_argument("--out", type=Path, default=Path("data/green_red_examples.json"))
    ap.add_argument("--num", type=int, default=10, help="Number of prompts to process")
    ap.add_argument(
        "--completion-index",
        type=int,
        default=0,
        help="Which completion index to annotate (0-based)",
    )
    ap.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device map for model (e.g., 'auto', 'cpu', 'cuda')",
    )
    ap.add_argument(
        "--display",
        type=str,
        default="text",
        choices=["text", "internal"],
        help="How to render tokens in output (default: text)",
    )
    args = ap.parse_args()

    base_data = load_json(args.base)
    wm_data = load_json(args.watermarked)

    # Basic consistency checks
    assert base_data.get("mode") == "base", "Base file must have mode=base"
    assert wm_data.get("mode") == "watermarked", "WM file must have mode=watermarked"
    assert base_data.get("model") == wm_data.get("model"), "Model mismatch"

    model_id: str = str(wm_data["model"])

    # Tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token

    # Load model with the requested device mapping
    from transformers import AutoModelForCausalLM

    # Choose a safe default dtype for the current device
    use_cuda = torch.cuda.is_available() and (args.device != "cpu")
    dtype = torch.bfloat16 if use_cuda else torch.float32

    # Try loading with flash attention, fallback to default if unsupported
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map=args.device,
            attn_implementation="flash_attention_2",
        )
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map=args.device,
        )
    model.eval()

    wm_params = wm_data.get("watermark_params") or {}
    kgw = build_kgw_from_params(tokenizer, wm_params)

    per_prompt_base = int(base_data["generation_params"].get("per_prompt", 1))
    per_prompt_wm = int(wm_data["generation_params"].get("per_prompt", 1))
    assert (
        per_prompt_base == per_prompt_wm
    ), f"per_prompt mismatch: base={per_prompt_base}, wm={per_prompt_wm}"

    idx = int(args.completion_index)
    assert 0 <= idx < per_prompt_base, f"completion-index out of range [0,{per_prompt_base})"

    base_samples = base_data.get("samples", [])
    wm_samples = wm_data.get("samples", [])
    n = min(args.num, len(base_samples), len(wm_samples))

    results: List[Dict[str, Any]] = []

    for i in range(n):
        base_sample = base_samples[i]
        wm_sample = wm_samples[i]
        prompt: str = wm_sample["prompt"]

        # Sanity: prompts should match
        if base_sample["prompt"] != prompt:
            # Fall back to wm prompt
            pass

        # Rebuild chat-templated prompt
        templated_prompt = apply_chat_template(tokenizer, prompt)

        # Get chosen completion for each file (skip if missing)
        try:
            base_text = base_sample["completions"][idx]
            wm_text = wm_sample["completions"][idx]
        except Exception:
            continue

        # Classify tokens for base and watermarked completions with the same KGW params
        base_ids, base_green = classify_completion_ids(
            tokenizer, model, kgw, templated_prompt, base_text
        )
        wm_ids, wm_green = classify_completion_ids(
            tokenizer, model, kgw, templated_prompt, wm_text
        )

        base_tokens = ids_to_tokens(tokenizer, base_ids, display=args.display)
        wm_tokens = ids_to_tokens(tokenizer, wm_ids, display=args.display)

        results.append(
            {
                "prompt": prompt,
                "base": {"tokens": base_tokens, "green": base_green},
                "watermarked": {"tokens": wm_tokens, "green": wm_green},
            }
        )

    # Write output
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(results)} annotated examples to {args.out}")


if __name__ == "__main__":
    main()
