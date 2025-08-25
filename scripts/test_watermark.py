#!/usr/bin/env python3


from src.kgw import WatermarkedLM

MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"

# Generation settings
NUM_PROMPTS = 10
MAX_NEW_TOKENS = 256


def build_prompts(n: int) -> list[str]:
    topics = [
        "astronomy",
        "medieval history",
        "quantum computing",
        "ecology",
        "culinary arts",
        "classical music",
        "urban planning",
        "health and fitness",
        "machine learning",
        "travel tips",
        "photography",
        "gardening",
    ]
    topics = topics[: max(1, n)]
    return [f"Write a concise paragraph explaining basics of {t}." for t in topics[:n]]


if __name__ == "__main__":
    lm = WatermarkedLM(MODEL_ID)

    # Prepare templated prompts for chat-style models
    raw_prompts = build_prompts(NUM_PROMPTS)
    templated_prompts = [
        lm.tokenizer.apply_chat_template(
            [{"role": "user", "content": p}],
            add_generation_prompt=True,
            tokenize=False,
        )
        for p in raw_prompts
    ]

    # Generate plain (unwatermarked) completions by temporarily disabling the logits processor
    original_processor = lm.logits_processor
    lm.logits_processor = None
    plain_samples = lm.generate(
        templated_prompts,
        max_new_tokens=MAX_NEW_TOKENS,
    )
    lm.logits_processor = original_processor  # restore watermarking

    # Generate watermarked completions
    watermarked_samples = lm.generate(
        templated_prompts,
        max_new_tokens=MAX_NEW_TOKENS,
    )

    assert len(plain_samples) >= 10 and len(watermarked_samples) >= 10

    # Detect on plain completions (expect False)
    plain_flags = []
    for i, text in enumerate(plain_samples):
        is_wm, z = lm.detect_watermark(text)
        plain_flags.append(is_wm)
        print(f"[PLAIN {i:02d}] z={z:.2f} flagged={is_wm}")

    # Detect on watermarked completions (expect True)
    wm_flags = []
    for i, text in enumerate(watermarked_samples):
        is_wm, z = lm.detect_watermark(text)
        wm_flags.append(is_wm)
        print(f"[WATERMARKED {i:02d}] z={z:.2f} flagged={is_wm}")

    assert all(wm_flags), "Expected all watermarked generations to be detected"
    assert not all(plain_flags), "Expected some plain generations to not be detected"

    print("All watermark detection assertions passed.")
