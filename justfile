test:
  uv run python -m scripts.test_watermark

generate_samples_base:
  uv run python -m scripts.generate_samples --mode=base

generate_samples_wm:
  uv run python -m scripts.generate_samples --mode=watermarked

generate_samples_wm_sh:
  uv run python -m scripts.generate_samples --mode=watermarked --self_hash
