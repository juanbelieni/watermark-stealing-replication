test:
  uv run python -m scripts.test_watermark

web:
  uv run streamlit run web/app.py

generate_samples_base:
  uv run python -m scripts.generate_samples --mode=base

generate_samples_wm:
  uv run python -m scripts.generate_samples --mode=watermarked

generate_samples_wm_sh:
  uv run python -m scripts.generate_samples --mode=watermarked --self_hash

extract_examples:
  uv run python -m scripts.mark_green_red_tokens  \
    --base data/samples_meta_llama_llama_3.2_3b_instruct_base_20250826T165913Z.json  \
    --watermarked data/samples_meta_llama_llama_3.2_3b_instruct_watermarked_20250827T133902Z.json \
    --out data/green_red_examples.json \
    --num 5 --completion-index 0 --display text
