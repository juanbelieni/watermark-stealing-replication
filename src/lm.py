from typing import List, Optional, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer
from transformers.generation.logits_process import LogitsProcessorList


class LM:
    """
    Thin wrapper around a Hugging Face causal LM that exposes a
    `logits_processor` hook passed directly to `transformers.generate`.

    The `logits_processor` can be set/overridden by subclasses to inject
    watermarking or other per-step biases without reimplementing generation.
    The `generate` method accepts a list of prompts and returns a list of
    completion strings (prompt stripped).
    """

    def __init__(self, model_name_or_path: str) -> None:
        """
        Initialize and load a Hugging Face causal LM on CUDA with bfloat16.
        """
        # Tokenizer
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            use_fast=True,
            padding_side="left",
        )

        # Model in bfloat16
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.bfloat16,
            device_map=torch.device("cuda"),
        )
        self.model.eval()

        if self.tokenizer.pad_token_id is None:
            # Ensure a pad token exists for batching; fall back to eos when needed.
            self.tokenizer.pad_token = (
                self.tokenizer.eos_token or self.tokenizer.unk_token
            )

        # Optional custom logits processor to be forwarded to `generate`.
        # By default, no extra processor is used.
        self.logits_processor: Optional[LogitsProcessorList] = None

    # ---- Generation ----------------------------------------------------------------
    @torch.inference_mode()
    def generate(
        self,
        prompts: Sequence[str],
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 0,
        repetition_penalty: float = 1.0,
        eos_token_id: Optional[int] = None,
    ) -> List[str]:
        """
        Generate completions for a batch of prompts.

        Returns a list of strings containing only the newly generated text
        (without the original prompt).
        """
        if not prompts:
            return []

        # Resolve EOS and PAD ids
        effective_eos = (
            eos_token_id
            if eos_token_id is not None
            else (
                self.tokenizer.eos_token_id
                if self.tokenizer.eos_token_id is not None
                else None
            )
        )
        pad_id = self.tokenizer.pad_token_id

        enc = self.tokenizer(
            list(prompts),
            return_tensors="pt",
            padding=True,
            truncation=False,
        ).to(self.model.device)

        input_ids: torch.LongTensor = enc["input_ids"]
        attention_mask: torch.LongTensor = enc["attention_mask"]

        # Use HF generation with optional custom logits processor
        gen_ids = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            eos_token_id=effective_eos,
            pad_token_id=pad_id,
            logits_processor=self.logits_processor,
        )

        # Slice off the prompts to return only newly generated text
        prompt_lengths = attention_mask.sum(dim=1).tolist()
        completions: List[str] = []
        gen_ids = gen_ids.detach().cpu()
        for i, start in enumerate(prompt_lengths):
            seq = gen_ids[i].tolist()
            gen_tokens = seq[int(start) :]
            if effective_eos is not None and effective_eos in gen_tokens:
                eos_pos = gen_tokens.index(effective_eos)
                gen_tokens = gen_tokens[:eos_pos]
            text = self.tokenizer.decode(
                gen_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )
            completions.append(text)

        return completions
