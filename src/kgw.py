from dataclasses import dataclass
from typing import Optional, Sequence

import hashlib

import torch
from transformers.generation.logits_process import LogitsProcessor, LogitsProcessorList

from src.lm import LM


@dataclass
class KGWConfig:
    """Configuration for the KGW (SelfHash) watermark.

    - gamma: Fraction of the vocabulary assigned to the green list per step.
    - delta: Logit bias added to green-list tokens.
    - hash_key: Secret integer key used to seed the per-step partition.
    - window_size: Use the last `h` tokens of context for seeding.
    - ignore_token_ids: Tokens that will never be boosted (e.g., special tokens).
    """

    gamma: float
    delta: float
    hash_key: int
    window_size: int
    ignore_token_ids: Optional[set[int]] = None


class KGWLogitsProcessor(LogitsProcessor):
    """KGW/SelfHash watermark logits processor.

    For each decoding step and each sequence in the batch, deterministically
    partitions the vocabulary into green/red sets via a PRF seeded by the
    secret key and the current context (last token id and position). Green-list
    tokens receive a positive logit bias `delta`.
    """

    def __init__(self, config: KGWConfig) -> None:
        super().__init__()
        assert 0.0 < config.gamma <= 1.0, "gamma must be in (0, 1]"
        assert torch.isfinite(torch.tensor(config.delta)), "delta must be finite"
        assert config.window_size >= 1, "window_size must be >= 1"

        self.config = config

    @staticmethod
    def _derive_seed(key: int, window: Sequence[int]) -> int:
        """Derive a 64-bit seed from key and last h tokens via SHA-256.

        The `window` is the slice of the last `h` token ids (or fewer if
        the sequence is shorter). The order matters and is included in the hash.
        """
        payload = key.to_bytes(8, "little", signed=False)
        payload += int(len(window)).to_bytes(4, "little", signed=False)
        for t in window:
            payload += int(t).to_bytes(8, "little", signed=False)

        hash = hashlib.sha256(payload).digest()

        # Take the first 8 bytes as an unsigned 64-bit int
        return int.from_bytes(hash[:8], "little", signed=False)

    def _green_mask_for_batch(
        self,
        input_ids: torch.LongTensor,
        vocab_size: int,
    ) -> torch.BoolTensor:
        """Compute a [batch, vocab] boolean mask of green tokens for this step.

        The mask is deterministic given (key, last_token, position) per batch row.
        """
        batch_size = input_ids.size(0)
        green_count = max(1, int(self.config.gamma * vocab_size))

        mask = torch.zeros((batch_size, vocab_size), dtype=torch.bool)

        ignore_ids: Sequence[int] = (
            tuple(self.config.ignore_token_ids) if self.config.ignore_token_ids else ()
        )

        for i in range(batch_size):
            # Use the last h (window_size) tokens as context hash
            seq = input_ids[i]

            h = max(0, seq.size(0) - self.config.window_size)
            window = seq[h:].tolist()
            seed = self._derive_seed(self.config.hash_key, window)

            gen = torch.Generator().manual_seed(seed)
            # Random permutation to choose the green set indices deterministically
            perm = torch.randperm(vocab_size, generator=gen)
            chosen = perm[:green_count]

            mask[i, chosen] = True

            # Ensure ignored tokens are never marked green
            mask[i, ignore_ids] = False

        return mask

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
    ) -> torch.FloatTensor:
        # scores: [batch, vocab]
        assert scores.dim() == 2, "Expected scores to be [batch, vocab]"
        assert input_ids.device == scores.device, (
            "input_ids and scores must be on the same device"
        )
        batch, vocab_size = scores.shape

        # Build green mask and add delta to green tokens
        mask = self._green_mask_for_batch(input_ids, vocab_size).to(scores.device)

        delta = torch.as_tensor(
            self.config.delta,
            dtype=scores.dtype,
            device=scores.device,
        )

        scores = scores.clone()
        scores[mask] += delta
        return scores


class WatermarkedLM(LM):
    """LM wrapper that applies KGW watermark during generation via logits processing."""

    def __init__(
        self,
        model_name_or_path: str,
        *,
        gamma: float = 0.5,
        delta: float = 2.0,
        hash_key: int = 42,
        window_size: int = 3,
        ignore_special_tokens: bool = True,
    ) -> None:
        super().__init__(model_name_or_path)

        ignore_ids: set[int] = set()
        if ignore_special_tokens:
            if self.tokenizer.pad_token_id is not None:
                ignore_ids.add(int(self.tokenizer.pad_token_id))
            if self.tokenizer.eos_token_id is not None:
                ignore_ids.add(int(self.tokenizer.eos_token_id))
            if self.tokenizer.bos_token_id is not None:
                ignore_ids.add(int(self.tokenizer.bos_token_id))

        config = KGWConfig(
            gamma=gamma,
            delta=delta,
            hash_key=hash_key,
            window_size=window_size,
            ignore_token_ids=ignore_ids or None,
        )

        self.kgw = KGWLogitsProcessor(config)
        self.logits_processor = LogitsProcessorList([self.kgw])

    @torch.inference_mode()
    def detect_watermark(
        self,
        text: str,
        threshold: float = 2.0,  # Threshold for z-score (default is 2.0, which corresponds to a 95% confidence level)
    ) -> tuple[bool, float]:
        """Detects the presence of a KGW watermark in the given text.

        Args:
            text (str): The text to analyze for watermark presence.
            threshold (float): The z-score threshold above which watermark presence is confirmed.

        Returns:
            tuple[bool, float]: A tuple containing a boolean indicating watermark presence
                               and the computed z-score.
        """

        # Tokenize without adding special tokens; we analyze the raw text tokens.
        enc = self.tokenizer(
            text,
            return_tensors="pt",
            add_special_tokens=False,
        ).to(self.model.device)
        input_ids: torch.LongTensor = enc["input_ids"]
        vocab_size: int = self.model.config.vocab_size

        ignore_ids = self.kgw.config.ignore_token_ids or set()
        s = 0  # number of green tokens observed
        n_eff = 0  # effective token count (excluding ignored tokens)

        # Iterate tokens; build the green mask from previously seen tokens.
        n_total = input_ids.size(1)

        for i in range(n_total):
            tok_id = int(input_ids[0, i])
            if tok_id in ignore_ids:
                continue

            mask = self.kgw._green_mask_for_batch(input_ids[:, :i], vocab_size).to(
                input_ids.device
            )

            # Check if current token is in the green list
            if mask[0, tok_id].item():
                s += 1
            n_eff += 1

        # Formula: z = (s - n * gamma) / sqrt(n * gamma * (1 - gamma))
        gamma = self.kgw.config.gamma
        if n_eff == 0:
            z = 0.0
        else:
            z = (s - n_eff * gamma) / ((n_eff * gamma * (1 - gamma)) ** 0.5)

        return (z > threshold, z)
