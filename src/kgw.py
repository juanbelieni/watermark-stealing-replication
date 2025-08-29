"""KGW watermarking components.

This module implements a GPU-friendly KGW greenlist selector (as a
``LogitsProcessor``) and a lightweight ``WatermarkedLM`` wrapper that exposes
generation and watermark detection helpers.
"""

import hashlib
import torch
from dataclasses import dataclass
from transformers.generation.logits_process import LogitsProcessor, LogitsProcessorList

from src.lm import LM


@dataclass
class KGWConfig:
    gamma: float  # Fraction of the vocabulary assigned to the green list per step.
    delta: float  # Logit bias added to green-list tokens.
    hash_key: int
    window_size: int
    vocab_size: int
    self_hash: bool


def _mix32(x: torch.Tensor) -> torch.Tensor:
    """32-bit mix on int64 tensor (CUDA-safe)."""
    device = x.device
    mask32 = torch.tensor(0xFFFFFFFF, dtype=torch.int64, device=device)
    prime = torch.tensor(16777619, dtype=torch.int64, device=device)
    h = torch.tensor(2166136261, dtype=torch.int64, device=device)
    # FNV-1a style: XOR then multiply, masked to 32 bits
    h = (h ^ (x & mask32)) & mask32
    h = (h * prime) & mask32
    # one extra round to improve diffusion
    h = (h ^ (h >> 15)) & mask32
    h = (h * prime) & mask32
    return h  # int64, lower 32 bits carry the state


def _u01_from_mix32(h: torch.Tensor) -> torch.Tensor:
    """Map 32-bit state to [0, 1)."""
    return (h & 0xFFFFFFFF).to(torch.float32) / 4294967296.0


# PRF(seed, token, key) -> u in [0,1)
def _prf_u01_gpu(
    seed_i64: torch.Tensor, token_i64: torch.Tensor, key_i64: torch.Tensor
) -> torch.Tensor:
    """Pseudorandom float in [0, 1) from int64 inputs (broadcastable)."""
    h = _mix32(seed_i64 ^ key_i64)
    h = _mix32(h ^ token_i64)
    return _u01_from_mix32(h)


def _derive_seed(key: int, tokens: list[int]) -> int:
    token_hashes = [
        hashlib.sha256(t.to_bytes(8, "little", signed=False)).digest() for t in tokens
    ]

    token_hash = min(token_hashes)
    payload = token_hash + key.to_bytes(8, "little", signed=False)
    hash = hashlib.sha256(payload).digest()

    return int.from_bytes(hash[:4], "little", signed=False)


class KGWLogitsProcessor(LogitsProcessor):
    def __init__(self, config: KGWConfig) -> None:
        super().__init__()
        assert 0.0 < config.gamma <= 1.0, "gamma must be in (0, 1]"
        assert torch.isfinite(torch.tensor(config.delta)), "delta must be finite"
        assert config.window_size >= 1, "window_size must be >= 1"

        self.config = config

    def get_green_ids(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
    ) -> torch.LongTensor:
        if not self.config.self_hash:
            h = min(self.config.window_size, input_ids.size(0))

            seed = _derive_seed(
                self.config.hash_key,
                input_ids[-h:].tolist(),
            )

            gen = torch.Generator(device=input_ids.device).manual_seed(seed)

            green_count = max(1, int(self.config.vocab_size * self.config.gamma))
            perm = torch.randperm(self.config.vocab_size, generator=gen, device=input_ids.device)

            green_ids = perm[:green_count]
        else:
            # Ensure key lives on the same device as inputs
            hash_key = torch.tensor(
                self.config.hash_key, dtype=torch.long, device=input_ids.device
            )
            candidates = scores.argsort(dim=-1, descending=True)

            h = min(self.config.window_size, input_ids.size(0))
            prev_tokens = input_ids[-h:]
            prev_tokens = prev_tokens ^ hash_key

            prev_tokens_hash = _mix32(prev_tokens[None, :]).min(dim=1).values
            seed = _mix32(prev_tokens_hash ^ candidates)

            u = _prf_u01_gpu(seed, candidates, hash_key)
            u = u.to(candidates.device)
            green_ids = candidates[u < self.config.gamma]

        return green_ids

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
    ) -> torch.FloatTensor:
        batch_size = input_ids.size(0)
        scores = scores.clone()

        # Process each sequence in the batch independently
        for i in range(batch_size):
            green_ids = self.get_green_ids(
                input_ids[i],
                scores[i],
            )

            scores[i, green_ids] += self.config.delta

        return scores


class WatermarkedLM(LM):
    def __init__(
        self,
        model_name_or_path: str,
        *,
        gamma: float = 0.5,
        delta: float = 2.0,
        hash_key: int = 42,
        window_size: int = 3,
        self_hash: bool = True,
        device: str = "auto",
    ) -> None:
        super().__init__(model_name_or_path, device)

        config = KGWConfig(
            gamma=gamma,
            delta=delta,
            hash_key=hash_key,
            window_size=window_size,
            vocab_size=self.tokenizer.vocab_size,
            self_hash=self_hash,
        )

        self.kgw = KGWLogitsProcessor(config)
        self.logits_processor = LogitsProcessorList([self.kgw])

    @torch.inference_mode()
    def detect_watermark(
        self,
        text: str,
        threshold: float = 2.0,
    ) -> tuple[bool, float]:
        enc = self.tokenizer(
            text,
            return_tensors="pt",
            add_special_tokens=False,
        ).to(self.model.device)
        input_ids: torch.LongTensor = enc["input_ids"][0]

        green_count = 0
        num_tokens = input_ids.size(0)

        if num_tokens == 0:
            return (False, 0.0)

        scores: torch.FloatTensor = self.model(input_ids.unsqueeze(0)).logits[0]

        for i in range(1, num_tokens):
            curr_input_ids = input_ids[:i]
            curr_scores = scores[i - 1]
            token_id = input_ids[i].item()

            green_ids: torch.LongTensor = self.kgw.get_green_ids(
                curr_input_ids,
                curr_scores,
            )

            if token_id in green_ids.tolist():
                green_count += 1

        gamma = self.kgw.config.gamma
        z = (green_count - num_tokens * gamma) / (
            (num_tokens * gamma * (1 - gamma)) ** 0.5
        )

        return (z > threshold, z)
