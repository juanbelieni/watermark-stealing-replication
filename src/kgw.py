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


# ---- 32-bit mix on int64 (CUDA-safe) ----
def _mix32(x: torch.Tensor) -> torch.Tensor:
    # x: int64 tensor (CPU or CUDA)
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
    # map 32-bit state -> [0,1)
    return (h & 0xFFFFFFFF).to(torch.float32) / 4294967296.0


# PRF(seed, token, key) -> u in [0,1)
def _prf_u01_gpu(
    seed_i64: torch.Tensor, token_i64: torch.Tensor, key_i64: torch.Tensor
) -> torch.Tensor:
    # all int64, broadcastable
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

            gen = torch.Generator().manual_seed(seed)

            green_count = max(1, int(self.config.vocab_size * self.config.gamma))
            perm = torch.randperm(self.config.vocab_size, generator=gen)

            green_ids = perm[:green_count].to(input_ids.device)
        else:
            hash_key = torch.tensor(self.config.hash_key, dtype=torch.long)
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
    ) -> None:
        super().__init__(model_name_or_path)

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
        threshold: float = 2.0,  # Threshold for z-score (default is 2.0, which corresponds to a 95% confidence level)
    ) -> tuple[bool, float]:
        enc = self.tokenizer(
            text,
            return_tensors="pt",
            add_special_tokens=False,
        ).to(self.model.device)
        input_ids: torch.LongTensor = enc["input_ids"][0]

        s = 0  # number of green tokens observed
        n = input_ids.size(0)

        if n == 0:
            return (False, 0.0)

        scores: torch.FloatTensor = self.model(input_ids.unsqueeze(0)).logits[0]

        for i in range(1, n):
            curr_input_ids = input_ids[:i]
            curr_scores = scores[i - 1]
            token_id = input_ids[i].item()

            green_ids: torch.LongTensor = self.kgw.get_green_ids(
                curr_input_ids,
                curr_scores,
            )

            # Check if current token is in the green list
            if token_id in green_ids.tolist():
                s += 1

        # Formula: z = (s - n * gamma) / sqrt(n * gamma * (1 - gamma))
        gamma = self.kgw.config.gamma
        z = (s - n * gamma) / ((n * gamma * (1 - gamma)) ** 0.5)

        # print(f"Detected {s} green tokens out of {n} total tokens. z = {z:.2f}")

        return (z > threshold, z)
