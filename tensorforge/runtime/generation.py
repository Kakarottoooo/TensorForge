"""Correctness-first autoregressive generation without a KV cache."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True, slots=True)
class GenerationConfig:
    """Controls deterministic greedy generation."""

    max_new_tokens: int
    eos_token_id: int | None = None

    def __post_init__(self) -> None:
        if self.max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")
        if self.eos_token_id is not None and self.eos_token_id < 0:
            raise ValueError("eos_token_id must be non-negative")


@torch.inference_mode()
def greedy_generate(model: nn.Module, input_ids: Tensor, config: GenerationConfig) -> Tensor:
    """Generate tokens by recomputing the full prefix on every decode step.

    This O(sequence^2) path is intentionally slow and becomes the correctness
    oracle for the paged KV-cache runtime implemented in Phase 4.
    """

    if input_ids.ndim != 2 or input_ids.shape[1] == 0:
        raise ValueError("input_ids must have shape [batch, non-empty sequence]")
    if input_ids.dtype != torch.long:
        raise TypeError("input_ids must use torch.long")

    generated = input_ids.clone()
    finished = torch.zeros(input_ids.shape[0], device=input_ids.device, dtype=torch.bool)

    for _ in range(config.max_new_tokens):
        logits = model(generated)
        next_tokens = logits[:, -1, :].argmax(dim=-1)
        if config.eos_token_id is not None:
            eos = torch.full_like(next_tokens, config.eos_token_id)
            next_tokens = torch.where(finished, eos, next_tokens)
            finished |= next_tokens.eq(config.eos_token_id)
        generated = torch.cat((generated, next_tokens[:, None]), dim=1)
        if finished.all():
            break

    return generated
