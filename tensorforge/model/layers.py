"""Individual transformer operators used by the correctness baseline."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from tensorforge.model.config import ModelConfig


class RMSNorm(nn.Module):
    """Root-mean-square normalization with FP32 accumulation.

    The explicit implementation is the reference oracle for the later Triton
    kernel. Accumulating squares in FP32 avoids excessive error in FP16/BF16.
    """

    def __init__(self, hidden_size: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states: Tensor) -> Tensor:
        input_dtype = hidden_states.dtype
        variance = hidden_states.float().pow(2).mean(dim=-1, keepdim=True)
        normalized = hidden_states.float() * torch.rsqrt(variance + self.eps)
        return (normalized * self.weight.float()).to(input_dtype)


class RotaryEmbedding(nn.Module):
    """Rotary position embedding generated for exactly the requested sequence."""

    def __init__(self, head_dim: int, theta: float) -> None:
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, positions: Tensor, *, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        frequencies = torch.outer(positions.float(), self.inv_freq)
        embeddings = torch.cat((frequencies, frequencies), dim=-1)
        return embeddings.cos().to(dtype), embeddings.sin().to(dtype)


def _rotate_half(x: Tensor) -> Tensor:
    first, second = x.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def apply_rotary_embedding(q: Tensor, k: Tensor, cos: Tensor, sin: Tensor) -> tuple[Tensor, Tensor]:
    """Apply RoPE to tensors shaped ``[batch, heads, sequence, head_dim]``."""

    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    return (q * cos) + (_rotate_half(q) * sin), (k * cos) + (_rotate_half(k) * sin)


class CausalSelfAttention(nn.Module):
    """Transparent grouped-query causal attention baseline.

    This intentionally uses explicit matmul, masking, and softmax instead of
    SDPA. It provides an inspectable reference and profiler baseline before
    optimized attention is introduced in later phases.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(
            config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.rotary = RotaryEmbedding(self.head_dim, config.rope_theta)

    def _shape(self, tensor: Tensor, heads: int) -> Tensor:
        batch, sequence, _ = tensor.shape
        return tensor.view(batch, sequence, heads, self.head_dim).transpose(1, 2)

    def forward(self, hidden_states: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        batch, sequence, _ = hidden_states.shape
        query = self._shape(self.q_proj(hidden_states), self.num_heads)
        key = self._shape(self.k_proj(hidden_states), self.num_key_value_heads)
        value = self._shape(self.v_proj(hidden_states), self.num_key_value_heads)

        positions = torch.arange(sequence, device=hidden_states.device)
        cos, sin = self.rotary(positions, dtype=query.dtype)
        query, key = apply_rotary_embedding(query, key, cos, sin)

        if self.num_key_value_groups > 1:
            key = key.repeat_interleave(self.num_key_value_groups, dim=1)
            value = value.repeat_interleave(self.num_key_value_groups, dim=1)

        scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        causal_mask = torch.ones(
            (sequence, sequence), device=hidden_states.device, dtype=torch.bool
        ).triu(diagonal=1)
        scores = scores.masked_fill(causal_mask, torch.finfo(scores.dtype).min)

        if attention_mask is not None:
            if attention_mask.shape != (batch, sequence):
                raise ValueError(
                    f"attention_mask must have shape {(batch, sequence)}, "
                    f"got {tuple(attention_mask.shape)}"
                )
            key_padding_mask = ~attention_mask.to(device=hidden_states.device, dtype=torch.bool)
            scores = scores.masked_fill(
                key_padding_mask[:, None, None, :], torch.finfo(scores.dtype).min
            )

        probabilities = F.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
        context = torch.matmul(probabilities, value)
        context = context.transpose(1, 2).contiguous().view(batch, sequence, -1)
        return self.o_proj(context)


class SwiGLU(nn.Module):
    """Llama-style gated MLP: down(silu(gate(x)) * up(x))."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, hidden_states: Tensor) -> Tensor:
        return self.down_proj(F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


class TransformerBlock(nn.Module):
    """Pre-normalized decoder block with explicit residual paths."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.input_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.attention = CausalSelfAttention(config)
        self.post_attention_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = SwiGLU(config)

    def forward(self, hidden_states: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        hidden_states = hidden_states + self.attention(
            self.input_norm(hidden_states), attention_mask=attention_mask
        )
        return hidden_states + self.mlp(self.post_attention_norm(hidden_states))


def initialize_weights(module: nn.Module) -> None:
    """Initialize all trainable matrices consistently for reproducible tests."""

    if isinstance(module, nn.Linear | nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
