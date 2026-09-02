"""Numerical oracles for inference-only fused kernels."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F


def rms_norm_reference(inputs: Tensor, weight: Tensor, eps: float) -> Tensor:
    """Explicit RMSNorm with FP32 reduction and multiplication."""

    variance = inputs.float().square().mean(dim=-1, keepdim=True)
    normalized = inputs.float() * torch.rsqrt(variance + eps)
    return (normalized * weight.float()).to(inputs.dtype)


def torch_rms_norm(inputs: Tensor, weight: Tensor, eps: float) -> Tensor:
    """Standard PyTorch operator used as the performance baseline."""

    return F.rms_norm(inputs, (inputs.shape[-1],), weight, eps)


def residual_rms_norm_reference(
    inputs: Tensor, residual: Tensor, weight: Tensor, eps: float
) -> tuple[Tensor, Tensor]:
    """Return the residual sum and its RMS-normalized representation."""

    residual_output = inputs + residual
    return residual_output, rms_norm_reference(residual_output, weight, eps)


def torch_residual_rms_norm(
    inputs: Tensor, residual: Tensor, weight: Tensor, eps: float
) -> tuple[Tensor, Tensor]:
    residual_output = inputs + residual
    return residual_output, torch_rms_norm(residual_output, weight, eps)


def swiglu_reference(gate: Tensor, up: Tensor) -> Tensor:
    """SwiGLU activation after the two matrix projections."""

    return F.silu(gate) * up


def paged_gqa_decode_reference(
    query: Tensor,
    key_cache: Tensor,
    value_cache: Tensor,
    block_tables: Tensor,
    context_lengths: Tensor,
    scale: float | None = None,
) -> Tensor:
    """Readable single-token GQA over logical sequences stored in physical pages."""

    if query.ndim != 3:
        raise ValueError("query must have shape [batch, query_heads, head_dim]")
    if key_cache.ndim != 4 or value_cache.shape != key_cache.shape:
        raise ValueError("key/value cache must share [blocks, block_size, kv_heads, head_dim]")
    batch, query_heads, head_dim = query.shape
    _, block_size, kv_heads, cache_head_dim = key_cache.shape
    if head_dim != cache_head_dim or query_heads % kv_heads != 0:
        raise ValueError("query/cache head dimensions do not form valid GQA groups")
    if block_tables.ndim != 2 or block_tables.shape[0] != batch:
        raise ValueError("block_tables must have one row per query")
    if context_lengths.shape != (batch,):
        raise ValueError("context_lengths must have one value per query")

    attention_scale = head_dim**-0.5 if scale is None else scale
    kv_indices = torch.arange(query_heads, device=query.device) // (query_heads // kv_heads)
    outputs: list[Tensor] = []
    for batch_index in range(batch):
        context_length = int(context_lengths[batch_index].item())
        if context_length <= 0:
            raise ValueError("context lengths must be positive")
        logical_blocks = (context_length + block_size - 1) // block_size
        if logical_blocks > block_tables.shape[1]:
            raise ValueError("context length exceeds block table capacity")
        key_chunks: list[Tensor] = []
        value_chunks: list[Tensor] = []
        for logical_block in range(logical_blocks):
            physical_block = int(block_tables[batch_index, logical_block].item())
            if not 0 <= physical_block < key_cache.shape[0]:
                raise ValueError("block table contains an invalid physical block")
            valid_tokens = min(block_size, context_length - logical_block * block_size)
            key_chunks.append(key_cache[physical_block, :valid_tokens])
            value_chunks.append(value_cache[physical_block, :valid_tokens])
        keys = torch.cat(key_chunks, dim=0)[:, kv_indices]
        values = torch.cat(value_chunks, dim=0)[:, kv_indices]
        scores = torch.einsum("hd,thd->ht", query[batch_index].float(), keys.float())
        probabilities = F.softmax(scores * attention_scale, dim=-1)
        outputs.append(torch.einsum("ht,thd->hd", probabilities, values.float()))
    return torch.stack(outputs).to(query.dtype)
