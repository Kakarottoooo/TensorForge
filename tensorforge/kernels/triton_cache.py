# mypy: disable-error-code="import-not-found,import-untyped,no-untyped-def,untyped-decorator"
"""Capture-safe masked writes into paged KV storage."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None

_PAGED_KV_WRITE_KERNEL: Any = None


if triton is not None:

    @triton.jit
    def _paged_kv_write_kernel(
        keys_ptr,
        values_ptr,
        key_cache_ptr,
        value_cache_ptr,
        physical_blocks_ptr,
        block_offsets_ptr,
        active_mask_ptr,
        source_batch_stride,
        source_head_stride,
        cache_block_stride,
        cache_token_stride,
        cache_head_stride,
        NUM_KV_HEADS: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        BLOCK_HEAD_DIM: tl.constexpr,
    ):
        program = tl.program_id(axis=0)
        batch_index = program // NUM_KV_HEADS
        kv_head = program % NUM_KV_HEADS
        active = tl.load(active_mask_ptr + batch_index) != 0
        dimensions = tl.arange(0, BLOCK_HEAD_DIM)
        mask = active & (dimensions < HEAD_DIM)
        source_offsets = (
            batch_index * source_batch_stride
            + kv_head * source_head_stride
            + dimensions
        )
        physical_block = tl.load(physical_blocks_ptr + batch_index)
        block_offset = tl.load(block_offsets_ptr + batch_index)
        cache_offsets = (
            physical_block * cache_block_stride
            + block_offset * cache_token_stride
            + kv_head * cache_head_stride
            + dimensions
        )
        keys = tl.load(keys_ptr + source_offsets, mask=mask, other=0.0)
        values = tl.load(values_ptr + source_offsets, mask=mask, other=0.0)
        tl.store(key_cache_ptr + cache_offsets, keys, mask=mask)
        tl.store(value_cache_ptr + cache_offsets, values, mask=mask)

    _PAGED_KV_WRITE_KERNEL = _paged_kv_write_kernel


def paged_kv_write_token(
    keys: Tensor,
    values: Tensor,
    key_cache: Tensor,
    value_cache: Tensor,
    physical_blocks: Tensor,
    block_offsets: Tensor,
    active_mask: Tensor,
) -> None:
    """Write at most one KV token per batch lane using device-side destinations."""

    if triton is None:
        raise RuntimeError("Triton is unavailable; install the kernels extra on Linux")
    if keys.ndim != 3 or values.shape != keys.shape:
        raise ValueError("keys and values must share [batch, kv_heads, head_dim]")
    if key_cache.ndim != 4 or value_cache.shape != key_cache.shape:
        raise ValueError("cache tensors must share [blocks, block_size, kv_heads, head_dim]")
    if keys.shape[1:] != key_cache.shape[2:]:
        raise ValueError("source and cache KV head dimensions must match")
    tensors = (
        keys,
        values,
        key_cache,
        value_cache,
        physical_blocks,
        block_offsets,
        active_mask,
    )
    if any(not tensor.is_cuda for tensor in tensors):
        raise ValueError("paged KV writes require CUDA tensors")
    if any(not tensor.is_contiguous() for tensor in tensors):
        raise ValueError("paged KV writes require contiguous tensors")
    if (
        values.dtype != keys.dtype
        or key_cache.dtype != keys.dtype
        or value_cache.dtype != keys.dtype
    ):
        raise ValueError("source and cache tensors must share dtype")
    batch = keys.shape[0]
    for name, tensor in (
        ("physical_blocks", physical_blocks),
        ("block_offsets", block_offsets),
        ("active_mask", active_mask),
    ):
        if tensor.shape != (batch,) or tensor.dtype != torch.int32:
            raise ValueError(f"{name} must be int32 with shape [batch]")
    kv_heads, head_dim = keys.shape[1:]
    head_bucket = triton.next_power_of_2(head_dim)
    _PAGED_KV_WRITE_KERNEL[(batch * kv_heads,)](
        keys,
        values,
        key_cache,
        value_cache,
        physical_blocks,
        block_offsets,
        active_mask,
        keys.stride(0),
        keys.stride(1),
        key_cache.stride(0),
        key_cache.stride(1),
        key_cache.stride(2),
        NUM_KV_HEADS=kv_heads,
        HEAD_DIM=head_dim,
        BLOCK_HEAD_DIM=head_bucket,
        num_warps=4,
    )
