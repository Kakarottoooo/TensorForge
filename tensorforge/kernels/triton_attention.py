# mypy: disable-error-code="import-not-found,import-untyped,no-untyped-def,untyped-decorator"
"""Online-softmax Triton GQA decode attention over a paged KV cache."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None

_PAGED_GQA_DECODE_KERNEL: Any = None
_PAGED_GQA_SPLIT_KERNEL: Any = None
_PAGED_GQA_REDUCE_KERNEL: Any = None
_LAST_ATTENTION_CONFIG: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class PagedAttentionWorkspace:
    """Caller-owned split-KV buffers whose addresses can remain graph-stable."""

    partial_max: Tensor
    partial_sum: Tensor
    partial_accumulator: Tensor

    @classmethod
    def allocate(
        cls,
        *,
        batch_capacity: int,
        query_heads: int,
        head_dim: int,
        max_context_length: int,
        device: torch.device,
    ) -> PagedAttentionWorkspace:
        dimensions = (batch_capacity, query_heads, head_dim, max_context_length)
        if any(value <= 0 for value in dimensions):
            raise ValueError("workspace dimensions must be positive")
        split_size = 256
        num_splits = (max_context_length + split_size - 1) // split_size
        programs = batch_capacity * query_heads * num_splits
        head_bucket = 1 << (head_dim - 1).bit_length()
        partial_max = torch.empty(programs, device=device, dtype=torch.float32)
        return cls(
            partial_max=partial_max,
            partial_sum=torch.empty_like(partial_max),
            partial_accumulator=torch.empty(
                (programs, head_bucket), device=device, dtype=torch.float32
            ),
        )


if triton is not None:
    _ATTENTION_CONFIGS = [
        triton.Config({"BLOCK_TOKENS": 16}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_TOKENS": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_TOKENS": 64}, num_warps=8, num_stages=2),
    ]

    @triton.autotune(
        configs=_ATTENTION_CONFIGS,
        key=["NUM_QUERY_HEADS", "NUM_KV_HEADS", "HEAD_DIM", "MAX_CONTEXT"],
        warmup=5,
        rep=20,
    )
    @triton.jit
    def _paged_gqa_decode_kernel(
        query_ptr,
        key_cache_ptr,
        value_cache_ptr,
        block_tables_ptr,
        context_lengths_ptr,
        output_ptr,
        query_batch_stride,
        query_head_stride,
        cache_block_stride,
        cache_token_stride,
        cache_head_stride,
        table_batch_stride,
        output_batch_stride,
        output_head_stride,
        SCALE: tl.constexpr,
        NUM_QUERY_HEADS: tl.constexpr,
        NUM_KV_HEADS: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
        BLOCK_TABLE_WIDTH: tl.constexpr,
        MAX_CONTEXT: tl.constexpr,
        BLOCK_HEAD_DIM: tl.constexpr,
        BLOCK_TOKENS: tl.constexpr,
    ):
        program = tl.program_id(axis=0)
        batch_index = program // NUM_QUERY_HEADS
        query_head = program % NUM_QUERY_HEADS
        kv_head = query_head // (NUM_QUERY_HEADS // NUM_KV_HEADS)
        context_length = tl.load(context_lengths_ptr + batch_index)

        dimensions = tl.arange(0, BLOCK_HEAD_DIM)
        dimension_mask = dimensions < HEAD_DIM
        query = tl.load(
            query_ptr
            + batch_index * query_batch_stride
            + query_head * query_head_stride
            + dimensions,
            mask=dimension_mask,
            other=0.0,
        ).to(tl.float32)

        running_max = -float("inf")
        running_sum = 0.0
        accumulator = tl.zeros((BLOCK_HEAD_DIM,), dtype=tl.float32)

        # Online softmax makes working memory O(head_dim), independent of context length.
        for token_start in tl.range(0, MAX_CONTEXT, BLOCK_TOKENS):
            positions = token_start + tl.arange(0, BLOCK_TOKENS)
            valid_tokens = positions < context_length
            logical_blocks = positions // BLOCK_SIZE
            block_offsets = positions % BLOCK_SIZE
            physical_blocks = tl.load(
                block_tables_ptr
                + batch_index * table_batch_stride
                + logical_blocks,
                mask=valid_tokens & (logical_blocks < BLOCK_TABLE_WIDTH),
                other=0,
            )
            cache_offsets = (
                physical_blocks[:, None] * cache_block_stride
                + block_offsets[:, None] * cache_token_stride
                + kv_head * cache_head_stride
                + dimensions[None, :]
            )
            cache_mask = valid_tokens[:, None] & dimension_mask[None, :]
            keys = tl.load(key_cache_ptr + cache_offsets, mask=cache_mask, other=0.0).to(
                tl.float32
            )
            scores = tl.sum(keys * query[None, :], axis=1) * SCALE
            scores = tl.where(valid_tokens, scores, -float("inf"))

            block_max = tl.max(scores, axis=0)
            new_max = tl.maximum(running_max, block_max)
            old_scale = tl.exp(running_max - new_max)
            probabilities = tl.exp(scores - new_max)
            block_sum = tl.sum(probabilities, axis=0)

            values = tl.load(value_cache_ptr + cache_offsets, mask=cache_mask, other=0.0).to(
                tl.float32
            )
            accumulator = accumulator * old_scale + tl.sum(
                probabilities[:, None] * values, axis=0
            )
            running_sum = running_sum * old_scale + block_sum
            running_max = new_max

        output = accumulator / running_sum
        tl.store(
            output_ptr
            + batch_index * output_batch_stride
            + query_head * output_head_stride
            + dimensions,
            output,
            mask=dimension_mask,
        )

    _PAGED_GQA_DECODE_KERNEL = _paged_gqa_decode_kernel

    @triton.jit
    def _paged_gqa_split_kernel(
        query_ptr,
        key_cache_ptr,
        value_cache_ptr,
        block_tables_ptr,
        context_lengths_ptr,
        partial_max_ptr,
        partial_sum_ptr,
        partial_accumulator_ptr,
        query_batch_stride,
        query_head_stride,
        cache_block_stride,
        cache_token_stride,
        cache_head_stride,
        table_batch_stride,
        SCALE: tl.constexpr,
        NUM_QUERY_HEADS: tl.constexpr,
        NUM_KV_HEADS: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
        BLOCK_TABLE_WIDTH: tl.constexpr,
        NUM_SPLITS: tl.constexpr,
        SPLIT_SIZE: tl.constexpr,
        BLOCK_HEAD_DIM: tl.constexpr,
        BLOCK_TOKENS: tl.constexpr,
    ):
        program = tl.program_id(axis=0)
        split_index = program % NUM_SPLITS
        batch_head = program // NUM_SPLITS
        query_head = batch_head % NUM_QUERY_HEADS
        batch_index = batch_head // NUM_QUERY_HEADS
        kv_head = query_head // (NUM_QUERY_HEADS // NUM_KV_HEADS)
        context_length = tl.load(context_lengths_ptr + batch_index)
        split_start = split_index * SPLIT_SIZE

        dimensions = tl.arange(0, BLOCK_HEAD_DIM)
        dimension_mask = dimensions < HEAD_DIM
        query = tl.load(
            query_ptr
            + batch_index * query_batch_stride
            + query_head * query_head_stride
            + dimensions,
            mask=dimension_mask,
            other=0.0,
        ).to(tl.float32)
        running_max = -float("inf")
        running_sum = 0.0
        accumulator = tl.zeros((BLOCK_HEAD_DIM,), dtype=tl.float32)

        for relative_start in tl.range(0, SPLIT_SIZE, BLOCK_TOKENS):
            positions = split_start + relative_start + tl.arange(0, BLOCK_TOKENS)
            valid_tokens = positions < context_length
            logical_blocks = positions // BLOCK_SIZE
            block_offsets = positions % BLOCK_SIZE
            physical_blocks = tl.load(
                block_tables_ptr
                + batch_index * table_batch_stride
                + logical_blocks,
                mask=valid_tokens & (logical_blocks < BLOCK_TABLE_WIDTH),
                other=0,
            )
            cache_offsets = (
                physical_blocks[:, None] * cache_block_stride
                + block_offsets[:, None] * cache_token_stride
                + kv_head * cache_head_stride
                + dimensions[None, :]
            )
            cache_mask = valid_tokens[:, None] & dimension_mask[None, :]
            keys = tl.load(key_cache_ptr + cache_offsets, mask=cache_mask, other=0.0).to(
                tl.float32
            )
            scores = tl.sum(keys * query[None, :], axis=1) * SCALE
            scores = tl.where(valid_tokens, scores, -float("inf"))
            block_max = tl.max(scores, axis=0)
            new_max = tl.maximum(running_max, block_max)
            old_scale = tl.exp(running_max - new_max)
            probabilities = tl.exp(scores - new_max)
            values = tl.load(value_cache_ptr + cache_offsets, mask=cache_mask, other=0.0).to(
                tl.float32
            )
            accumulator = accumulator * old_scale + tl.sum(
                probabilities[:, None] * values, axis=0
            )
            running_sum = running_sum * old_scale + tl.sum(probabilities, axis=0)
            running_max = new_max

        tl.store(partial_max_ptr + program, running_max)
        tl.store(partial_sum_ptr + program, running_sum)
        tl.store(
            partial_accumulator_ptr + program * BLOCK_HEAD_DIM + dimensions,
            accumulator,
            mask=dimension_mask,
        )

    @triton.jit
    def _paged_gqa_reduce_kernel(
        partial_max_ptr,
        partial_sum_ptr,
        partial_accumulator_ptr,
        output_ptr,
        output_batch_stride,
        output_head_stride,
        NUM_QUERY_HEADS: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        NUM_SPLITS: tl.constexpr,
        BLOCK_HEAD_DIM: tl.constexpr,
        BLOCK_SPLITS: tl.constexpr,
    ):
        program = tl.program_id(axis=0)
        batch_index = program // NUM_QUERY_HEADS
        query_head = program % NUM_QUERY_HEADS
        splits = tl.arange(0, BLOCK_SPLITS)
        split_mask = splits < NUM_SPLITS
        partial_indices = program * NUM_SPLITS + splits
        partial_maxima = tl.load(
            partial_max_ptr + partial_indices, mask=split_mask, other=-float("inf")
        )
        global_max = tl.max(partial_maxima, axis=0)
        correction = tl.exp(partial_maxima - global_max)
        partial_sums = tl.load(
            partial_sum_ptr + partial_indices, mask=split_mask, other=0.0
        )
        denominator = tl.sum(partial_sums * correction, axis=0)

        dimensions = tl.arange(0, BLOCK_HEAD_DIM)
        dimension_mask = dimensions < HEAD_DIM
        accumulator_offsets = (
            partial_indices[:, None] * BLOCK_HEAD_DIM + dimensions[None, :]
        )
        partial_accumulators = tl.load(
            partial_accumulator_ptr + accumulator_offsets,
            mask=split_mask[:, None] & dimension_mask[None, :],
            other=0.0,
        )
        numerator = tl.sum(partial_accumulators * correction[:, None], axis=0)
        tl.store(
            output_ptr
            + batch_index * output_batch_stride
            + query_head * output_head_stride
            + dimensions,
            numerator / denominator,
            mask=dimension_mask,
        )

    _PAGED_GQA_SPLIT_KERNEL = _paged_gqa_split_kernel
    _PAGED_GQA_REDUCE_KERNEL = _paged_gqa_reduce_kernel


def is_triton_attention_available() -> bool:
    return triton is not None


def paged_gqa_decode_attention(
    query: Tensor,
    key_cache: Tensor,
    value_cache: Tensor,
    block_tables: Tensor,
    context_lengths: Tensor,
    *,
    scale: float | None = None,
    max_context_length: int | None = None,
    output: Tensor | None = None,
    workspace: PagedAttentionWorkspace | None = None,
    attention_path: str = "auto",
) -> Tensor:
    """Attend one query token per sequence over logically ordered physical KV pages."""

    if triton is None:
        raise RuntimeError("Triton is unavailable; install the `kernels` extra on Linux")
    tensors = (query, key_cache, value_cache, block_tables, context_lengths)
    if any(not tensor.is_cuda for tensor in tensors):
        raise ValueError("paged GQA attention requires CUDA tensors")
    if any(not tensor.is_contiguous() for tensor in tensors):
        raise ValueError("paged GQA attention requires contiguous tensors")
    if query.ndim != 3:
        raise ValueError("query must have shape [batch, query_heads, head_dim]")
    if key_cache.ndim != 4 or value_cache.shape != key_cache.shape:
        raise ValueError("key/value cache must share [blocks, block_size, kv_heads, head_dim]")
    if query.dtype not in {torch.float16, torch.bfloat16, torch.float32}:
        raise TypeError(f"unsupported query dtype: {query.dtype}")
    if key_cache.dtype != query.dtype or value_cache.dtype != query.dtype:
        raise ValueError("query, key cache, and value cache must share dtype")
    if block_tables.dtype != torch.int32 or context_lengths.dtype != torch.int32:
        raise TypeError("block tables and context lengths must use torch.int32")
    batch, query_heads, head_dim = query.shape
    _, block_size, kv_heads, cache_head_dim = key_cache.shape
    if head_dim != cache_head_dim or query_heads % kv_heads != 0:
        raise ValueError("query/cache head dimensions do not form valid GQA groups")
    if head_dim > 256:
        raise ValueError("decode attention supports head dimensions up to 256")
    if block_tables.ndim != 2 or block_tables.shape[0] != batch:
        raise ValueError("block_tables must have one row per query")
    if context_lengths.shape != (batch,):
        raise ValueError("context_lengths must have one value per query")
    if query.requires_grad and torch.is_grad_enabled():
        raise RuntimeError("TensorForge Triton kernels are inference-only")
    if attention_path not in {"auto", "one_pass", "split_kv"}:
        raise ValueError("attention_path must be auto, one_pass, or split_kv")

    context_capacity = block_tables.shape[1] * block_size
    if max_context_length is None:
        launch_context = int(context_lengths.max().item())
        if launch_context <= 0:
            raise ValueError("context lengths must be positive")
    else:
        # A scheduler-owned bucket avoids a GPU-to-host synchronization and is
        # the capture-safe path for future CUDA Graph execution. The caller is
        # responsible for ensuring every device-side length fits the bucket.
        launch_context = max_context_length
    if launch_context <= 0 or launch_context > context_capacity:
        raise ValueError("max_context_length must be positive and within table capacity")
    max_context_bucket = triton.next_power_of_2(launch_context)
    if max_context_bucket > context_capacity:
        max_context_bucket = context_capacity
    head_bucket = triton.next_power_of_2(head_dim)
    if output is None:
        output = torch.empty_like(query)
    elif (
        output.shape != query.shape
        or output.dtype != query.dtype
        or output.device != query.device
        or not output.is_contiguous()
    ):
        raise ValueError("output must be contiguous and share query shape, dtype, and device")
    attention_scale = head_dim**-0.5 if scale is None else scale
    global _LAST_ATTENTION_CONFIG
    use_split_kv = attention_path == "split_kv" or (
        attention_path == "auto"
        and max_context_bucket >= 1024
        and batch * query_heads <= 32
    )
    if use_split_kv:
        split_size = 256
        num_splits = triton.cdiv(max_context_bucket, split_size)
        partial_programs = batch * query_heads * num_splits
        if workspace is None:
            workspace = PagedAttentionWorkspace.allocate(
                batch_capacity=batch,
                query_heads=query_heads,
                head_dim=head_dim,
                max_context_length=max_context_bucket,
                device=query.device,
            )
        if (
            workspace.partial_max.device != query.device
            or workspace.partial_sum.device != query.device
            or workspace.partial_accumulator.device != query.device
            or workspace.partial_max.dtype != torch.float32
            or workspace.partial_sum.dtype != torch.float32
            or workspace.partial_accumulator.dtype != torch.float32
            or workspace.partial_max.numel() < partial_programs
            or workspace.partial_sum.numel() < partial_programs
            or workspace.partial_accumulator.shape[0] < partial_programs
            or workspace.partial_accumulator.shape[1] != head_bucket
        ):
            raise ValueError("split-KV workspace has insufficient or incompatible capacity")
        partial_max = workspace.partial_max[:partial_programs]
        partial_sum = workspace.partial_sum[:partial_programs]
        partial_accumulator = workspace.partial_accumulator[:partial_programs]
        _PAGED_GQA_SPLIT_KERNEL[(partial_programs,)](
            query,
            key_cache,
            value_cache,
            block_tables,
            context_lengths,
            partial_max,
            partial_sum,
            partial_accumulator,
            query.stride(0),
            query.stride(1),
            key_cache.stride(0),
            key_cache.stride(1),
            key_cache.stride(2),
            block_tables.stride(0),
            SCALE=attention_scale,
            NUM_QUERY_HEADS=query_heads,
            NUM_KV_HEADS=kv_heads,
            HEAD_DIM=head_dim,
            BLOCK_SIZE=block_size,
            BLOCK_TABLE_WIDTH=block_tables.shape[1],
            NUM_SPLITS=num_splits,
            SPLIT_SIZE=split_size,
            BLOCK_HEAD_DIM=head_bucket,
            BLOCK_TOKENS=64,
            num_warps=4,
            num_stages=2,
        )
        _PAGED_GQA_REDUCE_KERNEL[(batch * query_heads,)](
            partial_max,
            partial_sum,
            partial_accumulator,
            output,
            output.stride(0),
            output.stride(1),
            NUM_QUERY_HEADS=query_heads,
            HEAD_DIM=head_dim,
            NUM_SPLITS=num_splits,
            BLOCK_HEAD_DIM=head_bucket,
            BLOCK_SPLITS=triton.next_power_of_2(num_splits),
            num_warps=4,
        )
        _LAST_ATTENTION_CONFIG = {
            "path": "split_kv",
            "split_size": split_size,
            "num_splits": num_splits,
        }
    else:
        _PAGED_GQA_DECODE_KERNEL[(batch * query_heads,)](
            query,
            key_cache,
            value_cache,
            block_tables,
            context_lengths,
            output,
            query.stride(0),
            query.stride(1),
            key_cache.stride(0),
            key_cache.stride(1),
            key_cache.stride(2),
            block_tables.stride(0),
            output.stride(0),
            output.stride(1),
            SCALE=attention_scale,
            NUM_QUERY_HEADS=query_heads,
            NUM_KV_HEADS=kv_heads,
            HEAD_DIM=head_dim,
            BLOCK_SIZE=block_size,
            BLOCK_TABLE_WIDTH=block_tables.shape[1],
            MAX_CONTEXT=max_context_bucket,
            BLOCK_HEAD_DIM=head_bucket,
        )
        config = getattr(_PAGED_GQA_DECODE_KERNEL, "best_config", None)
        _LAST_ATTENTION_CONFIG = (
            None
            if config is None
            else {
                "path": "one_pass",
                "kwargs": dict(config.kwargs),
                "num_warps": config.num_warps,
                "num_stages": config.num_stages,
                "num_ctas": config.num_ctas,
                "maxnreg": config.maxnreg,
            }
        )
    return output


def selected_attention_autotune_config() -> dict[str, Any] | None:
    return _LAST_ATTENTION_CONFIG
