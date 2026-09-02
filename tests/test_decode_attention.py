from __future__ import annotations

import pytest
import torch

from tensorforge.kernels.reference import paged_gqa_decode_reference


def test_paged_gqa_reference_follows_noncontiguous_block_table() -> None:
    query = torch.randn(1, 4, 8)
    key_cache = torch.randn(3, 2, 2, 8)
    value_cache = torch.randn_like(key_cache)
    block_tables = torch.tensor([[2, 0]], dtype=torch.int32)
    context_lengths = torch.tensor([3], dtype=torch.int32)

    actual = paged_gqa_decode_reference(
        query, key_cache, value_cache, block_tables, context_lengths
    )

    dense_keys = torch.cat((key_cache[2], key_cache[0, :1]), dim=0)
    dense_values = torch.cat((value_cache[2], value_cache[0, :1]), dim=0)
    dense_keys = dense_keys.repeat_interleave(2, dim=1)
    dense_values = dense_values.repeat_interleave(2, dim=1)
    scores = torch.einsum("bhd,thd->bht", query.float(), dense_keys.float()) / 8**0.5
    expected = torch.einsum(
        "bht,thd->bhd", scores.softmax(dim=-1), dense_values.float()
    ).to(query.dtype)

    torch.testing.assert_close(actual, expected)


def _require_triton_cuda() -> None:
    pytest.importorskip("triton")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")


@pytest.mark.cuda
@pytest.mark.triton
def test_triton_paged_gqa_decode_matches_reference() -> None:
    _require_triton_cuda()
    from tensorforge.kernels.triton_attention import paged_gqa_decode_attention

    query = torch.randn(1, 4, 64, device="cuda", dtype=torch.float32)
    key_cache = torch.randn(3, 2, 2, 64, device="cuda", dtype=torch.float32)
    value_cache = torch.randn_like(key_cache)
    block_tables = torch.tensor([[2, 0]], device="cuda", dtype=torch.int32)
    context_lengths = torch.tensor([3], device="cuda", dtype=torch.int32)

    actual = paged_gqa_decode_attention(
        query, key_cache, value_cache, block_tables, context_lengths
    )
    expected = paged_gqa_decode_reference(
        query, key_cache, value_cache, block_tables, context_lengths
    )

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.cuda
@pytest.mark.triton
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    ("batch", "query_heads", "kv_heads", "head_dim", "block_size", "lengths"),
    [
        (2, 8, 2, 80, 4, [1, 7]),
        (2, 32, 8, 128, 16, [17, 31]),
        (1, 8, 1, 64, 16, [129]),
        (1, 32, 8, 128, 16, [2048]),
    ],
)
def test_triton_paged_gqa_decode_shape_dtype_grid(
    batch: int,
    query_heads: int,
    kv_heads: int,
    head_dim: int,
    block_size: int,
    lengths: list[int],
    dtype: torch.dtype,
) -> None:
    _require_triton_cuda()
    from tensorforge.kernels.triton_attention import paged_gqa_decode_attention

    logical_blocks = (max(lengths) + block_size - 1) // block_size
    physical_blocks = batch * logical_blocks
    query = torch.randn(batch, query_heads, head_dim, device="cuda", dtype=dtype)
    key_cache = torch.randn(
        physical_blocks, block_size, kv_heads, head_dim, device="cuda", dtype=dtype
    )
    value_cache = torch.randn_like(key_cache)
    block_tables = torch.arange(
        physical_blocks, device="cuda", dtype=torch.int32
    ).view(batch, logical_blocks).flip(1).contiguous()
    context_lengths = torch.tensor(lengths, device="cuda", dtype=torch.int32)

    actual = paged_gqa_decode_attention(
        query, key_cache, value_cache, block_tables, context_lengths
    )
    expected = paged_gqa_decode_reference(
        query, key_cache, value_cache, block_tables, context_lengths
    )

    tolerance = 1e-5 if dtype == torch.float32 else 3e-3 if dtype == torch.float16 else 3e-2
    torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)


@pytest.mark.cuda
@pytest.mark.triton
def test_long_context_with_few_heads_selects_split_kv_path() -> None:
    _require_triton_cuda()
    from tensorforge.kernels.triton_attention import (
        PagedAttentionWorkspace,
        paged_gqa_decode_attention,
        selected_attention_autotune_config,
    )

    query = torch.randn(1, 8, 64, device="cuda", dtype=torch.float16)
    key_cache = torch.randn(128, 16, 4, 64, device="cuda", dtype=torch.float16)
    value_cache = torch.randn_like(key_cache)
    block_tables = torch.arange(128, device="cuda", dtype=torch.int32).view(1, 128)
    context_lengths = torch.tensor([2048], device="cuda", dtype=torch.int32)

    output = torch.empty_like(query)
    workspace = PagedAttentionWorkspace.allocate(
        batch_capacity=1,
        query_heads=8,
        head_dim=64,
        max_context_length=2048,
        device=torch.device("cuda"),
    )
    actual = paged_gqa_decode_attention(
        query,
        key_cache,
        value_cache,
        block_tables,
        context_lengths,
        max_context_length=2048,
        output=output,
        workspace=workspace,
    )
    expected = paged_gqa_decode_reference(
        query, key_cache, value_cache, block_tables, context_lengths
    )

    assert actual.data_ptr() == output.data_ptr()
    torch.testing.assert_close(actual, expected, rtol=3e-3, atol=3e-3)
    assert selected_attention_autotune_config() == {
        "path": "split_kv",
        "split_size": 256,
        "num_splits": 8,
    }
