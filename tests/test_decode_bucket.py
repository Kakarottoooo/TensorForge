from __future__ import annotations

import pytest
import torch

from tensorforge.cache.paged import PagedKVCache, PagedKVCacheConfig
from tensorforge.model.config import tiny_config
from tensorforge.model.llama import LlamaForCausalLM

pytestmark = [pytest.mark.cuda, pytest.mark.triton]


def _require_triton_cuda() -> None:
    pytest.importorskip("triton")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")


def test_eager_decode_bucket_reuses_addresses_and_matches_full_prefix() -> None:
    _require_triton_cuda()
    from tensorforge.runtime.decode_bucket import (
        BucketedPagedDecodeExecutor,
        DecodeExecutionMode,
    )

    config = tiny_config(num_hidden_layers=1, max_position_embeddings=8)
    torch.manual_seed(17)
    model = LlamaForCausalLM(config).to(device="cuda", dtype=torch.float32).eval()
    cache = PagedKVCache(
        PagedKVCacheConfig(
            num_layers=1,
            num_blocks=8,
            block_size=2,
            num_kv_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            max_sequences=2,
            max_sequence_length=8,
            dtype=torch.float32,
            device=torch.device("cuda"),
        )
    )
    executor = BucketedPagedDecodeExecutor(
        model,
        cache,
        mode=DecodeExecutionMode.EAGER,
        batch_buckets=(2,),
        context_buckets=(8,),
    )
    executor.create_request("a")
    executor.create_request("b")

    first = executor.append_tokens({"a": 3, "b": 9})
    first_a = first.logits["a"].clone()
    first_addresses = executor.buffer_addresses()
    second = executor.append_tokens({"a": 5, "b": 11})
    second_addresses = executor.buffer_addresses()

    assert not first.errors
    assert not second.errors
    assert first_addresses == second_addresses
    with torch.inference_mode():
        torch.testing.assert_close(first_a, model(torch.tensor([[3]], device="cuda"))[0, -1])
        torch.testing.assert_close(
            second.logits["a"], model(torch.tensor([[3, 5]], device="cuda"))[0, -1]
        )
        torch.testing.assert_close(
            second.logits["b"], model(torch.tensor([[9, 11]], device="cuda"))[0, -1]
        )
    assert executor.metrics().eager_calls == 2


def test_compiled_decode_bucket_matches_full_prefix_and_records_setup() -> None:
    _require_triton_cuda()
    from torch._logging._internal import trace_log

    from tensorforge.runtime.decode_bucket import (
        BucketedPagedDecodeExecutor,
        DecodeExecutionMode,
    )

    config = tiny_config(num_hidden_layers=1, max_position_embeddings=8)
    torch.manual_seed(19)
    model = LlamaForCausalLM(config).to(device="cuda", dtype=torch.float16).eval()
    cache = PagedKVCache(
        PagedKVCacheConfig(
            num_layers=1,
            num_blocks=4,
            block_size=2,
            num_kv_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            max_sequences=1,
            max_sequence_length=8,
            dtype=torch.float16,
            device=torch.device("cuda"),
        )
    )
    executor = BucketedPagedDecodeExecutor(
        model,
        cache,
        mode=DecodeExecutionMode.COMPILE,
        batch_buckets=(1,),
        context_buckets=(8,),
    )
    executor.create_request("request")

    # Pytest attaches a DEBUG trace handler. PyTorch 2.5 then probes
    # `nvcc --version` while rendering optional compiler artifacts; this WSL
    # runtime intentionally uses Triton's bundled compiler and has no nvcc.
    trace_handlers = list(trace_log.handlers)
    trace_log.handlers.clear()
    try:
        result = executor.append_tokens({"request": 7})
    finally:
        trace_log.handlers.extend(trace_handlers)

    assert not result.errors
    with torch.inference_mode():
        expected = model(torch.tensor([[7]], device="cuda"))[0, -1]
    torch.testing.assert_close(result.logits["request"], expected, rtol=4e-3, atol=4e-3)
    metrics = executor.metrics()
    assert metrics.compile_count == 1
    assert metrics.compiled_calls == 1
    assert metrics.compile_time_ms > 0


def test_cuda_graph_bucket_records_capture_miss_then_replay_hit() -> None:
    _require_triton_cuda()
    from tensorforge.runtime.decode_bucket import (
        BucketedPagedDecodeExecutor,
        DecodeExecutionMode,
    )

    config = tiny_config(num_hidden_layers=1, max_position_embeddings=8)
    torch.manual_seed(23)
    model = LlamaForCausalLM(config).to(device="cuda", dtype=torch.float16).eval()
    cache = PagedKVCache(
        PagedKVCacheConfig(
            num_layers=1,
            num_blocks=4,
            block_size=2,
            num_kv_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            max_sequences=1,
            max_sequence_length=8,
            dtype=torch.float16,
            device=torch.device("cuda"),
        )
    )
    executor = BucketedPagedDecodeExecutor(
        model,
        cache,
        mode=DecodeExecutionMode.CUDA_GRAPH,
        batch_buckets=(1,),
        context_buckets=(8,),
    )
    executor.create_request("request")

    first = executor.append_tokens({"request": 7})
    first_logits = first.logits["request"].clone()
    addresses_after_capture = executor.buffer_addresses()
    second = executor.append_tokens({"request": 13})

    assert not first.errors
    assert not second.errors
    assert addresses_after_capture == executor.buffer_addresses()
    with torch.inference_mode():
        torch.testing.assert_close(
            first_logits,
            model(torch.tensor([[7]], device="cuda"))[0, -1],
            rtol=4e-3,
            atol=4e-3,
        )
        torch.testing.assert_close(
            second.logits["request"],
            model(torch.tensor([[7, 13]], device="cuda"))[0, -1],
            rtol=4e-3,
            atol=4e-3,
        )
    metrics = executor.metrics()
    assert metrics.capture_count == 1
    assert metrics.capture_time_ms > 0
    assert metrics.capture_warmup_time_ms > 0
    assert metrics.graph_calls == 2
    assert metrics.graph_misses == 1
    assert metrics.graph_hits == 1


def test_cuda_graph_shape_overflow_falls_back_to_dynamic_eager_with_reason() -> None:
    _require_triton_cuda()
    from tensorforge.runtime.decode_bucket import (
        BucketedPagedDecodeExecutor,
        DecodeExecutionMode,
    )

    config = tiny_config(num_hidden_layers=1, max_position_embeddings=8)
    torch.manual_seed(29)
    model = LlamaForCausalLM(config).to(device="cuda", dtype=torch.float16).eval()
    cache = PagedKVCache(
        PagedKVCacheConfig(
            num_layers=1,
            num_blocks=4,
            block_size=2,
            num_kv_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            max_sequences=2,
            max_sequence_length=8,
            dtype=torch.float16,
            device=torch.device("cuda"),
        )
    )
    executor = BucketedPagedDecodeExecutor(
        model,
        cache,
        mode=DecodeExecutionMode.CUDA_GRAPH,
        batch_buckets=(1,),
        context_buckets=(8,),
    )
    executor.create_request("a")
    executor.create_request("b")

    result = executor.append_tokens({"a": 3, "b": 5})

    assert not result.errors
    metrics = executor.metrics()
    assert metrics.shape_fallbacks == 1
    assert metrics.eager_fallback_calls == 1
    assert metrics.graph_misses == 1
    assert metrics.graph_hits == 0
    assert metrics.capture_count == 0
    assert metrics.fallback_reasons == {"batch_overflow": 1}


def test_cuda_graph_padded_lane_is_masked_and_cache_is_reclaimable() -> None:
    _require_triton_cuda()
    from tensorforge.runtime.decode_bucket import (
        BucketedPagedDecodeExecutor,
        DecodeExecutionMode,
    )

    config = tiny_config(num_hidden_layers=1, max_position_embeddings=8)
    torch.manual_seed(31)
    model = LlamaForCausalLM(config).to(device="cuda", dtype=torch.float16).eval()
    cache = PagedKVCache(
        PagedKVCacheConfig(
            num_layers=1,
            num_blocks=4,
            block_size=2,
            num_kv_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            max_sequences=2,
            max_sequence_length=8,
            dtype=torch.float16,
            device=torch.device("cuda"),
        )
    )
    executor = BucketedPagedDecodeExecutor(
        model,
        cache,
        mode=DecodeExecutionMode.CUDA_GRAPH,
        batch_buckets=(2,),
        context_buckets=(8,),
    )
    executor.create_request("only")

    first = executor.append_tokens({"only": 3})
    first_logits = first.logits["only"].clone()
    second = executor.append_tokens({"only": 5})

    assert not first.errors
    assert not second.errors
    with torch.inference_mode():
        torch.testing.assert_close(
            first_logits,
            model(torch.tensor([[3]], device="cuda"))[0, -1],
            rtol=4e-3,
            atol=4e-3,
        )
        torch.testing.assert_close(
            second.logits["only"],
            model(torch.tensor([[3, 5]], device="cuda"))[0, -1],
            rtol=4e-3,
            atol=4e-3,
        )
    assert executor.metrics().padded_lanes == 2
    executor.release_request("only")
    assert cache.stats().active_sequences == 0
    assert cache.stats().used_blocks == 0
