from __future__ import annotations

import pytest
import torch

from tensorforge.cache.paged import PagedKVCache, PagedKVCacheConfig
from tensorforge.model.config import tiny_config
from tensorforge.model.llama import LlamaForCausalLM
from tensorforge.scheduler.continuous import (
    BatchingPolicy,
    ContinuousBatchScheduler,
    RequestInput,
    RequestState,
    SchedulerConfig,
)

pytestmark = [pytest.mark.cuda, pytest.mark.triton]


def test_continuous_scheduler_tokens_match_independent_full_prefix_generation() -> None:
    pytest.importorskip("triton")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    from tensorforge.runtime.batched_decode import BatchedPagedDecodeExecutor

    config = tiny_config(num_hidden_layers=2, max_position_embeddings=16)
    model = LlamaForCausalLM(config).to(device="cuda", dtype=torch.float16).eval()
    cache = PagedKVCache(
        PagedKVCacheConfig(
            num_layers=config.num_hidden_layers,
            num_blocks=12,
            block_size=4,
            num_kv_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            max_sequences=3,
            max_sequence_length=16,
            dtype=torch.float16,
            device=torch.device("cuda"),
        )
    )
    executor = BatchedPagedDecodeExecutor(model, cache)
    scheduler = ContinuousBatchScheduler(
        executor,
        SchedulerConfig(
            policy=BatchingPolicy.CONTINUOUS,
            max_batch_requests=2,
            max_batch_tokens=2,
            max_active_token_budget=32,
            max_request_tokens=16,
        ),
    )
    requests = (
        RequestInput("a", (3, 5, 7), 3),
        RequestInput("b", (11,), 2),
        RequestInput("c", (13, 17, 19, 23), 2),
    )
    for request in requests:
        scheduler.submit(request)

    scheduler.run_until_complete()

    with torch.inference_mode():
        for request in requests:
            sequence = torch.tensor([request.prompt_token_ids], device="cuda")
            expected: list[int] = []
            for _ in range(request.max_new_tokens):
                token = int(model(sequence)[0, -1].argmax().item())
                expected.append(token)
                sequence = torch.cat(
                    (sequence, torch.tensor([[token]], device="cuda")), dim=1
                )
            snapshot = scheduler.snapshot(request.request_id)
            assert snapshot.state is RequestState.COMPLETED
            assert snapshot.generated_token_ids == tuple(expected)

    assert cache.stats().active_sequences == 0
    assert cache.stats().used_blocks == 0


def test_real_cache_exhaustion_fails_one_request_and_reclaims_everything() -> None:
    pytest.importorskip("triton")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    from tensorforge.runtime.batched_decode import BatchedPagedDecodeExecutor

    config = tiny_config(num_hidden_layers=1, max_position_embeddings=8)
    model = LlamaForCausalLM(config).to(device="cuda", dtype=torch.float16).eval()
    cache = PagedKVCache(
        PagedKVCacheConfig(
            num_layers=1,
            num_blocks=1,
            block_size=2,
            num_kv_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            max_sequences=2,
            max_sequence_length=8,
            dtype=torch.float16,
            device=torch.device("cuda"),
        )
    )
    scheduler = ContinuousBatchScheduler(
        BatchedPagedDecodeExecutor(model, cache),
        SchedulerConfig(
            policy=BatchingPolicy.CONTINUOUS,
            max_batch_requests=2,
            max_batch_tokens=2,
            max_active_token_budget=8,
            max_request_tokens=4,
        ),
    )
    scheduler.submit(RequestInput("commits", (3,), 1))
    scheduler.submit(RequestInput("exhausted", (5,), 1))

    scheduler.run_until_complete()

    assert scheduler.snapshot("commits").state is RequestState.COMPLETED
    assert scheduler.snapshot("exhausted").state is RequestState.FAILED
    assert "CacheExhaustedError" in (scheduler.snapshot("exhausted").error or "")
    assert scheduler.outstanding_token_budget == 0
    assert cache.stats().active_sequences == 0
    assert cache.stats().used_blocks == 0
    assert cache.stats().reserved_tokens == 0


def test_active_gpu_request_cancellation_reclaims_pages_and_budget() -> None:
    pytest.importorskip("triton")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    from tensorforge.runtime.batched_decode import BatchedPagedDecodeExecutor

    config = tiny_config(num_hidden_layers=1, max_position_embeddings=8)
    model = LlamaForCausalLM(config).to(device="cuda", dtype=torch.float16).eval()
    cache = PagedKVCache(
        PagedKVCacheConfig(
            num_layers=1,
            num_blocks=2,
            block_size=2,
            num_kv_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            max_sequences=1,
            max_sequence_length=8,
            dtype=torch.float16,
            device=torch.device("cuda"),
        )
    )
    scheduler = ContinuousBatchScheduler(
        BatchedPagedDecodeExecutor(model, cache),
        SchedulerConfig(
            policy=BatchingPolicy.CONTINUOUS,
            max_batch_requests=1,
            max_batch_tokens=1,
            max_active_token_budget=8,
            max_request_tokens=8,
        ),
    )
    scheduler.submit(RequestInput("cancelled", (3, 5, 7), 2))
    scheduler.step()
    assert cache.stats().used_blocks == 1

    scheduler.cancel("cancelled")

    assert scheduler.snapshot("cancelled").state is RequestState.CANCELLED
    assert scheduler.outstanding_token_budget == 0
    assert cache.stats().active_sequences == 0
    assert cache.stats().used_blocks == 0
    assert cache.stats().reserved_tokens == 0
