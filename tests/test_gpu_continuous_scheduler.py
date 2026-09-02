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
