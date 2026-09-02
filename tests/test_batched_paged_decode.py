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


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_batched_decode_matches_independent_full_prefixes(dtype: torch.dtype) -> None:
    _require_triton_cuda()
    from tensorforge.runtime.batched_decode import BatchedPagedDecodeExecutor

    config = tiny_config(num_hidden_layers=2, max_position_embeddings=8)
    model = LlamaForCausalLM(config).to(device="cuda", dtype=dtype).eval()
    cache = PagedKVCache(
        PagedKVCacheConfig(
            num_layers=config.num_hidden_layers,
            num_blocks=4,
            block_size=2,
            num_kv_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            max_sequences=2,
            max_sequence_length=config.max_position_embeddings,
            dtype=dtype,
            device=torch.device("cuda"),
        )
    )
    executor = BatchedPagedDecodeExecutor(model, cache)
    executor.create_request("a")
    executor.create_request("b")
    prefixes: dict[str, list[int]] = {"a": [], "b": []}

    for tokens in ({"a": 3, "b": 9}, {"a": 5}, {"a": 7, "b": 11}):
        result = executor.append_tokens(tokens)
        assert not result.errors
        for request_id, token_id in tokens.items():
            prefixes[request_id].append(token_id)
            input_ids = torch.tensor([prefixes[request_id]], device="cuda")
            expected = model(input_ids)[0, -1]
            tolerance = (
                1e-5 if dtype == torch.float32 else 4e-3 if dtype == torch.float16 else 4e-2
            )
            torch.testing.assert_close(
                result.logits[request_id], expected, rtol=tolerance, atol=tolerance
            )

    assert cache.layer_view("a", 0).context_length == 3
    assert cache.layer_view("b", 0).context_length == 2


def test_cache_exhaustion_is_isolated_to_request_that_needs_a_page() -> None:
    _require_triton_cuda()
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
    executor = BatchedPagedDecodeExecutor(model, cache)
    executor.create_request("fits-existing-page")
    executor.create_request("needs-new-page")
    assert not executor.append_tokens({"fits-existing-page": 3}).errors

    result = executor.append_tokens({"fits-existing-page": 5, "needs-new-page": 9})

    assert "fits-existing-page" in result.logits
    assert "needs-new-page" in result.errors
    assert "CacheExhaustedError" in result.errors["needs-new-page"]
    assert cache.layer_view("fits-existing-page", 0).context_length == 2
    assert cache.layer_view("needs-new-page", 0).context_length == 0
