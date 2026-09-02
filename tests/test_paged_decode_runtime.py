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
def test_paged_decode_logits_match_full_prefix_oracle(dtype: torch.dtype) -> None:
    _require_triton_cuda()
    from tensorforge.runtime.paged_decode import PagedDecodeExecutor

    config = tiny_config(num_hidden_layers=2, max_position_embeddings=16)
    model = LlamaForCausalLM(config).to(device="cuda", dtype=dtype).eval()
    cache = PagedKVCache(
        PagedKVCacheConfig(
            num_layers=config.num_hidden_layers,
            num_blocks=4,
            block_size=4,
            num_kv_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            max_sequences=1,
            max_sequence_length=config.max_position_embeddings,
            dtype=dtype,
            device=torch.device("cuda"),
        )
    )
    executor = PagedDecodeExecutor(model, cache)
    executor.create_request("request")
    input_ids = torch.tensor([[11, 29, 7, 91, 3, 44]], device="cuda", dtype=torch.long)

    for position in range(input_ids.shape[1]):
        expected = model(input_ids[:, : position + 1])[:, -1]
        actual = executor.append_token("request", int(input_ids[0, position].item()))
        tolerance = 1e-5 if dtype == torch.float32 else 4e-3 if dtype == torch.float16 else 4e-2
        torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)

    assert cache.layer_view("request", 0).context_length == input_ids.shape[1]
