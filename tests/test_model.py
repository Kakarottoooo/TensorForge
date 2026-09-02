from __future__ import annotations

import pytest
import torch

from tensorforge.model.config import ModelConfig
from tensorforge.model.llama import LlamaForCausalLM


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("sequence_length", [1, 17, 64])
def test_forward_shape(model_config: ModelConfig, batch_size: int, sequence_length: int) -> None:
    model = LlamaForCausalLM(model_config).eval()
    input_ids = torch.randint(model_config.vocab_size, (batch_size, sequence_length))

    logits = model(input_ids)

    assert logits.shape == (batch_size, sequence_length, model_config.vocab_size)
    assert torch.isfinite(logits).all()


def test_attention_is_causal(model_config: ModelConfig) -> None:
    model = LlamaForCausalLM(model_config).eval()
    original = torch.randint(model_config.vocab_size, (1, 24))
    changed_future = original.clone()
    changed_future[:, 12:] = torch.randint(model_config.vocab_size, (1, 12))

    with torch.inference_mode():
        original_logits = model(original)
        changed_logits = model(changed_future)

    torch.testing.assert_close(original_logits[:, :12], changed_logits[:, :12], rtol=0, atol=0)


def test_padding_mask_hides_key_tokens(model_config: ModelConfig) -> None:
    model = LlamaForCausalLM(model_config).eval()
    inputs = torch.randint(model_config.vocab_size, (2, 8))
    mask = torch.ones_like(inputs, dtype=torch.bool)
    mask[:, :2] = False

    logits = model(inputs, attention_mask=mask)

    assert logits.shape == (2, 8, model_config.vocab_size)
    assert torch.isfinite(logits).all()


def test_invalid_inputs_fail_early(model_config: ModelConfig) -> None:
    model = LlamaForCausalLM(model_config)
    with pytest.raises(TypeError, match=r"torch\.long"):
        model(torch.ones(2, 4))
    with pytest.raises(ValueError, match="exceeds"):
        model(torch.ones(1, model_config.max_position_embeddings + 1, dtype=torch.long))


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_cuda_precision_modes(model_config: ModelConfig, dtype: torch.dtype) -> None:
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("GPU does not support BF16")
    model = LlamaForCausalLM(model_config).to(device="cuda", dtype=dtype).eval()
    inputs = torch.randint(model_config.vocab_size, (2, 16), device="cuda")

    with torch.inference_mode():
        logits = model(inputs)

    assert logits.dtype == dtype
    assert torch.isfinite(logits).all()
