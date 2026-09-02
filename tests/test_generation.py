from __future__ import annotations

import torch

from tensorforge.model.config import ModelConfig
from tensorforge.model.llama import LlamaForCausalLM
from tensorforge.runtime.generation import GenerationConfig, greedy_generate


def test_greedy_generation_matches_manual_full_prefix_decode(model_config: ModelConfig) -> None:
    model = LlamaForCausalLM(model_config).eval()
    prompt = torch.randint(model_config.vocab_size, (3, 9))

    actual = greedy_generate(model, prompt, GenerationConfig(max_new_tokens=5))

    expected = prompt.clone()
    with torch.inference_mode():
        for _ in range(5):
            token = model(expected)[:, -1].argmax(-1, keepdim=True)
            expected = torch.cat((expected, token), dim=1)
    torch.testing.assert_close(actual, expected)


def test_zero_generation_returns_value_equal_copy(model_config: ModelConfig) -> None:
    model = LlamaForCausalLM(model_config).eval()
    prompt = torch.randint(model_config.vocab_size, (2, 7))

    output = greedy_generate(model, prompt, GenerationConfig(max_new_tokens=0))

    torch.testing.assert_close(output, prompt)
    assert output.data_ptr() != prompt.data_ptr()


class AlwaysEos(torch.nn.Module):
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        logits = torch.zeros((*input_ids.shape, 8), device=input_ids.device)
        logits[..., 3] = 1
        return logits


def test_generation_stops_when_every_request_reaches_eos() -> None:
    prompt = torch.tensor([[1, 2], [4, 5]])

    output = greedy_generate(
        AlwaysEos(), prompt, GenerationConfig(max_new_tokens=10, eos_token_id=3)
    )

    assert output.tolist() == [[1, 2, 3], [4, 5, 3]]
