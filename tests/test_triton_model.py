from __future__ import annotations

import pytest
import torch

from tensorforge.model.config import tiny_config
from tensorforge.model.llama import LlamaForCausalLM

pytestmark = [pytest.mark.cuda, pytest.mark.triton]


def _require_triton_cuda() -> None:
    pytest.importorskip("triton")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", [(1, 7), (2, 17), (4, 33)])
def test_integrated_triton_forward_matches_reference(
    dtype: torch.dtype, shape: tuple[int, int]
) -> None:
    _require_triton_cuda()
    from tensorforge.runtime.triton_model import TritonModelExecutor

    torch.manual_seed(1234)
    config = tiny_config(max_position_embeddings=64)
    model = LlamaForCausalLM(config).to(device="cuda", dtype=dtype).eval()
    executor = TritonModelExecutor(model)
    input_ids = torch.randint(config.vocab_size, shape, device="cuda")

    with torch.inference_mode():
        expected = model(input_ids)
        actual = executor.forward(input_ids)

    tolerance = 5e-3 if dtype == torch.float16 else 3e-2
    torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
    top_two = expected.float().topk(2, dim=-1).values
    reference_margin = top_two[..., 0] - top_two[..., 1]
    stable_positions = reference_margin > 2 * tolerance
    torch.testing.assert_close(
        actual.argmax(-1)[stable_positions], expected.argmax(-1)[stable_positions]
    )


def test_integrated_executor_rejects_training_and_cpu_models() -> None:
    _require_triton_cuda()
    from tensorforge.runtime.triton_model import TritonModelExecutor

    with pytest.raises(ValueError, match="CUDA"):
        TritonModelExecutor(LlamaForCausalLM(tiny_config()).eval())
    with pytest.raises(ValueError, match="eval mode"):
        TritonModelExecutor(LlamaForCausalLM(tiny_config()).cuda().train())
