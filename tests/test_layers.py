from __future__ import annotations

import torch
from torch.nn import functional as F

from tensorforge.model.config import ModelConfig
from tensorforge.model.layers import RMSNorm, SwiGLU


def test_rms_norm_matches_explicit_fp32_reference() -> None:
    layer = RMSNorm(hidden_size=64, eps=1e-6)
    layer.weight.data.uniform_(0.5, 1.5)
    inputs = torch.randn(3, 17, 64)

    actual = layer(inputs)
    variance = inputs.float().square().mean(-1, keepdim=True)
    expected = inputs.float() * torch.rsqrt(variance + layer.eps) * layer.weight.float()

    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)


def test_swiglu_matches_unfused_reference(model_config: ModelConfig) -> None:
    layer = SwiGLU(model_config)
    inputs = torch.randn(2, 11, model_config.hidden_size)

    actual = layer(inputs)
    expected = F.linear(
        F.silu(F.linear(inputs, layer.gate_proj.weight)) * F.linear(inputs, layer.up_proj.weight),
        layer.down_proj.weight,
    )

    torch.testing.assert_close(actual, expected)
