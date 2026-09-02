"""Numerical oracles for inference-only fused kernels."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F


def rms_norm_reference(inputs: Tensor, weight: Tensor, eps: float) -> Tensor:
    """Explicit RMSNorm with FP32 reduction and multiplication."""

    variance = inputs.float().square().mean(dim=-1, keepdim=True)
    normalized = inputs.float() * torch.rsqrt(variance + eps)
    return (normalized * weight.float()).to(inputs.dtype)


def torch_rms_norm(inputs: Tensor, weight: Tensor, eps: float) -> Tensor:
    """Standard PyTorch operator used as the performance baseline."""

    return F.rms_norm(inputs, (inputs.shape[-1],), weight, eps)


def residual_rms_norm_reference(
    inputs: Tensor, residual: Tensor, weight: Tensor, eps: float
) -> tuple[Tensor, Tensor]:
    """Return the residual sum and its RMS-normalized representation."""

    residual_output = inputs + residual
    return residual_output, rms_norm_reference(residual_output, weight, eps)


def torch_residual_rms_norm(
    inputs: Tensor, residual: Tensor, weight: Tensor, eps: float
) -> tuple[Tensor, Tensor]:
    residual_output = inputs + residual
    return residual_output, torch_rms_norm(residual_output, weight, eps)


def swiglu_reference(gate: Tensor, up: Tensor) -> Tensor:
    """SwiGLU activation after the two matrix projections."""

    return F.silu(gate) * up
