# mypy: disable-error-code="import-not-found,import-untyped,no-untyped-def,untyped-decorator"
"""Autotuned Triton inference kernels with explicit PyTorch-facing contracts.

The Triton DSL is intentionally isolated in this module because its pointer
arguments and constexpr annotations are not representable in normal Python
type stubs. Public wrappers remain strictly typed and validate every boundary.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None

_RMS_NORM_KERNEL: Any = None
_RESIDUAL_RMS_NORM_KERNEL: Any = None
_SWIGLU_KERNEL: Any = None


if triton is not None:
    _NORM_CONFIGS = [
        triton.Config({}, num_warps=1, num_stages=1),
        triton.Config({}, num_warps=2, num_stages=1),
        triton.Config({}, num_warps=4, num_stages=1),
        triton.Config({}, num_warps=8, num_stages=1),
    ]

    @triton.autotune(configs=_NORM_CONFIGS, key=["N_ROWS", "N_COLS"], warmup=5, rep=20)
    @triton.jit
    def _rms_norm_kernel(
        input_ptr,
        weight_ptr,
        output_ptr,
        input_row_stride,
        output_row_stride,
        N_ROWS: tl.constexpr,
        N_COLS: tl.constexpr,
        EPS: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        # One program owns one logical row. A power-of-two block enables a
        # single-program reduction; the mask prevents reads beyond odd widths.
        row = tl.program_id(axis=0)
        columns = tl.arange(0, BLOCK_SIZE)
        mask = columns < N_COLS

        # Statistics and scaling use FP32 even when storage is FP16/BF16.
        values = tl.load(input_ptr + row * input_row_stride + columns, mask=mask, other=0.0).to(
            tl.float32
        )
        mean_square = tl.sum(values * values, axis=0) / N_COLS
        inverse_rms = tl.rsqrt(mean_square + EPS)
        weights = tl.load(weight_ptr + columns, mask=mask, other=0.0).to(tl.float32)
        tl.store(
            output_ptr + row * output_row_stride + columns,
            values * inverse_rms * weights,
            mask=mask,
        )

    @triton.autotune(configs=_NORM_CONFIGS, key=["N_ROWS", "N_COLS"], warmup=5, rep=20)
    @triton.jit
    def _residual_rms_norm_kernel(
        input_ptr,
        residual_ptr,
        weight_ptr,
        residual_output_ptr,
        norm_output_ptr,
        row_stride,
        N_ROWS: tl.constexpr,
        N_COLS: tl.constexpr,
        EPS: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        # Fusing the residual add removes an intermediate read by RMSNorm while
        # still materializing residual_output for the next transformer branch.
        row = tl.program_id(axis=0)
        columns = tl.arange(0, BLOCK_SIZE)
        mask = columns < N_COLS
        offsets = row * row_stride + columns

        inputs = tl.load(input_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        residual = tl.load(residual_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        summed = inputs + residual
        tl.store(residual_output_ptr + offsets, summed, mask=mask)

        mean_square = tl.sum(summed * summed, axis=0) / N_COLS
        inverse_rms = tl.rsqrt(mean_square + EPS)
        weights = tl.load(weight_ptr + columns, mask=mask, other=0.0).to(tl.float32)
        tl.store(norm_output_ptr + offsets, summed * inverse_rms * weights, mask=mask)

    _SWIGLU_CONFIGS = [
        triton.Config({"BLOCK_SIZE": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE": 256}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=8, num_stages=2),
    ]

    @triton.autotune(configs=_SWIGLU_CONFIGS, key=["N_ELEMENTS"], warmup=5, rep=20)
    @triton.jit
    def _swiglu_kernel(
        gate_ptr,
        up_ptr,
        output_ptr,
        N_ELEMENTS: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        # Programs cover contiguous chunks. Grid size depends on the autotuned
        # block; the mask makes arbitrary tensor sizes safe.
        offsets = tl.program_id(axis=0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N_ELEMENTS
        gate = tl.load(gate_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        up = tl.load(up_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        activated = gate * tl.sigmoid(gate)
        tl.store(output_ptr + offsets, activated * up, mask=mask)

    _RMS_NORM_KERNEL = _rms_norm_kernel
    _RESIDUAL_RMS_NORM_KERNEL = _residual_rms_norm_kernel
    _SWIGLU_KERNEL = _swiglu_kernel


def is_triton_available() -> bool:
    return triton is not None


def _require_triton() -> None:
    if triton is None:
        raise RuntimeError("Triton is unavailable; install the `kernels` extra on Linux")


def _validate_common(inputs: Tensor, weight: Tensor | None = None) -> None:
    if not inputs.is_cuda:
        raise ValueError("Triton kernels require CUDA tensors")
    if inputs.dtype not in {torch.float16, torch.bfloat16, torch.float32}:
        raise TypeError(f"unsupported dtype: {inputs.dtype}")
    if inputs.numel() == 0 or inputs.ndim == 0:
        raise ValueError("inputs must be a non-empty tensor with a hidden dimension")
    if not inputs.is_contiguous():
        raise ValueError("inputs must be contiguous")
    if inputs.requires_grad and torch.is_grad_enabled():
        raise RuntimeError("TensorForge Triton kernels are inference-only")
    if weight is not None:
        if weight.shape != (inputs.shape[-1],):
            raise ValueError("weight must match the final input dimension")
        if weight.device != inputs.device or weight.dtype != inputs.dtype:
            raise ValueError("weight must share input device and dtype")
        if not weight.is_contiguous():
            raise ValueError("weight must be contiguous")


def _norm_launch_shape(inputs: Tensor) -> tuple[int, int, int]:
    columns = inputs.shape[-1]
    if columns > 16_384:
        raise ValueError("single-program RMSNorm supports at most 16,384 columns")
    assert triton is not None
    return inputs.numel() // columns, columns, triton.next_power_of_2(columns)


def rms_norm(inputs: Tensor, weight: Tensor, eps: float = 1e-6) -> Tensor:
    """Run autotuned RMSNorm and return a tensor shaped like ``inputs``."""

    _require_triton()
    _validate_common(inputs, weight)
    if eps <= 0:
        raise ValueError("eps must be positive")
    rows, columns, block_size = _norm_launch_shape(inputs)
    output = torch.empty_like(inputs)
    _RMS_NORM_KERNEL[(rows,)](
        inputs,
        weight,
        output,
        columns,
        columns,
        N_ROWS=rows,
        N_COLS=columns,
        EPS=eps,
        BLOCK_SIZE=block_size,
    )
    return output


def residual_rms_norm(
    inputs: Tensor, residual: Tensor, weight: Tensor, eps: float = 1e-6
) -> tuple[Tensor, Tensor]:
    """Fuse residual addition and RMSNorm while preserving the residual sum."""

    _require_triton()
    _validate_common(inputs, weight)
    _validate_common(residual)
    if residual.shape != inputs.shape or residual.dtype != inputs.dtype:
        raise ValueError("residual must share input shape and dtype")
    if residual.device != inputs.device:
        raise ValueError("residual must share input device")
    if eps <= 0:
        raise ValueError("eps must be positive")
    rows, columns, block_size = _norm_launch_shape(inputs)
    residual_output = torch.empty_like(inputs)
    norm_output = torch.empty_like(inputs)
    _RESIDUAL_RMS_NORM_KERNEL[(rows,)](
        inputs,
        residual,
        weight,
        residual_output,
        norm_output,
        columns,
        N_ROWS=rows,
        N_COLS=columns,
        EPS=eps,
        BLOCK_SIZE=block_size,
    )
    return residual_output, norm_output


def swiglu(gate: Tensor, up: Tensor) -> Tensor:
    """Fuse SiLU and gating after the two SwiGLU matrix projections."""

    _require_triton()
    _validate_common(gate)
    _validate_common(up)
    if up.shape != gate.shape or up.dtype != gate.dtype or up.device != gate.device:
        raise ValueError("gate and up tensors must share shape, dtype, and device")
    output = torch.empty_like(gate)

    def grid(meta: dict[str, Any]) -> tuple[int]:
        return (triton.cdiv(gate.numel(), meta["BLOCK_SIZE"]),)

    _SWIGLU_KERNEL[grid](gate, up, output, N_ELEMENTS=gate.numel())
    return output


def selected_autotune_config(operation: str) -> dict[str, Any] | None:
    """Return the most recently selected config for a benchmark result."""

    kernels = {
        "rms_norm": _RMS_NORM_KERNEL,
        "residual_rms_norm": _RESIDUAL_RMS_NORM_KERNEL,
        "swiglu": _SWIGLU_KERNEL,
    }
    if operation not in kernels:
        raise ValueError(f"unknown operation: {operation}")
    kernel = kernels[operation]
    config = getattr(kernel, "best_config", None)
    if config is None:
        return None
    return {
        "kwargs": dict(config.kwargs),
        "num_warps": config.num_warps,
        "num_stages": config.num_stages,
        "num_ctas": config.num_ctas,
        "maxnreg": config.maxnreg,
    }


def autotune_cache_size(operation: str) -> int:
    kernels = {
        "rms_norm": _RMS_NORM_KERNEL,
        "residual_rms_norm": _RESIDUAL_RMS_NORM_KERNEL,
        "swiglu": _SWIGLU_KERNEL,
    }
    if operation not in kernels:
        raise ValueError(f"unknown operation: {operation}")
    cache = getattr(kernels[operation], "cache", {})
    return len(cache)
