from __future__ import annotations

import pytest
import torch

from tensorforge.kernels.reference import (
    residual_rms_norm_reference,
    rms_norm_reference,
    swiglu_reference,
)

pytestmark = [pytest.mark.cuda, pytest.mark.triton]


def _require_triton_cuda() -> None:
    pytest.importorskip("triton")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")


def _tolerance(dtype: torch.dtype) -> float:
    if dtype == torch.float32:
        return 1e-5
    if dtype == torch.float16:
        return 3e-3
    return 3e-2


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", [(1, 1), (3, 7), (7, 513), (16, 1376), (4, 4096), (2, 8192)])
def test_rms_norm_shape_dtype_grid(shape: tuple[int, int], dtype: torch.dtype) -> None:
    _require_triton_cuda()
    from tensorforge.kernels.triton_ops import rms_norm

    inputs = torch.randn(shape, device="cuda", dtype=dtype)
    weight = torch.randn(shape[-1], device="cuda", dtype=dtype)

    actual = rms_norm(inputs, weight)
    expected = rms_norm_reference(inputs, weight, 1e-6)

    tolerance = _tolerance(dtype)
    torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", [(1, 7), (7, 513), (16, 1376), (4, 4096)])
def test_residual_rms_norm_shape_dtype_grid(shape: tuple[int, int], dtype: torch.dtype) -> None:
    _require_triton_cuda()
    from tensorforge.kernels.triton_ops import residual_rms_norm

    inputs = torch.randn(shape, device="cuda", dtype=dtype)
    residual = torch.randn_like(inputs)
    weight = torch.randn(shape[-1], device="cuda", dtype=dtype)

    actual_residual, actual_norm = residual_rms_norm(inputs, residual, weight)
    expected_residual, expected_norm = residual_rms_norm_reference(inputs, residual, weight, 1e-6)

    tolerance = _tolerance(dtype)
    torch.testing.assert_close(actual_residual, expected_residual, rtol=tolerance, atol=tolerance)
    torch.testing.assert_close(actual_norm, expected_norm, rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", [(1, 1), (3, 7), (7, 513), (16, 1376), (2, 11008)])
def test_swiglu_shape_dtype_grid(shape: tuple[int, int], dtype: torch.dtype) -> None:
    _require_triton_cuda()
    from tensorforge.kernels.triton_ops import swiglu

    gate = torch.randn(shape, device="cuda", dtype=dtype)
    up = torch.randn_like(gate)

    actual = swiglu(gate, up)
    expected = swiglu_reference(gate, up)

    tolerance = _tolerance(dtype)
    torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize("fill", [0.0, 1e-5, 100.0, -100.0])
def test_rms_norm_adversarial_values(fill: float) -> None:
    _require_triton_cuda()
    from tensorforge.kernels.triton_ops import rms_norm

    inputs = torch.full((3, 513), fill, device="cuda", dtype=torch.float32)
    weight = torch.linspace(-2, 2, 513, device="cuda")

    actual = rms_norm(inputs, weight)
    expected = rms_norm_reference(inputs, weight, 1e-6)

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
    assert torch.isfinite(actual).all()


def test_autotune_selection_is_observable() -> None:
    _require_triton_cuda()
    from tensorforge.kernels.triton_ops import (
        autotune_cache_size,
        rms_norm,
        selected_autotune_config,
    )

    inputs = torch.randn(4, 513, device="cuda", dtype=torch.float16)
    weight = torch.ones(513, device="cuda", dtype=torch.float16)
    rms_norm(inputs, weight)
    rms_norm(torch.randn(16, 513, device="cuda", dtype=torch.float16), weight)

    config = selected_autotune_config("rms_norm")
    assert config is not None
    assert config["num_warps"] in {1, 2, 4, 8}
    assert autotune_cache_size("rms_norm") >= 2


def test_residual_cancellation_and_swiglu_extremes() -> None:
    _require_triton_cuda()
    from tensorforge.kernels.triton_ops import residual_rms_norm, swiglu

    inputs = torch.linspace(-10, 10, 513, device="cuda").repeat(3, 1)
    residual = -inputs + 1e-5
    weight = torch.linspace(0.5, 1.5, 513, device="cuda")
    actual_residual, actual_norm = residual_rms_norm(inputs, residual, weight)
    expected_residual, expected_norm = residual_rms_norm_reference(inputs, residual, weight, 1e-6)
    torch.testing.assert_close(actual_residual, expected_residual, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(actual_norm, expected_norm, rtol=1e-4, atol=1e-4)

    gate = torch.linspace(-20, 20, 513, device="cuda").repeat(3, 1)
    up = torch.linspace(2, -2, 513, device="cuda").repeat(3, 1)
    torch.testing.assert_close(swiglu(gate, up), swiglu_reference(gate, up), rtol=1e-5, atol=1e-5)


def test_triton_kernels_reject_cpu_and_noncontiguous_inputs() -> None:
    _require_triton_cuda()
    from tensorforge.kernels.triton_ops import rms_norm

    with pytest.raises(ValueError, match="CUDA"):
        rms_norm(torch.randn(2, 8), torch.ones(8))

    inputs = torch.randn(8, 8, device="cuda").transpose(0, 1)
    weight = torch.ones(8, device="cuda")
    with pytest.raises(ValueError, match="contiguous"):
        rms_norm(inputs, weight)


def test_inference_only_contract_rejects_gradients() -> None:
    _require_triton_cuda()
    from tensorforge.kernels.triton_ops import swiglu

    gate = torch.randn(2, 8, device="cuda", requires_grad=True)
    up = torch.randn_like(gate)
    with pytest.raises(RuntimeError, match="inference-only"):
        swiglu(gate, up)


def test_caller_owned_outputs_are_reused_and_numerically_correct() -> None:
    _require_triton_cuda()
    from tensorforge.kernels.triton_ops import residual_rms_norm, rms_norm, swiglu

    inputs = torch.randn(3, 64, device="cuda", dtype=torch.float16)
    residual = torch.randn_like(inputs)
    weight = torch.randn(64, device="cuda", dtype=torch.float16)
    gate = torch.randn(3, 160, device="cuda", dtype=torch.float16)
    up = torch.randn_like(gate)
    norm_output = torch.empty_like(inputs)
    residual_output = torch.empty_like(inputs)
    post_norm_output = torch.empty_like(inputs)
    swiglu_output = torch.empty_like(gate)
    addresses = tuple(
        tensor.data_ptr()
        for tensor in (
            norm_output,
            residual_output,
            post_norm_output,
            swiglu_output,
        )
    )

    assert rms_norm(inputs, weight, output=norm_output) is norm_output
    actual_residual, actual_post_norm = residual_rms_norm(
        inputs,
        residual,
        weight,
        residual_output=residual_output,
        norm_output=post_norm_output,
    )
    assert actual_residual is residual_output
    assert actual_post_norm is post_norm_output
    assert swiglu(gate, up, output=swiglu_output) is swiglu_output
    assert addresses == tuple(
        tensor.data_ptr()
        for tensor in (
            norm_output,
            residual_output,
            post_norm_output,
            swiglu_output,
        )
    )
    expected_residual, expected_post_norm = residual_rms_norm_reference(
        inputs, residual, weight, 1e-6
    )
    torch.testing.assert_close(
        norm_output,
        rms_norm_reference(inputs, weight, 1e-6),
        rtol=3e-3,
        atol=3e-3,
    )
    torch.testing.assert_close(
        residual_output, expected_residual, rtol=3e-3, atol=3e-3
    )
    torch.testing.assert_close(
        post_norm_output, expected_post_norm, rtol=3e-3, atol=3e-3
    )
    torch.testing.assert_close(
        swiglu_output, swiglu_reference(gate, up), rtol=3e-3, atol=3e-3
    )
