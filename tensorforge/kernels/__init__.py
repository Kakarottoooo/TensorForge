"""Triton kernels and PyTorch numerical oracles."""

from tensorforge.kernels.reference import (
    residual_rms_norm_reference,
    rms_norm_reference,
    swiglu_reference,
)
from tensorforge.kernels.triton_cache import paged_kv_write_token
from tensorforge.kernels.triton_ops import residual_rms_norm, rms_norm, swiglu

__all__ = [
    "paged_kv_write_token",
    "residual_rms_norm",
    "residual_rms_norm_reference",
    "rms_norm",
    "rms_norm_reference",
    "swiglu",
    "swiglu_reference",
]
