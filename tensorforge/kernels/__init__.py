"""Triton kernels and PyTorch numerical oracles."""

from tensorforge.kernels.reference import (
    residual_rms_norm_reference,
    rms_norm_reference,
    swiglu_reference,
)
from tensorforge.kernels.triton_ops import residual_rms_norm, rms_norm, swiglu

__all__ = [
    "residual_rms_norm",
    "residual_rms_norm_reference",
    "rms_norm",
    "rms_norm_reference",
    "swiglu",
    "swiglu_reference",
]
