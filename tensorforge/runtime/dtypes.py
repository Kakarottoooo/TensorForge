"""Precision selection and device capability checks."""

from __future__ import annotations

import torch

_DTYPES = {
    "fp32": torch.float32,
    "float32": torch.float32,
    "fp16": torch.float16,
    "float16": torch.float16,
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
}


def parse_dtype(name: str) -> torch.dtype:
    """Convert a CLI precision name to a PyTorch dtype."""

    try:
        return _DTYPES[name.lower()]
    except KeyError as error:
        supported = ", ".join(sorted(_DTYPES))
        raise ValueError(f"unsupported dtype {name!r}; choose one of: {supported}") from error


def validate_dtype_support(dtype: torch.dtype, device: torch.device) -> None:
    """Fail early when a requested low-precision mode is unsupported."""

    if device.type == "cpu" and dtype == torch.float16:
        raise ValueError("FP16 inference requires a CUDA device in TensorForge")
    if device.type == "cuda" and dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        raise ValueError("this CUDA device does not support BF16")
