from __future__ import annotations

import pytest
import torch

from tensorforge.runtime.dtypes import parse_dtype, validate_dtype_support


@pytest.mark.parametrize(
    ("name", "expected"),
    [("fp32", torch.float32), ("FP16", torch.float16), ("bf16", torch.bfloat16)],
)
def test_parse_dtype(name: str, expected: torch.dtype) -> None:
    assert parse_dtype(name) == expected


def test_unsupported_dtype_is_rejected() -> None:
    with pytest.raises(ValueError, match="unsupported dtype"):
        parse_dtype("int8")


def test_cpu_fp16_is_rejected() -> None:
    with pytest.raises(ValueError, match="requires a CUDA"):
        validate_dtype_support(torch.float16, torch.device("cpu"))
