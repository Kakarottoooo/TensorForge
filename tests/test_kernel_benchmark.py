from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from tensorforge.benchmark.kernel_benchmark import KernelCase, load_kernel_manifest


def test_kernel_case_rejects_unknown_operation() -> None:
    with pytest.raises(ValueError, match="unsupported operation"):
        KernelCase("bad", "unknown", 1, 64, "fp16")


def test_kernel_manifest_loads_compute_ceiling(tmp_path: Path) -> None:
    path = tmp_path / "kernels.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "compute_ceiling": {
                    "fp32_gflops": 10.0,
                    "derivation": "test",
                    "source_url": "https://example.test",
                },
                "cases": [
                    {
                        "name": "rms",
                        "operation": "rms_norm",
                        "rows": 1,
                        "columns": 64,
                        "precision": "fp16",
                    }
                ],
            }
        )
    )

    manifest = load_kernel_manifest(path)

    assert manifest.compute_ceiling.fp32_gflops == 10.0
    assert manifest.cases[0].columns == 64


def test_kernel_manifest_rejects_wrong_schema(tmp_path: Path) -> None:
    path = tmp_path / "kernels.json"
    path.write_text(json.dumps({"schema_version": "0", "cases": []}))
    with pytest.raises(ValueError, match="expected kernel schema"):
        load_kernel_manifest(path)


def test_kernel_manifest_rejects_unknown_case_keys(tmp_path: Path) -> None:
    path = tmp_path / "kernels.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "compute_ceiling": {
                    "fp32_gflops": 10.0,
                    "derivation": "test",
                    "source_url": "https://example.test",
                },
                "cases": [
                    {
                        "name": "rms",
                        "operation": "rms_norm",
                        "rows": 1,
                        "columns": 64,
                        "precision": "fp16",
                        "mystery": True,
                    }
                ],
            }
        )
    )
    with pytest.raises(ValueError, match="unknown kernel case keys"):
        load_kernel_manifest(path)


@pytest.mark.cuda
@pytest.mark.triton
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_copy_ceiling_smoke() -> None:
    pytest.importorskip("triton")
    from tensorforge.benchmark.kernel_benchmark import measure_copy_ceiling

    result = measure_copy_ceiling(
        device=torch.device("cuda"), buffer_mib=8, warmup_ms=2, repetition_ms=5
    )

    assert result["p50_ms"] > 0
    assert result["effective_bandwidth_gbps"] > 0
