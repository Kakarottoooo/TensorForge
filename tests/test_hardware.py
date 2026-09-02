from __future__ import annotations

from dataclasses import replace

from tensorforge.benchmark.hardware import HardwareMetadata


def _metadata() -> HardwareMetadata:
    return HardwareMetadata(
        operating_system="os",
        kernel="kernel",
        python_version="3.11",
        cpu="cpu",
        logical_cpu_count=8,
        torch_version="2.5",
        torch_cuda_runtime="12.4",
        cudnn_version=90100,
        triton_version="3.1",
        nvidia_driver_cuda_version="12.8",
        cuda_available=True,
        cuda_device_count=1,
        visible_cuda_devices=None,
        gpus=({"name": "gpu"},),
        git_commit="first",
        git_dirty=False,
    )


def test_hardware_fingerprint_excludes_repository_state() -> None:
    clean = _metadata()
    other_commit = replace(clean, git_commit="second", git_dirty=True)

    assert clean.to_dict()["fingerprint_sha256"] == other_commit.to_dict()["fingerprint_sha256"]


def test_hardware_fingerprint_changes_with_software_stack() -> None:
    original = _metadata()
    other_runtime = replace(original, torch_cuda_runtime="12.6")

    assert original.to_dict()["fingerprint_sha256"] != other_runtime.to_dict()["fingerprint_sha256"]
