"""Exact hardware and software identity attached to every benchmark suite."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import torch


def _run_text(command: list[str]) -> str | None:
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip() or None


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _cpu_name() -> str:
    if os.name == "nt":
        value = _run_text(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "(Get-CimInstance Win32_Processor | Select-Object -First 1).Name",
            ]
        )
        if value:
            return value.strip()
    if platform.system() == "Linux":
        try:
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
        except OSError:
            pass
    return platform.processor() or "unknown"


def _nvidia_smi_rows() -> list[dict[str, str]]:
    output = _run_text(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ]
    )
    if output is None:
        return []
    rows: list[dict[str, str]] = []
    for line in output.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) == 5:
            rows.append(
                {
                    "physical_index": parts[0],
                    "uuid": parts[1],
                    "name": parts[2],
                    "vram_mib": parts[3],
                    "driver_version": parts[4],
                }
            )
    return rows


def _nvidia_driver_cuda_version() -> str | None:
    output = _run_text(["nvidia-smi"])
    if output is None:
        return None
    match = re.search(r"CUDA Version:\s*([0-9.]+)", output)
    return match.group(1) if match else None


def resolve_physical_gpu_selector(visible_index: int) -> str:
    """Map a PyTorch-visible index to an nvidia-smi index or UUID."""

    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not visible:
        return str(visible_index)
    selectors = [selector.strip() for selector in visible.split(",") if selector.strip()]
    if visible_index >= len(selectors):
        return str(visible_index)
    return selectors[visible_index]


def _smi_row_for_visible(rows: list[dict[str, str]], visible_index: int) -> dict[str, str]:
    selector = resolve_physical_gpu_selector(visible_index)
    for row in rows:
        if row["physical_index"] == selector or row["uuid"].startswith(selector):
            return row
    return rows[visible_index] if visible_index < len(rows) else {}


def _git_identity(repository: Path | None) -> tuple[str | None, bool | None]:
    if repository is None:
        return None, None
    commit = _run_text(["git", "-C", str(repository), "rev-parse", "HEAD"])
    try:
        status = subprocess.run(
            ["git", "-C", str(repository), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
    except (OSError, subprocess.SubprocessError):
        return commit, None
    return commit, bool(status.stdout.strip())


@dataclass(frozen=True, slots=True)
class HardwareMetadata:
    operating_system: str
    kernel: str
    python_version: str
    cpu: str
    logical_cpu_count: int | None
    torch_version: str
    torch_cuda_runtime: str | None
    cudnn_version: int | None
    triton_version: str | None
    nvidia_driver_cuda_version: str | None
    cuda_available: bool
    cuda_device_count: int
    visible_cuda_devices: str | None
    gpus: tuple[dict[str, Any], ...]
    git_commit: str | None
    git_dirty: bool | None

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        fingerprint_fields = {
            key: item for key, item in value.items() if key not in {"git_commit", "git_dirty"}
        }
        encoded = json.dumps(fingerprint_fields, sort_keys=True, separators=(",", ":"), default=str)
        value["fingerprint_sha256"] = hashlib.sha256(encoded.encode()).hexdigest()
        return value


def collect_hardware_metadata(repository: Path | None = None) -> HardwareMetadata:
    """Collect metadata without requiring NVML or a CUDA toolkit installation."""

    smi_rows = _nvidia_smi_rows()
    gpus: list[dict[str, Any]] = []
    if torch.cuda.is_available():
        for visible_index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(visible_index)
            smi = _smi_row_for_visible(smi_rows, visible_index)
            gpus.append(
                {
                    "visible_index": visible_index,
                    "physical_index": smi.get("physical_index"),
                    "gpu_uuid_sha256_16": hashlib.sha256(smi["uuid"].encode()).hexdigest()[:16]
                    if smi.get("uuid")
                    else None,
                    "name": properties.name,
                    "compute_capability": f"{properties.major}.{properties.minor}",
                    "total_memory_bytes": properties.total_memory,
                    "vram_mib_nvidia_smi": smi.get("vram_mib"),
                    "driver_version": smi.get("driver_version"),
                    "multiprocessor_count": properties.multi_processor_count,
                }
            )
    commit, dirty = _git_identity(repository)
    cudnn_backend: Any = torch.backends.cudnn
    cudnn_version = (
        cast(int | None, cudnn_backend.version()) if cudnn_backend.is_available() else None
    )
    return HardwareMetadata(
        operating_system=platform.platform(),
        kernel=platform.release(),
        python_version=platform.python_version(),
        cpu=_cpu_name(),
        logical_cpu_count=os.cpu_count(),
        torch_version=torch.__version__,
        torch_cuda_runtime=torch.version.cuda,
        cudnn_version=cudnn_version,
        triton_version=_package_version("triton"),
        nvidia_driver_cuda_version=_nvidia_driver_cuda_version(),
        cuda_available=torch.cuda.is_available(),
        cuda_device_count=torch.cuda.device_count(),
        visible_cuda_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        gpus=tuple(gpus),
        git_commit=commit,
        git_dirty=dirty,
    )
