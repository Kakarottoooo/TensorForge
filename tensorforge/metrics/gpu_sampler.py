"""Low-frequency out-of-process GPU utilization sampling via nvidia-smi."""

from __future__ import annotations

import os
import subprocess
import threading
from dataclasses import dataclass
from statistics import fmean
from typing import IO

from tensorforge.benchmark.schema import UtilizationSummary


@dataclass(frozen=True, slots=True)
class _Sample:
    utilization_pct: float
    memory_used_mib: float


class NvidiaSmiSampler:
    """Sample one physical GPU without importing optional NVML bindings.

    Sampling is deliberately low frequency and runs in a separate process. The
    interval and sample count are persisted so utilization is never presented
    without its measurement resolution.
    """

    def __init__(self, *, gpu_selector: str, interval_ms: int) -> None:
        self._gpu_selector = gpu_selector
        self._interval_ms = interval_ms
        self._process: subprocess.Popen[str] | None = None
        self._reader: threading.Thread | None = None
        self._samples: list[_Sample] = []
        self._error: str | None = None

    def __enter__(self) -> NvidiaSmiSampler:
        command = [
            "nvidia-smi",
            "-i",
            self._gpu_selector,
            "--query-gpu=utilization.gpu,memory.used",
            "--format=csv,noheader,nounits",
            f"--loop-ms={self._interval_ms}",
        ]
        try:
            self._process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                bufsize=1,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
        except OSError as error:
            self._error = f"nvidia-smi unavailable: {error}"
            return self
        if self._process.stdout is None:
            self._error = "nvidia-smi stdout pipe unavailable"
            return self
        self._reader = threading.Thread(
            target=self._read, args=(self._process.stdout,), daemon=True
        )
        self._reader.start()
        return self

    def _read(self, stream: IO[str]) -> None:
        try:
            for line in stream:
                parts = [part.strip() for part in line.split(",")]
                if len(parts) != 2:
                    continue
                try:
                    self._samples.append(_Sample(float(parts[0]), float(parts[1])))
                except ValueError:
                    continue
        except OSError as error:
            self._error = f"nvidia-smi sampling failed: {error}"

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        if self._process is not None:
            self._process.terminate()
            try:
                self._process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=3)
        if self._reader is not None:
            self._reader.join(timeout=3)

    def summary(self) -> UtilizationSummary:
        if not self._samples:
            return UtilizationSummary(
                source="nvidia-smi",
                sample_count=0,
                gpu_utilization_mean_pct=None,
                gpu_utilization_max_pct=None,
                device_memory_used_max_mib=None,
                unavailable_reason=self._error or "run shorter than sampler startup/interval",
            )
        utilization = [sample.utilization_pct for sample in self._samples]
        memory = [sample.memory_used_mib for sample in self._samples]
        return UtilizationSummary(
            source=f"nvidia-smi[{self._gpu_selector}]@{self._interval_ms}ms",
            sample_count=len(self._samples),
            gpu_utilization_mean_pct=fmean(utilization),
            gpu_utilization_max_pct=max(utilization),
            device_memory_used_max_mib=max(memory),
        )
