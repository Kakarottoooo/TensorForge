"""OOM-aware benchmark orchestration with explicit synchronization and warmup."""

from __future__ import annotations

import gc
import random
from collections.abc import Callable
from dataclasses import replace

import torch

from tensorforge.benchmark.backends import BenchmarkBackend, EagerBaselineBackend
from tensorforge.benchmark.hardware import resolve_physical_gpu_selector
from tensorforge.benchmark.schema import (
    CaseResult,
    RequestSpec,
    RunSample,
    UtilizationSummary,
    WorkloadSpec,
)
from tensorforge.metrics.gpu_sampler import NvidiaSmiSampler
from tensorforge.metrics.statistics import aggregate_samples
from tensorforge.runtime.dtypes import parse_dtype, validate_dtype_support

BackendFactory = Callable[[WorkloadSpec, torch.device], BenchmarkBackend]


def build_requests(workload: WorkloadSpec, executed_batch_size: int) -> tuple[RequestSpec, ...]:
    """Create deterministic prompts without touching the global Torch RNG."""

    generator = random.Random(workload.seed)
    arrivals = workload.arrival_offsets_ms or (0.0,) * workload.concurrency
    return tuple(
        RequestSpec(
            request_id=f"request-{index:05d}",
            prompt_token_ids=tuple(
                generator.randrange(workload.model.vocab_size)
                for _ in range(workload.prompt_length)
            ),
            max_new_tokens=workload.generation_length,
            arrival_offset_ns=int(arrivals[index] * 1e6),
        )
        for index in range(executed_batch_size)
    )


def _cleanup_cuda(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize(device)


def _memory_value(device: torch.device, name: str) -> int | None:
    if device.type != "cuda":
        return None
    function = getattr(torch.cuda, name)
    return int(function(device))


def run_case(
    workload: WorkloadSpec,
    *,
    device: torch.device,
    backend_factory: BackendFactory = EagerBaselineBackend.create,
) -> CaseResult:
    """Run one case, reducing only batch size after a genuine CUDA OOM."""

    validate_dtype_support(parse_dtype(workload.precision.value), device)
    executed_batch_size = workload.batch_size
    adjustment_reason: str | None = None

    while executed_batch_size >= 1:
        backend: BenchmarkBackend | None = None
        try:
            executed_workload = replace(workload, batch_size=executed_batch_size)
            requests = build_requests(executed_workload, executed_batch_size)
            backend = backend_factory(executed_workload, device)

            for _ in range(workload.warmup_repetitions):
                backend.run(requests)

            samples: list[RunSample] = []
            for repetition in range(workload.measured_repetitions):
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                    torch.cuda.reset_peak_memory_stats(device)
                memory_before = _memory_value(device, "memory_allocated")
                if device.type == "cuda":
                    sampler = NvidiaSmiSampler(
                        gpu_selector=resolve_physical_gpu_selector(device.index or 0),
                        interval_ms=workload.gpu_util_sample_ms,
                    )
                    with sampler:
                        backend_run = backend.run(requests)
                    utilization = sampler.summary()
                else:
                    backend_run = backend.run(requests)
                    utilization = UtilizationSummary(
                        source=None,
                        sample_count=0,
                        gpu_utilization_mean_pct=None,
                        gpu_utilization_max_pct=None,
                        device_memory_used_max_mib=None,
                        unavailable_reason="CPU execution",
                    )
                samples.append(
                    RunSample(
                        repetition=repetition,
                        traces=backend_run.traces,
                        cuda_elapsed_ms=backend_run.cuda_elapsed_ms,
                        memory_allocated_before_bytes=memory_before,
                        memory_allocated_after_bytes=_memory_value(device, "memory_allocated"),
                        peak_memory_allocated_bytes=_memory_value(device, "max_memory_allocated"),
                        utilization=utilization,
                        counters=backend_run.counters,
                    )
                )
            sample_tuple = tuple(samples)
            return CaseResult(
                case_id=workload.case_id,
                requested=workload,
                executed_batch_size=executed_batch_size,
                status="completed",
                adjustment_reason=adjustment_reason,
                samples=sample_tuple,
                aggregate=aggregate_samples(sample_tuple),
            )
        except torch.OutOfMemoryError as error:
            _cleanup_cuda(device)
            if not workload.allow_batch_size_reduction or executed_batch_size == 1:
                return CaseResult(
                    case_id=workload.case_id,
                    requested=workload,
                    executed_batch_size=None,
                    status="oom",
                    adjustment_reason=adjustment_reason,
                    samples=(),
                    aggregate={},
                    error=f"{type(error).__name__}: {error}",
                )
            previous = executed_batch_size
            executed_batch_size = max(1, executed_batch_size // 2)
            adjustment_reason = f"CUDA OOM at batch {previous}; reduced to {executed_batch_size}"
            continue
        finally:
            if backend is not None:
                backend.close()
            _cleanup_cuda(device)

    raise AssertionError("unreachable batch-size planner state")
