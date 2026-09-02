"""Request-level latency and throughput statistics without NumPy dependency."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from statistics import fmean

from tensorforge.benchmark.schema import RequestTrace, RunSample


def percentile(values: Iterable[float], quantile: float) -> float:
    """Return a linearly interpolated percentile (Hyndman-Fan type 7)."""

    samples = sorted(values)
    if not samples:
        raise ValueError("at least one sample is required")
    if not 0 <= quantile <= 1:
        raise ValueError("quantile must be between zero and one")
    position = (len(samples) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return samples[lower]
    weight = position - lower
    return samples[lower] * (1 - weight) + samples[upper] * weight


@dataclass(frozen=True, slots=True)
class Distribution:
    count: int
    minimum: float
    maximum: float
    mean: float
    p50: float
    p95: float
    p99: float


def summarize(values: Iterable[float]) -> Distribution:
    samples = list(values)
    if not samples:
        raise ValueError("at least one sample is required")
    return Distribution(
        count=len(samples),
        minimum=min(samples),
        maximum=max(samples),
        mean=fmean(samples),
        p50=percentile(samples, 0.50),
        p95=percentile(samples, 0.95),
        p99=percentile(samples, 0.99),
    )


def _trace_metrics(trace: RequestTrace) -> dict[str, float | None]:
    first_token_ns = trace.token_timestamps_ns[0]
    ttft_ms = (first_token_ns - trace.arrival_ns) / 1e6
    latency_ms = (trace.completed_ns - trace.arrival_ns) / 1e6
    queue_ms = (trace.started_ns - trace.arrival_ns) / 1e6
    if trace.generated_tokens > 1:
        tpot_ms = (trace.completed_ns - first_token_ns) / (trace.generated_tokens - 1) / 1e6
    else:
        tpot_ms = None
    return {
        "ttft_ms": ttft_ms,
        "tpot_ms": tpot_ms,
        "request_latency_ms": latency_ms,
        "queue_ms": queue_ms,
    }


def aggregate_samples(samples: tuple[RunSample, ...]) -> dict[str, object]:
    """Aggregate per-request samples while retaining per-run throughput."""

    if not samples:
        return {}
    trace_metrics = [_trace_metrics(trace) for sample in samples for trace in sample.traces]
    aggregate: dict[str, object] = {}
    for metric in ("ttft_ms", "tpot_ms", "request_latency_ms", "queue_ms"):
        values: list[float] = []
        for item in trace_metrics:
            value = item[metric]
            if value is not None:
                values.append(value)
        if values:
            aggregate[metric] = asdict(summarize(values))

    throughput_values: list[float] = []
    for sample in samples:
        earliest = min(trace.arrival_ns for trace in sample.traces)
        latest = max(trace.completed_ns for trace in sample.traces)
        elapsed_seconds = (latest - earliest) / 1e9
        if elapsed_seconds <= 0:
            raise ValueError("sample lifetime must be positive to compute throughput")
        generated_tokens = sum(trace.generated_tokens for trace in sample.traces)
        throughput_values.append(generated_tokens / elapsed_seconds)
    aggregate["output_tokens_per_second"] = asdict(summarize(throughput_values))

    cuda_values = [
        sample.cuda_elapsed_ms for sample in samples if sample.cuda_elapsed_ms is not None
    ]
    if cuda_values:
        aggregate["cuda_elapsed_ms"] = asdict(summarize(cuda_values))

    peak_values = [
        float(sample.peak_memory_allocated_bytes)
        for sample in samples
        if sample.peak_memory_allocated_bytes is not None
    ]
    if peak_values:
        aggregate["peak_memory_allocated_bytes"] = asdict(summarize(peak_values))

    utilization_values = [
        sample.utilization.gpu_utilization_mean_pct
        for sample in samples
        if sample.utilization.gpu_utilization_mean_pct is not None
    ]
    if utilization_values:
        aggregate["gpu_utilization_mean_pct"] = asdict(summarize(utilization_values))
    return aggregate
