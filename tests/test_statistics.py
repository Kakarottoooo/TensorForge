from __future__ import annotations

import pytest

from tensorforge.benchmark.schema import RequestTrace, RunSample, UtilizationSummary
from tensorforge.metrics.statistics import aggregate_samples, percentile, summarize


def test_percentile_uses_linear_interpolation() -> None:
    values = [1.0, 2.0, 3.0, 4.0]

    assert percentile(values, 0.5) == 2.5
    assert percentile(values, 0.95) == pytest.approx(3.85)
    assert percentile(values, 0.99) == pytest.approx(3.97)


def test_distribution_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="at least one"):
        summarize([])


def test_request_level_aggregation() -> None:
    trace = RequestTrace(
        request_id="r1",
        arrival_ns=0,
        started_ns=1_000_000,
        token_timestamps_ns=(3_000_000, 5_000_000, 7_000_000),
        completed_ns=7_000_000,
        prompt_tokens=8,
        generated_tokens=3,
        finish_reason="length",
    )
    sample = RunSample(
        repetition=0,
        traces=(trace,),
        cuda_elapsed_ms=5.5,
        memory_allocated_before_bytes=10,
        memory_allocated_after_bytes=10,
        peak_memory_allocated_bytes=20,
        utilization=UtilizationSummary("test", 1, 50.0, 50.0, 100.0),
        counters={},
    )

    aggregate = aggregate_samples((sample,))

    assert aggregate["ttft_ms"]["p50"] == 3.0  # type: ignore[index]
    assert aggregate["tpot_ms"]["p50"] == 2.0  # type: ignore[index]
    assert aggregate["request_latency_ms"]["p50"] == 7.0  # type: ignore[index]
    assert aggregate["output_tokens_per_second"]["mean"] == pytest.approx(3 / 0.007)  # type: ignore[index]
