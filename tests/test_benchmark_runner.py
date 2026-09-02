from __future__ import annotations

import time

import torch

from tensorforge.benchmark.backends import EagerBaselineBackend
from tensorforge.benchmark.runner import run_case
from tensorforge.benchmark.schema import (
    BackendCapabilities,
    BackendRun,
    ModelSpec,
    Precision,
    RequestSpec,
    RequestTrace,
    WorkloadSpec,
)


def _cpu_workload(**overrides: object) -> WorkloadSpec:
    values: dict[str, object] = {
        "name": "cpu-smoke",
        "model": ModelSpec(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=64,
        ),
        "prompt_length": 8,
        "generation_length": 2,
        "batch_size": 2,
        "concurrency": 2,
        "precision": Precision.FP32,
        "warmup_repetitions": 0,
        "measured_repetitions": 2,
    }
    values.update(overrides)
    return WorkloadSpec(**values)  # type: ignore[arg-type]


def test_cpu_eager_benchmark_produces_request_metrics() -> None:
    result = run_case(_cpu_workload(), device=torch.device("cpu"))

    assert result.status == "completed"
    assert result.executed_batch_size == 2
    assert len(result.samples) == 2
    assert result.aggregate["ttft_ms"]
    assert result.aggregate["output_tokens_per_second"]


class OomUntilBatchOne:
    name = "oom-test"
    capabilities = BackendCapabilities()

    def __init__(self, workload: WorkloadSpec) -> None:
        self.workload = workload

    def run(self, requests: tuple[RequestSpec, ...]) -> BackendRun:
        if len(requests) > 1:
            raise torch.OutOfMemoryError("synthetic allocator boundary")
        now = time.perf_counter_ns()
        trace = RequestTrace(
            request_id=requests[0].request_id,
            arrival_ns=now,
            started_ns=now,
            token_timestamps_ns=(now + 1,),
            completed_ns=now + 2,
            prompt_tokens=len(requests[0].prompt_token_ids),
            generated_tokens=1,
            finish_reason="length",
        )
        return BackendRun((trace,), None, {})

    def close(self) -> None:
        pass


def test_oom_reduction_is_explicit() -> None:
    workload = _cpu_workload(generation_length=1, measured_repetitions=1)

    result = run_case(
        workload,
        device=torch.device("cpu"),
        backend_factory=lambda case, _device: OomUntilBatchOne(case),
    )

    assert result.status == "completed"
    assert result.executed_batch_size == 1
    assert result.adjustment_reason == "CUDA OOM at batch 2; reduced to 1"


@torch.inference_mode()
def test_backend_reports_recomputation_counter() -> None:
    workload = _cpu_workload(batch_size=1, concurrency=1, measured_repetitions=1)
    backend = EagerBaselineBackend.create(workload, torch.device("cpu"))
    request = RequestSpec("r", tuple(range(8)), 2)

    result = backend.run((request,))

    assert result.counters["model_forward_calls"] == 2
    assert result.counters["tokens_recomputed"] == 17
