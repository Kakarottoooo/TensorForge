from __future__ import annotations

from dataclasses import replace

import pytest

from tensorforge.benchmark.schema import (
    DecodeStrategy,
    ExecutionSpec,
    ModelSpec,
    RequestTrace,
    WorkloadSpec,
)


def test_case_id_is_stable_and_sensitive_to_workload() -> None:
    workload = WorkloadSpec(name="test", model=ModelSpec(), measured_repetitions=2)

    assert workload.case_id == workload.case_id
    assert workload.case_id != replace(workload, batch_size=2, concurrency=2).case_id


def test_context_capacity_is_enforced() -> None:
    with pytest.raises(ValueError, match="context capacity"):
        WorkloadSpec(
            name="too-long",
            model=ModelSpec(max_position_embeddings=128),
            prompt_length=128,
            generation_length=1,
        )


def test_speculative_identity_requires_a_draft_model() -> None:
    with pytest.raises(ValueError, match="draft_model"):
        ExecutionSpec(decode=DecodeStrategy.SPECULATIVE)


def test_trace_rejects_non_monotonic_token_events() -> None:
    with pytest.raises(ValueError, match="monotonic"):
        RequestTrace(
            request_id="bad",
            arrival_ns=0,
            started_ns=1,
            token_timestamps_ns=(4, 3),
            completed_ns=5,
            prompt_tokens=2,
            generated_tokens=2,
            finish_reason="length",
        )
