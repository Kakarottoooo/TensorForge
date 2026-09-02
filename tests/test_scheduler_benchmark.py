from __future__ import annotations

import json
from pathlib import Path

import pytest

from tensorforge.benchmark.scheduler_benchmark import (
    SchedulerWorkload,
    build_request_plan,
    load_scheduler_manifest,
)


def test_phase5_manifest_is_valid_and_seeded_plan_is_reproducible() -> None:
    repository = Path(__file__).resolve().parents[1]
    manifest = load_scheduler_manifest(repository / "benchmarks" / "phase5-scheduler.json")

    first = build_request_plan(manifest.workloads[1], vocab_size=manifest.model.vocab_size)
    second = build_request_plan(manifest.workloads[1], vocab_size=manifest.model.vocab_size)

    assert first == second
    assert len(first) == 24
    assert sum(item.cancellation_step is not None for item in first) == 6
    assert {item.arrival_step for item in first} <= set(range(13))


def test_scheduler_workload_rejects_invalid_cancellation_fraction() -> None:
    with pytest.raises(ValueError, match="cancellation_fraction"):
        SchedulerWorkload(
            name="invalid",
            request_count=1,
            prompt_length_choices=(1,),
            generation_length_choices=(1,),
            max_arrival_step=0,
            cancellation_fraction=1.1,
            cancellation_delay_steps=0,
            seed=1,
        )


def test_scheduler_manifest_rejects_unknown_workload_fields(tmp_path: Path) -> None:
    path = tmp_path / "scheduler.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "model": {
                    "vocab_size": 128,
                    "hidden_size": 64,
                    "intermediate_size": 160,
                    "num_hidden_layers": 2,
                    "num_attention_heads": 4,
                    "num_key_value_heads": 2,
                    "max_position_embeddings": 64
                },
                "workloads": [
                    {
                        "name": "invalid",
                        "request_count": 8,
                        "prompt_length_choices": [4, 8],
                        "generation_length_choices": [2, 4],
                        "max_arrival_step": 0,
                        "cancellation_fraction": 0.0,
                        "cancellation_delay_steps": 1,
                        "seed": 7,
                        "uncontrolled": True,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unknown scheduler workload keys"):
        load_scheduler_manifest(path)
