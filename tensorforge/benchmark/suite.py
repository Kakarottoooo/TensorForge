"""Manifest loading and multi-case suite execution."""

from __future__ import annotations

import json
import uuid
from dataclasses import fields
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from tensorforge.benchmark.hardware import collect_hardware_metadata
from tensorforge.benchmark.runner import run_case
from tensorforge.benchmark.schema import (
    SCHEMA_VERSION,
    BenchmarkSuiteResult,
    CachePolicy,
    CaseResult,
    DecodeStrategy,
    ExecutionMode,
    ExecutionSpec,
    ModelSpec,
    Precision,
    SchedulerPolicy,
    WorkloadSpec,
)


def _strict_keys(value: dict[str, Any], allowed: set[str], context: str) -> None:
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"unknown {context} keys: {sorted(unknown)}")


def _workload_from_dict(value: dict[str, Any]) -> WorkloadSpec:
    allowed = {field.name for field in fields(WorkloadSpec)}
    _strict_keys(value, allowed, "workload")
    raw = dict(value)
    model_value = raw.pop("model")
    execution_value = raw.pop("execution", {})
    if not isinstance(model_value, dict) or not isinstance(execution_value, dict):
        raise TypeError("model and execution entries must be objects")
    _strict_keys(model_value, {field.name for field in fields(ModelSpec)}, "model")
    _strict_keys(execution_value, {field.name for field in fields(ExecutionSpec)}, "execution")
    execution_raw = dict(execution_value)
    if "mode" in execution_raw:
        execution_raw["mode"] = ExecutionMode(execution_raw["mode"])
    if "cache" in execution_raw:
        execution_raw["cache"] = CachePolicy(execution_raw["cache"])
    if "scheduler" in execution_raw:
        execution_raw["scheduler"] = SchedulerPolicy(execution_raw["scheduler"])
    if "decode" in execution_raw:
        execution_raw["decode"] = DecodeStrategy(execution_raw["decode"])
    if "extra_tags" in execution_raw:
        execution_raw["extra_tags"] = tuple(execution_raw["extra_tags"])
    if "precision" in raw:
        raw["precision"] = Precision(raw["precision"])
    if "arrival_offsets_ms" in raw:
        raw["arrival_offsets_ms"] = tuple(raw["arrival_offsets_ms"])
    return WorkloadSpec(
        model=ModelSpec(**model_value),
        execution=ExecutionSpec(**execution_raw),
        **raw,
    )


def load_manifest(path: Path) -> tuple[WorkloadSpec, ...]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"manifest must declare schema_version {SCHEMA_VERSION!r}")
    cases = value.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("manifest cases must be a non-empty list")
    return tuple(_workload_from_dict(case) for case in cases)


def run_suite(
    workloads: tuple[WorkloadSpec, ...], *, device: torch.device, repository: Path
) -> BenchmarkSuiteResult:
    started = datetime.now(UTC)
    hardware = collect_hardware_metadata(repository).to_dict()
    cases: list[CaseResult] = []
    for workload in workloads:
        try:
            cases.append(run_case(workload, device=device))
        except Exception as error:
            cases.append(
                CaseResult(
                    case_id=workload.case_id,
                    requested=workload,
                    executed_batch_size=None,
                    status="failed",
                    adjustment_reason=None,
                    samples=(),
                    aggregate={},
                    error=f"{type(error).__name__}: {error}",
                )
            )
    finished = datetime.now(UTC)
    return BenchmarkSuiteResult(
        schema_version=SCHEMA_VERSION,
        suite_id=str(uuid.uuid4()),
        started_at_utc=started.isoformat(),
        finished_at_utc=finished.isoformat(),
        hardware=hardware,
        cases=tuple(cases),
    )
