"""Manifest and reporting for real-checkpoint backend comparisons."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import re
import uuid
from dataclasses import asdict, dataclass, fields, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from tensorforge.benchmark.checkpoint_backends import (
    TensorForgeCheckpointBackend,
    TransformersCheckpointBackend,
    VllmCheckpointBackend,
    VllmCheckpointEngine,
)
from tensorforge.benchmark.hardware import (
    collect_gpu_runtime_state,
    collect_hardware_metadata,
)
from tensorforge.benchmark.reporting import write_csv, write_json
from tensorforge.benchmark.runner import BackendFactory, run_case
from tensorforge.benchmark.schema import (
    SCHEMA_VERSION,
    BenchmarkSuiteResult,
    CachePolicy,
    CaseResult,
    ExecutionMode,
    SchedulerPolicy,
    WorkloadSpec,
)
from tensorforge.benchmark.suite import workload_from_dict
from tensorforge.runtime.decode_bucket import DecodeExecutionMode, DecodeFusionLevel

CHECKPOINT_BENCHMARK_SCHEMA_VERSION = "1.0"
_COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")


@dataclass(frozen=True, slots=True)
class CheckpointIdentity:
    repository: str
    revision: str

    def __post_init__(self) -> None:
        if not self.repository or "/" not in self.repository:
            raise ValueError("checkpoint repository must be an owner/name identifier")
        if not _COMMIT_PATTERN.fullmatch(self.revision):
            raise ValueError("checkpoint revision must be a 40-character commit SHA")


@dataclass(frozen=True, slots=True)
class CheckpointRuntimeSpec:
    block_size: int
    num_blocks: int
    mode: DecodeExecutionMode
    fusion_level: DecodeFusionLevel
    batch_buckets: tuple[int, ...]
    context_buckets: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.block_size <= 0 or self.num_blocks <= 0:
            raise ValueError("cache dimensions must be positive")
        if not self.batch_buckets or any(value <= 0 for value in self.batch_buckets):
            raise ValueError("batch buckets must contain positive capacities")
        if not self.context_buckets or any(value <= 0 for value in self.context_buckets):
            raise ValueError("context buckets must contain positive capacities")


@dataclass(frozen=True, slots=True)
class CheckpointBenchmarkManifest:
    checkpoint: CheckpointIdentity
    runtime: CheckpointRuntimeSpec
    workloads: tuple[WorkloadSpec, ...]

    def __post_init__(self) -> None:
        if not self.workloads:
            raise ValueError("checkpoint benchmark requires at least one workload")
        for workload in self.workloads:
            if workload.batch_size != workload.concurrency:
                raise ValueError("real-model cases require batch_size == concurrency")
            if workload.batch_size > max(self.runtime.batch_buckets):
                raise ValueError(f"case {workload.name} exceeds the largest batch bucket")
            maximum_context = workload.prompt_length + workload.generation_length
            if maximum_context > max(self.runtime.context_buckets):
                raise ValueError(f"case {workload.name} exceeds the largest context bucket")
            required_blocks = (
                (maximum_context + self.runtime.block_size - 1)
                // self.runtime.block_size
                * workload.batch_size
            )
            if required_blocks > self.runtime.num_blocks:
                raise ValueError(f"case {workload.name} exceeds paged KV capacity")


def _strict_keys(value: dict[str, Any], allowed: set[str], context: str) -> None:
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"unknown {context} keys: {sorted(unknown)}")


def load_checkpoint_manifest(path: Path) -> CheckpointBenchmarkManifest:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("checkpoint manifest must be an object")
    _strict_keys(payload, {"schema_version", "checkpoint", "runtime", "cases"}, "manifest")
    if payload.get("schema_version") != CHECKPOINT_BENCHMARK_SCHEMA_VERSION:
        raise ValueError(
            f"manifest must declare schema_version {CHECKPOINT_BENCHMARK_SCHEMA_VERSION!r}"
        )
    checkpoint_value = payload.get("checkpoint")
    runtime_value = payload.get("runtime")
    case_values = payload.get("cases")
    if not isinstance(checkpoint_value, dict) or not isinstance(runtime_value, dict):
        raise TypeError("checkpoint and runtime entries must be objects")
    if not isinstance(case_values, list) or not case_values:
        raise ValueError("cases must be a non-empty list")
    _strict_keys(
        checkpoint_value,
        {field.name for field in fields(CheckpointIdentity)},
        "checkpoint",
    )
    _strict_keys(
        runtime_value,
        {field.name for field in fields(CheckpointRuntimeSpec)},
        "runtime",
    )
    runtime_raw = dict(runtime_value)
    runtime_raw["mode"] = DecodeExecutionMode(runtime_raw["mode"])
    runtime_raw["fusion_level"] = DecodeFusionLevel(runtime_raw["fusion_level"])
    runtime_raw["batch_buckets"] = tuple(runtime_raw["batch_buckets"])
    runtime_raw["context_buckets"] = tuple(runtime_raw["context_buckets"])
    parsed_workloads = tuple(workload_from_dict(case) for case in case_values)
    workloads = tuple(
        replace(
            workload,
            execution=replace(
                workload.execution,
                backend="tensorforge",
                mode=ExecutionMode(runtime_raw["mode"].value),
                cache=CachePolicy.PAGED,
                scheduler=SchedulerPolicy.CONTINUOUS,
            ),
        )
        for workload in parsed_workloads
    )
    return CheckpointBenchmarkManifest(
        checkpoint=CheckpointIdentity(**checkpoint_value),
        runtime=CheckpointRuntimeSpec(**runtime_raw),
        workloads=workloads,
    )


def _checkpoint_metadata(
    manifest: CheckpointBenchmarkManifest, checkpoint_dir: Path
) -> dict[str, Any]:
    config_path = checkpoint_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"missing checkpoint config: {config_path}")
    weights_path = checkpoint_dir / "model.safetensors"
    if not weights_path.is_file():
        raise FileNotFoundError(f"missing checkpoint weights: {weights_path}")

    weights_digest = hashlib.sha256()
    with weights_path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            weights_digest.update(chunk)
    return {
        **asdict(manifest.checkpoint),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "weights_sha256": weights_digest.hexdigest(),
        "weights_size_bytes": weights_path.stat().st_size,
        "local_files_only": True,
    }


def _installed_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _backend_factory(
    name: str,
    *,
    checkpoint_dir: Path,
    runtime: CheckpointRuntimeSpec,
    vllm_engine: VllmCheckpointEngine | None = None,
) -> BackendFactory:
    if name == "tensorforge":
        return lambda workload, device: TensorForgeCheckpointBackend(
            workload=workload,
            device=device,
            checkpoint_dir=checkpoint_dir,
            block_size=runtime.block_size,
            num_blocks=runtime.num_blocks,
            mode=runtime.mode,
            fusion_level=runtime.fusion_level,
            batch_buckets=runtime.batch_buckets,
            context_buckets=runtime.context_buckets,
        )
    if name == "transformers_sdpa":
        return lambda workload, device: TransformersCheckpointBackend(
            workload=workload,
            device=device,
            checkpoint_dir=checkpoint_dir,
        )
    if name == "vllm":
        if vllm_engine is None:
            raise ValueError("vLLM backend requires a shared engine")
        return lambda workload, device: VllmCheckpointBackend(
            workload=workload,
            engine=vllm_engine,
        )
    raise ValueError(f"unsupported checkpoint backend: {name}")


def _with_output_identity(case: CaseResult) -> CaseResult:
    digests = {
        str(sample.counters["generated_token_sha256"])
        for sample in case.samples
        if "generated_token_sha256" in sample.counters
    }
    aggregate = dict(case.aggregate)
    if digests:
        aggregate["generated_token_sha256"] = next(iter(digests))
        aggregate["generated_tokens_deterministic"] = len(digests) == 1
    return replace(case, aggregate=aggregate)


def run_checkpoint_suite(
    manifest: CheckpointBenchmarkManifest,
    *,
    backend: str,
    checkpoint_dir: Path,
    device: torch.device,
    repository: Path,
) -> BenchmarkSuiteResult:
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("real-checkpoint benchmarks require CUDA")
    started = datetime.now(UTC)
    hardware = collect_hardware_metadata(repository).to_dict()
    hardware["gpu_runtime_state_before"] = collect_gpu_runtime_state(device.index or 0)
    hardware["checkpoint"] = _checkpoint_metadata(manifest, checkpoint_dir)
    hardware["benchmark_backend"] = backend
    hardware["backend_packages"] = {
        package: _installed_version(package)
        for package in ("transformers", "safetensors", "vllm")
    }
    hardware["measurement_boundary"] = {
        "tensorforge": (
            "scheduled host token IDs through terminal request release; model loading and "
            "tokenization excluded; scheduler token selection synchronizes each decode step"
        ),
        "transformers_sdpa": (
            "simultaneous host token-ID batch through final greedy token; model loading and "
            "tokenization excluded; CUDA explicitly synchronized at each emitted token"
        ),
        "vllm": (
            "offline LLM.generate token-ID batch through completed outputs; model loading and "
            "tokenization excluded; TTFT/finish use vLLM metrics and intermediate token "
            "timestamps are interpolated"
        ),
    }[backend]
    hardware["runtime_config"] = {
        "precision": manifest.workloads[0].precision.value,
        "block_size": manifest.runtime.block_size,
        "num_blocks": manifest.runtime.num_blocks,
        "decode_mode": manifest.runtime.mode.value,
        "fusion_level": manifest.runtime.fusion_level.value,
        "batch_buckets": manifest.runtime.batch_buckets,
        "context_buckets": manifest.runtime.context_buckets,
        "vllm_gpu_memory_utilization": 0.70 if backend == "vllm" else None,
        "vllm_enforce_eager": False if backend == "vllm" else None,
    }
    vllm_engine = (
        VllmCheckpointEngine(
            checkpoint_dir=checkpoint_dir,
            precision=manifest.workloads[0].precision.value,
            max_model_len=max(manifest.runtime.context_buckets),
            max_num_seqs=max(manifest.runtime.batch_buckets),
        )
        if backend == "vllm"
        else None
    )
    factory = _backend_factory(
        backend,
        checkpoint_dir=checkpoint_dir,
        runtime=manifest.runtime,
        vllm_engine=vllm_engine,
    )
    cases: list[CaseResult] = []
    for workload in manifest.workloads:
        execution = replace(
            workload.execution,
            backend=backend,
            mode=(
                ExecutionMode.EXTERNAL
                if backend == "transformers_sdpa"
                else ExecutionMode(manifest.runtime.mode.value)
            ),
            cache=(CachePolicy.CONTIGUOUS if backend == "transformers_sdpa" else CachePolicy.PAGED),
            scheduler=(
                SchedulerPolicy.NONE
                if backend == "transformers_sdpa"
                else SchedulerPolicy.CONTINUOUS
            ),
        )
        requested = replace(workload, execution=execution)
        try:
            cases.append(
                _with_output_identity(
                    run_case(requested, device=device, backend_factory=factory)
                )
            )
        except Exception as error:
            cases.append(
                CaseResult(
                    case_id=requested.case_id,
                    requested=requested,
                    executed_batch_size=None,
                    status="failed",
                    adjustment_reason=None,
                    samples=(),
                    aggregate={},
                    error=f"{type(error).__name__}: {error}",
                )
            )
    hardware["gpu_runtime_state_after"] = collect_gpu_runtime_state(device.index or 0)
    result = BenchmarkSuiteResult(
        schema_version=SCHEMA_VERSION,
        suite_id=str(uuid.uuid4()),
        started_at_utc=started.isoformat(),
        finished_at_utc=datetime.now(UTC).isoformat(),
        hardware=hardware,
        cases=tuple(cases),
    )
    if vllm_engine is not None:
        vllm_engine.close()
    return result


def _metric(case: CaseResult, metric: str, statistic: str) -> float | None:
    distribution = case.aggregate.get(metric)
    if not isinstance(distribution, dict):
        return None
    value = distribution.get(statistic)
    return float(value) if isinstance(value, int | float) else None


def _display(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def write_checkpoint_reports(
    result: BenchmarkSuiteResult, output_directory: Path
) -> tuple[Path, Path, Path]:
    output_directory.mkdir(parents=True, exist_ok=True)
    json_path = output_directory / "checkpoint.json"
    csv_path = output_directory / "checkpoint.csv"
    markdown_path = output_directory / "checkpoint.md"
    write_json(result, json_path)
    write_csv(result, csv_path)
    checkpoint = result.hardware["checkpoint"]
    gpu = result.hardware["gpus"][0]
    lines = [
        "# TensorForge real-checkpoint benchmark",
        "",
        f"- Backend: `{result.hardware['benchmark_backend']}`",
        f"- Checkpoint: `{checkpoint['repository']}@{checkpoint['revision']}`",
        f"- Config SHA256: `{checkpoint['config_sha256']}`",
        f"- GPU: {gpu['name']} (compute capability {gpu['compute_capability']})",
        f"- Git: `{result.hardware['git_commit']}`; dirty: `{result.hardware['git_dirty']}`",
        "",
        "| Case | Batch | Prompt/output | Latency P50/P95/P99 ms | "
        "TTFT P50/P95/P99 ms | TPOT P50/P95/P99 ms | Output tok/s | Status |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for case in result.cases:
        latency = "/".join(
            _display(_metric(case, "request_latency_ms", quantile))
            for quantile in ("p50", "p95", "p99")
        )
        ttft = "/".join(
            _display(_metric(case, "ttft_ms", quantile))
            for quantile in ("p50", "p95", "p99")
        )
        tpot = "/".join(
            _display(_metric(case, "tpot_ms", quantile))
            for quantile in ("p50", "p95", "p99")
        )
        lines.append(
            f"| {case.requested.name} | {case.executed_batch_size or 'n/a'} | "
            f"{case.requested.prompt_length}/{case.requested.generation_length} | "
            f"{latency} | {ttft} | {tpot} | "
            f"{_display(_metric(case, 'output_tokens_per_second', 'mean'))} | "
            f"{case.status} |"
        )
        if case.error:
            lines.append(f"\nFailure for `{case.requested.name}`: `{case.error}`\n")
    lines.extend(
        [
            "",
            "## Measurement and claim boundary",
            "",
            result.hardware["measurement_boundary"],
            "",
            "Only rows with the same pinned checkpoint, input-token plan, precision, GPU, "
            "generation semantics, and synchronization boundary are comparable. TensorForge "
            "uses parallel SDPA prefill followed by its transactional paged decode path; the "
            "Transformers reference uses SDPA and its standard contiguous KV cache. Loading and "
            "tokenization are excluded. Output-token SHA256 values in JSON expose semantic drift.",
            "",
        ]
    )
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, csv_path, markdown_path
