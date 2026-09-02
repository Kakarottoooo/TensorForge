"""Lossless JSON plus flat CSV and review-oriented Markdown reports."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from tensorforge.benchmark.schema import BenchmarkSuiteResult, CaseResult


def write_json(result: BenchmarkSuiteResult, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(result), indent=2, sort_keys=True, default=str) + "\n")


def _metric(case: CaseResult, metric: str, statistic: str) -> float | int | None:
    value = case.aggregate.get(metric)
    if not isinstance(value, dict):
        return None
    result = value.get(statistic)
    return result if isinstance(result, int | float) else None


def _case_row(result: BenchmarkSuiteResult, case: CaseResult) -> dict[str, Any]:
    hardware = result.hardware
    gpus = hardware.get("gpus") or []
    first_gpu = gpus[0] if gpus else {}
    return {
        "schema_version": result.schema_version,
        "suite_id": result.suite_id,
        "case_id": case.case_id,
        "name": case.requested.name,
        "status": case.status,
        "backend": case.requested.execution.backend,
        "execution_mode": case.requested.execution.mode.value,
        "precision": case.requested.precision.value,
        "prompt_length": case.requested.prompt_length,
        "generation_length": case.requested.generation_length,
        "requested_batch_size": case.requested.batch_size,
        "executed_batch_size": case.executed_batch_size,
        "concurrency": case.requested.concurrency,
        "warmups": case.requested.warmup_repetitions,
        "repetitions": case.requested.measured_repetitions,
        "ttft_p50_ms": _metric(case, "ttft_ms", "p50"),
        "ttft_p95_ms": _metric(case, "ttft_ms", "p95"),
        "ttft_p99_ms": _metric(case, "ttft_ms", "p99"),
        "tpot_p50_ms": _metric(case, "tpot_ms", "p50"),
        "tpot_p95_ms": _metric(case, "tpot_ms", "p95"),
        "tpot_p99_ms": _metric(case, "tpot_ms", "p99"),
        "request_latency_p50_ms": _metric(case, "request_latency_ms", "p50"),
        "request_latency_p95_ms": _metric(case, "request_latency_ms", "p95"),
        "request_latency_p99_ms": _metric(case, "request_latency_ms", "p99"),
        "output_tokens_per_second_mean": _metric(case, "output_tokens_per_second", "mean"),
        "cuda_elapsed_mean_ms": _metric(case, "cuda_elapsed_ms", "mean"),
        "peak_memory_allocated_max_bytes": _metric(case, "peak_memory_allocated_bytes", "maximum"),
        "gpu_utilization_mean_pct": _metric(case, "gpu_utilization_mean_pct", "mean"),
        "gpu_name": first_gpu.get("name"),
        "gpu_total_memory_bytes": first_gpu.get("total_memory_bytes"),
        "driver_version": first_gpu.get("driver_version"),
        "torch_version": hardware.get("torch_version"),
        "torch_cuda_runtime": hardware.get("torch_cuda_runtime"),
        "triton_version": hardware.get("triton_version"),
        "hardware_fingerprint": hardware.get("fingerprint_sha256"),
        "git_commit": hardware.get("git_commit"),
        "git_dirty": hardware.get("git_dirty"),
        "adjustment_reason": case.adjustment_reason,
        "error": case.error,
    }


def write_csv(result: BenchmarkSuiteResult, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [_case_row(result, case) for case in result.cases]
    if not rows:
        raise ValueError("cannot write a CSV report with no cases")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _format_metric(value: float | int | None, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, int):
        return str(value)
    return f"{value:.{digits}f}"


def write_markdown(result: BenchmarkSuiteResult, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    hardware = result.hardware
    gpus = hardware.get("gpus") or []
    gpu_names = ", ".join(str(gpu.get("name", "unknown")) for gpu in gpus) or "none"
    lines = [
        "# TensorForge benchmark report",
        "",
        f"- Suite ID: `{result.suite_id}`",
        f"- Schema: `{result.schema_version}`",
        f"- Started: `{result.started_at_utc}`",
        f"- Finished: `{result.finished_at_utc}`",
        f"- GPU(s): {gpu_names}",
        f"- PyTorch: `{hardware.get('torch_version')}`; CUDA runtime: "
        f"`{hardware.get('torch_cuda_runtime')}`; driver: "
        f"`{gpus[0].get('driver_version') if gpus else None}`",
        f"- Hardware fingerprint: `{hardware.get('fingerprint_sha256')}`",
        "",
        "Host percentiles are computed from request-level samples using linear interpolation. "
        "CUDA elapsed time is separately measured with device events. GPU utilization is sampled "
        "out-of-process and must be interpreted with its sample count in the raw JSON.",
        "",
        "| Case | Precision | Prompt / output | Batch requested / run | TTFT P50 / P95 / P99 "
        "(ms) | TPOT P50 / P95 / P99 (ms) | Output tok/s | Peak allocated MiB | Status |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for case in result.cases:
        peak = _metric(case, "peak_memory_allocated_bytes", "maximum")
        peak_mib = float(peak) / 2**20 if peak is not None else None
        ttft = " / ".join(
            _format_metric(_metric(case, "ttft_ms", statistic))
            for statistic in ("p50", "p95", "p99")
        )
        tpot = " / ".join(
            _format_metric(_metric(case, "tpot_ms", statistic))
            for statistic in ("p50", "p95", "p99")
        )
        lines.append(
            f"| {case.requested.name} | {case.requested.precision.value} | "
            f"{case.requested.prompt_length} / {case.requested.generation_length} | "
            f"{case.requested.batch_size} / {case.executed_batch_size or 'n/a'} | {ttft} | "
            f"{tpot} | {_format_metric(_metric(case, 'output_tokens_per_second', 'mean'))} | "
            f"{_format_metric(peak_mib)} | {case.status} |"
        )
        if case.adjustment_reason:
            lines.append(f"\nAdjustment for `{case.requested.name}`: {case.adjustment_reason}\n")
        if case.error:
            lines.append(f"\nFailure for `{case.requested.name}`: `{case.error}`\n")
    lines.extend(
        [
            "",
            "## Claim boundary",
            "",
            "These measurements apply only to the exact random-weight model dimensions, "
            "workload, software stack, commit, and hardware fingerprint recorded above. They "
            "are not language-quality results and should not be compared with another runtime "
            "unless model semantics, "
            "precision, request set, synchronization, warmup, and measurement boundaries match.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_suite_reports(result: BenchmarkSuiteResult, output_directory: Path) -> tuple[Path, ...]:
    output_directory.mkdir(parents=True, exist_ok=True)
    paths = (
        output_directory / "benchmark.json",
        output_directory / "benchmark.csv",
        output_directory / "benchmark.md",
    )
    write_json(result, paths[0])
    write_csv(result, paths[1])
    write_markdown(result, paths[2])
    return paths
