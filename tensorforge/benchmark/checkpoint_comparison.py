"""Cross-backend comparison for separately executed checkpoint reports."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"checkpoint report must be an object: {path}")
    return value


def _case_identity(case: dict[str, Any]) -> dict[str, Any]:
    requested = case["requested"]
    return {
        "name": requested["name"],
        "model": requested["model"],
        "prompt_length": requested["prompt_length"],
        "generation_length": requested["generation_length"],
        "batch_size": requested["batch_size"],
        "concurrency": requested["concurrency"],
        "precision": requested["precision"],
        "seed": requested["seed"],
        "arrival_offsets_ms": requested["arrival_offsets_ms"],
    }


def _mean(case: dict[str, Any], metric: str) -> float | None:
    distribution = case.get("aggregate", {}).get(metric)
    if not isinstance(distribution, dict):
        return None
    value = distribution.get("mean")
    return float(value) if isinstance(value, int | float) else None


def compare_checkpoint_reports(paths: tuple[Path, ...]) -> dict[str, Any]:
    if len(paths) < 2:
        raise ValueError("comparison requires at least two backend reports")
    reports = [_load(path) for path in paths]
    checkpoint_identity = reports[0]["hardware"]["checkpoint"]
    identity_fields = (
        "repository",
        "revision",
        "config_sha256",
        "weights_sha256",
        "weights_size_bytes",
    )
    expected_checkpoint = {
        field: checkpoint_identity[field] for field in identity_fields
    }
    for report in reports[1:]:
        actual = {
            field: report["hardware"]["checkpoint"][field]
            for field in identity_fields
        }
        if actual != expected_checkpoint:
            raise ValueError("checkpoint identity differs between backend reports")
    expected_gpu = {
        key: reports[0]["hardware"]["gpus"][0][key]
        for key in ("name", "total_memory_bytes")
    }
    for report in reports[1:]:
        actual_gpu = {
            key: report["hardware"]["gpus"][0][key]
            for key in ("name", "total_memory_bytes")
        }
        if actual_gpu != expected_gpu:
            raise ValueError("GPU identity differs between backend reports")

    by_backend: dict[str, dict[str, Any]] = {}
    for report in reports:
        backend = str(report["hardware"]["benchmark_backend"])
        if backend in by_backend:
            raise ValueError(f"duplicate backend report: {backend}")
        by_backend[backend] = report
    reference_cases = reports[0]["cases"]
    reference_identities = {
        case["requested"]["name"]: _case_identity(case)
        for case in reference_cases
    }
    rows = []
    tensorforge_by_name = {
        case["requested"]["name"]: case
        for case in by_backend.get("tensorforge", {}).get("cases", [])
    }
    for backend, report in by_backend.items():
        for case in report["cases"]:
            name = case["requested"]["name"]
            if (
                name not in reference_identities
                or _case_identity(case) != reference_identities[name]
            ):
                raise ValueError(f"workload identity differs for case {name}/{backend}")
            throughput = _mean(case, "output_tokens_per_second")
            tensorforge_case = tensorforge_by_name.get(name)
            tensorforge_throughput = (
                _mean(tensorforge_case, "output_tokens_per_second")
                if tensorforge_case is not None
                else None
            )
            digest = case.get("aggregate", {}).get("generated_token_sha256")
            tensorforge_digest = (
                tensorforge_case.get("aggregate", {}).get("generated_token_sha256")
                if tensorforge_case is not None
                else None
            )
            rows.append(
                {
                    "case": name,
                    "backend": backend,
                    "status": case["status"],
                    "output_tokens_per_second_mean": throughput,
                    "throughput_relative_to_tensorforge": (
                        throughput / tensorforge_throughput
                        if throughput is not None
                        and tensorforge_throughput is not None
                        and tensorforge_throughput > 0
                        else None
                    ),
                    "request_latency_mean_ms": _mean(case, "request_latency_ms"),
                    "ttft_mean_ms": _mean(case, "ttft_ms"),
                    "tpot_mean_ms": _mean(case, "tpot_ms"),
                    "generated_token_sha256": digest,
                    "greedy_tokens_match_tensorforge": (
                        digest == tensorforge_digest
                        if digest is not None and tensorforge_digest is not None
                        else None
                    ),
                    "error": case.get("error"),
                }
            )
    return {
        "schema_version": "1.0",
        "checkpoint": expected_checkpoint,
        "gpu": expected_gpu,
        "backends": {
            backend: {
                "torch_version": report["hardware"].get("torch_version"),
                "torch_cuda_runtime": report["hardware"].get("torch_cuda_runtime"),
                "triton_version": report["hardware"].get("triton_version"),
                "hardware_fingerprint": report["hardware"].get("fingerprint_sha256"),
                "measurement_boundary": report["hardware"].get("measurement_boundary"),
                "backend_packages": report["hardware"].get("backend_packages"),
            }
            for backend, report in by_backend.items()
        },
        "rows": rows,
    }


def _display(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int | float):
        return f"{value:.{digits}f}"
    return str(value)


def write_checkpoint_comparison(
    comparison: dict[str, Any], output_directory: Path
) -> tuple[Path, Path, Path]:
    output_directory.mkdir(parents=True, exist_ok=True)
    json_path = output_directory / "comparison.json"
    csv_path = output_directory / "comparison.csv"
    markdown_path = output_directory / "comparison.md"
    json_path.write_text(
        json.dumps(comparison, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    rows = comparison["rows"]
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    checkpoint = comparison["checkpoint"]
    lines = [
        "# Real-checkpoint backend comparison",
        "",
        f"- Checkpoint: `{checkpoint['repository']}@{checkpoint['revision']}`",
        f"- GPU: {comparison['gpu']['name']}",
        "",
        "| Case | Backend | Output tok/s | Relative to TensorForge | Mean latency ms | "
        "Mean TTFT ms | Mean TPOT ms | Greedy tokens match | Status |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        relative = row["throughput_relative_to_tensorforge"]
        relative_text = "n/a" if relative is None else f"{relative:.3f}x"
        lines.append(
            f"| {row['case']} | {row['backend']} | "
            f"{_display(row['output_tokens_per_second_mean'])} | "
            f"{relative_text} | "
            f"{_display(row['request_latency_mean_ms'])} | "
            f"{_display(row['ttft_mean_ms'])} | {_display(row['tpot_mean_ms'])} | "
            f"{_display(row['greedy_tokens_match_tensorforge'])} | {row['status']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "This is an external reference comparison, not a one-factor kernel ablation. vLLM "
            "uses its production-oriented FlashAttention, scheduler, paged cache, sampling path, "
            "and CUDA Graph defaults; Transformers uses SDPA and a contiguous cache; TensorForge "
            "uses parallel SDPA prefill and its custom paged Triton decode path. Model loading and "
            "tokenization are excluded for all backends. A greedy-token mismatch can result from "
            "BF16 argmax ties and must be interpreted with the separate teacher-forced logit gate.",
            "",
        ]
    )
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, csv_path, markdown_path
