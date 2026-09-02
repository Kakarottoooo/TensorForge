"""Correctness-gated paged GQA decode-attention benchmarks."""

from __future__ import annotations

import csv
import importlib
import json
import time
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F

from tensorforge.benchmark.hardware import collect_gpu_runtime_state, collect_hardware_metadata
from tensorforge.benchmark.kernel_benchmark import measure_copy_ceiling
from tensorforge.kernels.reference import paged_gqa_decode_reference
from tensorforge.kernels.triton_attention import (
    PagedAttentionWorkspace,
    paged_gqa_decode_attention,
    selected_attention_autotune_config,
)

ATTENTION_BENCHMARK_SCHEMA_VERSION = "1.0"


@dataclass(frozen=True, slots=True)
class AttentionCase:
    name: str
    batch: int
    query_heads: int
    kv_heads: int
    head_dim: int
    context_length: int
    block_size: int
    precision: str
    seed: int = 17

    def __post_init__(self) -> None:
        dimensions = {
            "batch": self.batch,
            "query_heads": self.query_heads,
            "kv_heads": self.kv_heads,
            "head_dim": self.head_dim,
            "context_length": self.context_length,
            "block_size": self.block_size,
        }
        for name, value in dimensions.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.query_heads % self.kv_heads != 0:
            raise ValueError("query_heads must be divisible by kv_heads")
        if self.head_dim > 256:
            raise ValueError("head_dim must not exceed the kernel limit of 256")
        if self.precision not in {"fp32", "fp16", "bf16"}:
            raise ValueError(f"unsupported precision: {self.precision}")


@dataclass(frozen=True, slots=True)
class AttentionManifest:
    cases: tuple[AttentionCase, ...]
    warmup_ms: int = 25
    repetition_ms: int = 100
    copy_buffer_mib: int = 256

    def __post_init__(self) -> None:
        if not self.cases:
            raise ValueError("attention manifest requires at least one case")
        if self.warmup_ms <= 0 or self.repetition_ms <= 0 or self.copy_buffer_mib <= 0:
            raise ValueError("benchmark durations and copy buffer must be positive")


@dataclass(frozen=True, slots=True)
class AttentionCorrectness:
    maximum_absolute_error: float
    maximum_relative_error: float
    rtol: float
    atol: float


@dataclass(frozen=True, slots=True)
class AttentionTiming:
    implementation: str
    p05_ms: float
    p50_ms: float
    p95_ms: float
    speedup_vs_torch: float
    performance_outcome: str
    logical_bytes: int
    modeled_flops: int
    arithmetic_intensity_flops_per_byte: float
    effective_logical_bandwidth_gbps: float
    achieved_gflops: float
    empirical_copy_bandwidth_pct: float


@dataclass(frozen=True, slots=True)
class AttentionCaseResult:
    case: AttentionCase
    correctness: AttentionCorrectness
    first_call_wall_ms: float
    selected_autotune_config: dict[str, Any] | None
    timings: tuple[AttentionTiming, ...]


def _dtype(name: str) -> torch.dtype:
    return {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[name]


def _tolerance(dtype: torch.dtype) -> float:
    if dtype == torch.float32:
        return 1e-5
    if dtype == torch.float16:
        return 3e-3
    return 3e-2


def torch_expanded_gqa_decode(query: Tensor, keys: Tensor, values: Tensor) -> Tensor:
    """Phase 1-equivalent PyTorch decode baseline over contiguous logical K/V."""

    groups = query.shape[1] // keys.shape[2]
    expanded_keys = keys.repeat_interleave(groups, dim=2)
    expanded_values = values.repeat_interleave(groups, dim=2)
    scores = torch.einsum("bhd,bthd->bht", query, expanded_keys) * query.shape[-1] ** -0.5
    probabilities = F.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
    return torch.einsum("bht,bthd->bhd", probabilities, expanded_values)


def _bench(
    function: Callable[[], object], warmup_ms: int, repetition_ms: int
) -> tuple[float, float, float]:
    triton_testing: Any = importlib.import_module("triton.testing")
    result = triton_testing.do_bench(
        function,
        warmup=warmup_ms,
        rep=repetition_ms,
        quantiles=[0.05, 0.5, 0.95],
        return_mode="median",
    )
    return float(result[0]), float(result[1]), float(result[2])


def _logical_work(case: AttentionCase, element_size: int) -> tuple[int, int]:
    q_elements = case.batch * case.query_heads * case.head_dim
    kv_elements = case.batch * case.context_length * case.kv_heads * case.head_dim
    logical_blocks = (case.context_length + case.block_size - 1) // case.block_size
    table_bytes = case.batch * (logical_blocks + 1) * 4
    logical_bytes = (2 * q_elements + 2 * kv_elements) * element_size + table_bytes
    score_flops = case.batch * case.query_heads * case.context_length * (
        2 * case.head_dim - 1
    )
    softmax_flops = 3 * case.batch * case.query_heads * case.context_length
    value_flops = case.batch * case.query_heads * case.head_dim * (
        2 * case.context_length - 1
    )
    return logical_bytes, score_flops + softmax_flops + value_flops


def _case_functions(
    case: AttentionCase, device: torch.device
) -> tuple[Callable[[], Tensor], Callable[[], Tensor], Tensor]:
    dtype = _dtype(case.precision)
    generator = torch.Generator(device=device).manual_seed(case.seed)
    logical_blocks = (case.context_length + case.block_size - 1) // case.block_size
    physical_blocks = case.batch * logical_blocks
    query = torch.randn(
        (case.batch, case.query_heads, case.head_dim),
        generator=generator,
        device=device,
        dtype=dtype,
    )
    key_cache = torch.randn(
        (physical_blocks, case.block_size, case.kv_heads, case.head_dim),
        generator=generator,
        device=device,
        dtype=dtype,
    )
    value_cache = torch.randn(
        key_cache.shape, generator=generator, device=device, dtype=dtype
    )
    block_tables = torch.arange(
        physical_blocks, device=device, dtype=torch.int32
    ).view(case.batch, logical_blocks)
    # Reverse each request's physical order so the kernel cannot rely on contiguous pages.
    block_tables = block_tables.flip(1).contiguous()
    context_lengths = torch.full(
        (case.batch,), case.context_length, device=device, dtype=torch.int32
    )
    positions = torch.arange(case.context_length, device=device)
    logical_indices = positions // case.block_size
    block_offsets = positions % case.block_size
    physical_indices = block_tables[:, logical_indices]
    dense_keys = key_cache[physical_indices, block_offsets.unsqueeze(0)]
    dense_values = value_cache[physical_indices, block_offsets.unsqueeze(0)]
    expected = paged_gqa_decode_reference(
        query, key_cache, value_cache, block_tables, context_lengths
    )
    output = torch.empty_like(query)
    workspace = PagedAttentionWorkspace.allocate(
        batch_capacity=case.batch,
        query_heads=case.query_heads,
        head_dim=case.head_dim,
        max_context_length=case.context_length,
        device=device,
    )

    def torch_run() -> Tensor:
        return torch_expanded_gqa_decode(query, dense_keys, dense_values)

    def triton_run() -> Tensor:
        return paged_gqa_decode_attention(
            query,
            key_cache,
            value_cache,
            block_tables,
            context_lengths,
            max_context_length=case.context_length,
            output=output,
            workspace=workspace,
        )

    return torch_run, triton_run, expected


def run_attention_case(
    case: AttentionCase,
    *,
    device: torch.device,
    copy_bandwidth_gbps: float,
    warmup_ms: int,
    repetition_ms: int,
) -> AttentionCaseResult:
    torch_function, triton_function, expected = _case_functions(case, device)
    torch.cuda.synchronize(device)
    first_call_start = time.perf_counter_ns()
    actual = triton_function()
    torch.cuda.synchronize(device)
    first_call_wall_ms = (time.perf_counter_ns() - first_call_start) / 1e6

    tolerance = _tolerance(_dtype(case.precision))
    torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
    torch.testing.assert_close(torch_function(), expected, rtol=tolerance, atol=tolerance)
    difference = (actual.float() - expected.float()).abs()
    relative = difference / expected.float().abs().clamp_min(1e-7)
    correctness = AttentionCorrectness(
        maximum_absolute_error=float(difference.max().item()),
        maximum_relative_error=float(relative.max().item()),
        rtol=tolerance,
        atol=tolerance,
    )

    torch_p05, torch_p50, torch_p95 = _bench(torch_function, warmup_ms, repetition_ms)
    triton_p05, triton_p50, triton_p95 = _bench(triton_function, warmup_ms, repetition_ms)
    logical_bytes, modeled_flops = _logical_work(case, actual.element_size())
    intensity = modeled_flops / logical_bytes

    def timing(
        implementation: str, p05: float, p50: float, p95: float, speedup: float
    ) -> AttentionTiming:
        bandwidth = logical_bytes / (p50 * 1e6)
        return AttentionTiming(
            implementation=implementation,
            p05_ms=p05,
            p50_ms=p50,
            p95_ms=p95,
            speedup_vs_torch=speedup,
            performance_outcome=(
                "baseline"
                if implementation == "torch_expanded_gqa"
                else ("improvement" if speedup >= 1 else "regression")
            ),
            logical_bytes=logical_bytes,
            modeled_flops=modeled_flops,
            arithmetic_intensity_flops_per_byte=intensity,
            effective_logical_bandwidth_gbps=bandwidth,
            achieved_gflops=modeled_flops / (p50 * 1e6),
            empirical_copy_bandwidth_pct=100 * bandwidth / copy_bandwidth_gbps,
        )

    return AttentionCaseResult(
        case=case,
        correctness=correctness,
        first_call_wall_ms=first_call_wall_ms,
        selected_autotune_config=selected_attention_autotune_config(),
        timings=(
            timing("torch_expanded_gqa", torch_p05, torch_p50, torch_p95, 1.0),
            timing(
                "triton_paged_gqa_preallocated",
                triton_p05,
                triton_p50,
                triton_p95,
                torch_p50 / triton_p50,
            ),
        ),
    )


def load_attention_manifest(path: Path) -> AttentionManifest:
    payload = json.loads(path.read_text(encoding="utf-8"))
    allowed_manifest = {
        "schema_version",
        "warmup_ms",
        "repetition_ms",
        "copy_buffer_mib",
        "cases",
    }
    unknown_manifest = set(payload) - allowed_manifest
    if unknown_manifest:
        raise ValueError(f"unknown attention manifest keys: {sorted(unknown_manifest)}")
    if payload.get("schema_version") != ATTENTION_BENCHMARK_SCHEMA_VERSION:
        raise ValueError(f"expected attention schema {ATTENTION_BENCHMARK_SCHEMA_VERSION}")
    case_keys = {
        "name",
        "batch",
        "query_heads",
        "kv_heads",
        "head_dim",
        "context_length",
        "block_size",
        "precision",
        "seed",
    }
    for case in payload["cases"]:
        unknown_case = set(case) - case_keys
        if unknown_case:
            raise ValueError(f"unknown attention case keys: {sorted(unknown_case)}")
    return AttentionManifest(
        cases=tuple(AttentionCase(**case) for case in payload["cases"]),
        warmup_ms=int(payload.get("warmup_ms", 25)),
        repetition_ms=int(payload.get("repetition_ms", 100)),
        copy_buffer_mib=int(payload.get("copy_buffer_mib", 256)),
    )


def run_attention_suite(
    manifest: AttentionManifest, *, device: torch.device, repository: Path
) -> dict[str, Any]:
    started = datetime.now(UTC)
    runtime_state_before = collect_gpu_runtime_state(device.index or 0)
    copy_ceiling = measure_copy_ceiling(
        device=device,
        buffer_mib=manifest.copy_buffer_mib,
        warmup_ms=manifest.warmup_ms,
        repetition_ms=manifest.repetition_ms,
    )
    copy_bandwidth = float(copy_ceiling["effective_bandwidth_gbps"])
    results = tuple(
        run_attention_case(
            case,
            device=device,
            copy_bandwidth_gbps=copy_bandwidth,
            warmup_ms=manifest.warmup_ms,
            repetition_ms=manifest.repetition_ms,
        )
        for case in manifest.cases
    )
    return {
        "schema_version": ATTENTION_BENCHMARK_SCHEMA_VERSION,
        "suite_id": str(uuid.uuid4()),
        "started_at_utc": started.isoformat(),
        "finished_at_utc": datetime.now(UTC).isoformat(),
        "hardware": collect_hardware_metadata(repository).to_dict(),
        "gpu_runtime_state_before": runtime_state_before,
        "gpu_runtime_state_after": collect_gpu_runtime_state(device.index or 0),
        "measurement": {
            "warmup_ms": manifest.warmup_ms,
            "repetition_ms": manifest.repetition_ms,
            "latency_quantiles": [0.05, 0.5, 0.95],
            "autotune_excluded_from_steady_state": True,
            "torch_baseline": (
                "contiguous logical KV with timed repeat_interleave, score matmul, "
                "FP32 softmax, and value aggregation"
            ),
            "triton_buffers": "output and split-KV workspace are preallocated outside timing",
        },
        "empirical_copy_ceiling": copy_ceiling,
        "logical_work_model": {
            "bytes": "query + output + unique logical K/V + block table + context length",
            "flops": "QK dot + 3 softmax ops per score + probability/value dot",
            "caveat": (
                "Logical bytes are not measured DRAM transactions. The PyTorch baseline expands "
                "GQA heads in the timed region, so semantic-work bandwidth is not physical traffic."
            ),
        },
        "cases": [asdict(result) for result in results],
    }


def write_attention_reports(
    result: dict[str, Any], output_directory: Path
) -> tuple[Path, Path, Path]:
    output_directory.mkdir(parents=True, exist_ok=True)
    json_path = output_directory / "attention.json"
    csv_path = output_directory / "attention.csv"
    markdown_path = output_directory / "attention.md"
    json_path.write_text(
        json.dumps(result, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )

    rows: list[dict[str, Any]] = []
    for case_result in result["cases"]:
        for timing_result in case_result["timings"]:
            rows.append(
                {
                    **case_result["case"],
                    **timing_result,
                    **case_result["correctness"],
                    "first_call_wall_ms": case_result["first_call_wall_ms"],
                    "selected_autotune_config": json.dumps(
                        case_result["selected_autotune_config"], sort_keys=True
                    ),
                    "suite_id": result["suite_id"],
                    "hardware_fingerprint": result["hardware"]["fingerprint_sha256"],
                    "git_commit": result["hardware"]["git_commit"],
                }
            )
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    hardware = result["hardware"]
    gpu = hardware["gpus"][0]
    regressions = sum(
        timing_result["performance_outcome"] == "regression"
        for case_result in result["cases"]
        for timing_result in case_result["timings"]
    )
    lines = [
        "# TensorForge Phase 4 paged GQA decode-attention report",
        "",
        f"- Suite: `{result['suite_id']}`",
        f"- GPU: {gpu['name']} (compute capability {gpu['compute_capability']})",
        f"- PyTorch / CUDA / Triton: `{hardware['torch_version']}` / "
        f"`{hardware['torch_cuda_runtime']}` / `{hardware['triton_version']}`",
        f"- Git: `{hardware['git_commit']}`; dirty: `{hardware['git_dirty']}`",
        f"- Empirical copy ceiling: "
        f"{result['empirical_copy_ceiling']['effective_bandwidth_gbps']:.2f} GB/s",
        f"- Triton regressions versus PyTorch expanded GQA: **{regressions}**",
        "",
        "Steady-state timings exclude first-call JIT/autotune. The PyTorch baseline performs "
        "GQA head expansion inside the timed region; Triton output and split workspace are "
        "preallocated. Effective bandwidth uses semantic logical work and is not a DRAM-counter "
        "measurement.",
        "",
        "| Case | Dtype | B | Q/KV heads | D | Context | Implementation | P50 ms | P95 ms | "
        "Speedup | Logical GB/s | Outcome |",
        "|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---|",
    ]
    for case_result in result["cases"]:
        case = case_result["case"]
        for timing_result in case_result["timings"]:
            lines.append(
                f"| {case['name']} | {case['precision']} | {case['batch']} | "
                f"{case['query_heads']}/{case['kv_heads']} | {case['head_dim']} | "
                f"{case['context_length']} | {timing_result['implementation']} | "
                f"{timing_result['p50_ms']:.6f} | {timing_result['p95_ms']:.6f} | "
                f"{timing_result['speedup_vs_torch']:.3f}x | "
                f"{timing_result['effective_logical_bandwidth_gbps']:.2f} | "
                f"{timing_result['performance_outcome']} |"
            )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            result["logical_work_model"]["caveat"],
        ]
    )
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, csv_path, markdown_path
