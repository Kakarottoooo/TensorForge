"""Correctness-gated Triton microbenchmarks and empirical roofline accounting."""

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

from tensorforge.benchmark.hardware import collect_gpu_runtime_state, collect_hardware_metadata
from tensorforge.kernels.reference import (
    residual_rms_norm_reference,
    rms_norm_reference,
    swiglu_reference,
    torch_residual_rms_norm,
    torch_rms_norm,
)
from tensorforge.kernels.triton_ops import (
    residual_rms_norm,
    rms_norm,
    selected_autotune_config,
    swiglu,
)

KERNEL_BENCHMARK_SCHEMA_VERSION = "1.0"


@dataclass(frozen=True, slots=True)
class KernelCase:
    name: str
    operation: str
    rows: int
    columns: int
    precision: str
    seed: int = 7
    eps: float = 1e-6

    def __post_init__(self) -> None:
        if self.operation not in {"rms_norm", "residual_rms_norm", "swiglu"}:
            raise ValueError(f"unsupported operation: {self.operation}")
        if self.rows <= 0 or self.columns <= 0:
            raise ValueError("rows and columns must be positive")
        if self.precision not in {"fp32", "fp16", "bf16"}:
            raise ValueError(f"unsupported precision: {self.precision}")
        if self.eps <= 0:
            raise ValueError("eps must be positive")


@dataclass(frozen=True, slots=True)
class ComputeCeiling:
    fp32_gflops: float
    derivation: str
    source_url: str

    def __post_init__(self) -> None:
        if self.fp32_gflops <= 0:
            raise ValueError("fp32_gflops must be positive")
        if not self.derivation or not self.source_url.startswith("https://"):
            raise ValueError("compute ceiling requires a derivation and HTTPS source")


@dataclass(frozen=True, slots=True)
class KernelManifest:
    cases: tuple[KernelCase, ...]
    compute_ceiling: ComputeCeiling
    warmup_ms: int = 25
    repetition_ms: int = 100
    copy_buffer_mib: int = 256

    def __post_init__(self) -> None:
        if not self.cases:
            raise ValueError("kernel manifest requires at least one case")
        if self.warmup_ms <= 0 or self.repetition_ms <= 0 or self.copy_buffer_mib <= 0:
            raise ValueError("benchmark durations and copy buffer must be positive")


@dataclass(frozen=True, slots=True)
class CorrectnessError:
    maximum_absolute_error: float
    maximum_relative_error: float
    rtol: float
    atol: float


@dataclass(frozen=True, slots=True)
class KernelTiming:
    implementation: str
    p05_ms: float
    p50_ms: float
    p95_ms: float
    logical_bytes: int
    modeled_flops: int
    arithmetic_intensity_flops_per_byte: float
    effective_bandwidth_gbps: float
    achieved_gflops: float
    empirical_memory_roofline_gflops: float
    roofline_efficiency_pct: float
    empirical_copy_bandwidth_pct: float
    speedup_vs_torch: float
    performance_outcome: str


@dataclass(frozen=True, slots=True)
class KernelCaseResult:
    case: KernelCase
    correctness: CorrectnessError
    first_call_wall_ms: float
    selected_autotune_config: dict[str, Any] | None
    timings: tuple[KernelTiming, ...]


def _dtype(name: str) -> torch.dtype:
    return {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[name]


def _tolerances(dtype: torch.dtype) -> tuple[float, float]:
    if dtype == torch.float32:
        return 1e-5, 1e-5
    if dtype == torch.float16:
        return 3e-3, 3e-3
    return 3e-2, 3e-2


def _error(actual: Tensor, expected: Tensor, rtol: float, atol: float) -> CorrectnessError:
    difference = (actual.float() - expected.float()).abs()
    relative = difference / expected.float().abs().clamp_min(1e-7)
    return CorrectnessError(
        maximum_absolute_error=float(difference.max().item()),
        maximum_relative_error=float(relative.max().item()),
        rtol=rtol,
        atol=atol,
    )


def _logical_work(case: KernelCase, element_size: int) -> tuple[int, int]:
    elements = case.rows * case.columns
    if case.operation == "rms_norm":
        # Unique logical traffic: input + output + one shared weight vector.
        logical_bytes = (2 * elements + case.columns) * element_size
        modeled_flops = case.rows * (4 * case.columns + 2)
    elif case.operation == "residual_rms_norm":
        # input + residual + residual_output + norm_output + shared weight.
        logical_bytes = (4 * elements + case.columns) * element_size
        modeled_flops = case.rows * (5 * case.columns + 2)
    else:
        logical_bytes = 3 * elements * element_size
        # Sigmoid counts as one special-function operation, plus two multiplies.
        modeled_flops = 3 * elements
    return logical_bytes, modeled_flops


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


def measure_copy_ceiling(
    *, device: torch.device, buffer_mib: int, warmup_ms: int, repetition_ms: int
) -> dict[str, float | int | str]:
    elements = buffer_mib * 2**20 // 4
    source = torch.randn(elements, device=device, dtype=torch.float32)
    destination = torch.empty_like(source)
    p05, p50, p95 = _bench(lambda: destination.copy_(source), warmup_ms, repetition_ms)
    logical_bytes = 2 * source.numel() * source.element_size()
    return {
        "implementation": "torch.copy_",
        "buffer_mib": buffer_mib,
        "logical_bytes": logical_bytes,
        "p05_ms": p05,
        "p50_ms": p50,
        "p95_ms": p95,
        "effective_bandwidth_gbps": logical_bytes / (p50 * 1e6),
    }


def _case_functions(
    case: KernelCase, device: torch.device
) -> tuple[
    Callable[[], Tensor | tuple[Tensor, Tensor]],
    Callable[[], Tensor | tuple[Tensor, Tensor]],
    Tensor,
]:
    dtype = _dtype(case.precision)
    generator = torch.Generator(device=device).manual_seed(case.seed)
    first = torch.randn((case.rows, case.columns), generator=generator, device=device, dtype=dtype)
    second = torch.randn((case.rows, case.columns), generator=generator, device=device, dtype=dtype)
    weight = torch.randn(case.columns, generator=generator, device=device, dtype=dtype)

    if case.operation == "rms_norm":
        expected = rms_norm_reference(first, weight, case.eps)

        def torch_rms_run() -> Tensor:
            return torch_rms_norm(first, weight, case.eps)

        def triton_rms_run() -> Tensor:
            return rms_norm(first, weight, case.eps)

        return torch_rms_run, triton_rms_run, expected
    if case.operation == "residual_rms_norm":
        _, expected = residual_rms_norm_reference(first, second, weight, case.eps)

        def torch_residual_run() -> tuple[Tensor, Tensor]:
            return torch_residual_rms_norm(first, second, weight, case.eps)

        def triton_residual_run() -> tuple[Tensor, Tensor]:
            return residual_rms_norm(first, second, weight, case.eps)

        return torch_residual_run, triton_residual_run, expected

    expected = swiglu_reference(first, second)

    def torch_swiglu_run() -> Tensor:
        return swiglu_reference(first, second)

    def triton_swiglu_run() -> Tensor:
        return swiglu(first, second)

    return torch_swiglu_run, triton_swiglu_run, expected


def run_kernel_case(
    case: KernelCase,
    *,
    device: torch.device,
    copy_bandwidth_gbps: float,
    compute_ceiling_gflops: float,
    warmup_ms: int,
    repetition_ms: int,
) -> KernelCaseResult:
    # A fresh process starts with empty autotune state. This wall measurement
    # includes JIT cache lookup/compile and config search, never steady state.
    torch_function, triton_function, expected = _case_functions(case, device)
    torch.cuda.synchronize(device)
    first_call_start = time.perf_counter_ns()
    raw_actual = triton_function()
    torch.cuda.synchronize(device)
    first_call_wall_ms = (time.perf_counter_ns() - first_call_start) / 1e6
    actual = raw_actual[1] if isinstance(raw_actual, tuple) else raw_actual

    rtol, atol = _tolerances(_dtype(case.precision))
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
    correctness = _error(actual, expected, rtol, atol)
    autotune_config = selected_autotune_config(case.operation)

    torch_p05, torch_p50, torch_p95 = _bench(torch_function, warmup_ms, repetition_ms)
    triton_p05, triton_p50, triton_p95 = _bench(triton_function, warmup_ms, repetition_ms)
    logical_bytes, modeled_flops = _logical_work(case, actual.element_size())
    arithmetic_intensity = modeled_flops / logical_bytes
    memory_roofline_gflops = copy_bandwidth_gbps * arithmetic_intensity
    roofline_gflops = min(memory_roofline_gflops, compute_ceiling_gflops)

    def timing(
        implementation: str, p05: float, p50: float, p95: float, speedup: float
    ) -> KernelTiming:
        achieved_gflops = modeled_flops / (p50 * 1e6)
        bandwidth = logical_bytes / (p50 * 1e6)
        return KernelTiming(
            implementation=implementation,
            p05_ms=p05,
            p50_ms=p50,
            p95_ms=p95,
            logical_bytes=logical_bytes,
            modeled_flops=modeled_flops,
            arithmetic_intensity_flops_per_byte=arithmetic_intensity,
            effective_bandwidth_gbps=bandwidth,
            achieved_gflops=achieved_gflops,
            empirical_memory_roofline_gflops=roofline_gflops,
            roofline_efficiency_pct=100 * achieved_gflops / roofline_gflops,
            empirical_copy_bandwidth_pct=100 * bandwidth / copy_bandwidth_gbps,
            speedup_vs_torch=speedup,
            performance_outcome=(
                "baseline"
                if implementation == "torch_standard"
                else ("improvement" if speedup >= 1 else "regression")
            ),
        )

    return KernelCaseResult(
        case=case,
        correctness=correctness,
        first_call_wall_ms=first_call_wall_ms,
        selected_autotune_config=autotune_config,
        timings=(
            timing("torch_standard", torch_p05, torch_p50, torch_p95, 1.0),
            timing("triton_autotuned", triton_p05, triton_p50, triton_p95, torch_p50 / triton_p50),
        ),
    )


def load_kernel_manifest(path: Path) -> KernelManifest:
    payload = json.loads(path.read_text(encoding="utf-8"))
    allowed_manifest = {
        "schema_version",
        "warmup_ms",
        "repetition_ms",
        "copy_buffer_mib",
        "compute_ceiling",
        "cases",
    }
    unknown_manifest = set(payload) - allowed_manifest
    if unknown_manifest:
        raise ValueError(f"unknown kernel manifest keys: {sorted(unknown_manifest)}")
    if payload.get("schema_version") != KERNEL_BENCHMARK_SCHEMA_VERSION:
        raise ValueError(f"expected kernel schema {KERNEL_BENCHMARK_SCHEMA_VERSION}")
    case_keys = {"name", "operation", "rows", "columns", "precision", "seed", "eps"}
    for case in payload["cases"]:
        unknown_case = set(case) - case_keys
        if unknown_case:
            raise ValueError(f"unknown kernel case keys: {sorted(unknown_case)}")
    ceiling_keys = {"fp32_gflops", "derivation", "source_url"}
    unknown_ceiling = set(payload["compute_ceiling"]) - ceiling_keys
    if unknown_ceiling:
        raise ValueError(f"unknown compute ceiling keys: {sorted(unknown_ceiling)}")
    cases = tuple(KernelCase(**case) for case in payload["cases"])
    return KernelManifest(
        cases=cases,
        compute_ceiling=ComputeCeiling(**payload["compute_ceiling"]),
        warmup_ms=int(payload.get("warmup_ms", 25)),
        repetition_ms=int(payload.get("repetition_ms", 100)),
        copy_buffer_mib=int(payload.get("copy_buffer_mib", 256)),
    )


def run_kernel_suite(
    manifest: KernelManifest, *, device: torch.device, repository: Path
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
        run_kernel_case(
            case,
            device=device,
            copy_bandwidth_gbps=copy_bandwidth,
            compute_ceiling_gflops=manifest.compute_ceiling.fp32_gflops,
            warmup_ms=manifest.warmup_ms,
            repetition_ms=manifest.repetition_ms,
        )
        for case in manifest.cases
    )
    runtime_state_after = collect_gpu_runtime_state(device.index or 0)
    return {
        "schema_version": KERNEL_BENCHMARK_SCHEMA_VERSION,
        "suite_id": str(uuid.uuid4()),
        "started_at_utc": started.isoformat(),
        "finished_at_utc": datetime.now(UTC).isoformat(),
        "hardware": collect_hardware_metadata(repository).to_dict(),
        "gpu_runtime_state_before": runtime_state_before,
        "gpu_runtime_state_after": runtime_state_after,
        "measurement": {
            "warmup_ms": manifest.warmup_ms,
            "repetition_ms": manifest.repetition_ms,
            "latency_quantiles": [0.05, 0.5, 0.95],
            "autotune_excluded_from_steady_state": True,
        },
        "compute_ceiling": asdict(manifest.compute_ceiling),
        "empirical_copy_ceiling": copy_ceiling,
        "logical_work_model": {
            "rms_norm": "bytes=(2*rows+1)*cols*element_size; flops=rows*(4*cols+2)",
            "residual_rms_norm": "bytes=(4*rows+1)*cols*element_size; flops=rows*(5*cols+2)",
            "swiglu": "bytes=3*rows*cols*element_size; flops=3*rows*cols; sigmoid counts as 1",
            "caveat": (
                "Logical unique bytes are not measured DRAM transactions; cache effects apply."
            ),
        },
        "cases": [asdict(result) for result in results],
    }


def write_kernel_reports(result: dict[str, Any], output_directory: Path) -> tuple[Path, Path, Path]:
    output_directory.mkdir(parents=True, exist_ok=True)
    json_path = output_directory / "kernels.json"
    csv_path = output_directory / "kernels.csv"
    markdown_path = output_directory / "kernels.md"
    json_path.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n")

    rows: list[dict[str, Any]] = []
    for case_result in result["cases"]:
        for timing in case_result["timings"]:
            rows.append(
                {
                    **case_result["case"],
                    **timing,
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
    state_before = result["gpu_runtime_state_before"]
    state_after = result["gpu_runtime_state_after"]
    regressions = sum(
        timing["performance_outcome"] == "regression"
        for case in result["cases"]
        for timing in case["timings"]
    )
    lines = [
        "# TensorForge Phase 3 kernel report",
        "",
        f"- Suite: `{result['suite_id']}`",
        f"- GPU: {gpu['name']} (compute capability {gpu['compute_capability']})",
        f"- PyTorch / CUDA / Triton: `{hardware['torch_version']}` / "
        f"`{hardware['torch_cuda_runtime']}` / `{hardware['triton_version']}`",
        f"- Git: `{hardware['git_commit']}`; dirty: `{hardware['git_dirty']}`",
        f"- Empirical copy ceiling: "
        f"{result['empirical_copy_ceiling']['effective_bandwidth_gbps']:.2f} GB/s",
        f"- FP32 compute ceiling: {result['compute_ceiling']['fp32_gflops']:.1f} GFLOP/s "
        f"({result['compute_ceiling']['derivation']})",
        f"- GPU state before: `{state_before}`",
        f"- GPU state after: `{state_after}`",
        f"- Triton regressions versus PyTorch: **{regressions}**",
        "",
        "Steady-state timings exclude first-call JIT/autotune. P05/P50/P95 are CUDA-event "
        "microbenchmark quantiles. Roofline uses logical unique bytes and the empirical copy "
        "ceiling; it is not a DRAM-counter measurement.",
        "",
        "| Case | Op | Dtype | Shape | Implementation | P50 ms | P95 ms | Speedup | GB/s | "
        "Copy ceiling % | Outcome |",
        "|---|---|---:|---:|---|---:|---:|---:|---:|---:|---|",
    ]
    for case_result in result["cases"]:
        case = case_result["case"]
        for timing in case_result["timings"]:
            lines.append(
                f"| {case['name']} | {case['operation']} | {case['precision']} | "
                f"{case['rows']}x{case['columns']} | {timing['implementation']} | "
                f"{timing['p50_ms']:.6f} | {timing['p95_ms']:.6f} | "
                f"{timing['speedup_vs_torch']:.3f}x | {timing['effective_bandwidth_gbps']:.2f} | "
                f"{timing['empirical_copy_bandwidth_pct']:.1f}% | "
                f"{timing['performance_outcome']} |"
            )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "Arithmetic intensity counts rsqrt/sigmoid as one special-function operation. Logical "
            "bytes count unique tensor traffic and shared weights once; actual DRAM transactions "
            "require Nsight Compute. Low arithmetic intensity makes these memory/launch "
            "candidates, "
            "but the report does not promote modeled traffic to measured bandwidth.",
            "",
        ]
    )
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, csv_path, markdown_path
