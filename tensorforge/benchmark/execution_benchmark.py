"""Phase 6 eager, torch.compile, and explicit CUDA Graph decode experiments."""

from __future__ import annotations

import csv
import json
import random
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from tensorforge.benchmark.hardware import collect_gpu_runtime_state, collect_hardware_metadata
from tensorforge.benchmark.schema import ModelSpec
from tensorforge.cache.paged import PagedKVCache, PagedKVCacheConfig
from tensorforge.metrics.statistics import summarize
from tensorforge.model.llama import LlamaForCausalLM
from tensorforge.runtime.decode_bucket import (
    BucketedPagedDecodeExecutor,
    DecodeExecutionMetrics,
    DecodeExecutionMode,
)

EXECUTION_BENCHMARK_SCHEMA_VERSION = "1.0"
_MODES = tuple(DecodeExecutionMode)


@dataclass(frozen=True, slots=True)
class ExecutionCase:
    name: str
    batch_size: int
    initial_context: int
    decode_steps: int
    batch_buckets: tuple[int, ...]
    context_buckets: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("execution case name must be non-empty")
        for name in ("batch_size", "initial_context", "decode_steps"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if not self.batch_buckets or any(value <= 0 for value in self.batch_buckets):
            raise ValueError("batch_buckets must contain positive capacities")
        if not self.context_buckets or any(value <= 0 for value in self.context_buckets):
            raise ValueError("context_buckets must contain positive capacities")


@dataclass(frozen=True, slots=True)
class ExecutionManifest:
    model: ModelSpec
    cases: tuple[ExecutionCase, ...]
    precision: str = "fp16"
    block_size: int = 16
    num_blocks: int = 64
    warmup_repetitions: int = 1
    measured_repetitions: int = 5
    seed: int = 7

    def __post_init__(self) -> None:
        if not self.cases:
            raise ValueError("execution manifest requires at least one case")
        if self.precision not in {"fp16", "bf16", "fp32"}:
            raise ValueError(f"unsupported precision: {self.precision}")
        if self.block_size <= 0 or self.num_blocks <= 0:
            raise ValueError("cache dimensions must be positive")
        if self.warmup_repetitions <= 0 or self.measured_repetitions <= 0:
            raise ValueError("invalid repetition count")
        for case in self.cases:
            total_length = case.initial_context + case.decode_steps
            if total_length > self.model.max_position_embeddings:
                raise ValueError(f"case {case.name} exceeds model context capacity")
            if max(case.context_buckets) > self.model.max_position_embeddings:
                raise ValueError(f"case {case.name} context bucket exceeds model capacity")
            required_blocks = (
                (total_length + self.block_size - 1) // self.block_size * case.batch_size
            )
            if required_blocks > self.num_blocks:
                raise ValueError(f"case {case.name} exceeds KV block capacity")


@dataclass(frozen=True, slots=True)
class ExecutionRun:
    repetition: int
    mode: str
    setup_wall_ms: float
    measured_host_ms: float
    cuda_step_latency_ms: tuple[float, ...]
    tokens_per_second: float
    address_stable: bool
    setup_metrics: dict[str, Any]
    measured_metrics: dict[str, Any]


def _dtype(name: str) -> torch.dtype:
    return {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[name]


def load_execution_manifest(path: Path) -> ExecutionManifest:
    payload = json.loads(path.read_text(encoding="utf-8"))
    allowed = {
        "schema_version",
        "model",
        "precision",
        "block_size",
        "num_blocks",
        "warmup_repetitions",
        "measured_repetitions",
        "seed",
        "cases",
    }
    unknown = set(payload) - allowed
    if unknown:
        raise ValueError(f"unknown execution manifest keys: {sorted(unknown)}")
    if payload.get("schema_version") != EXECUTION_BENCHMARK_SCHEMA_VERSION:
        raise ValueError(f"expected execution schema {EXECUTION_BENCHMARK_SCHEMA_VERSION}")
    unknown_model = set(payload["model"]) - set(ModelSpec.__dataclass_fields__)
    if unknown_model:
        raise ValueError(f"unknown execution model keys: {sorted(unknown_model)}")
    case_keys = {
        "name",
        "batch_size",
        "initial_context",
        "decode_steps",
        "batch_buckets",
        "context_buckets",
    }
    cases = []
    for raw_case in payload["cases"]:
        unknown_case = set(raw_case) - case_keys
        if unknown_case:
            raise ValueError(f"unknown execution case keys: {sorted(unknown_case)}")
        cases.append(
            ExecutionCase(
                **{
                    **raw_case,
                    "batch_buckets": tuple(raw_case["batch_buckets"]),
                    "context_buckets": tuple(raw_case["context_buckets"]),
                }
            )
        )
    defaults = ExecutionManifest.__dataclass_fields__
    return ExecutionManifest(
        model=ModelSpec(**payload["model"]),
        cases=tuple(cases),
        precision=str(payload.get("precision", defaults["precision"].default)),
        block_size=int(payload.get("block_size", defaults["block_size"].default)),
        num_blocks=int(payload.get("num_blocks", defaults["num_blocks"].default)),
        warmup_repetitions=int(
            payload.get("warmup_repetitions", defaults["warmup_repetitions"].default)
        ),
        measured_repetitions=int(
            payload.get("measured_repetitions", defaults["measured_repetitions"].default)
        ),
        seed=int(payload.get("seed", defaults["seed"].default)),
    )


def _metrics_delta(
    before: DecodeExecutionMetrics, after: DecodeExecutionMetrics
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for field_name in DecodeExecutionMetrics.__dataclass_fields__:
        before_value = getattr(before, field_name)
        after_value = getattr(after, field_name)
        if isinstance(after_value, dict):
            keys = set(before_value) | set(after_value)
            result[field_name] = {
                key: after_value.get(key, 0) - before_value.get(key, 0) for key in keys
            }
        else:
            result[field_name] = after_value - before_value
    return result


def _run_once(
    manifest: ExecutionManifest,
    case: ExecutionCase,
    *,
    model: LlamaForCausalLM,
    device: torch.device,
    mode: DecodeExecutionMode,
    repetition: int,
) -> ExecutionRun:
    dtype = _dtype(manifest.precision)
    config = manifest.model.to_model_config()
    cache = PagedKVCache(
        PagedKVCacheConfig(
            num_layers=config.num_hidden_layers,
            num_blocks=manifest.num_blocks,
            block_size=manifest.block_size,
            num_kv_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            max_sequences=max(case.batch_size, max(case.batch_buckets)),
            max_sequence_length=config.max_position_embeddings,
            dtype=dtype,
            device=device,
        )
    )
    executor = BucketedPagedDecodeExecutor(
        model=model,
        cache=cache,
        mode=mode,
        batch_buckets=case.batch_buckets,
        context_buckets=case.context_buckets,
    )
    request_ids = [f"request-{index}" for index in range(case.batch_size)]
    for request_id in request_ids:
        executor.create_request(request_id)
    generator = random.Random(manifest.seed + repetition * 10_007 + case.batch_size)
    token_rows = [
        [
            generator.randrange(config.vocab_size)
            for _ in range(case.initial_context + case.decode_steps)
        ]
        for _ in request_ids
    ]

    torch.cuda.synchronize(device)
    setup_started_ns = time.perf_counter_ns()
    for position in range(case.initial_context):
        result = executor.append_tokens(
            {
                request_id: token_rows[index][position]
                for index, request_id in enumerate(request_ids)
            }
        )
        if result.errors:
            raise RuntimeError(f"setup failed: {result.errors}")
    torch.cuda.synchronize(device)
    setup_wall_ms = (time.perf_counter_ns() - setup_started_ns) / 1e6
    addresses_before = executor.buffer_addresses()
    setup_metrics = executor.metrics()

    event_factory: Any = torch.cuda.Event
    start_events = [event_factory(enable_timing=True) for _ in range(case.decode_steps)]
    end_events = [event_factory(enable_timing=True) for _ in range(case.decode_steps)]
    measured_before = executor.metrics()
    first_logits: torch.Tensor | None = None
    measured_started_ns = time.perf_counter_ns()
    for step in range(case.decode_steps):
        position = case.initial_context + step
        start_events[step].record()
        result = executor.append_tokens(
            {
                request_id: token_rows[index][position]
                for index, request_id in enumerate(request_ids)
            }
        )
        end_events[step].record()
        if result.errors:
            raise RuntimeError(f"measured decode failed: {result.errors}")
        if step == 0:
            first_logits = result.logits[request_ids[0]].clone()
    torch.cuda.synchronize(device)
    measured_host_ms = (time.perf_counter_ns() - measured_started_ns) / 1e6
    measured_after = executor.metrics()
    addresses_after = executor.buffer_addresses()
    cuda_step_latency_ms = tuple(
        float(start.elapsed_time(end)) for start, end in zip(start_events, end_events, strict=True)
    )

    assert first_logits is not None
    with torch.inference_mode():
        correctness_prefix = torch.tensor(
            [token_rows[0][: case.initial_context + 1]], device=device
        )
        expected = model(correctness_prefix)[0, -1]
    torch.testing.assert_close(first_logits, expected, rtol=4e-3, atol=4e-3)
    for request_id in request_ids:
        executor.release_request(request_id)
    stats = cache.stats()
    if stats.active_sequences or stats.used_blocks or stats.reserved_tokens:
        raise RuntimeError(f"cache state leaked after {case.name}/{mode}")
    return ExecutionRun(
        repetition=repetition,
        mode=mode.value,
        setup_wall_ms=setup_wall_ms,
        measured_host_ms=measured_host_ms,
        cuda_step_latency_ms=cuda_step_latency_ms,
        tokens_per_second=(case.batch_size * case.decode_steps) / (measured_host_ms / 1_000),
        address_stable=addresses_before == addresses_after,
        setup_metrics=asdict(setup_metrics),
        measured_metrics=_metrics_delta(measured_before, measured_after),
    )


def _distribution(values: list[float]) -> dict[str, Any]:
    return asdict(summarize(values))


def _aggregate(runs: list[ExecutionRun]) -> dict[str, Any]:
    step_latencies = [latency for run in runs for latency in run.cuda_step_latency_ms]
    metric_totals: dict[str, Any] = {}
    for name in DecodeExecutionMetrics.__dataclass_fields__:
        values = [run.measured_metrics[name] for run in runs]
        if values and isinstance(values[0], dict):
            keys = {key for value in values for key in value}
            metric_totals[name] = {key: sum(value.get(key, 0) for value in values) for key in keys}
        else:
            metric_totals[name] = sum(values)
    return {
        "cuda_step_latency_ms": _distribution(step_latencies),
        "tokens_per_second": _distribution([run.tokens_per_second for run in runs]),
        "measured_host_ms": _distribution([run.measured_host_ms for run in runs]),
        "setup_wall_ms": _distribution([run.setup_wall_ms for run in runs]),
        "capture_time_ms": _distribution(
            [float(run.setup_metrics["capture_time_ms"]) for run in runs]
        ),
        "capture_warmup_time_ms": _distribution(
            [float(run.setup_metrics["capture_warmup_time_ms"]) for run in runs]
        ),
        "compile_time_ms": _distribution(
            [float(run.setup_metrics["compile_time_ms"]) for run in runs]
        ),
        "address_stable_all_runs": all(run.address_stable for run in runs),
        "measured_metric_totals": metric_totals,
    }


def run_execution_suite(
    manifest: ExecutionManifest, *, device: torch.device, repository: Path
) -> dict[str, Any]:
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("execution benchmark requires an available CUDA device")
    started = datetime.now(UTC)
    runtime_before = collect_gpu_runtime_state(device.index or 0)
    dtype = _dtype(manifest.precision)
    torch.manual_seed(manifest.seed)
    torch.cuda.manual_seed_all(manifest.seed)
    model = LlamaForCausalLM(manifest.model.to_model_config()).to(device=device, dtype=dtype)
    model.eval()
    case_results: list[dict[str, Any]] = []
    for case in manifest.cases:
        warmup_runs_by_mode: dict[DecodeExecutionMode, list[ExecutionRun]] = {
            mode: [] for mode in _MODES
        }
        for warmup in range(manifest.warmup_repetitions):
            for mode in _MODES:
                warmup_runs_by_mode[mode].append(
                    _run_once(
                        manifest,
                        case,
                        model=model,
                        device=device,
                        mode=mode,
                        repetition=-(warmup + 1),
                    )
                )
        runs_by_mode: dict[DecodeExecutionMode, list[ExecutionRun]] = {
            mode: [] for mode in _MODES
        }
        orders = []
        for repetition in range(manifest.measured_repetitions):
            order = list(_MODES)
            random.Random(manifest.seed + repetition + case.batch_size).shuffle(order)
            orders.append([mode.value for mode in order])
            for mode in order:
                runs_by_mode[mode].append(
                    _run_once(
                        manifest,
                        case,
                        model=model,
                        device=device,
                        mode=mode,
                        repetition=repetition,
                    )
                )
        eager = _aggregate(runs_by_mode[DecodeExecutionMode.EAGER])
        modes = []
        for mode in _MODES:
            aggregate = _aggregate(runs_by_mode[mode])
            eager_latency = eager["cuda_step_latency_ms"]["mean"]
            latency = aggregate["cuda_step_latency_ms"]["mean"]
            aggregate["mean_latency_speedup_vs_eager"] = eager_latency / latency
            aggregate["mean_latency_change_pct_vs_eager"] = (latency / eager_latency - 1) * 100
            cold_setup = warmup_runs_by_mode[mode][0]
            aggregate["cold_setup_wall_ms"] = cold_setup.setup_wall_ms
            aggregate["cold_capture_time_ms"] = cold_setup.setup_metrics[
                "capture_time_ms"
            ]
            aggregate["cold_compile_time_ms"] = cold_setup.setup_metrics[
                "compile_time_ms"
            ]
            modes.append(
                {
                    "mode": mode.value,
                    "warmup_runs": [
                        asdict(run) for run in warmup_runs_by_mode[mode]
                    ],
                    "runs": [asdict(run) for run in runs_by_mode[mode]],
                    "aggregate": aggregate,
                }
            )
        case_results.append(
            {"case": asdict(case), "mode_order_by_repetition": orders, "modes": modes}
        )
    return {
        "schema_version": EXECUTION_BENCHMARK_SCHEMA_VERSION,
        "suite_id": str(uuid.uuid4()),
        "started_at_utc": started.isoformat(),
        "finished_at_utc": datetime.now(UTC).isoformat(),
        "hardware": collect_hardware_metadata(repository).to_dict(),
        "gpu_runtime_state_before": runtime_before,
        "gpu_runtime_state_after": collect_gpu_runtime_state(device.index or 0),
        "manifest": asdict(manifest),
        "measurement": {
            "quantiles": [0.50, 0.95, 0.99],
            "timing_boundary": (
                "CUDA events bracket each append after setup; host throughput includes cache "
                "transactions, two pinned-control H2D copies, dispatch, and final synchronization"
            ),
            "setup_excluded": True,
            "warmups_excluded": True,
            "correctness_gate": "first measured logit versus independent full-prefix model",
            "mode_order": "seed-shuffled for every measured repetition",
            "compile_boundary": (
                "PyTorch model segments use torch.compile with Inductor cudagraphs disabled; "
                "custom Triton KV-write and paged-attention launches remain explicit graph breaks"
            ),
            "cuda_graph_boundary": (
                "explicit torch.cuda.CUDAGraph captures bucketed model/KV kernels; cache ownership "
                "and pinned-host control staging remain outside capture"
            ),
        },
        "cases": case_results,
    }


def write_execution_reports(
    result: dict[str, Any], output_directory: Path
) -> tuple[Path, Path, Path]:
    output_directory.mkdir(parents=True, exist_ok=True)
    json_path = output_directory / "execution.json"
    csv_path = output_directory / "execution.csv"
    markdown_path = output_directory / "execution.md"
    json_path.write_text(
        json.dumps(result, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )
    rows = []
    for case_result in result["cases"]:
        for mode_result in case_result["modes"]:
            aggregate = mode_result["aggregate"]
            latency = aggregate["cuda_step_latency_ms"]
            metrics = aggregate["measured_metric_totals"]
            rows.append(
                {
                    "suite_id": result["suite_id"],
                    "case": case_result["case"]["name"],
                    "mode": mode_result["mode"],
                    "latency_p50_ms": latency["p50"],
                    "latency_p95_ms": latency["p95"],
                    "latency_p99_ms": latency["p99"],
                    "tokens_per_second_mean": aggregate["tokens_per_second"]["mean"],
                    "latency_speedup_vs_eager": aggregate["mean_latency_speedup_vs_eager"],
                    "latency_change_pct_vs_eager": aggregate[
                        "mean_latency_change_pct_vs_eager"
                    ],
                    "graph_hits": metrics["graph_hits"],
                    "graph_misses": metrics["graph_misses"],
                    "shape_fallbacks": metrics["shape_fallbacks"],
                    "capture_time_ms_mean": aggregate["capture_time_ms"]["mean"],
                    "compile_time_ms_mean": aggregate["compile_time_ms"]["mean"],
                    "cold_setup_wall_ms": aggregate["cold_setup_wall_ms"],
                    "cold_capture_time_ms": aggregate["cold_capture_time_ms"],
                    "cold_compile_time_ms": aggregate["cold_compile_time_ms"],
                    "addresses_stable": aggregate["address_stable_all_runs"],
                    "git_commit": result["hardware"]["git_commit"],
                    "git_dirty": result["hardware"]["git_dirty"],
                    "hardware_fingerprint": result["hardware"]["fingerprint_sha256"],
                }
            )
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    hardware = result["hardware"]
    gpu = hardware["gpus"][0]
    lines = [
        "# TensorForge Phase 6 execution-specialization report",
        "",
        f"- Suite: `{result['suite_id']}`",
        f"- GPU: {gpu['name']} (compute capability {gpu['compute_capability']})",
        f"- PyTorch / CUDA / Triton: `{hardware['torch_version']}` / "
        f"`{hardware['torch_cuda_runtime']}` / `{hardware['triton_version']}`",
        f"- Git: `{hardware['git_commit']}`; dirty: `{hardware['git_dirty']}`",
        "",
        "Setup/capture/compile costs are excluded from steady-state latency and reported "
        "separately. Every run passes an independent full-prefix numerical gate.",
        "",
        "| Case | Mode | CUDA latency P50/P95/P99 ms | Mean tok/s | Mean change vs eager | "
        "Graph hit/miss | Shape fallback | Cold setup/capture/compile ms |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['case']} | {row['mode']} | {row['latency_p50_ms']:.3f}/"
            f"{row['latency_p95_ms']:.3f}/{row['latency_p99_ms']:.3f} | "
            f"{row['tokens_per_second_mean']:.2f} | "
            f"{row['latency_change_pct_vs_eager']:+.1f}% | "
            f"{row['graph_hits']}/{row['graph_misses']} | {row['shape_fallbacks']} | "
            f"{row['cold_setup_wall_ms']:.2f}/{row['cold_capture_time_ms']:.2f}/"
            f"{row['cold_compile_time_ms']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "This isolates execution specialization over the same address-stable bucket, model "
            "weights, transactional PagedKVCache, Triton KV writer, and Triton GQA attention. "
            "`torch.compile` is intentionally a hybrid segmented path because custom Triton calls "
            "are explicit graph boundaries. The fallback case measures dynamic eager delegation, "
            "not a captured or compiled shape. Regressions are retained.",
        ]
    )
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, csv_path, markdown_path
