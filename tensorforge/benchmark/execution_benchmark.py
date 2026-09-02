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
from tensorforge.model.layers import apply_rotary_embedding
from tensorforge.model.llama import LlamaForCausalLM
from tensorforge.runtime.batch import BatchTokenExecutor
from tensorforge.runtime.batched_decode import BatchedPagedDecodeExecutor
from tensorforge.runtime.decode_bucket import (
    BucketedPagedDecodeExecutor,
    DecodeExecutionMetrics,
    DecodeExecutionMode,
    DecodeFusionLevel,
)

EXECUTION_BENCHMARK_SCHEMA_VERSION = "1.1"
_SUPPORTED_SCHEMA_VERSIONS = {"1.0", EXECUTION_BENCHMARK_SCHEMA_VERSION}


@dataclass(frozen=True, slots=True)
class ExecutionCase:
    name: str
    batch_size: int
    initial_context: int
    decode_steps: int
    batch_buckets: tuple[int, ...]
    context_buckets: tuple[int, ...]
    setup_method: str = "token_decode"

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
        if self.setup_method not in {"token_decode", "reference_prefill"}:
            raise ValueError("setup_method must be token_decode or reference_prefill")


@dataclass(frozen=True, slots=True)
class ExecutionVariant:
    name: str
    mode: DecodeExecutionMode
    fusion_level: DecodeFusionLevel = DecodeFusionLevel.NONE
    stable_bucket: bool = True
    parent: str | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("execution variant name must be non-empty")
        if not self.stable_bucket and (
            self.mode is not DecodeExecutionMode.EAGER
            or self.fusion_level is not DecodeFusionLevel.NONE
        ):
            raise ValueError("dynamic execution supports only unfused eager mode")


_DEFAULT_VARIANTS = (
    ExecutionVariant(name="eager", mode=DecodeExecutionMode.EAGER),
    ExecutionVariant(
        name="torch_compile", mode=DecodeExecutionMode.COMPILE, parent="eager"
    ),
    ExecutionVariant(
        name="cuda_graph", mode=DecodeExecutionMode.CUDA_GRAPH, parent="eager"
    ),
)


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
    variants: tuple[ExecutionVariant, ...] = _DEFAULT_VARIANTS

    def __post_init__(self) -> None:
        if not self.cases:
            raise ValueError("execution manifest requires at least one case")
        if not self.variants:
            raise ValueError("execution manifest requires at least one variant")
        if len({variant.name for variant in self.variants}) != len(self.variants):
            raise ValueError("execution variant names must be unique")
        names = {variant.name for variant in self.variants}
        for variant in self.variants:
            if variant.parent is not None and variant.parent not in names:
                raise ValueError(f"unknown parent variant: {variant.parent}")
            if variant.parent == variant.name:
                raise ValueError("execution variant cannot parent itself")
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
    variant: str
    execution_mode: str
    fusion_level: str
    stable_bucket: bool
    setup_wall_ms: float
    measured_host_ms: float
    cuda_step_latency_ms: tuple[float, ...]
    tokens_per_second: float
    address_stable: bool | None
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
        "variants",
        "cases",
    }
    unknown = set(payload) - allowed
    if unknown:
        raise ValueError(f"unknown execution manifest keys: {sorted(unknown)}")
    if payload.get("schema_version") not in _SUPPORTED_SCHEMA_VERSIONS:
        raise ValueError(
            f"expected execution schema in {sorted(_SUPPORTED_SCHEMA_VERSIONS)}"
        )
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
        "setup_method",
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
    variant_keys = {"name", "mode", "fusion_level", "stable_bucket", "parent"}
    variants = []
    for raw_variant in payload.get("variants", []):
        unknown_variant = set(raw_variant) - variant_keys
        if unknown_variant:
            raise ValueError(f"unknown execution variant keys: {sorted(unknown_variant)}")
        variants.append(
            ExecutionVariant(
                name=str(raw_variant["name"]),
                mode=DecodeExecutionMode(raw_variant["mode"]),
                fusion_level=DecodeFusionLevel(
                    raw_variant.get("fusion_level", DecodeFusionLevel.NONE.value)
                ),
                stable_bucket=bool(raw_variant.get("stable_bucket", True)),
                parent=(
                    None
                    if raw_variant.get("parent") is None
                    else str(raw_variant["parent"])
                ),
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
        variants=tuple(variants) if variants else _DEFAULT_VARIANTS,
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


def _empty_metrics() -> DecodeExecutionMetrics:
    return DecodeExecutionMetrics(
        eager_calls=0,
        compiled_calls=0,
        graph_calls=0,
        graph_hits=0,
        graph_misses=0,
        shape_fallbacks=0,
        eager_fallback_calls=0,
        capture_count=0,
        capture_time_ms=0.0,
        capture_warmup_time_ms=0.0,
        compile_count=0,
        compile_time_ms=0.0,
        padded_lanes=0,
        fallback_reasons={},
    )


def _executor_metrics(executor: BatchTokenExecutor) -> DecodeExecutionMetrics:
    if isinstance(executor, BucketedPagedDecodeExecutor):
        return executor.metrics()
    return _empty_metrics()


def _executor_addresses(executor: BatchTokenExecutor) -> dict[str, dict[str, int]]:
    if isinstance(executor, BucketedPagedDecodeExecutor):
        return executor.buffer_addresses()
    return {}


@torch.inference_mode()
def _prime_request_cache(
    model: LlamaForCausalLM,
    cache: PagedKVCache,
    request_id: str,
    token_ids: list[int],
) -> None:
    """Populate a benchmark prefix transaction with the readable full-prefix path."""

    if not token_ids:
        return
    device = next(model.parameters()).device
    reservation = cache.begin_append(request_id, token_count=len(token_ids))
    try:
        tokens = torch.tensor([token_ids], dtype=torch.long, device=device)
        positions = torch.arange(len(token_ids), dtype=torch.long, device=device)
        hidden_states = model.model.embedding(tokens)
        for layer_index, layer in enumerate(model.model.layers):
            attention = layer.attention
            normalized = layer.input_norm(hidden_states)
            query = attention._shape(attention.q_proj(normalized), attention.num_heads)
            key = attention._shape(
                attention.k_proj(normalized), attention.num_key_value_heads
            )
            value = attention._shape(
                attention.v_proj(normalized), attention.num_key_value_heads
            )
            cos, sin = attention.rotary(positions, dtype=query.dtype)
            _, key = apply_rotary_embedding(query, key, cos, sin)
            cache.write_layer(
                reservation,
                layer_index,
                key[0].transpose(0, 1).contiguous(),
                value[0].transpose(0, 1).contiguous(),
            )
            hidden_states = hidden_states + attention(normalized)
            hidden_states = hidden_states + layer.mlp(
                layer.post_attention_norm(hidden_states)
            )
        cache.commit(reservation)
    except Exception:
        cache.rollback(reservation)
        raise


def _run_once(
    manifest: ExecutionManifest,
    case: ExecutionCase,
    *,
    model: LlamaForCausalLM,
    device: torch.device,
    variant: ExecutionVariant,
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
    executor: BatchTokenExecutor = (
        BucketedPagedDecodeExecutor(
            model=model,
            cache=cache,
            mode=variant.mode,
            batch_buckets=case.batch_buckets,
            context_buckets=case.context_buckets,
            fusion_level=variant.fusion_level,
        )
        if variant.stable_bucket
        else BatchedPagedDecodeExecutor(model=model, cache=cache)
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
    setup_start_position = 0
    if case.setup_method == "reference_prefill":
        prefix_length = case.initial_context - 1
        for index, request_id in enumerate(request_ids):
            _prime_request_cache(
                model,
                cache,
                request_id,
                token_rows[index][:prefix_length],
            )
        setup_start_position = prefix_length
    for position in range(setup_start_position, case.initial_context):
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
    addresses_before = _executor_addresses(executor)
    setup_metrics = _executor_metrics(executor)

    event_factory: Any = torch.cuda.Event
    start_events = [event_factory(enable_timing=True) for _ in range(case.decode_steps)]
    end_events = [event_factory(enable_timing=True) for _ in range(case.decode_steps)]
    measured_before = _executor_metrics(executor)
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
    measured_after = _executor_metrics(executor)
    addresses_after = _executor_addresses(executor)
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
        raise RuntimeError(f"cache state leaked after {case.name}/{variant.name}")
    return ExecutionRun(
        repetition=repetition,
        variant=variant.name,
        execution_mode=variant.mode.value,
        fusion_level=variant.fusion_level.value,
        stable_bucket=variant.stable_bucket,
        setup_wall_ms=setup_wall_ms,
        measured_host_ms=measured_host_ms,
        cuda_step_latency_ms=cuda_step_latency_ms,
        tokens_per_second=(case.batch_size * case.decode_steps) / (measured_host_ms / 1_000),
        address_stable=(addresses_before == addresses_after) if addresses_before else None,
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
        "address_stable_all_runs": (
            all(run.address_stable is True for run in runs)
            if any(run.address_stable is not None for run in runs)
            else None
        ),
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
        warmup_runs_by_variant: dict[str, list[ExecutionRun]] = {
            variant.name: [] for variant in manifest.variants
        }
        for warmup in range(manifest.warmup_repetitions):
            for variant in manifest.variants:
                warmup_runs_by_variant[variant.name].append(
                    _run_once(
                        manifest,
                        case,
                        model=model,
                        device=device,
                        variant=variant,
                        repetition=-(warmup + 1),
                    )
                )
        runs_by_variant: dict[str, list[ExecutionRun]] = {
            variant.name: [] for variant in manifest.variants
        }
        orders = []
        for repetition in range(manifest.measured_repetitions):
            order = list(manifest.variants)
            random.Random(manifest.seed + repetition + case.batch_size).shuffle(order)
            orders.append([variant.name for variant in order])
            for variant in order:
                runs_by_variant[variant.name].append(
                    _run_once(
                        manifest,
                        case,
                        model=model,
                        device=device,
                        variant=variant,
                        repetition=repetition,
                    )
                )
        aggregates = {
            variant.name: _aggregate(runs_by_variant[variant.name])
            for variant in manifest.variants
        }
        baseline = aggregates[manifest.variants[0].name]
        variants = []
        for variant in manifest.variants:
            aggregate = aggregates[variant.name]
            baseline_latency = baseline["cuda_step_latency_ms"]["mean"]
            latency = aggregate["cuda_step_latency_ms"]["mean"]
            aggregate["mean_latency_speedup_vs_baseline"] = baseline_latency / latency
            aggregate["mean_latency_change_pct_vs_baseline"] = (
                latency / baseline_latency - 1
            ) * 100
            parent = (
                aggregates[variant.parent]
                if variant.parent is not None
                else aggregate
            )
            parent_latency = parent["cuda_step_latency_ms"]["mean"]
            aggregate["mean_latency_speedup_vs_parent"] = parent_latency / latency
            aggregate["mean_latency_change_pct_vs_parent"] = (
                latency / parent_latency - 1
            ) * 100
            cold_setup = warmup_runs_by_variant[variant.name][0]
            aggregate["cold_setup_wall_ms"] = cold_setup.setup_wall_ms
            aggregate["cold_capture_time_ms"] = cold_setup.setup_metrics[
                "capture_time_ms"
            ]
            aggregate["cold_compile_time_ms"] = cold_setup.setup_metrics[
                "compile_time_ms"
            ]
            variants.append(
                {
                    "variant": asdict(variant),
                    "warmup_runs": [
                        asdict(run) for run in warmup_runs_by_variant[variant.name]
                    ],
                    "runs": [asdict(run) for run in runs_by_variant[variant.name]],
                    "aggregate": aggregate,
                }
            )
        case_results.append(
            {
                "case": asdict(case),
                "variant_order_by_repetition": orders,
                "variants": variants,
            }
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
            "variant_order": "seed-shuffled for every measured repetition",
            "reference_prefill": (
                "optional setup-only full-prefix cache priming writes a multi-token transaction; "
                "it is excluded from timing and is not a production prefill-kernel claim"
            ),
            "compile_boundary": (
                "PyTorch model segments use torch.compile with Inductor cudagraphs disabled; "
                "custom Triton norm, activation, KV-write, and paged-attention launches remain "
                "explicit graph breaks"
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
        for variant_result in case_result["variants"]:
            aggregate = variant_result["aggregate"]
            variant = variant_result["variant"]
            latency = aggregate["cuda_step_latency_ms"]
            metrics = aggregate["measured_metric_totals"]
            rows.append(
                {
                    "suite_id": result["suite_id"],
                    "case": case_result["case"]["name"],
                    "variant": variant["name"],
                    "execution_mode": variant["mode"],
                    "fusion_level": variant["fusion_level"],
                    "stable_bucket": variant["stable_bucket"],
                    "parent": variant["parent"],
                    "latency_p50_ms": latency["p50"],
                    "latency_p95_ms": latency["p95"],
                    "latency_p99_ms": latency["p99"],
                    "tokens_per_second_mean": aggregate["tokens_per_second"]["mean"],
                    "latency_speedup_vs_baseline": aggregate[
                        "mean_latency_speedup_vs_baseline"
                    ],
                    "latency_change_pct_vs_baseline": aggregate[
                        "mean_latency_change_pct_vs_baseline"
                    ],
                    "latency_speedup_vs_parent": aggregate[
                        "mean_latency_speedup_vs_parent"
                    ],
                    "latency_change_pct_vs_parent": aggregate[
                        "mean_latency_change_pct_vs_parent"
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
    is_cumulative = len(result["manifest"]["variants"]) > 3
    lines = [
        (
            "# TensorForge Phase 7A cumulative decode-ablation report"
            if is_cumulative
            else "# TensorForge Phase 6 execution-specialization report"
        ),
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
        "| Case | Variant | Mode/fusion | CUDA latency P50/P95/P99 ms | Mean tok/s | "
        "Change vs baseline/parent | Graph hit/miss | Cold setup/capture/compile ms |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['case']} | {row['variant']} | {row['execution_mode']}/"
            f"{row['fusion_level']} | {row['latency_p50_ms']:.3f}/"
            f"{row['latency_p95_ms']:.3f}/{row['latency_p99_ms']:.3f} | "
            f"{row['tokens_per_second_mean']:.2f} | "
            f"{row['latency_change_pct_vs_baseline']:+.1f}%/"
            f"{row['latency_change_pct_vs_parent']:+.1f}% | "
            f"{row['graph_hits']}/{row['graph_misses']} | "
            f"{row['cold_setup_wall_ms']:.2f}/{row['cold_capture_time_ms']:.2f}/"
            f"{row['cold_compile_time_ms']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "Rows share model weights, transactional PagedKVCache, and paged GQA attention. Each "
            "variant names its causal parent; compile and CUDA Graph both compare with the fully "
            "fused eager parent rather than with each other. `torch.compile` remains segmented "
            "because custom Triton calls are explicit graph boundaries. Reference prefill is an "
            "excluded setup mechanism, not a production prefill claim. Regressions are retained.",
        ]
    )
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, csv_path, markdown_path
