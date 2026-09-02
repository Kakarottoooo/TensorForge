"""Reproducible continuous-batching experiments over the canonical paged cache."""

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
from tensorforge.benchmark.schema import ModelSpec, RequestTrace
from tensorforge.cache.paged import PagedKVCache, PagedKVCacheConfig
from tensorforge.metrics.statistics import summarize
from tensorforge.model.llama import LlamaForCausalLM
from tensorforge.runtime.batched_decode import BatchedPagedDecodeExecutor
from tensorforge.scheduler.continuous import (
    BatchingPolicy,
    ContinuousBatchScheduler,
    RequestInput,
    RequestSnapshot,
    RequestState,
    SchedulerConfig,
)

SCHEDULER_BENCHMARK_SCHEMA_VERSION = "1.0"
_TERMINAL_STATES = {
    RequestState.COMPLETED,
    RequestState.FAILED,
    RequestState.CANCELLED,
}
_POLICIES = (
    BatchingPolicy.NO_BATCHING,
    BatchingPolicy.STATIC,
    BatchingPolicy.CONTINUOUS,
)


@dataclass(frozen=True, slots=True)
class SchedulerWorkload:
    name: str
    request_count: int
    prompt_length_choices: tuple[int, ...]
    generation_length_choices: tuple[int, ...]
    max_arrival_step: int
    cancellation_fraction: float
    cancellation_delay_steps: int
    seed: int

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("workload name must be non-empty")
        if self.request_count <= 0:
            raise ValueError("request_count must be positive")
        if not self.prompt_length_choices or any(
            value <= 0 for value in self.prompt_length_choices
        ):
            raise ValueError("prompt_length_choices must contain positive lengths")
        if not self.generation_length_choices or any(
            value <= 0 for value in self.generation_length_choices
        ):
            raise ValueError("generation_length_choices must contain positive lengths")
        if self.max_arrival_step < 0:
            raise ValueError("max_arrival_step must be non-negative")
        if not 0 <= self.cancellation_fraction <= 1:
            raise ValueError("cancellation_fraction must be between zero and one")
        if self.cancellation_delay_steps < 0:
            raise ValueError("cancellation_delay_steps must be non-negative")


@dataclass(frozen=True, slots=True)
class SchedulerManifest:
    model: ModelSpec
    workloads: tuple[SchedulerWorkload, ...]
    precision: str = "fp16"
    block_size: int = 16
    num_blocks: int = 64
    max_batch_requests: int = 8
    max_batch_tokens: int = 8
    max_active_token_budget: int = 2_048
    max_request_tokens: int = 64
    warmup_repetitions: int = 1
    measured_repetitions: int = 3

    def __post_init__(self) -> None:
        if not self.workloads:
            raise ValueError("scheduler manifest requires at least one workload")
        if self.precision not in {"fp16", "bf16", "fp32"}:
            raise ValueError(f"unsupported precision: {self.precision}")
        dimensions = {
            "block_size": self.block_size,
            "num_blocks": self.num_blocks,
            "max_batch_requests": self.max_batch_requests,
            "max_batch_tokens": self.max_batch_tokens,
            "max_active_token_budget": self.max_active_token_budget,
            "max_request_tokens": self.max_request_tokens,
            "measured_repetitions": self.measured_repetitions,
        }
        for name, value in dimensions.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.warmup_repetitions < 0:
            raise ValueError("warmup_repetitions must be non-negative")
        if self.max_batch_tokens > self.max_batch_requests:
            raise ValueError("max_batch_tokens cannot exceed max_batch_requests")
        if self.max_request_tokens > self.max_active_token_budget:
            raise ValueError("max_request_tokens cannot exceed the global budget")
        for workload in self.workloads:
            maximum_length = max(workload.prompt_length_choices) + max(
                workload.generation_length_choices
            )
            if maximum_length > self.model.max_position_embeddings:
                raise ValueError(f"workload {workload.name} exceeds model context capacity")
            if maximum_length > self.max_request_tokens:
                raise ValueError(f"workload {workload.name} exceeds per-request token budget")
            if maximum_length * workload.request_count > self.max_active_token_budget:
                raise ValueError(
                    f"workload {workload.name} may exceed the global admission budget"
                )
        maximum_sequence = max(
            max(workload.prompt_length_choices) + max(workload.generation_length_choices)
            for workload in self.workloads
        )
        maximum_blocks_per_batch = (
            (maximum_sequence + self.block_size - 1)
            // self.block_size
            * self.max_batch_requests
        )
        if maximum_blocks_per_batch > self.num_blocks:
            raise ValueError("KV cache cannot hold the configured worst-case active batch")


@dataclass(frozen=True, slots=True)
class PlannedRequest:
    request: RequestInput
    arrival_step: int
    cancellation_step: int | None


@dataclass(frozen=True, slots=True)
class SchedulerRun:
    repetition: int
    policy: BatchingPolicy
    traces: tuple[RequestTrace, ...]
    host_elapsed_ms: float
    cuda_elapsed_ms: float
    incremental_peak_memory_bytes: int
    completed_requests: int
    cancelled_requests: int
    failed_requests: int
    planned_cancellations: int
    missed_cancellations: int
    generated_tokens: int
    executed_input_tokens: int
    scheduler_steps: int
    maximum_observed_batch: int


def _dtype(name: str) -> torch.dtype:
    return {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[name]


def build_request_plan(
    workload: SchedulerWorkload, *, vocab_size: int
) -> tuple[PlannedRequest, ...]:
    """Generate one immutable seeded plan reused by all policy variants."""

    generator = random.Random(workload.seed)
    arrivals = [
        generator.randint(0, workload.max_arrival_step)
        for _ in range(workload.request_count)
    ]
    cancellation_count = round(workload.request_count * workload.cancellation_fraction)
    cancelled_indices = set(generator.sample(range(workload.request_count), cancellation_count))
    plan = []
    for index, arrival_step in enumerate(arrivals):
        prompt_length = generator.choice(workload.prompt_length_choices)
        generation_length = generator.choice(workload.generation_length_choices)
        prompt = tuple(generator.randrange(vocab_size) for _ in range(prompt_length))
        cancellation_step = (
            arrival_step + workload.cancellation_delay_steps
            if index in cancelled_indices
            else None
        )
        plan.append(
            PlannedRequest(
                request=RequestInput(
                    request_id=f"{workload.name}-{index:04d}",
                    prompt_token_ids=prompt,
                    max_new_tokens=generation_length,
                ),
                arrival_step=arrival_step,
                cancellation_step=cancellation_step,
            )
        )
    return tuple(sorted(plan, key=lambda item: (item.arrival_step, item.request.request_id)))


def load_scheduler_manifest(path: Path) -> SchedulerManifest:
    payload = json.loads(path.read_text(encoding="utf-8"))
    allowed_manifest = {
        "schema_version",
        "model",
        "precision",
        "block_size",
        "num_blocks",
        "max_batch_requests",
        "max_batch_tokens",
        "max_active_token_budget",
        "max_request_tokens",
        "warmup_repetitions",
        "measured_repetitions",
        "workloads",
    }
    unknown_manifest = set(payload) - allowed_manifest
    if unknown_manifest:
        raise ValueError(f"unknown scheduler manifest keys: {sorted(unknown_manifest)}")
    if payload.get("schema_version") != SCHEDULER_BENCHMARK_SCHEMA_VERSION:
        raise ValueError(f"expected scheduler schema {SCHEDULER_BENCHMARK_SCHEMA_VERSION}")
    model_keys = set(ModelSpec.__dataclass_fields__)
    unknown_model = set(payload["model"]) - model_keys
    if unknown_model:
        raise ValueError(f"unknown scheduler model keys: {sorted(unknown_model)}")
    workload_keys = {
        "name",
        "request_count",
        "prompt_length_choices",
        "generation_length_choices",
        "max_arrival_step",
        "cancellation_fraction",
        "cancellation_delay_steps",
        "seed",
    }
    for workload in payload["workloads"]:
        unknown_workload = set(workload) - workload_keys
        if unknown_workload:
            raise ValueError(f"unknown scheduler workload keys: {sorted(unknown_workload)}")
    workloads = tuple(
        SchedulerWorkload(
            **{
                **workload,
                "prompt_length_choices": tuple(workload["prompt_length_choices"]),
                "generation_length_choices": tuple(workload["generation_length_choices"]),
            }
        )
        for workload in payload["workloads"]
    )
    defaults = SchedulerManifest.__dataclass_fields__
    return SchedulerManifest(
        model=ModelSpec(**payload["model"]),
        workloads=workloads,
        precision=str(payload.get("precision", defaults["precision"].default)),
        block_size=int(payload.get("block_size", defaults["block_size"].default)),
        num_blocks=int(payload.get("num_blocks", defaults["num_blocks"].default)),
        max_batch_requests=int(
            payload.get("max_batch_requests", defaults["max_batch_requests"].default)
        ),
        max_batch_tokens=int(
            payload.get("max_batch_tokens", defaults["max_batch_tokens"].default)
        ),
        max_active_token_budget=int(
            payload.get(
                "max_active_token_budget", defaults["max_active_token_budget"].default
            )
        ),
        max_request_tokens=int(
            payload.get("max_request_tokens", defaults["max_request_tokens"].default)
        ),
        warmup_repetitions=int(
            payload.get("warmup_repetitions", defaults["warmup_repetitions"].default)
        ),
        measured_repetitions=int(
            payload.get("measured_repetitions", defaults["measured_repetitions"].default)
        ),
    )


def _make_cache(
    manifest: SchedulerManifest, *, device: torch.device, dtype: torch.dtype
) -> PagedKVCache:
    config = manifest.model.to_model_config()
    return PagedKVCache(
        PagedKVCacheConfig(
            num_layers=config.num_hidden_layers,
            num_blocks=manifest.num_blocks,
            block_size=manifest.block_size,
            num_kv_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            max_sequences=manifest.max_batch_requests,
            max_sequence_length=config.max_position_embeddings,
            dtype=dtype,
            device=device,
        )
    )


def _trace(snapshot: RequestSnapshot) -> RequestTrace:
    assert snapshot.started_ns is not None
    assert snapshot.completed_ns is not None
    assert snapshot.finish_reason is not None
    return RequestTrace(
        request_id=snapshot.request_id,
        arrival_ns=snapshot.arrival_ns,
        started_ns=snapshot.started_ns,
        token_timestamps_ns=snapshot.token_timestamps_ns,
        completed_ns=snapshot.completed_ns,
        prompt_tokens=len(snapshot.prompt_token_ids),
        generated_tokens=len(snapshot.generated_token_ids),
        finish_reason=snapshot.finish_reason,
    )


def _run_once(
    manifest: SchedulerManifest,
    workload: SchedulerWorkload,
    plan: tuple[PlannedRequest, ...],
    *,
    model: LlamaForCausalLM,
    device: torch.device,
    policy: BatchingPolicy,
    repetition: int,
) -> SchedulerRun:
    dtype = _dtype(manifest.precision)
    cache = _make_cache(manifest, device=device, dtype=dtype)
    scheduler = ContinuousBatchScheduler(
        BatchedPagedDecodeExecutor(model=model, cache=cache),
        SchedulerConfig(
            policy=policy,
            max_batch_requests=manifest.max_batch_requests,
            max_batch_tokens=manifest.max_batch_tokens,
            max_active_token_budget=manifest.max_active_token_budget,
            max_request_tokens=manifest.max_request_tokens,
        ),
    )
    plans_by_arrival: dict[int, list[PlannedRequest]] = {}
    plans_by_cancellation: dict[int, list[PlannedRequest]] = {}
    for item in plan:
        plans_by_arrival.setdefault(item.arrival_step, []).append(item)
        if item.cancellation_step is not None:
            plans_by_cancellation.setdefault(item.cancellation_step, []).append(item)

    memory_before = torch.cuda.memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)
    event_factory: Any = torch.cuda.Event
    start_event = event_factory(enable_timing=True)
    end_event = event_factory(enable_timing=True)
    missed_cancellations = 0
    tick = 0
    maximum_tick = max(
        [item.arrival_step for item in plan]
        + [item.cancellation_step for item in plan if item.cancellation_step is not None]
    )
    start_ns = time.perf_counter_ns()
    start_event.record()
    while True:
        for item in plans_by_arrival.get(tick, ()):
            scheduler.submit(item.request)
        for item in plans_by_cancellation.get(tick, ()):
            snapshot = scheduler.snapshot(item.request.request_id)
            if snapshot.state in _TERMINAL_STATES:
                missed_cancellations += 1
            else:
                scheduler.cancel(item.request.request_id)
        snapshots = scheduler.snapshots()
        has_nonterminal = any(snapshot.state not in _TERMINAL_STATES for snapshot in snapshots)
        if has_nonterminal:
            scheduler.step()
        if tick >= maximum_tick and not any(
            snapshot.state not in _TERMINAL_STATES for snapshot in scheduler.snapshots()
        ):
            break
        tick += 1
        if tick > 1_000_000:
            raise RuntimeError(f"scheduler did not converge for workload {workload.name}")
    end_event.record()
    torch.cuda.synchronize(device)
    host_elapsed_ms = (time.perf_counter_ns() - start_ns) / 1e6
    cuda_elapsed_ms = float(start_event.elapsed_time(end_event))

    snapshots = scheduler.snapshots()
    completed = tuple(
        snapshot for snapshot in snapshots if snapshot.state is RequestState.COMPLETED
    )
    cancelled = tuple(
        snapshot for snapshot in snapshots if snapshot.state is RequestState.CANCELLED
    )
    failed = tuple(snapshot for snapshot in snapshots if snapshot.state is RequestState.FAILED)
    stats = cache.stats()
    if (
        stats.active_sequences
        or stats.used_blocks
        or stats.reserved_tokens
        or scheduler.outstanding_token_budget
    ):
        raise RuntimeError(f"cache or scheduler state leaked after {workload.name}/{policy}")
    return SchedulerRun(
        repetition=repetition,
        policy=policy,
        traces=tuple(_trace(snapshot) for snapshot in completed),
        host_elapsed_ms=host_elapsed_ms,
        cuda_elapsed_ms=cuda_elapsed_ms,
        incremental_peak_memory_bytes=max(
            0, torch.cuda.max_memory_allocated(device) - memory_before
        ),
        completed_requests=len(completed),
        cancelled_requests=len(cancelled),
        failed_requests=len(failed),
        planned_cancellations=sum(item.cancellation_step is not None for item in plan),
        missed_cancellations=missed_cancellations,
        generated_tokens=sum(len(snapshot.generated_token_ids) for snapshot in completed),
        executed_input_tokens=scheduler.executed_token_count,
        scheduler_steps=scheduler.step_count,
        maximum_observed_batch=scheduler.maximum_observed_batch,
    )


def _distribution(values: list[float]) -> dict[str, Any]:
    return asdict(summarize(values))


def _aggregate_runs(runs: tuple[SchedulerRun, ...]) -> dict[str, Any]:
    traces = [trace for run in runs for trace in run.traces]
    if not traces:
        raise RuntimeError("benchmark policy produced no completed requests")
    request_latency_ms = [(trace.completed_ns - trace.arrival_ns) / 1e6 for trace in traces]
    ttft_ms = [(trace.token_timestamps_ns[0] - trace.arrival_ns) / 1e6 for trace in traces]
    queue_ms = [(trace.started_ns - trace.arrival_ns) / 1e6 for trace in traces]
    tpot_ms = [
        (trace.completed_ns - trace.token_timestamps_ns[0])
        / (trace.generated_tokens - 1)
        / 1e6
        for trace in traces
        if trace.generated_tokens > 1
    ]
    output_throughput = [
        run.generated_tokens / (run.host_elapsed_ms / 1_000) for run in runs
    ]
    input_throughput = [
        run.executed_input_tokens / (run.host_elapsed_ms / 1_000) for run in runs
    ]
    result: dict[str, Any] = {
        "request_latency_ms": _distribution(request_latency_ms),
        "ttft_ms": _distribution(ttft_ms),
        "queue_ms": _distribution(queue_ms),
        "output_tokens_per_second": _distribution(output_throughput),
        "executed_input_tokens_per_second": _distribution(input_throughput),
        "host_elapsed_ms": _distribution([run.host_elapsed_ms for run in runs]),
        "cuda_elapsed_ms": _distribution([run.cuda_elapsed_ms for run in runs]),
        "incremental_peak_memory_bytes": _distribution(
            [float(run.incremental_peak_memory_bytes) for run in runs]
        ),
        "scheduler_steps": _distribution([float(run.scheduler_steps) for run in runs]),
        "maximum_observed_batch": max(run.maximum_observed_batch for run in runs),
        "completed_requests": sum(run.completed_requests for run in runs),
        "cancelled_requests": sum(run.cancelled_requests for run in runs),
        "failed_requests": sum(run.failed_requests for run in runs),
        "planned_cancellations": sum(run.planned_cancellations for run in runs),
        "missed_cancellations": sum(run.missed_cancellations for run in runs),
    }
    if tpot_ms:
        result["tpot_ms"] = _distribution(tpot_ms)
    return result


def run_scheduler_suite(
    manifest: SchedulerManifest, *, device: torch.device, repository: Path
) -> dict[str, Any]:
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("scheduler benchmark requires an available CUDA device")
    started = datetime.now(UTC)
    runtime_state_before = collect_gpu_runtime_state(device.index or 0)
    dtype = _dtype(manifest.precision)
    torch.manual_seed(7)
    torch.cuda.manual_seed_all(7)
    model = LlamaForCausalLM(manifest.model.to_model_config()).to(
        device=device, dtype=dtype
    )
    model.eval()
    workload_results: list[dict[str, Any]] = []
    for workload in manifest.workloads:
        plan = build_request_plan(workload, vocab_size=manifest.model.vocab_size)
        for policy in _POLICIES:
            for warmup in range(manifest.warmup_repetitions):
                _run_once(
                    manifest,
                    workload,
                    plan,
                    model=model,
                    device=device,
                    policy=policy,
                    repetition=-(warmup + 1),
                )
        runs_by_policy: dict[BatchingPolicy, list[SchedulerRun]] = {
            policy: [] for policy in _POLICIES
        }
        policy_orders: list[list[str]] = []
        for repetition in range(manifest.measured_repetitions):
            order = list(_POLICIES)
            random.Random(workload.seed + repetition).shuffle(order)
            policy_orders.append([policy.value for policy in order])
            for policy in order:
                runs_by_policy[policy].append(
                    _run_once(
                        manifest,
                        workload,
                        plan,
                        model=model,
                        device=device,
                        policy=policy,
                        repetition=repetition,
                    )
                )
        workload_results.append(
            {
                "workload": asdict(workload),
                "request_plan": [asdict(item) for item in plan],
                "policy_order_by_repetition": policy_orders,
                "policies": [
                    {
                        "policy": policy.value,
                        "runs": [asdict(run) for run in runs_by_policy[policy]],
                        "aggregate": _aggregate_runs(tuple(runs_by_policy[policy])),
                    }
                    for policy in _POLICIES
                ],
            }
        )
    return {
        "schema_version": SCHEDULER_BENCHMARK_SCHEMA_VERSION,
        "suite_id": str(uuid.uuid4()),
        "started_at_utc": started.isoformat(),
        "finished_at_utc": datetime.now(UTC).isoformat(),
        "hardware": collect_hardware_metadata(repository).to_dict(),
        "gpu_runtime_state_before": runtime_state_before,
        "gpu_runtime_state_after": collect_gpu_runtime_state(device.index or 0),
        "manifest": asdict(manifest),
        "measurement": {
            "request_quantiles": [0.50, 0.95, 0.99],
            "arrival_clock": "seeded logical scheduler steps",
            "timing_boundary": (
                "host submission through terminal release; batched argmax synchronizes before "
                "token timestamps and CUDA events are synchronized at run end"
            ),
            "warmups_excluded": True,
            "policy_order": "seed-shuffled independently for each measured repetition",
            "shared_path": (
                "identical model weights, paged KV allocator, Triton GQA attention, and batched "
                "executor; only admission/refill policy changes"
            ),
        },
        "workloads": workload_results,
    }


def write_scheduler_reports(
    result: dict[str, Any], output_directory: Path
) -> tuple[Path, Path, Path]:
    output_directory.mkdir(parents=True, exist_ok=True)
    json_path = output_directory / "scheduler.json"
    csv_path = output_directory / "scheduler.csv"
    markdown_path = output_directory / "scheduler.md"
    json_path.write_text(
        json.dumps(result, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )
    rows = []
    for workload_result in result["workloads"]:
        for policy_result in workload_result["policies"]:
            aggregate = policy_result["aggregate"]
            rows.append(
                {
                    "suite_id": result["suite_id"],
                    "workload": workload_result["workload"]["name"],
                    "policy": policy_result["policy"],
                    "request_latency_p50_ms": aggregate["request_latency_ms"]["p50"],
                    "request_latency_p95_ms": aggregate["request_latency_ms"]["p95"],
                    "request_latency_p99_ms": aggregate["request_latency_ms"]["p99"],
                    "ttft_p50_ms": aggregate["ttft_ms"]["p50"],
                    "ttft_p95_ms": aggregate["ttft_ms"]["p95"],
                    "ttft_p99_ms": aggregate["ttft_ms"]["p99"],
                    "tpot_p50_ms": aggregate.get("tpot_ms", {}).get("p50"),
                    "tpot_p95_ms": aggregate.get("tpot_ms", {}).get("p95"),
                    "tpot_p99_ms": aggregate.get("tpot_ms", {}).get("p99"),
                    "output_tokens_per_second_mean": aggregate["output_tokens_per_second"][
                        "mean"
                    ],
                    "maximum_observed_batch": aggregate["maximum_observed_batch"],
                    "completed_requests": aggregate["completed_requests"],
                    "cancelled_requests": aggregate["cancelled_requests"],
                    "failed_requests": aggregate["failed_requests"],
                    "missed_cancellations": aggregate["missed_cancellations"],
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
        "# TensorForge Phase 5 continuous-batching report",
        "",
        f"- Suite: `{result['suite_id']}`",
        f"- GPU: {gpu['name']} (compute capability {gpu['compute_capability']})",
        f"- PyTorch / CUDA / Triton: `{hardware['torch_version']}` / "
        f"`{hardware['torch_cuda_runtime']}` / `{hardware['triton_version']}`",
        f"- Git: `{hardware['git_commit']}`; dirty: `{hardware['git_dirty']}`",
        "",
        "All rows use identical model weights, the same transactional PagedKVCache, and the same "
        "Triton GQA decode path. Only admission/refill policy changes. Arrival offsets are seeded "
        "logical scheduler steps, not wall-clock Poisson arrivals.",
        "",
        "| Workload | Policy | Latency P50/P95/P99 ms | TTFT P50/P95/P99 ms | "
        "TPOT P50/P95/P99 ms | Output tok/s | Max batch | Completed/Cancelled/Failed |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for workload_result in result["workloads"]:
        for policy_result in workload_result["policies"]:
            aggregate = policy_result["aggregate"]
            latency = aggregate["request_latency_ms"]
            ttft = aggregate["ttft_ms"]
            tpot = aggregate.get("tpot_ms")
            tpot_text = (
                f"{tpot['p50']:.3f}/{tpot['p95']:.3f}/{tpot['p99']:.3f}"
                if tpot
                else "n/a"
            )
            lines.append(
                f"| {workload_result['workload']['name']} | {policy_result['policy']} | "
                f"{latency['p50']:.3f}/{latency['p95']:.3f}/{latency['p99']:.3f} | "
                f"{ttft['p50']:.3f}/{ttft['p95']:.3f}/{ttft['p99']:.3f} | "
                f"{tpot_text} | {aggregate['output_tokens_per_second']['mean']:.2f} | "
                f"{aggregate['maximum_observed_batch']} | {aggregate['completed_requests']}/"
                f"{aggregate['cancelled_requests']}/{aggregate['failed_requests']} |"
            )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "This is a scheduler-policy ablation, not a comparison with the Phase 1 full-prefix "
            "oracle, vLLM, FlashAttention, or a production prefill kernel. Prefill currently walks "
            "the same one-token paged decode executor. Completed-token throughput excludes work "
            "discarded by cancellation; executed-input throughput remains available in JSON.",
        ]
    )
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, csv_path, markdown_path
