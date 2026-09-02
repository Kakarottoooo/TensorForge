"""Real-checkpoint backends with one explicit end-to-end timing contract."""

from __future__ import annotations

import hashlib
import json
import time
from contextlib import suppress
from dataclasses import dataclass, field
from importlib import metadata
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from tensorforge.benchmark.backends import BenchmarkBackend
from tensorforge.benchmark.schema import (
    BackendCapabilities,
    BackendRun,
    RequestSpec,
    RequestTrace,
    WorkloadSpec,
)
from tensorforge.cache.paged import PagedKVCache, PagedKVCacheConfig
from tensorforge.model.huggingface import load_huggingface_checkpoint
from tensorforge.model.llama import LlamaForCausalLM
from tensorforge.runtime.decode_bucket import (
    BucketedPagedDecodeExecutor,
    DecodeExecutionMode,
    DecodeFusionLevel,
)
from tensorforge.scheduler.continuous import (
    BatchingPolicy,
    ContinuousBatchScheduler,
    PrefillMode,
    RequestInput,
    RequestSnapshot,
    RequestState,
    SchedulerConfig,
)


def _package_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "unavailable"


def generated_token_digest(outputs: dict[str, tuple[int, ...]]) -> str:
    """Return an order-independent digest for cross-backend greedy-token checks."""

    normalized = {request_id: list(outputs[request_id]) for request_id in sorted(outputs)}
    payload = json.dumps(normalized, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def _trace(snapshot: RequestSnapshot) -> RequestTrace:
    if snapshot.started_ns is None or snapshot.completed_ns is None:
        raise RuntimeError(f"request {snapshot.request_id} has incomplete timing state")
    if snapshot.finish_reason is None:
        raise RuntimeError(f"request {snapshot.request_id} has no finish reason")
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


@dataclass(slots=True)
class TensorForgeCheckpointBackend(BenchmarkBackend):
    """Parallel-prefill + paged continuous-decode backend for a real checkpoint."""

    workload: WorkloadSpec
    device: torch.device
    checkpoint_dir: Path
    block_size: int
    num_blocks: int
    mode: DecodeExecutionMode
    fusion_level: DecodeFusionLevel
    batch_buckets: tuple[int, ...]
    context_buckets: tuple[int, ...]
    model: LlamaForCausalLM = field(init=False)

    def __post_init__(self) -> None:
        if self.device.type != "cuda":
            raise ValueError("TensorForge checkpoint benchmark requires CUDA")
        dtype = {
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
            "fp32": torch.float32,
        }[self.workload.precision.value]
        self.model = load_huggingface_checkpoint(
            self.checkpoint_dir, device=self.device, dtype=dtype
        )
        if self.model.config != self.workload.model.to_model_config():
            raise ValueError("workload model dimensions do not match checkpoint config")

    @property
    def name(self) -> str:
        return "tensorforge"

    @property
    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            variable_prompt_lengths=True,
            asynchronous_arrivals=True,
            continuous_batching=True,
            paged_kv_cache=True,
            cuda_graphs=self.mode is DecodeExecutionMode.CUDA_GRAPH,
        )

    @torch.inference_mode()
    def run(self, requests: tuple[RequestSpec, ...]) -> BackendRun:
        if not requests:
            raise ValueError("at least one request is required")
        config = self.model.config
        cache = PagedKVCache(
            PagedKVCacheConfig(
                num_layers=config.num_hidden_layers,
                num_blocks=self.num_blocks,
                block_size=self.block_size,
                num_kv_heads=config.num_key_value_heads,
                head_dim=config.head_dim,
                max_sequences=len(requests),
                max_sequence_length=config.max_position_embeddings,
                dtype=next(self.model.parameters()).dtype,
                device=self.device,
            )
        )
        executor = BucketedPagedDecodeExecutor(
            model=self.model,
            cache=cache,
            mode=self.mode,
            fusion_level=self.fusion_level,
            batch_buckets=tuple(
                capacity
                for capacity in self.batch_buckets
                if capacity <= len(requests)
            ),
            context_buckets=self.context_buckets,
        )
        maximum_request_tokens = max(
            len(request.prompt_token_ids) + request.max_new_tokens
            for request in requests
        )
        scheduler = ContinuousBatchScheduler(
            executor,
            SchedulerConfig(
                policy=BatchingPolicy.CONTINUOUS,
                prefill_mode=PrefillMode.PARALLEL,
                max_batch_requests=len(requests),
                max_batch_tokens=sum(len(request.prompt_token_ids) for request in requests),
                max_active_token_budget=sum(
                    len(request.prompt_token_ids) + request.max_new_tokens
                    for request in requests
                ),
                max_request_tokens=maximum_request_tokens,
            ),
        )
        pending = sorted(requests, key=lambda request: request.arrival_offset_ns)
        torch.cuda.synchronize(self.device)
        benchmark_start_ns = time.perf_counter_ns()
        while pending or any(
            snapshot.state
            not in {RequestState.COMPLETED, RequestState.FAILED, RequestState.CANCELLED}
            for snapshot in scheduler.snapshots()
        ):
            now_ns = time.perf_counter_ns()
            while pending and pending[0].arrival_offset_ns <= now_ns - benchmark_start_ns:
                request = pending.pop(0)
                scheduler.submit(
                    RequestInput(
                        request_id=request.request_id,
                        prompt_token_ids=request.prompt_token_ids,
                        max_new_tokens=request.max_new_tokens,
                        arrival_ns=benchmark_start_ns + request.arrival_offset_ns,
                    )
                )
            has_work = any(
                snapshot.state
                not in {RequestState.COMPLETED, RequestState.FAILED, RequestState.CANCELLED}
                for snapshot in scheduler.snapshots()
            )
            if has_work:
                scheduler.step()
            elif pending:
                sleep_ns = (
                    benchmark_start_ns
                    + pending[0].arrival_offset_ns
                    - time.perf_counter_ns()
                )
                if sleep_ns > 0:
                    time.sleep(sleep_ns / 1e9)
        torch.cuda.synchronize(self.device)

        snapshots = scheduler.snapshots()
        failed = [snapshot for snapshot in snapshots if snapshot.state is RequestState.FAILED]
        if failed:
            details = "; ".join(f"{item.request_id}: {item.error}" for item in failed)
            raise RuntimeError(f"TensorForge requests failed: {details}")
        outputs = {
            snapshot.request_id: snapshot.generated_token_ids for snapshot in snapshots
        }
        stats = cache.stats()
        if stats.active_sequences or stats.used_blocks or scheduler.outstanding_token_budget:
            raise RuntimeError("TensorForge backend leaked cache or scheduler state")
        metrics = executor.metrics()
        return BackendRun(
            traces=tuple(_trace(snapshot) for snapshot in snapshots),
            cuda_elapsed_ms=None,
            counters={
                "generated_token_sha256": generated_token_digest(outputs),
                "executed_input_tokens": scheduler.executed_token_count,
                "scheduler_steps": scheduler.step_count,
                "maximum_observed_batch": scheduler.maximum_observed_batch,
                "graph_hits": metrics.graph_hits,
                "graph_misses": metrics.graph_misses,
                "shape_fallbacks": metrics.shape_fallbacks,
                "capture_count": metrics.capture_count,
                "capture_time_ms": metrics.capture_time_ms,
                "transformers_version": _package_version("transformers"),
            },
        )

    def close(self) -> None:
        del self.model


@dataclass(slots=True)
class TransformersCheckpointBackend(BenchmarkBackend):
    """Hugging Face SDPA backend for homogeneous, simultaneous offline batches."""

    workload: WorkloadSpec
    device: torch.device
    checkpoint_dir: Path
    model: Any = field(init=False)

    def __post_init__(self) -> None:
        try:
            from transformers import AutoModelForCausalLM  # type: ignore[import-untyped]
        except ImportError as error:
            raise RuntimeError("install TensorForge with the models extra") from error
        dtype = {
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
            "fp32": torch.float32,
        }[self.workload.precision.value]
        self.model = AutoModelForCausalLM.from_pretrained(
            self.checkpoint_dir,
            local_files_only=True,
            torch_dtype=dtype,
            attn_implementation="sdpa",
        ).to(self.device)
        self.model.eval()

    @property
    def name(self) -> str:
        return "transformers_sdpa"

    @property
    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities()

    @torch.inference_mode()
    def run(self, requests: tuple[RequestSpec, ...]) -> BackendRun:
        if not requests:
            raise ValueError("at least one request is required")
        if {request.arrival_offset_ns for request in requests} != {0}:
            raise ValueError("Transformers comparison supports simultaneous arrivals only")
        if len({len(request.prompt_token_ids) for request in requests}) != 1:
            raise ValueError("Transformers comparison requires equal prompt lengths")
        if len({request.max_new_tokens for request in requests}) != 1:
            raise ValueError("Transformers comparison requires equal output lengths")

        input_ids = torch.tensor(
            [request.prompt_token_ids for request in requests],
            dtype=torch.long,
            device=self.device,
        )
        generated: list[list[int]] = [[] for _ in requests]
        token_timestamps: list[int] = []
        torch.cuda.synchronize(self.device)
        started_ns = time.perf_counter_ns()
        event_factory: Any = torch.cuda.Event
        start_event = event_factory(enable_timing=True)
        end_event = event_factory(enable_timing=True)
        start_event.record()

        output = self.model(input_ids=input_ids, use_cache=True)
        next_tokens: Tensor = output.logits[:, -1].argmax(dim=-1)
        for row, token in zip(generated, next_tokens.tolist(), strict=True):
            row.append(token)
        torch.cuda.synchronize(self.device)
        token_timestamps.append(time.perf_counter_ns())
        for _ in range(requests[0].max_new_tokens - 1):
            output = self.model(
                input_ids=next_tokens[:, None],
                past_key_values=output.past_key_values,
                use_cache=True,
            )
            next_tokens = output.logits[:, -1].argmax(dim=-1)
            for row, token in zip(generated, next_tokens.tolist(), strict=True):
                row.append(token)
            torch.cuda.synchronize(self.device)
            token_timestamps.append(time.perf_counter_ns())
        end_event.record()
        end_event.synchronize()
        completed_ns = time.perf_counter_ns()
        outputs = {
            request.request_id: tuple(generated[index])
            for index, request in enumerate(requests)
        }
        return BackendRun(
            traces=tuple(
                RequestTrace(
                    request_id=request.request_id,
                    arrival_ns=started_ns,
                    started_ns=started_ns,
                    token_timestamps_ns=tuple(token_timestamps),
                    completed_ns=completed_ns,
                    prompt_tokens=len(request.prompt_token_ids),
                    generated_tokens=len(generated[index]),
                    finish_reason="length",
                )
                for index, request in enumerate(requests)
            ),
            cuda_elapsed_ms=float(start_event.elapsed_time(end_event)),
            counters={
                "generated_token_sha256": generated_token_digest(outputs),
                "forward_calls": requests[0].max_new_tokens,
                "attention_implementation": "sdpa",
                "transformers_version": _package_version("transformers"),
            },
        )

    def close(self) -> None:
        del self.model


@dataclass(slots=True)
class VllmCheckpointEngine:
    """One reusable vLLM engine shared across comparison cases."""

    checkpoint_dir: Path
    precision: str
    max_model_len: int
    max_num_seqs: int
    gpu_memory_utilization: float = 0.70
    llm: Any = field(init=False)
    sampling_params_type: Any = field(init=False)

    def __post_init__(self) -> None:
        try:
            from vllm import LLM, SamplingParams  # type: ignore[import-not-found]
        except ImportError as error:
            raise RuntimeError("vLLM must be installed in an isolated environment") from error
        self.sampling_params_type = SamplingParams
        self.llm = LLM(
            model=str(self.checkpoint_dir),
            tokenizer=str(self.checkpoint_dir),
            dtype={"fp16": "float16", "bf16": "bfloat16", "fp32": "float32"}[
                self.precision
            ],
            max_model_len=self.max_model_len,
            max_num_seqs=self.max_num_seqs,
            gpu_memory_utilization=self.gpu_memory_utilization,
            disable_log_stats=True,
            enforce_eager=False,
        )

    def close(self) -> None:
        del self.llm
        with suppress(ImportError):
            from vllm.distributed.parallel_state import (  # type: ignore[import-not-found]
                destroy_distributed_environment,
                destroy_model_parallel,
            )

            destroy_model_parallel()
            destroy_distributed_environment()


@dataclass(slots=True)
class VllmCheckpointBackend(BenchmarkBackend):
    """vLLM offline reference; exact TTFT/finish metrics, derived intermediate timestamps."""

    workload: WorkloadSpec
    engine: VllmCheckpointEngine

    @property
    def name(self) -> str:
        return "vllm"

    @property
    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            variable_prompt_lengths=True,
            continuous_batching=True,
            paged_kv_cache=True,
            cuda_graphs=True,
        )

    def run(self, requests: tuple[RequestSpec, ...]) -> BackendRun:
        if not requests:
            raise ValueError("at least one request is required")
        if {request.arrival_offset_ns for request in requests} != {0}:
            raise ValueError("vLLM offline comparison supports simultaneous arrivals only")
        if len({request.max_new_tokens for request in requests}) != 1:
            raise ValueError("vLLM comparison requires equal output lengths")
        parameters = self.engine.sampling_params_type(
            temperature=0.0,
            max_tokens=requests[0].max_new_tokens,
            ignore_eos=True,
            detokenize=False,
        )
        call_started_ns = time.perf_counter_ns()
        outputs = self.engine.llm.generate(
            prompts=None,
            prompt_token_ids=[list(request.prompt_token_ids) for request in requests],
            sampling_params=parameters,
            use_tqdm=False,
        )
        call_completed_ns = time.perf_counter_ns()
        generated = {
            request.request_id: tuple(outputs[index].outputs[0].token_ids)
            for index, request in enumerate(requests)
        }
        traces = []
        for index, request in enumerate(requests):
            metrics = outputs[index].metrics
            if (
                metrics is None
                or metrics.first_scheduled_time is None
                or metrics.first_token_time is None
                or metrics.finished_time is None
            ):
                raise RuntimeError("vLLM did not return complete per-request metrics")
            arrival_ns = call_started_ns + int(
                (metrics.arrival_time - outputs[0].metrics.arrival_time) * 1e9
            )
            started_ns = arrival_ns + int(
                (metrics.first_scheduled_time - metrics.arrival_time) * 1e9
            )
            first_token_ns = arrival_ns + int(
                (metrics.first_token_time - metrics.arrival_time) * 1e9
            )
            completed_ns = arrival_ns + int(
                (metrics.finished_time - metrics.arrival_time) * 1e9
            )
            token_count = len(generated[request.request_id])
            token_timestamps: tuple[int, ...]
            if token_count == 1:
                token_timestamps = (completed_ns,)
            else:
                decode_span = completed_ns - first_token_ns
                token_timestamps = tuple(
                    first_token_ns + round(decode_span * token / (token_count - 1))
                    for token in range(token_count)
                )
            traces.append(
                RequestTrace(
                    request_id=request.request_id,
                    arrival_ns=arrival_ns,
                    started_ns=started_ns,
                    token_timestamps_ns=token_timestamps,
                    completed_ns=completed_ns,
                    prompt_tokens=len(request.prompt_token_ids),
                    generated_tokens=token_count,
                    finish_reason="length",
                )
            )
        return BackendRun(
            traces=tuple(traces),
            cuda_elapsed_ms=None,
            counters={
                "generated_token_sha256": generated_token_digest(generated),
                "vllm_version": _package_version("vllm"),
                "call_wall_ms": (call_completed_ns - call_started_ns) / 1e6,
                "intermediate_token_timestamps": "linear interpolation of TTFT/finish",
                "engine_mode": "vLLM default with CUDA Graphs",
            },
        )

    def close(self) -> None:
        # The suite owns the shared engine lifetime.
        return
