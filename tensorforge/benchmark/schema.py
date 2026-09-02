"""Versioned experiment contracts shared by every execution backend."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any

from tensorforge.model.config import ModelConfig

SCHEMA_VERSION = "1.0"


class Precision(StrEnum):
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"


class ExecutionMode(StrEnum):
    EAGER = "eager"
    COMPILE = "torch_compile"
    CUDA_GRAPH = "cuda_graph"
    EXTERNAL = "external"


class SchedulerPolicy(StrEnum):
    NONE = "none"
    STATIC = "static"
    CONTINUOUS = "continuous"


class CachePolicy(StrEnum):
    NONE = "none"
    CONTIGUOUS = "contiguous"
    PAGED = "paged"


class DecodeStrategy(StrEnum):
    STANDARD = "standard"
    SPECULATIVE = "speculative"


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """Serializable model dimensions; weights are deterministically initialized."""

    vocab_size: int = 32_000
    hidden_size: int = 512
    intermediate_size: int = 1_376
    num_hidden_layers: int = 8
    num_attention_heads: int = 8
    num_key_value_heads: int = 4
    max_position_embeddings: int = 4_096
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10_000.0
    tie_word_embeddings: bool = False

    def __post_init__(self) -> None:
        self.to_model_config()

    def to_model_config(self) -> ModelConfig:
        return ModelConfig(**asdict(self))

    @classmethod
    def from_model_config(cls, config: ModelConfig) -> ModelSpec:
        return cls(**asdict(config))


@dataclass(frozen=True, slots=True)
class ExecutionSpec:
    """One-factor-at-a-time execution identity for future ablation rows."""

    backend: str = "pytorch_reference"
    mode: ExecutionMode = ExecutionMode.EAGER
    attention: str = "explicit_pytorch"
    rms_norm: str = "pytorch_reference"
    swiglu: str = "pytorch_reference"
    cache: CachePolicy = CachePolicy.NONE
    scheduler: SchedulerPolicy = SchedulerPolicy.NONE
    decode: DecodeStrategy = DecodeStrategy.STANDARD
    tensor_parallel_size: int = 1
    draft_model: str | None = None
    extra_tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.tensor_parallel_size <= 0:
            raise ValueError("tensor_parallel_size must be positive")
        if self.decode == DecodeStrategy.SPECULATIVE and self.draft_model is None:
            raise ValueError("speculative decoding requires draft_model identity")


@dataclass(frozen=True, slots=True)
class WorkloadSpec:
    """Immutable requested workload, including measurement policy."""

    name: str
    model: ModelSpec
    execution: ExecutionSpec = field(default_factory=ExecutionSpec)
    prompt_length: int = 128
    generation_length: int = 32
    batch_size: int = 1
    concurrency: int = 1
    precision: Precision = Precision.FP16
    seed: int = 7
    warmup_repetitions: int = 3
    measured_repetitions: int = 10
    arrival_offsets_ms: tuple[float, ...] = ()
    gpu_util_sample_ms: int = 100
    allow_batch_size_reduction: bool = True

    def __post_init__(self) -> None:
        dimensions = {
            "prompt_length": self.prompt_length,
            "generation_length": self.generation_length,
            "batch_size": self.batch_size,
            "concurrency": self.concurrency,
            "measured_repetitions": self.measured_repetitions,
        }
        for name, value in dimensions.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if self.warmup_repetitions < 0:
            raise ValueError("warmup_repetitions must be non-negative")
        if self.prompt_length + self.generation_length > self.model.max_position_embeddings:
            raise ValueError("prompt_length + generation_length exceeds model context capacity")
        if self.concurrency < self.batch_size:
            raise ValueError("concurrency cannot be smaller than batch_size")
        if self.arrival_offsets_ms and len(self.arrival_offsets_ms) != self.concurrency:
            raise ValueError("arrival_offsets_ms must have exactly concurrency entries")
        if any(offset < 0 for offset in self.arrival_offsets_ms):
            raise ValueError("arrival offsets must be non-negative")
        if self.gpu_util_sample_ms < 50:
            raise ValueError("gpu_util_sample_ms must be at least 50")

    @property
    def case_id(self) -> str:
        encoded = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(encoded.encode()).hexdigest()[:16]


@dataclass(frozen=True, slots=True)
class RequestSpec:
    request_id: str
    prompt_token_ids: tuple[int, ...]
    max_new_tokens: int
    arrival_offset_ns: int = 0


@dataclass(frozen=True, slots=True)
class RequestTrace:
    """Host-observed request lifecycle.

    Token timestamps are completion timestamps. A speculative step may append
    several timestamps, while continuous batching may interleave requests.
    """

    request_id: str
    arrival_ns: int
    started_ns: int
    token_timestamps_ns: tuple[int, ...]
    completed_ns: int
    prompt_tokens: int
    generated_tokens: int
    finish_reason: str

    def __post_init__(self) -> None:
        if self.prompt_tokens <= 0 or self.generated_tokens < 0:
            raise ValueError("token counts must be non-negative and prompts must be non-empty")
        if self.started_ns < self.arrival_ns:
            raise ValueError("request started before arrival")
        if self.completed_ns < self.started_ns:
            raise ValueError("request completed before start")
        if self.generated_tokens and self.completed_ns <= self.arrival_ns:
            raise ValueError("completed request lifetime must be positive")
        if len(self.token_timestamps_ns) != self.generated_tokens:
            raise ValueError("one completion timestamp is required per generated token")
        previous = self.started_ns
        for timestamp in self.token_timestamps_ns:
            if timestamp < previous or timestamp > self.completed_ns:
                raise ValueError("token timestamps must be monotonic and within request lifetime")
            previous = timestamp


@dataclass(frozen=True, slots=True)
class BackendCapabilities:
    variable_prompt_lengths: bool = False
    asynchronous_arrivals: bool = False
    continuous_batching: bool = False
    paged_kv_cache: bool = False
    cuda_graphs: bool = False
    speculative_decoding: bool = False
    tensor_parallelism: bool = False


@dataclass(frozen=True, slots=True)
class BackendRun:
    traces: tuple[RequestTrace, ...]
    cuda_elapsed_ms: float | None
    counters: dict[str, int | float | str]


@dataclass(frozen=True, slots=True)
class UtilizationSummary:
    source: str | None
    sample_count: int
    gpu_utilization_mean_pct: float | None
    gpu_utilization_max_pct: float | None
    device_memory_used_max_mib: float | None
    unavailable_reason: str | None = None


@dataclass(frozen=True, slots=True)
class RunSample:
    repetition: int
    traces: tuple[RequestTrace, ...]
    cuda_elapsed_ms: float | None
    memory_allocated_before_bytes: int | None
    memory_allocated_after_bytes: int | None
    peak_memory_allocated_bytes: int | None
    utilization: UtilizationSummary
    counters: dict[str, int | float | str]


@dataclass(frozen=True, slots=True)
class CaseResult:
    case_id: str
    requested: WorkloadSpec
    executed_batch_size: int | None
    status: str
    adjustment_reason: str | None
    samples: tuple[RunSample, ...]
    aggregate: dict[str, Any]
    error: str | None = None


@dataclass(frozen=True, slots=True)
class BenchmarkSuiteResult:
    schema_version: str
    suite_id: str
    started_at_utc: str
    finished_at_utc: str
    hardware: dict[str, Any]
    cases: tuple[CaseResult, ...]
