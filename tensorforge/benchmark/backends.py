"""Execution backend protocol and the correctness-equivalent eager baseline."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Protocol

import torch
from torch import Tensor

from tensorforge.benchmark.schema import (
    BackendCapabilities,
    BackendRun,
    RequestSpec,
    RequestTrace,
    WorkloadSpec,
)
from tensorforge.model.llama import LlamaForCausalLM


class BenchmarkBackend(Protocol):
    """Minimal boundary implemented by eager, compiled, graph, and external runtimes."""

    @property
    def name(self) -> str: ...

    @property
    def capabilities(self) -> BackendCapabilities: ...

    def run(self, requests: tuple[RequestSpec, ...]) -> BackendRun: ...

    def close(self) -> None: ...


@dataclass(slots=True)
class EagerBaselineBackend:
    """Synchronous PyTorch baseline that recomputes the full prefix per token."""

    workload: WorkloadSpec
    device: torch.device
    model: LlamaForCausalLM

    @classmethod
    def create(cls, workload: WorkloadSpec, device: torch.device) -> EagerBaselineBackend:
        dtype = {
            "fp32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }[workload.precision.value]
        torch.manual_seed(workload.seed)
        model = LlamaForCausalLM(workload.model.to_model_config()).to(device=device, dtype=dtype)
        return cls(workload=workload, device=device, model=model.eval())

    @property
    def name(self) -> str:
        return "pytorch_reference"

    @property
    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities()

    def _validate_requests(self, requests: tuple[RequestSpec, ...]) -> None:
        if not requests:
            raise ValueError("at least one request is required")
        prompt_lengths = {len(request.prompt_token_ids) for request in requests}
        generation_lengths = {request.max_new_tokens for request in requests}
        arrival_offsets = {request.arrival_offset_ns for request in requests}
        if len(prompt_lengths) != 1 or len(generation_lengths) != 1:
            raise ValueError("eager baseline requires homogeneous prompt and generation lengths")
        if arrival_offsets != {0}:
            raise ValueError("eager baseline does not support asynchronous arrivals")

    @torch.inference_mode()
    def run(self, requests: tuple[RequestSpec, ...]) -> BackendRun:
        self._validate_requests(requests)
        prompts = torch.tensor(
            [request.prompt_token_ids for request in requests],
            device=self.device,
            dtype=torch.long,
        )
        generated = prompts
        token_timestamps: list[int] = []
        started_ns = time.perf_counter_ns()

        start_event: Any | None = None
        end_event: Any | None = None
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            started_ns = time.perf_counter_ns()
            event_factory: Any = torch.cuda.Event
            start_event = event_factory(enable_timing=True)
            end_event = event_factory(enable_timing=True)
            start_event.record()

        decode_steps = requests[0].max_new_tokens
        for _ in range(decode_steps):
            logits = self.model(generated)
            next_tokens: Tensor = logits[:, -1, :].argmax(dim=-1)
            generated = torch.cat((generated, next_tokens[:, None]), dim=1)
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            token_timestamps.append(time.perf_counter_ns())

        cuda_elapsed_ms: float | None = None
        if end_event is not None and start_event is not None:
            end_event.record()
            end_event.synchronize()
            cuda_elapsed_ms = start_event.elapsed_time(end_event)
        completed_ns = time.perf_counter_ns()

        traces = tuple(
            RequestTrace(
                request_id=request.request_id,
                arrival_ns=started_ns + request.arrival_offset_ns,
                started_ns=started_ns,
                token_timestamps_ns=tuple(token_timestamps),
                completed_ns=completed_ns,
                prompt_tokens=len(request.prompt_token_ids),
                generated_tokens=len(token_timestamps),
                finish_reason="length",
            )
            for request in requests
        )
        return BackendRun(
            traces=traces,
            cuda_elapsed_ms=cuda_elapsed_ms,
            counters={
                "model_forward_calls": decode_steps,
                "tokens_recomputed": sum(
                    len(request.prompt_token_ids) * decode_steps
                    + (decode_steps * (decode_steps - 1)) // 2
                    for request in requests
                ),
                "synchronizations": decode_steps + (2 if self.device.type == "cuda" else 0),
            },
        )

    def close(self) -> None:
        del self.model
