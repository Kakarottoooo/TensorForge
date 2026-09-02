"""Explicit request lifecycle and token-budgeted batching policies."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import cast

import torch

from tensorforge.runtime.batch import (
    BatchExecutionResult,
    BatchPrefillExecutor,
    BatchTokenExecutor,
)


class RequestState(StrEnum):
    QUEUED = "queued"
    PREFILL = "prefill"
    DECODE = "decode"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class BatchingPolicy(StrEnum):
    NO_BATCHING = "no_batching"
    STATIC = "static_batching"
    CONTINUOUS = "continuous_batching"


class PrefillMode(StrEnum):
    TOKEN_DECODE = "token_decode"
    PARALLEL = "parallel"


class TokenBudgetExceededError(RuntimeError):
    """Raised before admission when a request exceeds a declared token budget."""


@dataclass(frozen=True, slots=True)
class RequestInput:
    request_id: str
    prompt_token_ids: tuple[int, ...]
    max_new_tokens: int
    arrival_ns: int | None = None


@dataclass(frozen=True, slots=True)
class SchedulerConfig:
    policy: BatchingPolicy
    max_batch_requests: int
    max_batch_tokens: int
    max_active_token_budget: int
    max_request_tokens: int
    eos_token_id: int | None = None
    prefill_mode: PrefillMode = PrefillMode.TOKEN_DECODE

    def __post_init__(self) -> None:
        dimensions = {
            "max_batch_requests": self.max_batch_requests,
            "max_batch_tokens": self.max_batch_tokens,
            "max_active_token_budget": self.max_active_token_budget,
            "max_request_tokens": self.max_request_tokens,
        }
        for name, value in dimensions.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.max_request_tokens > self.max_active_token_budget:
            raise ValueError("per-request budget cannot exceed the global active budget")


@dataclass(frozen=True, slots=True)
class RequestSnapshot:
    request_id: str
    state: RequestState
    state_history: tuple[RequestState, ...]
    prompt_token_ids: tuple[int, ...]
    generated_token_ids: tuple[int, ...]
    max_new_tokens: int
    arrival_ns: int
    started_ns: int | None
    token_timestamps_ns: tuple[int, ...]
    completed_ns: int | None
    finish_reason: str | None
    error: str | None


@dataclass(slots=True)
class _Request:
    request_id: str
    prompt_token_ids: tuple[int, ...]
    max_new_tokens: int
    arrival_ns: int
    state: RequestState = RequestState.QUEUED
    state_history: list[RequestState] = field(
        default_factory=lambda: [RequestState.QUEUED]
    )
    prompt_cursor: int = 0
    generated_token_ids: list[int] = field(default_factory=list)
    token_timestamps_ns: list[int] = field(default_factory=list)
    next_input_token: int | None = None
    started_ns: int | None = None
    completed_ns: int | None = None
    finish_reason: str | None = None
    error: str | None = None
    cache_created: bool = False

    @property
    def token_budget(self) -> int:
        return len(self.prompt_token_ids) + self.max_new_tokens


_TERMINAL_STATES = {
    RequestState.COMPLETED,
    RequestState.FAILED,
    RequestState.CANCELLED,
}


class ContinuousBatchScheduler:
    """Policy-parametric scheduler with one canonical request lifecycle."""

    def __init__(
        self,
        executor: BatchTokenExecutor,
        config: SchedulerConfig,
        *,
        clock_ns: Callable[[], int] = time.perf_counter_ns,
    ) -> None:
        self.executor = executor
        self.config = config
        self._clock_ns = clock_ns
        self._requests: dict[str, _Request] = {}
        self._queue: list[str] = []
        self._active: list[str] = []
        self._outstanding_token_budget = 0
        self.step_count = 0
        self.executed_token_count = 0
        self.maximum_observed_batch = 0
        if config.prefill_mode is PrefillMode.PARALLEL and not callable(
            getattr(executor, "prefill_prompts", None)
        ):
            raise TypeError("parallel prefill requires a BatchPrefillExecutor")

    def submit(self, request: RequestInput) -> None:
        if not request.request_id:
            raise ValueError("request_id must be non-empty")
        if request.request_id in self._requests:
            raise ValueError(f"duplicate request_id: {request.request_id}")
        if not request.prompt_token_ids:
            raise ValueError("prompt must contain at least one token")
        if request.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        observed_ns = self._clock_ns()
        if request.arrival_ns is not None and not 0 <= request.arrival_ns <= observed_ns:
            raise ValueError("external arrival_ns must be a past monotonic timestamp")
        if any(not 0 <= token < self.executor.vocab_size for token in request.prompt_token_ids):
            raise ValueError("prompt token is outside the executor vocabulary")
        requested_budget = len(request.prompt_token_ids) + request.max_new_tokens
        if (
            self.config.prefill_mode is PrefillMode.PARALLEL
            and len(request.prompt_token_ids) > self.config.max_batch_tokens
        ):
            raise TokenBudgetExceededError("prompt exceeds parallel prefill token budget")
        if requested_budget > self.config.max_request_tokens:
            raise TokenBudgetExceededError("request exceeds per-request token budget")
        if (
            self._outstanding_token_budget + requested_budget
            > self.config.max_active_token_budget
        ):
            raise TokenBudgetExceededError("request exceeds global active token budget")
        internal = _Request(
            request_id=request.request_id,
            prompt_token_ids=request.prompt_token_ids,
            max_new_tokens=request.max_new_tokens,
            arrival_ns=observed_ns if request.arrival_ns is None else request.arrival_ns,
        )
        self._requests[request.request_id] = internal
        self._queue.append(request.request_id)
        self._outstanding_token_budget += requested_budget

    def snapshot(self, request_id: str) -> RequestSnapshot:
        request = self._request(request_id)
        return RequestSnapshot(
            request_id=request.request_id,
            state=request.state,
            state_history=tuple(request.state_history),
            prompt_token_ids=request.prompt_token_ids,
            generated_token_ids=tuple(request.generated_token_ids),
            max_new_tokens=request.max_new_tokens,
            arrival_ns=request.arrival_ns,
            started_ns=request.started_ns,
            token_timestamps_ns=tuple(request.token_timestamps_ns),
            completed_ns=request.completed_ns,
            finish_reason=request.finish_reason,
            error=request.error,
        )

    def snapshots(self) -> tuple[RequestSnapshot, ...]:
        return tuple(self.snapshot(request_id) for request_id in self._requests)

    @property
    def outstanding_token_budget(self) -> int:
        return self._outstanding_token_budget

    @property
    def active_request_count(self) -> int:
        return len(self._active)

    @property
    def queued_request_count(self) -> int:
        return len(self._queue)

    def cancel(self, request_id: str) -> None:
        request = self._request(request_id)
        if request.state in _TERMINAL_STATES:
            raise ValueError("cannot cancel a terminal request")
        self._finish(
            request,
            RequestState.CANCELLED,
            "cancelled",
            None,
            self._clock_ns(),
        )

    def step(self) -> int:
        self._activate_requests()
        selected = self._select_active()
        if not selected:
            return 0
        for request_id in selected:
            self._active.remove(request_id)
        prefill_prompts: dict[str, tuple[int, ...]] = {}
        tokens: dict[str, int] = {}
        for request_id in selected:
            request = self._requests[request_id]
            if (
                request.state is RequestState.PREFILL
                and self.config.prefill_mode is PrefillMode.PARALLEL
            ):
                prefill_prompts[request_id] = request.prompt_token_ids
            elif request.state is RequestState.PREFILL:
                tokens[request_id] = request.prompt_token_ids[request.prompt_cursor]
            else:
                assert request.state is RequestState.DECODE
                assert request.next_input_token is not None
                tokens[request_id] = request.next_input_token

        logits: dict[str, torch.Tensor] = {}
        errors: dict[str, str] = {}
        if prefill_prompts:
            prefill_executor = cast(BatchPrefillExecutor, self.executor)
            prefill_result = prefill_executor.prefill_prompts(prefill_prompts)
            logits.update(prefill_result.logits)
            errors.update(prefill_result.errors)
        if tokens:
            token_result = self.executor.append_tokens(tokens)
            logits.update(token_result.logits)
            errors.update(token_result.errors)
        result = BatchExecutionResult(logits=logits, errors=errors)
        successful_ids = list(result.logits)
        resolved_tokens = (
            dict(
                zip(
                    successful_ids,
                    torch.stack([result.logits[request_id] for request_id in successful_ids])
                    .argmax(dim=-1)
                    .cpu()
                    .tolist(),
                    strict=True,
                )
            )
            if successful_ids
            else {}
        )
        completed_at = self._clock_ns()
        self.step_count += 1
        self.executed_token_count += len(tokens) + sum(map(len, prefill_prompts.values()))
        self.maximum_observed_batch = max(self.maximum_observed_batch, len(selected))
        for request_id in selected:
            request = self._requests[request_id]
            error = result.errors.get(request_id)
            if error is not None:
                self._finish(request, RequestState.FAILED, "error", error, completed_at)
                continue
            request_logits = result.logits.get(request_id)
            if request_logits is None:
                self._finish(
                    request,
                    RequestState.FAILED,
                    "error",
                    "executor returned neither logits nor error",
                    completed_at,
                )
                continue
            if request.state is RequestState.PREFILL:
                if self.config.prefill_mode is PrefillMode.PARALLEL:
                    request.prompt_cursor = len(request.prompt_token_ids)
                else:
                    request.prompt_cursor += 1
                if request.prompt_cursor < len(request.prompt_token_ids):
                    self._active.append(request_id)
                    continue
                self._transition(request, RequestState.DECODE)

            next_token = resolved_tokens[request_id]
            request.generated_token_ids.append(next_token)
            request.token_timestamps_ns.append(completed_at)
            request.next_input_token = next_token
            reached_eos = (
                self.config.eos_token_id is not None
                and next_token == self.config.eos_token_id
            )
            if reached_eos or len(request.generated_token_ids) >= request.max_new_tokens:
                self._finish(
                    request,
                    RequestState.COMPLETED,
                    "eos" if reached_eos else "length",
                    None,
                    completed_at,
                )
            else:
                self._active.append(request_id)
        return len(selected)

    def _select_active(self) -> list[str]:
        selected: list[str] = []
        remaining_tokens = self.config.max_batch_tokens
        for request_id in self._active:
            if len(selected) >= self.config.max_batch_requests:
                break
            request = self._requests[request_id]
            cost = (
                len(request.prompt_token_ids)
                if request.state is RequestState.PREFILL
                and self.config.prefill_mode is PrefillMode.PARALLEL
                else 1
            )
            if cost <= remaining_tokens:
                selected.append(request_id)
                remaining_tokens -= cost
        return selected

    def run_until_complete(self, *, max_steps: int = 1_000_000) -> None:
        for _ in range(max_steps):
            if self.is_finished:
                return
            progressed = self.step()
            if progressed == 0 and not self._queue and not self._active:
                raise RuntimeError("scheduler cannot make progress")
        raise RuntimeError("scheduler exceeded max_steps")

    @property
    def is_finished(self) -> bool:
        return bool(self._requests) and all(
            request.state in _TERMINAL_STATES for request in self._requests.values()
        )

    def _activate_requests(self) -> None:
        if self.config.policy is BatchingPolicy.NO_BATCHING and self._active:
            return
        if self.config.policy is BatchingPolicy.STATIC and self._active:
            return
        capacity = self.config.max_batch_requests - len(self._active)
        if self.config.policy is BatchingPolicy.NO_BATCHING:
            capacity = min(capacity, 1)
        while capacity > 0 and self._queue:
            request_id = self._queue.pop(0)
            request = self._requests[request_id]
            try:
                self.executor.create_request(request_id)
            except Exception as error:
                self._finish(
                    request,
                    RequestState.FAILED,
                    "error",
                    str(error),
                    self._clock_ns(),
                )
                continue
            request.cache_created = True
            request.started_ns = self._clock_ns()
            self._transition(request, RequestState.PREFILL)
            self._active.append(request_id)
            capacity -= 1

    def _transition(self, request: _Request, state: RequestState) -> None:
        if request.state is state:
            return
        request.state = state
        request.state_history.append(state)

    def _finish(
        self,
        request: _Request,
        state: RequestState,
        finish_reason: str,
        error: str | None,
        completed_ns: int,
    ) -> None:
        self._transition(request, state)
        request.finish_reason = finish_reason
        request.error = error
        request.completed_ns = completed_ns
        if request.request_id in self._active:
            self._active.remove(request.request_id)
        if request.request_id in self._queue:
            self._queue.remove(request.request_id)
        if request.cache_created:
            self.executor.release_request(request.request_id)
            request.cache_created = False
        self._outstanding_token_budget -= request.token_budget

    def _request(self, request_id: str) -> _Request:
        try:
            return self._requests[request_id]
        except KeyError as error:
            raise KeyError(f"unknown request: {request_id}") from error
