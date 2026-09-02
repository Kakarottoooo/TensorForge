from __future__ import annotations

import random

import pytest
import torch

from tensorforge.runtime.batch import BatchExecutionResult
from tensorforge.scheduler.continuous import (
    BatchingPolicy,
    ContinuousBatchScheduler,
    RequestInput,
    RequestState,
    SchedulerConfig,
    TokenBudgetExceededError,
)


class DeterministicExecutor:
    vocab_size = 16

    def __init__(self) -> None:
        self.active: set[str] = set()
        self.batch_sizes: list[int] = []

    def create_request(self, request_id: str) -> None:
        self.active.add(request_id)

    def release_request(self, request_id: str) -> None:
        self.active.remove(request_id)

    def append_tokens(self, tokens: dict[str, int]) -> BatchExecutionResult:
        self.batch_sizes.append(len(tokens))
        logits: dict[str, torch.Tensor] = {}
        for request_id in tokens:
            value = torch.zeros(16)
            value[3] = 1
            logits[request_id] = value
        return BatchExecutionResult(logits=logits, errors={})


def test_request_transitions_to_completion_and_releases_cache() -> None:
    executor = DeterministicExecutor()
    scheduler = ContinuousBatchScheduler(
        executor,
        SchedulerConfig(
            policy=BatchingPolicy.CONTINUOUS,
            max_batch_requests=4,
            max_batch_tokens=4,
            max_active_token_budget=64,
            max_request_tokens=16,
        ),
    )
    scheduler.submit(RequestInput("request", (1, 2), max_new_tokens=2))

    scheduler.run_until_complete()

    request = scheduler.snapshot("request")
    assert request.state is RequestState.COMPLETED
    assert request.state_history == (
        RequestState.QUEUED,
        RequestState.PREFILL,
        RequestState.DECODE,
        RequestState.COMPLETED,
    )
    assert request.generated_token_ids == (3, 3)
    assert not executor.active


def test_cancellation_releases_cache_and_token_budget() -> None:
    executor = DeterministicExecutor()
    scheduler = ContinuousBatchScheduler(
        executor,
        SchedulerConfig(
            policy=BatchingPolicy.CONTINUOUS,
            max_batch_requests=1,
            max_batch_tokens=1,
            max_active_token_budget=4,
            max_request_tokens=4,
        ),
    )
    scheduler.submit(RequestInput("cancelled", (1, 2, 3), max_new_tokens=1))
    scheduler.step()
    assert executor.active == {"cancelled"}

    scheduler.cancel("cancelled")
    scheduler.submit(RequestInput("replacement", (4, 5, 6), max_new_tokens=1))

    cancelled = scheduler.snapshot("cancelled")
    assert cancelled.state is RequestState.CANCELLED
    assert cancelled.finish_reason == "cancelled"
    assert not executor.active


def test_global_and_per_request_token_budgets_are_enforced() -> None:
    scheduler = ContinuousBatchScheduler(
        DeterministicExecutor(),
        SchedulerConfig(
            policy=BatchingPolicy.CONTINUOUS,
            max_batch_requests=2,
            max_batch_tokens=2,
            max_active_token_budget=8,
            max_request_tokens=5,
        ),
    )
    scheduler.submit(RequestInput("first", (1, 2, 3), max_new_tokens=2))

    with pytest.raises(TokenBudgetExceededError, match="global"):
        scheduler.submit(RequestInput("global-overflow", (1, 2), max_new_tokens=2))
    with pytest.raises(TokenBudgetExceededError, match="per-request"):
        scheduler.submit(RequestInput("request-overflow", (1, 2, 3), max_new_tokens=3))

    scheduler.cancel("first")
    scheduler.submit(RequestInput("replacement", (1, 2), max_new_tokens=2))


class PartiallyFailingExecutor(DeterministicExecutor):
    def append_tokens(self, tokens: dict[str, int]) -> BatchExecutionResult:
        result = super().append_tokens(tokens)
        result.logits.pop("bad", None)
        errors = {"bad": "injected failure"} if "bad" in tokens else {}
        return BatchExecutionResult(logits=result.logits, errors=errors)


def test_one_request_failure_does_not_poison_batch_peer() -> None:
    executor = PartiallyFailingExecutor()
    scheduler = ContinuousBatchScheduler(
        executor,
        SchedulerConfig(
            policy=BatchingPolicy.CONTINUOUS,
            max_batch_requests=2,
            max_batch_tokens=2,
            max_active_token_budget=8,
            max_request_tokens=4,
        ),
    )
    scheduler.submit(RequestInput("good", (1,), max_new_tokens=1))
    scheduler.submit(RequestInput("bad", (2,), max_new_tokens=1))

    scheduler.run_until_complete()

    assert scheduler.snapshot("good").state is RequestState.COMPLETED
    assert scheduler.snapshot("bad").state is RequestState.FAILED
    assert scheduler.snapshot("bad").error == "injected failure"
    assert not executor.active


@pytest.mark.parametrize(
    ("policy", "expected_batch_sizes"),
    [
        (BatchingPolicy.NO_BATCHING, [1, 1, 1, 1, 1]),
        (BatchingPolicy.STATIC, [2, 1, 1, 1]),
        (BatchingPolicy.CONTINUOUS, [2, 2, 1]),
    ],
)
def test_batching_policies_have_distinct_refill_behavior(
    policy: BatchingPolicy, expected_batch_sizes: list[int]
) -> None:
    executor = DeterministicExecutor()
    scheduler = ContinuousBatchScheduler(
        executor,
        SchedulerConfig(
            policy=policy,
            max_batch_requests=2,
            max_batch_tokens=2,
            max_active_token_budget=16,
            max_request_tokens=8,
        ),
    )
    scheduler.submit(RequestInput("short-a", (1,), max_new_tokens=1))
    scheduler.submit(RequestInput("long", (2,), max_new_tokens=3))
    scheduler.submit(RequestInput("short-b", (3,), max_new_tokens=1))

    scheduler.run_until_complete()

    assert executor.batch_sizes == expected_batch_sizes
    assert scheduler.maximum_observed_batch == max(expected_batch_sizes)


def test_seeded_arrival_cancellation_and_completion_churn() -> None:
    randomizer = random.Random(20260902)
    executor = DeterministicExecutor()
    scheduler = ContinuousBatchScheduler(
        executor,
        SchedulerConfig(
            policy=BatchingPolicy.CONTINUOUS,
            max_batch_requests=4,
            max_batch_tokens=4,
            max_active_token_budget=64,
            max_request_tokens=8,
        ),
    )
    next_request = 0

    for _ in range(500):
        action = randomizer.random()
        if action < 0.45:
            prompt_length = randomizer.randint(1, 4)
            generation_length = randomizer.randint(1, 4)
            request = RequestInput(
                f"request-{next_request}",
                tuple(randomizer.randrange(executor.vocab_size) for _ in range(prompt_length)),
                max_new_tokens=generation_length,
            )
            try:
                scheduler.submit(request)
            except TokenBudgetExceededError:
                pass
            else:
                next_request += 1
        elif action < 0.65:
            cancellable = [
                request.request_id
                for request in scheduler.snapshots()
                if request.state not in {
                    RequestState.COMPLETED,
                    RequestState.FAILED,
                    RequestState.CANCELLED,
                }
            ]
            if cancellable:
                scheduler.cancel(randomizer.choice(cancellable))
        else:
            scheduler.step()

        assert scheduler.outstanding_token_budget <= 64
        assert scheduler.maximum_observed_batch <= 4

    scheduler.run_until_complete()
    assert scheduler.outstanding_token_budget == 0
    assert scheduler.active_request_count == 0
    assert scheduler.queued_request_count == 0
    assert not executor.active
