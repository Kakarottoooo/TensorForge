"""Batch-token execution contract shared by runtimes and schedulers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from torch import Tensor


@dataclass(frozen=True, slots=True)
class BatchExecutionResult:
    logits: dict[str, Tensor]
    errors: dict[str, str]


class BatchTokenExecutor(Protocol):
    @property
    def vocab_size(self) -> int: ...

    def create_request(self, request_id: str) -> None: ...

    def release_request(self, request_id: str) -> None: ...

    def append_tokens(self, tokens: dict[str, int]) -> BatchExecutionResult: ...
