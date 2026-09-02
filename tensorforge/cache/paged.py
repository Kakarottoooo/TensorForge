"""Transactional logical-to-physical paged KV-cache ownership."""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field

import torch
from torch import Tensor


class CacheExhaustedError(RuntimeError):
    """Raised when an append cannot reserve all required physical blocks."""


@dataclass(frozen=True, slots=True)
class CacheStats:
    total_blocks: int
    used_blocks: int
    free_blocks: int
    active_sequences: int
    committed_tokens: int
    reserved_tokens: int


@dataclass(frozen=True, slots=True)
class PagedKVCacheConfig:
    num_layers: int
    num_blocks: int
    block_size: int
    num_kv_heads: int
    head_dim: int
    max_sequences: int
    max_sequence_length: int
    dtype: torch.dtype
    device: torch.device

    def __post_init__(self) -> None:
        dimensions = {
            "num_layers": self.num_layers,
            "num_blocks": self.num_blocks,
            "block_size": self.block_size,
            "num_kv_heads": self.num_kv_heads,
            "head_dim": self.head_dim,
            "max_sequences": self.max_sequences,
            "max_sequence_length": self.max_sequence_length,
        }
        for name, value in dimensions.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.dtype not in {torch.float16, torch.bfloat16, torch.float32}:
            raise TypeError(f"unsupported cache dtype: {self.dtype}")
        if str(self.device) == "cuda":
            object.__setattr__(
                self, "device", torch.device("cuda", torch.cuda.current_device())
            )

    @property
    def max_blocks_per_sequence(self) -> int:
        return (self.max_sequence_length + self.block_size - 1) // self.block_size


@dataclass(frozen=True, slots=True)
class AppendReservation:
    request_id: str
    sequence_slot: int
    start_position: int
    token_count: int
    reservation_id: int

    @property
    def end_position(self) -> int:
        return self.start_position + self.token_count


@dataclass(frozen=True, slots=True)
class PagedLayerView:
    key_cache: Tensor
    value_cache: Tensor
    block_table: Tensor
    context_length: int


@dataclass(frozen=True, slots=True)
class CacheAppendLocation:
    physical_block: int
    block_offset: int


@dataclass(frozen=True, slots=True)
class SequenceLayout:
    physical_blocks: tuple[int, ...]
    context_length: int


@dataclass(slots=True)
class _SequenceState:
    slot: int
    length: int = 0
    blocks: list[int] = field(default_factory=list)
    active_reservation: int | None = None


@dataclass(slots=True)
class _ReservationState:
    public: AppendReservation
    newly_allocated_blocks: list[int]
    written_layers: set[int] = field(default_factory=set)


class PagedKVCache:
    """Own physical KV pages and expose transactional per-request append operations."""

    def __init__(self, config: PagedKVCacheConfig) -> None:
        self.config = config
        shape = (
            config.num_layers,
            config.num_blocks,
            config.block_size,
            config.num_kv_heads,
            config.head_dim,
        )
        self.key_cache = torch.empty(shape, dtype=config.dtype, device=config.device)
        self.value_cache = torch.empty_like(self.key_cache)
        self.block_tables = torch.full(
            (config.max_sequences, config.max_blocks_per_sequence),
            -1,
            dtype=torch.int32,
            device=config.device,
        )
        self.sequence_lengths = torch.zeros(
            config.max_sequences, dtype=torch.int32, device=config.device
        )
        self._free_blocks = list(range(config.num_blocks))
        self._free_slots = list(range(config.max_sequences))
        heapq.heapify(self._free_blocks)
        heapq.heapify(self._free_slots)
        self._block_owners: list[str | None] = [None] * config.num_blocks
        self._sequences: dict[str, _SequenceState] = {}
        self._reservations: dict[int, _ReservationState] = {}
        self._next_reservation_id = 0

    def create_sequence(self, request_id: str) -> None:
        if not request_id:
            raise ValueError("request_id must be non-empty")
        if request_id in self._sequences:
            raise ValueError(f"sequence already exists: {request_id}")
        if not self._free_slots:
            raise CacheExhaustedError("no free sequence slots")
        self._sequences[request_id] = _SequenceState(slot=heapq.heappop(self._free_slots))

    def begin_append(self, request_id: str, token_count: int) -> AppendReservation:
        sequence = self._sequence(request_id)
        if token_count <= 0:
            raise ValueError("token_count must be positive")
        if sequence.active_reservation is not None:
            raise RuntimeError("sequence already has an active append")
        end_position = sequence.length + token_count
        if end_position > self.config.max_sequence_length:
            raise CacheExhaustedError("append exceeds maximum sequence length")
        required_blocks = (end_position + self.config.block_size - 1) // self.config.block_size
        additional = required_blocks - len(sequence.blocks)
        if additional > len(self._free_blocks):
            raise CacheExhaustedError(
                f"KV cache needs {additional} blocks but only {len(self._free_blocks)} are free"
            )

        allocated = [heapq.heappop(self._free_blocks) for _ in range(additional)]
        for block in allocated:
            if self._block_owners[block] is not None:
                raise RuntimeError("allocator ownership corruption detected")
            self._block_owners[block] = request_id
        sequence.blocks.extend(allocated)
        for logical_index, physical_index in enumerate(sequence.blocks):
            self.block_tables[sequence.slot, logical_index] = physical_index

        reservation = AppendReservation(
            request_id=request_id,
            sequence_slot=sequence.slot,
            start_position=sequence.length,
            token_count=token_count,
            reservation_id=self._next_reservation_id,
        )
        self._next_reservation_id += 1
        self._reservations[reservation.reservation_id] = _ReservationState(
            public=reservation, newly_allocated_blocks=allocated
        )
        sequence.active_reservation = reservation.reservation_id
        return reservation

    def write_layer(
        self,
        reservation: AppendReservation,
        layer_index: int,
        keys: Tensor,
        values: Tensor,
    ) -> None:
        state = self._reservation(reservation)
        if not 0 <= layer_index < self.config.num_layers:
            raise IndexError("layer_index is out of range")
        expected_shape = (
            reservation.token_count,
            self.config.num_kv_heads,
            self.config.head_dim,
        )
        for name, tensor in (("keys", keys), ("values", values)):
            if tensor.shape != expected_shape:
                raise ValueError(f"{name} must have shape {expected_shape}")
            if tensor.dtype != self.config.dtype or tensor.device != self.config.device:
                raise ValueError(f"{name} must share cache dtype and device")
        sequence = self._sequence(reservation.request_id)
        for source_index in range(reservation.token_count):
            position = reservation.start_position + source_index
            logical_block, block_offset = divmod(position, self.config.block_size)
            physical_block = sequence.blocks[logical_block]
            self.key_cache[layer_index, physical_block, block_offset].copy_(keys[source_index])
            self.value_cache[layer_index, physical_block, block_offset].copy_(values[source_index])
        state.written_layers.add(layer_index)

    def commit(
        self, reservation: AppendReservation, accepted_tokens: int | None = None
    ) -> None:
        state = self._reservation(reservation)
        accepted = reservation.token_count if accepted_tokens is None else accepted_tokens
        if not 0 <= accepted <= reservation.token_count:
            raise ValueError("accepted_tokens must be within the reservation")
        if accepted > 0 and len(state.written_layers) != self.config.num_layers:
            raise RuntimeError("every cache layer must be written before commit")
        sequence = self._sequence(reservation.request_id)
        sequence.length = reservation.start_position + accepted
        retained_blocks = (
            sequence.length + self.config.block_size - 1
        ) // self.config.block_size
        for logical_index in range(retained_blocks, len(sequence.blocks)):
            physical_block = sequence.blocks[logical_index]
            self.block_tables[sequence.slot, logical_index] = -1
            self._block_owners[physical_block] = None
            heapq.heappush(self._free_blocks, physical_block)
        del sequence.blocks[retained_blocks:]
        self.sequence_lengths[sequence.slot] = sequence.length
        sequence.active_reservation = None
        del self._reservations[reservation.reservation_id]

    def rollback(self, reservation: AppendReservation) -> None:
        """Abort an append and return only pages allocated by that transaction."""

        state = self._reservation(reservation)
        sequence = self._sequence(reservation.request_id)
        first_removed_logical = len(sequence.blocks) - len(state.newly_allocated_blocks)
        for logical_index in range(first_removed_logical, len(sequence.blocks)):
            self.block_tables[sequence.slot, logical_index] = -1
        if state.newly_allocated_blocks:
            del sequence.blocks[first_removed_logical:]
            for block in state.newly_allocated_blocks:
                self._block_owners[block] = None
                heapq.heappush(self._free_blocks, block)
        sequence.active_reservation = None
        del self._reservations[reservation.reservation_id]

    def release(self, request_id: str) -> None:
        """Cancel pending work and return every page and slot owned by a request."""

        sequence = self._sequence(request_id)
        if sequence.active_reservation is not None:
            pending = self._reservations[sequence.active_reservation].public
            self.rollback(pending)
        for block in sequence.blocks:
            if self._block_owners[block] != request_id:
                raise RuntimeError("allocator ownership corruption detected")
            self._block_owners[block] = None
            heapq.heappush(self._free_blocks, block)
        self.block_tables[sequence.slot].fill_(-1)
        self.sequence_lengths[sequence.slot] = 0
        heapq.heappush(self._free_slots, sequence.slot)
        del self._sequences[request_id]

    def block_owners(self) -> tuple[str | None, ...]:
        """Expose immutable ownership state for accounting and stress verification."""

        return tuple(self._block_owners)

    def sequence_layout(
        self, request_id: str, reservation: AppendReservation | None = None
    ) -> SequenceLayout:
        """Return CPU-owned logical mapping metadata without a device synchronization."""

        sequence = self._sequence(request_id)
        context_length = sequence.length
        if reservation is not None:
            self._reservation(reservation)
            if reservation.request_id != request_id:
                raise ValueError("reservation belongs to another sequence")
            context_length = reservation.end_position
        return SequenceLayout(tuple(sequence.blocks), context_length)

    def append_location(
        self, reservation: AppendReservation, token_offset: int = 0
    ) -> CacheAppendLocation:
        """Resolve a reserved logical token to its physical page destination."""

        self._reservation(reservation)
        if not 0 <= token_offset < reservation.token_count:
            raise IndexError("token_offset is outside the append reservation")
        sequence = self._sequence(reservation.request_id)
        position = reservation.start_position + token_offset
        logical_block, block_offset = divmod(position, self.config.block_size)
        return CacheAppendLocation(sequence.blocks[logical_block], block_offset)

    def record_layer_write(
        self, reservation: AppendReservation, layer_index: int
    ) -> None:
        """Record that an optimized writer populated one complete reserved cache layer."""

        state = self._reservation(reservation)
        if not 0 <= layer_index < self.config.num_layers:
            raise IndexError("layer_index is out of range")
        state.written_layers.add(layer_index)

    def layer_view(
        self,
        request_id: str,
        layer_index: int,
        reservation: AppendReservation | None = None,
    ) -> PagedLayerView:
        sequence = self._sequence(request_id)
        if not 0 <= layer_index < self.config.num_layers:
            raise IndexError("layer_index is out of range")
        context_length = sequence.length
        if reservation is not None:
            self._reservation(reservation)
            if reservation.request_id != request_id:
                raise ValueError("reservation belongs to another sequence")
            context_length = reservation.end_position
        return PagedLayerView(
            key_cache=self.key_cache[layer_index],
            value_cache=self.value_cache[layer_index],
            block_table=self.block_tables[sequence.slot],
            context_length=context_length,
        )

    def materialize(self, request_id: str, layer_index: int) -> tuple[Tensor, Tensor]:
        view = self.layer_view(request_id, layer_index)
        keys = torch.empty(
            (view.context_length, self.config.num_kv_heads, self.config.head_dim),
            dtype=self.config.dtype,
            device=self.config.device,
        )
        values = torch.empty_like(keys)
        sequence = self._sequence(request_id)
        for position in range(view.context_length):
            logical_block, block_offset = divmod(position, self.config.block_size)
            physical_block = sequence.blocks[logical_block]
            keys[position].copy_(view.key_cache[physical_block, block_offset])
            values[position].copy_(view.value_cache[physical_block, block_offset])
        return keys, values

    def stats(self) -> CacheStats:
        """Return allocator accounting without synchronizing cache tensor contents."""

        return CacheStats(
            total_blocks=self.config.num_blocks,
            used_blocks=self.config.num_blocks - len(self._free_blocks),
            free_blocks=len(self._free_blocks),
            active_sequences=len(self._sequences),
            committed_tokens=sum(sequence.length for sequence in self._sequences.values()),
            reserved_tokens=sum(
                state.public.token_count for state in self._reservations.values()
            ),
        )

    def _sequence(self, request_id: str) -> _SequenceState:
        try:
            return self._sequences[request_id]
        except KeyError as error:
            raise KeyError(f"unknown sequence: {request_id}") from error

    def _reservation(self, reservation: AppendReservation) -> _ReservationState:
        try:
            state = self._reservations[reservation.reservation_id]
        except KeyError as error:
            raise RuntimeError("append reservation is no longer active") from error
        if state.public != reservation:
            raise RuntimeError("append reservation identity mismatch")
        return state
