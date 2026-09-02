# mypy: disable-error-code="untyped-decorator"
"""Address-stable bucketed decode execution over the canonical paged cache."""

from __future__ import annotations

import time
from collections.abc import Callable
from contextlib import suppress
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any, cast

import torch
from torch import Tensor

from tensorforge.cache.paged import AppendReservation, PagedKVCache
from tensorforge.kernels.triton_attention import (
    PagedAttentionWorkspace,
    paged_gqa_decode_attention,
)
from tensorforge.kernels.triton_cache import paged_kv_write_token
from tensorforge.kernels.triton_ops import residual_rms_norm, rms_norm, swiglu
from tensorforge.model.layers import apply_batched_rotary_embedding
from tensorforge.model.llama import LlamaForCausalLM
from tensorforge.runtime.batch import BatchExecutionResult
from tensorforge.runtime.batched_decode import BatchedPagedDecodeExecutor

compiler_disable: Any = torch.compiler.disable


@compiler_disable
def _write_paged_kv_eager(
    keys: Tensor,
    values: Tensor,
    key_cache: Tensor,
    value_cache: Tensor,
    physical_blocks: Tensor,
    block_offsets: Tensor,
    active_mask: Tensor,
) -> None:
    """Keep the custom Triton launch as an explicit torch.compile graph boundary."""

    paged_kv_write_token(
        keys,
        values,
        key_cache,
        value_cache,
        physical_blocks,
        block_offsets,
        active_mask,
    )


@compiler_disable
def _paged_attention_eager(
    query: Tensor,
    key_cache: Tensor,
    value_cache: Tensor,
    block_tables: Tensor,
    context_lengths: Tensor,
    *,
    scale: float,
    max_context_length: int,
    output: Tensor,
    workspace: PagedAttentionWorkspace,
) -> Tensor:
    """Keep the custom paged-attention launch outside Inductor code generation."""

    return paged_gqa_decode_attention(
        query,
        key_cache,
        value_cache,
        block_tables,
        context_lengths,
        scale=scale,
        max_context_length=max_context_length,
        output=output,
        workspace=workspace,
    )


@compiler_disable
def _rms_norm_eager(
    inputs: Tensor, weight: Tensor, eps: float, output: Tensor
) -> Tensor:
    return rms_norm(inputs, weight, eps, output=output)


@compiler_disable
def _residual_rms_norm_eager(
    inputs: Tensor,
    residual: Tensor,
    weight: Tensor,
    eps: float,
    residual_output: Tensor,
    norm_output: Tensor,
) -> tuple[Tensor, Tensor]:
    return residual_rms_norm(
        inputs,
        residual,
        weight,
        eps,
        residual_output=residual_output,
        norm_output=norm_output,
    )


@compiler_disable
def _swiglu_eager(gate: Tensor, up: Tensor, output: Tensor) -> Tensor:
    return swiglu(gate, up, output=output)


class DecodeExecutionMode(StrEnum):
    EAGER = "eager"
    COMPILE = "torch_compile"
    CUDA_GRAPH = "cuda_graph"


class DecodeFusionLevel(StrEnum):
    NONE = "none"
    RMS_NORM = "rms_norm"
    RESIDUAL_RMS_NORM = "residual_rms_norm"
    ALL = "all"

    @property
    def uses_triton_norm(self) -> bool:
        return self is not DecodeFusionLevel.NONE

    @property
    def uses_fused_residual_norm(self) -> bool:
        return self in {DecodeFusionLevel.RESIDUAL_RMS_NORM, DecodeFusionLevel.ALL}

    @property
    def uses_triton_swiglu(self) -> bool:
        return self is DecodeFusionLevel.ALL


@dataclass(frozen=True, slots=True)
class DecodeExecutionMetrics:
    eager_calls: int
    compiled_calls: int
    graph_calls: int
    graph_hits: int
    graph_misses: int
    shape_fallbacks: int
    eager_fallback_calls: int
    capture_count: int
    capture_time_ms: float
    capture_warmup_time_ms: float
    compile_count: int
    compile_time_ms: float
    padded_lanes: int
    fallback_reasons: dict[str, int]


@dataclass(slots=True)
class _MutableMetrics:
    eager_calls: int = 0
    compiled_calls: int = 0
    graph_calls: int = 0
    graph_hits: int = 0
    graph_misses: int = 0
    shape_fallbacks: int = 0
    eager_fallback_calls: int = 0
    capture_count: int = 0
    capture_time_ms: float = 0.0
    capture_warmup_time_ms: float = 0.0
    compile_count: int = 0
    compile_time_ms: float = 0.0
    padded_lanes: int = 0
    fallback_reasons: dict[str, int] = field(default_factory=dict)


class _DecodeBucket:
    def __init__(
        self,
        model: LlamaForCausalLM,
        cache: PagedKVCache,
        *,
        batch_capacity: int,
        context_capacity: int,
        mode: DecodeExecutionMode,
        fusion_level: DecodeFusionLevel,
        metrics: _MutableMetrics,
    ) -> None:
        self.model = model
        self.cache = cache
        self.batch_capacity = batch_capacity
        self.context_capacity = context_capacity
        self.mode = mode
        self.fusion_level = fusion_level
        self.metrics = metrics
        config = model.config
        parameter = next(model.parameters())
        device = parameter.device
        table_width = cache.config.max_blocks_per_sequence
        self._host_long_controls = torch.zeros(
            (2, batch_capacity), dtype=torch.long, pin_memory=True
        )
        self._long_controls = torch.zeros_like(
            self._host_long_controls, device=device
        )
        self.input_ids = self._long_controls[0].view(batch_capacity, 1)
        self.positions = self._long_controls[1]
        int_control_elements = batch_capacity * (table_width + 4)
        self._host_int_controls = torch.zeros(
            int_control_elements, dtype=torch.int32, pin_memory=True
        )
        self._int_controls = torch.zeros_like(
            self._host_int_controls, device=device
        )
        table_end = batch_capacity * table_width
        context_end = table_end + batch_capacity
        physical_end = context_end + batch_capacity
        offset_end = physical_end + batch_capacity
        self.block_tables = self._int_controls[:table_end].view(
            batch_capacity, table_width
        )
        self.context_lengths = self._int_controls[table_end:context_end]
        self.physical_blocks = self._int_controls[context_end:physical_end]
        self.block_offsets = self._int_controls[physical_end:offset_end]
        self.active_mask = self._int_controls[offset_end:]
        self._host_input_ids = self._host_long_controls[0].view(
            batch_capacity, 1
        )
        self._host_positions = self._host_long_controls[1]
        self._host_block_tables = self._host_int_controls[:table_end].view(
            batch_capacity, table_width
        )
        self._host_context_lengths = self._host_int_controls[table_end:context_end]
        self._host_physical_blocks = self._host_int_controls[context_end:physical_end]
        self._host_block_offsets = self._host_int_controls[physical_end:offset_end]
        self._host_active_mask = self._host_int_controls[offset_end:]
        self.logits = torch.empty(
            (batch_capacity, config.vocab_size), dtype=parameter.dtype, device=device
        )
        self.attention_outputs = tuple(
            torch.empty(
                (batch_capacity, config.num_attention_heads, config.head_dim),
                dtype=parameter.dtype,
                device=device,
            )
            for _ in range(config.num_hidden_layers)
        )
        self.attention_workspaces = tuple(
            PagedAttentionWorkspace.allocate(
                batch_capacity=batch_capacity,
                query_heads=config.num_attention_heads,
                head_dim=config.head_dim,
                max_context_length=context_capacity,
                device=device,
            )
            for _ in range(config.num_hidden_layers)
        )
        hidden_shape = (batch_capacity, 1, config.hidden_size)

        def allocate_hidden_outputs() -> tuple[Tensor, ...]:
            return tuple(
                torch.empty(hidden_shape, dtype=parameter.dtype, device=device)
                for _ in range(config.num_hidden_layers)
            )

        self.input_norm_outputs = (
            allocate_hidden_outputs()
            if fusion_level.uses_triton_norm
            else ()
        )
        self.residual_outputs = (
            allocate_hidden_outputs()
            if fusion_level.uses_triton_norm
            else ()
        )
        self.post_norm_outputs = (
            allocate_hidden_outputs()
            if fusion_level.uses_triton_norm
            else ()
        )
        self.hidden_outputs = (
            allocate_hidden_outputs()
            if fusion_level.uses_triton_norm
            else ()
        )
        self.swiglu_outputs = (
            tuple(
                torch.empty(
                    (batch_capacity, 1, config.intermediate_size),
                    dtype=parameter.dtype,
                    device=device,
                )
                for _ in range(config.num_hidden_layers)
            )
            if fusion_level.uses_triton_swiglu
            else ()
        )
        self.final_norm_output = (
            torch.empty(hidden_shape, dtype=parameter.dtype, device=device)
            if fusion_level.uses_triton_norm
            else None
        )
        self._compiled_forward: Callable[[], None] | None = None
        self._graph: Any | None = None

    def prepare(
        self,
        request_ids: list[str],
        tokens: dict[str, int],
        reservations: dict[str, AppendReservation],
    ) -> None:
        self._host_long_controls.zero_()
        self._host_int_controls.zero_()
        self._host_context_lengths.fill_(1)
        for batch_index, request_id in enumerate(request_ids):
            reservation = reservations[request_id]
            layout = self.cache.sequence_layout(request_id, reservation)
            location = self.cache.append_location(reservation)
            self._host_input_ids[batch_index, 0] = tokens[request_id]
            self._host_positions[batch_index] = reservation.start_position
            self._host_context_lengths[batch_index] = layout.context_length
            self._host_physical_blocks[batch_index] = location.physical_block
            self._host_block_offsets[batch_index] = location.block_offset
            self._host_active_mask[batch_index] = 1
            if layout.physical_blocks:
                self._host_block_tables[
                    batch_index, : len(layout.physical_blocks)
                ] = torch.tensor(layout.physical_blocks, dtype=torch.int32)
        self._long_controls.copy_(self._host_long_controls, non_blocking=True)
        self._int_controls.copy_(self._host_int_controls, non_blocking=True)

    def addresses(self) -> dict[str, int]:
        return {
            "input_ids": self.input_ids.data_ptr(),
            "positions": self.positions.data_ptr(),
            "block_tables": self.block_tables.data_ptr(),
            "context_lengths": self.context_lengths.data_ptr(),
            "physical_blocks": self.physical_blocks.data_ptr(),
            "block_offsets": self.block_offsets.data_ptr(),
            "active_mask": self.active_mask.data_ptr(),
            "logits": self.logits.data_ptr(),
            **{
                f"attention_output_{index}": output.data_ptr()
                for index, output in enumerate(self.attention_outputs)
            },
            **{
                f"input_norm_output_{index}": output.data_ptr()
                for index, output in enumerate(self.input_norm_outputs)
            },
            **{
                f"residual_output_{index}": output.data_ptr()
                for index, output in enumerate(self.residual_outputs)
            },
            **{
                f"post_norm_output_{index}": output.data_ptr()
                for index, output in enumerate(self.post_norm_outputs)
            },
            **{
                f"hidden_output_{index}": output.data_ptr()
                for index, output in enumerate(self.hidden_outputs)
            },
            **{
                f"swiglu_output_{index}": output.data_ptr()
                for index, output in enumerate(self.swiglu_outputs)
            },
            **(
                {"final_norm_output": self.final_norm_output.data_ptr()}
                if self.final_norm_output is not None
                else {}
            ),
        }

    def execute(self) -> None:
        if self.mode is DecodeExecutionMode.EAGER:
            self._forward()
            self.metrics.eager_calls += 1
            return
        if self.mode is DecodeExecutionMode.COMPILE:
            if self._compiled_forward is None:
                torch.cuda.synchronize(self.input_ids.device)
                started_ns = time.perf_counter_ns()
                self._compiled_forward = torch.compile(
                    self._forward,
                    fullgraph=False,
                    dynamic=False,
                    options={"triton.cudagraphs": False},
                )
                self._compiled_forward()
                torch.cuda.synchronize(self.input_ids.device)
                self.metrics.compile_time_ms += (
                    time.perf_counter_ns() - started_ns
                ) / 1e6
                self.metrics.compile_count += 1
            else:
                self._compiled_forward()
            self.metrics.compiled_calls += 1
            return
        if self._graph is None:
            current_stream = torch.cuda.current_stream(self.input_ids.device)
            stream_factory: Any = torch.cuda.Stream
            warmup_stream = stream_factory(device=self.input_ids.device)
            warmup_stream.wait_stream(current_stream)
            warmup_started_ns = time.perf_counter_ns()
            with torch.cuda.stream(warmup_stream):
                self._forward()
            warmup_stream.synchronize()
            current_stream.wait_stream(warmup_stream)
            self.metrics.capture_warmup_time_ms += (
                time.perf_counter_ns() - warmup_started_ns
            ) / 1e6

            graph_factory: Any = torch.cuda.CUDAGraph
            self._graph = graph_factory()
            torch.cuda.synchronize(self.input_ids.device)
            capture_started_ns = time.perf_counter_ns()
            with torch.cuda.graph(self._graph):
                self._forward()
            torch.cuda.synchronize(self.input_ids.device)
            self.metrics.capture_time_ms += (
                time.perf_counter_ns() - capture_started_ns
            ) / 1e6
            self.metrics.capture_count += 1
            self.metrics.graph_misses += 1
        else:
            self._graph.replay()
            self.metrics.graph_hits += 1
        self.metrics.graph_calls += 1

    def _forward(self) -> None:
        hidden_states = self.model.model.embedding(self.input_ids)
        for layer_index, layer in enumerate(self.model.model.layers):
            attention = layer.attention
            normalized = (
                _rms_norm_eager(
                    hidden_states,
                    layer.input_norm.weight,
                    layer.input_norm.eps,
                    self.input_norm_outputs[layer_index],
                )
                if self.fusion_level.uses_triton_norm
                else layer.input_norm(hidden_states)
            )
            query = attention._shape(attention.q_proj(normalized), attention.num_heads)
            key = attention._shape(
                attention.k_proj(normalized), attention.num_key_value_heads
            )
            value = attention._shape(
                attention.v_proj(normalized), attention.num_key_value_heads
            )
            cos, sin = attention.rotary(self.positions, dtype=query.dtype)
            query, key = apply_batched_rotary_embedding(query, key, cos, sin)
            _write_paged_kv_eager(
                key[:, :, 0, :].contiguous(),
                value[:, :, 0, :].contiguous(),
                self.cache.key_cache[layer_index],
                self.cache.value_cache[layer_index],
                self.physical_blocks,
                self.block_offsets,
                self.active_mask,
            )
            context = _paged_attention_eager(
                query[:, :, 0, :].contiguous(),
                self.cache.key_cache[layer_index],
                self.cache.value_cache[layer_index],
                self.block_tables,
                self.context_lengths,
                scale=attention.scale,
                max_context_length=self.context_capacity,
                output=self.attention_outputs[layer_index],
                workspace=self.attention_workspaces[layer_index],
            )
            attention_output = attention.o_proj(
                context.reshape(
                    self.batch_capacity, 1, self.model.config.hidden_size
                )
            )
            if self.fusion_level.uses_fused_residual_norm:
                residual_output, mlp_input = _residual_rms_norm_eager(
                    attention_output,
                    hidden_states,
                    layer.post_attention_norm.weight,
                    layer.post_attention_norm.eps,
                    self.residual_outputs[layer_index],
                    self.post_norm_outputs[layer_index],
                )
            elif self.fusion_level.uses_triton_norm:
                residual_output = torch.add(
                    hidden_states,
                    attention_output,
                    out=self.residual_outputs[layer_index],
                )
                mlp_input = _rms_norm_eager(
                    residual_output,
                    layer.post_attention_norm.weight,
                    layer.post_attention_norm.eps,
                    self.post_norm_outputs[layer_index],
                )
            else:
                residual_output = hidden_states + attention_output
                mlp_input = layer.post_attention_norm(residual_output)
            if self.fusion_level.uses_triton_swiglu:
                gate = layer.mlp.gate_proj(mlp_input)
                up = layer.mlp.up_proj(mlp_input)
                activation = _swiglu_eager(
                    gate, up, self.swiglu_outputs[layer_index]
                )
                mlp_output = layer.mlp.down_proj(activation)
            else:
                mlp_output = layer.mlp(mlp_input)
            hidden_states = (
                torch.add(
                    residual_output,
                    mlp_output,
                    out=self.hidden_outputs[layer_index],
                )
                if self.fusion_level.uses_triton_norm
                else residual_output + mlp_output
            )
        hidden_states = (
            _rms_norm_eager(
                hidden_states,
                self.model.model.final_norm.weight,
                self.model.model.final_norm.eps,
                cast(Tensor, self.final_norm_output),
            )
            if self.fusion_level.uses_triton_norm
            else self.model.model.final_norm(hidden_states)
        )
        output = cast(Tensor, self.model.output_projection(hidden_states)[:, 0])
        self.logits.copy_(output)


@dataclass
class BucketedPagedDecodeExecutor:
    """BatchTokenExecutor with persistent bucket buffers and explicit fallback metrics."""

    model: LlamaForCausalLM
    cache: PagedKVCache
    mode: DecodeExecutionMode
    batch_buckets: tuple[int, ...]
    context_buckets: tuple[int, ...]
    fusion_level: DecodeFusionLevel = DecodeFusionLevel.NONE

    def __post_init__(self) -> None:
        self._fallback = BatchedPagedDecodeExecutor(self.model, self.cache)
        if not self.batch_buckets or any(value <= 0 for value in self.batch_buckets):
            raise ValueError("batch_buckets must contain positive capacities")
        if not self.context_buckets or any(value <= 0 for value in self.context_buckets):
            raise ValueError("context_buckets must contain positive capacities")
        self.batch_buckets = tuple(sorted(set(self.batch_buckets)))
        self.context_buckets = tuple(sorted(set(self.context_buckets)))
        if self.batch_buckets[-1] > self.cache.config.max_sequences:
            raise ValueError("batch bucket exceeds cache sequence capacity")
        if self.context_buckets[-1] > self.cache.config.max_sequence_length:
            raise ValueError("context bucket exceeds cache sequence capacity")
        self._buckets: dict[tuple[int, int], _DecodeBucket] = {}
        self._metrics = _MutableMetrics()

    @property
    def vocab_size(self) -> int:
        return self.model.config.vocab_size

    def create_request(self, request_id: str) -> None:
        self.cache.create_sequence(request_id)

    def release_request(self, request_id: str) -> None:
        self.cache.release(request_id)

    def metrics(self) -> DecodeExecutionMetrics:
        return DecodeExecutionMetrics(**asdict(self._metrics))

    def buffer_addresses(self) -> dict[str, dict[str, int]]:
        return {
            f"b{batch}-c{context}": bucket.addresses()
            for (batch, context), bucket in self._buckets.items()
        }

    @torch.inference_mode()
    def append_tokens(self, tokens: dict[str, int]) -> BatchExecutionResult:
        if not tokens:
            raise ValueError("append_tokens requires at least one request")
        anticipated_contexts = [
            self.cache.sequence_layout(request_id).context_length + 1
            for request_id in tokens
        ]
        batch_capacity = next(
            (value for value in self.batch_buckets if value >= len(tokens)), None
        )
        context_capacity = next(
            (
                value
                for value in self.context_buckets
                if value >= max(anticipated_contexts)
            ),
            None,
        )
        if batch_capacity is None or context_capacity is None:
            self._metrics.shape_fallbacks += 1
            self._metrics.eager_fallback_calls += 1
            if batch_capacity is None and context_capacity is None:
                fallback_reason = "batch_and_context_overflow"
            elif batch_capacity is None:
                fallback_reason = "batch_overflow"
            else:
                fallback_reason = "context_overflow"
            self._metrics.fallback_reasons[fallback_reason] = (
                self._metrics.fallback_reasons.get(fallback_reason, 0) + 1
            )
            if self.mode is DecodeExecutionMode.CUDA_GRAPH:
                self._metrics.graph_misses += 1
            return self._fallback.append_tokens(tokens)

        errors: dict[str, str] = {}
        reservations: dict[str, AppendReservation] = {}
        for request_id, token_id in tokens.items():
            if not 0 <= token_id < self.vocab_size:
                errors[request_id] = "token_id is outside the model vocabulary"
                continue
            try:
                reservations[request_id] = self.cache.begin_append(
                    request_id, token_count=1
                )
            except Exception as error:
                errors[request_id] = f"{type(error).__name__}: {error}"
        if not reservations:
            return BatchExecutionResult(logits={}, errors=errors)

        request_ids = list(reservations)
        bucket_key = (batch_capacity, context_capacity)
        bucket = self._buckets.get(bucket_key)
        if bucket is None:
            bucket = _DecodeBucket(
                self.model,
                self.cache,
                batch_capacity=batch_capacity,
                context_capacity=context_capacity,
                mode=self.mode,
                fusion_level=self.fusion_level,
                metrics=self._metrics,
            )
            self._buckets[bucket_key] = bucket
        self._metrics.padded_lanes += batch_capacity - len(request_ids)
        try:
            bucket.prepare(request_ids, tokens, reservations)
            bucket.execute()
            for reservation in reservations.values():
                for layer_index in range(self.model.config.num_hidden_layers):
                    self.cache.record_layer_write(reservation, layer_index)
                self.cache.commit(reservation)
            return BatchExecutionResult(
                logits={
                    request_id: bucket.logits[batch_index]
                    for batch_index, request_id in enumerate(request_ids)
                },
                errors=errors,
            )
        except Exception as error:
            message = f"{type(error).__name__}: {error}"
            for request_id, reservation in reservations.items():
                with suppress(RuntimeError):
                    self.cache.rollback(reservation)
                errors[request_id] = message
            return BatchExecutionResult(logits={}, errors=errors)
