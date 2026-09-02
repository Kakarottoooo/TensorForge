"""Variable-length batched decode over one transactional paged KV cache."""

from __future__ import annotations

from contextlib import suppress
from dataclasses import dataclass
from typing import cast

import torch
from torch import Tensor

from tensorforge.cache.paged import AppendReservation, PagedKVCache
from tensorforge.kernels.triton_attention import paged_gqa_decode_attention
from tensorforge.model.layers import apply_batched_rotary_embedding
from tensorforge.model.llama import LlamaForCausalLM
from tensorforge.runtime.batch import BatchExecutionResult
from tensorforge.runtime.prefill import prefill_requests


@dataclass(slots=True)
class BatchedPagedDecodeExecutor:
    """Execute one token for each selected request through a shared model/cache."""

    model: LlamaForCausalLM
    cache: PagedKVCache

    def __post_init__(self) -> None:
        parameter = next(self.model.parameters())
        config = self.model.config
        cache_config = self.cache.config
        if not parameter.is_cuda:
            raise ValueError("BatchedPagedDecodeExecutor requires a CUDA model")
        if self.model.training:
            raise ValueError("BatchedPagedDecodeExecutor requires eval mode")
        expected = (
            config.num_hidden_layers,
            config.num_key_value_heads,
            config.head_dim,
            parameter.dtype,
            parameter.device,
        )
        actual = (
            cache_config.num_layers,
            cache_config.num_kv_heads,
            cache_config.head_dim,
            cache_config.dtype,
            cache_config.device,
        )
        if actual != expected:
            raise ValueError("cache dimensions, dtype, and device must match the model")
        if cache_config.max_sequence_length > config.max_position_embeddings:
            raise ValueError("cache sequence capacity exceeds the model context capacity")

    @property
    def vocab_size(self) -> int:
        return self.model.config.vocab_size

    def create_request(self, request_id: str) -> None:
        self.cache.create_sequence(request_id)

    def release_request(self, request_id: str) -> None:
        self.cache.release(request_id)

    def prefill_prompts(
        self, prompts: dict[str, tuple[int, ...]]
    ) -> BatchExecutionResult:
        return prefill_requests(self.model, self.cache, prompts)

    @torch.inference_mode()
    def append_tokens(self, tokens: dict[str, int]) -> BatchExecutionResult:
        if not tokens:
            raise ValueError("append_tokens requires at least one request")
        errors: dict[str, str] = {}
        reservations: dict[str, AppendReservation] = {}
        for request_id, token_id in tokens.items():
            if not 0 <= token_id < self.vocab_size:
                errors[request_id] = "token_id is outside the model vocabulary"
                continue
            try:
                reservations[request_id] = self.cache.begin_append(request_id, token_count=1)
            except Exception as error:
                errors[request_id] = f"{type(error).__name__}: {error}"
        if not reservations:
            return BatchExecutionResult(logits={}, errors=errors)

        request_ids = list(reservations)
        try:
            parameter = next(self.model.parameters())
            device = parameter.device
            token_tensor = torch.tensor(
                [[tokens[request_id]] for request_id in request_ids],
                dtype=torch.long,
                device=device,
            )
            positions = torch.tensor(
                [reservations[request_id].start_position for request_id in request_ids],
                dtype=torch.long,
                device=device,
            )
            hidden_states = self.model.model.embedding(token_tensor)

            for layer_index, layer in enumerate(self.model.model.layers):
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
                query, key = apply_batched_rotary_embedding(query, key, cos, sin)

                views = []
                for batch_index, request_id in enumerate(request_ids):
                    reservation = reservations[request_id]
                    self.cache.write_layer(
                        reservation,
                        layer_index,
                        key[batch_index, :, 0, :].unsqueeze(0).contiguous(),
                        value[batch_index, :, 0, :].unsqueeze(0).contiguous(),
                    )
                    views.append(
                        self.cache.layer_view(request_id, layer_index, reservation)
                    )
                block_tables = torch.stack([view.block_table for view in views])
                context_lengths = torch.tensor(
                    [view.context_length for view in views],
                    dtype=torch.int32,
                    device=device,
                )
                context = paged_gqa_decode_attention(
                    query[:, :, 0, :].contiguous(),
                    views[0].key_cache,
                    views[0].value_cache,
                    block_tables,
                    context_lengths,
                    scale=attention.scale,
                    max_context_length=max(view.context_length for view in views),
                )
                attention_output = attention.o_proj(
                    context.reshape(len(request_ids), 1, self.model.config.hidden_size)
                )
                hidden_states = hidden_states + attention_output
                hidden_states = hidden_states + layer.mlp(
                    layer.post_attention_norm(hidden_states)
                )

            hidden_states = self.model.model.final_norm(hidden_states)
            batch_logits = cast(Tensor, self.model.output_projection(hidden_states)[:, 0])
            for reservation in reservations.values():
                self.cache.commit(reservation)
            logits = {
                request_id: batch_logits[batch_index]
                for batch_index, request_id in enumerate(request_ids)
            }
            return BatchExecutionResult(logits=logits, errors=errors)
        except Exception as error:
            message = f"{type(error).__name__}: {error}"
            for request_id, reservation in reservations.items():
                with suppress(RuntimeError):
                    self.cache.rollback(reservation)
                errors[request_id] = message
            return BatchExecutionResult(logits={}, errors=errors)
