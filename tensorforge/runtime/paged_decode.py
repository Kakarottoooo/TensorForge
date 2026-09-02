"""Transactional single-request decode over paged KV storage."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch
from torch import Tensor

from tensorforge.cache.paged import PagedKVCache
from tensorforge.kernels.triton_attention import paged_gqa_decode_attention
from tensorforge.model.layers import apply_rotary_embedding
from tensorforge.model.llama import LlamaForCausalLM


@dataclass(slots=True)
class PagedDecodeExecutor:
    """Incremental Llama executor that commits one cache token transaction at a time.

    Phase 4 deliberately exposes single-request execution. Continuous batching must
    consume this stable cache/kernel boundary rather than introduce a second cache path.
    """

    model: LlamaForCausalLM
    cache: PagedKVCache

    def __post_init__(self) -> None:
        parameter = next(self.model.parameters())
        config = self.model.config
        cache_config = self.cache.config
        if not parameter.is_cuda:
            raise ValueError("PagedDecodeExecutor requires a CUDA model")
        if self.model.training:
            raise ValueError("PagedDecodeExecutor requires eval mode")
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

    def create_request(self, request_id: str) -> None:
        self.cache.create_sequence(request_id)

    def release_request(self, request_id: str) -> None:
        self.cache.release(request_id)

    @torch.inference_mode()
    def append_token(self, request_id: str, token_id: int) -> Tensor:
        """Append one token atomically and return its `[1, vocab]` logits."""

        if not 0 <= token_id < self.model.config.vocab_size:
            raise ValueError("token_id is outside the model vocabulary")
        reservation = self.cache.begin_append(request_id, token_count=1)
        try:
            device = next(self.model.parameters()).device
            token = torch.tensor([[token_id]], dtype=torch.long, device=device)
            position = torch.tensor(
                [reservation.start_position], dtype=torch.long, device=device
            )
            hidden_states = self.model.model.embedding(token)

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
                cos, sin = attention.rotary(position, dtype=query.dtype)
                query, key = apply_rotary_embedding(query, key, cos, sin)

                token_keys = key[0].transpose(0, 1).contiguous()
                token_values = value[0].transpose(0, 1).contiguous()
                self.cache.write_layer(
                    reservation, layer_index, token_keys, token_values
                )
                view = self.cache.layer_view(request_id, layer_index, reservation)
                context_lengths = torch.tensor(
                    [view.context_length], dtype=torch.int32, device=device
                )
                context = paged_gqa_decode_attention(
                    query[:, :, 0, :].contiguous(),
                    view.key_cache,
                    view.value_cache,
                    view.block_table.unsqueeze(0),
                    context_lengths,
                    scale=attention.scale,
                )
                attention_output = attention.o_proj(
                    context.reshape(1, 1, self.model.config.hidden_size)
                )
                hidden_states = hidden_states + attention_output
                hidden_states = hidden_states + layer.mlp(
                    layer.post_attention_norm(hidden_states)
                )

            hidden_states = self.model.model.final_norm(hidden_states)
            logits = cast(Tensor, self.model.output_projection(hidden_states)[:, 0])
            self.cache.commit(reservation)
            return logits
        except Exception:
            self.cache.rollback(reservation)
            raise
