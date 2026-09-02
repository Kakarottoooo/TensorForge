"""Parallel prompt ingestion into the canonical transactional paged KV cache."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import suppress
from typing import cast

import torch
from torch import Tensor
from torch.nn import functional as F

from tensorforge.cache.paged import AppendReservation, PagedKVCache
from tensorforge.model.layers import apply_rotary_embedding
from tensorforge.model.llama import LlamaForCausalLM
from tensorforge.runtime.batch import BatchExecutionResult


@torch.inference_mode()
def prefill_request(
    model: LlamaForCausalLM,
    cache: PagedKVCache,
    request_id: str,
    token_ids: Sequence[int],
) -> Tensor:
    """Ingest one complete prompt in parallel and return its final-token logits."""

    _validate_runtime(model, cache)
    _validate_prompt(model, cache, request_id, token_ids)
    result = prefill_requests(model, cache, {request_id: token_ids})
    if request_id in result.errors:
        raise RuntimeError(result.errors[request_id])
    return result.logits[request_id].unsqueeze(0)


@torch.inference_mode()
def prefill_requests(
    model: LlamaForCausalLM,
    cache: PagedKVCache,
    prompts: Mapping[str, Sequence[int]],
) -> BatchExecutionResult:
    """Ingest unequal prompts as one padded causal batch with per-request transactions."""

    _validate_runtime(model, cache)
    if not prompts:
        raise ValueError("prefill_requests requires at least one request")
    errors: dict[str, str] = {}
    reservations: dict[str, AppendReservation] = {}
    for request_id, token_ids in prompts.items():
        try:
            _validate_prompt(model, cache, request_id, token_ids)
            reservations[request_id] = cache.begin_append(
                request_id, token_count=len(token_ids)
            )
        except Exception as error:
            errors[request_id] = f"{type(error).__name__}: {error}"
    if not reservations:
        return BatchExecutionResult(logits={}, errors=errors)

    parameter = next(model.parameters())
    request_ids = list(reservations)
    try:
        lengths = torch.tensor(
            [len(prompts[request_id]) for request_id in request_ids],
            dtype=torch.long,
            device=parameter.device,
        )
        sequence_length = int(lengths.max().item())
        tokens = torch.zeros(
            (len(request_ids), sequence_length),
            dtype=torch.long,
            device=parameter.device,
        )
        for batch_index, request_id in enumerate(request_ids):
            prompt = torch.tensor(
                prompts[request_id], dtype=torch.long, device=parameter.device
            )
            tokens[batch_index, : prompt.numel()] = prompt
        positions = torch.arange(sequence_length, dtype=torch.long, device=parameter.device)
        token_positions = torch.arange(
            sequence_length, dtype=torch.long, device=parameter.device
        )
        valid = token_positions.unsqueeze(0) < lengths.unsqueeze(1)
        causal = token_positions.unsqueeze(1) >= token_positions.unsqueeze(0)
        attention_mask = (
            causal.unsqueeze(0) & valid.unsqueeze(1) & valid.unsqueeze(2)
        ).unsqueeze(1)
        hidden_states = model.model.embedding(tokens)
        for layer_index, layer in enumerate(model.model.layers):
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
            query, key = apply_rotary_embedding(query, key, cos, sin)
            for batch_index, request_id in enumerate(request_ids):
                token_count = len(prompts[request_id])
                cache.write_layer(
                    reservations[request_id],
                    layer_index,
                    key[batch_index, :, :token_count].transpose(0, 1).contiguous(),
                    value[batch_index, :, :token_count].transpose(0, 1).contiguous(),
                )
            context = F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=attention_mask,
                dropout_p=0.0,
                is_causal=False,
                scale=attention.scale,
                enable_gqa=True,
            )
            attention_output = attention.o_proj(
                context.transpose(1, 2)
                .contiguous()
                .view(len(request_ids), sequence_length, model.config.hidden_size)
            )
            hidden_states = hidden_states + attention_output
            hidden_states = hidden_states + layer.mlp(
                layer.post_attention_norm(hidden_states)
            )

        batch_indices = torch.arange(len(request_ids), device=parameter.device)
        final_hidden = model.model.final_norm(
            hidden_states[batch_indices, lengths - 1].unsqueeze(1)
        )
        batch_logits = cast(Tensor, model.output_projection(final_hidden)[:, 0])
        for reservation in reservations.values():
            cache.commit(reservation)
        return BatchExecutionResult(
            logits={
                request_id: batch_logits[batch_index]
                for batch_index, request_id in enumerate(request_ids)
            },
            errors=errors,
        )
    except Exception as error:
        message = f"{type(error).__name__}: {error}"
        for request_id, reservation in reservations.items():
            with suppress(RuntimeError):
                cache.rollback(reservation)
            errors[request_id] = message
        return BatchExecutionResult(logits={}, errors=errors)


def _validate_runtime(model: LlamaForCausalLM, cache: PagedKVCache) -> None:
    if model.training:
        raise ValueError("prefill requires an eval-mode model")
    parameter = next(model.parameters())
    expected_cache = (
        model.config.num_hidden_layers,
        model.config.num_key_value_heads,
        model.config.head_dim,
        parameter.dtype,
        parameter.device,
    )
    actual_cache = (
        cache.config.num_layers,
        cache.config.num_kv_heads,
        cache.config.head_dim,
        cache.config.dtype,
        torch.device(cache.config.device),
    )
    if actual_cache != expected_cache:
        raise ValueError("cache dimensions, dtype, and device must match the model")


def _validate_prompt(
    model: LlamaForCausalLM,
    cache: PagedKVCache,
    request_id: str,
    token_ids: Sequence[int],
) -> None:
    if not token_ids:
        raise ValueError("prefill requires at least one token")
    if any(not 0 <= token_id < model.config.vocab_size for token_id in token_ids):
        raise ValueError("prompt token is outside the model vocabulary")
    if cache.sequence_layout(request_id).context_length != 0:
        raise ValueError("prefill requires an empty request sequence")
