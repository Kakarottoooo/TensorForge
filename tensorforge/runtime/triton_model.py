"""Integrated Llama forward path using TensorForge Triton operators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch
from torch import Tensor

from tensorforge.kernels.triton_ops import residual_rms_norm, rms_norm, swiglu
from tensorforge.model.llama import LlamaForCausalLM


@dataclass(slots=True)
class TritonModelExecutor:
    """Inference-only executor that preserves the reference model's weights.

    Attention and GEMMs remain PyTorch in Phase 3. RMSNorm, the attention
    residual plus post-attention RMSNorm, and the SwiGLU activation use the
    custom kernels. This keeps the correctness baseline intact and makes the
    optimization boundary explicit.
    """

    model: LlamaForCausalLM

    def __post_init__(self) -> None:
        parameter = next(self.model.parameters())
        if not parameter.is_cuda:
            raise ValueError("TritonModelExecutor requires a CUDA model")
        if self.model.training:
            raise ValueError("TritonModelExecutor requires eval mode")

    @torch.inference_mode()
    def forward(self, input_ids: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        reference = self.model.model
        if input_ids.ndim != 2 or input_ids.dtype != torch.long:
            raise ValueError("input_ids must be rank-2 torch.long")
        if input_ids.shape[1] > reference.config.max_position_embeddings:
            raise ValueError("input sequence exceeds model context capacity")

        hidden_states = reference.embedding(input_ids)
        for layer in reference.layers:
            normalized = rms_norm(hidden_states, layer.input_norm.weight, layer.input_norm.eps)
            attention_output = layer.attention(normalized, attention_mask=attention_mask)
            residual_output, mlp_input = residual_rms_norm(
                attention_output,
                hidden_states,
                layer.post_attention_norm.weight,
                layer.post_attention_norm.eps,
            )
            gate = layer.mlp.gate_proj(mlp_input)
            up = layer.mlp.up_proj(mlp_input)
            mlp_output = layer.mlp.down_proj(swiglu(gate, up))
            hidden_states = residual_output + mlp_output

        hidden_states = rms_norm(
            hidden_states, reference.final_norm.weight, reference.final_norm.eps
        )
        return cast(Tensor, self.model.output_projection(hidden_states))
