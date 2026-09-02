"""Small Llama-style decoder-only transformer used as the baseline."""

from __future__ import annotations

from typing import cast

import torch
from torch import Tensor, nn

from tensorforge.model.config import ModelConfig
from tensorforge.model.layers import RMSNorm, TransformerBlock, initialize_weights


class LlamaModel(nn.Module):
    """Embedding, decoder blocks, and final normalization without an LM head."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            TransformerBlock(config) for _ in range(config.num_hidden_layers)
        )
        self.final_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.apply(initialize_weights)

    def forward(self, input_ids: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        if input_ids.ndim != 2:
            raise ValueError(f"input_ids must have rank 2, got shape {tuple(input_ids.shape)}")
        if input_ids.dtype != torch.long:
            raise TypeError(f"input_ids must use torch.long, got {input_ids.dtype}")
        if input_ids.shape[1] > self.config.max_position_embeddings:
            raise ValueError(
                f"sequence length {input_ids.shape[1]} exceeds configured maximum "
                f"{self.config.max_position_embeddings}"
            )

        hidden_states = self.embedding(input_ids)
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask=attention_mask)
        return cast(Tensor, self.final_norm(hidden_states))


class LlamaForCausalLM(nn.Module):
    """Reference language model with an intentionally unoptimized forward pass."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.model = LlamaModel(config)
        self.output_projection = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        initialize_weights(self.output_projection)
        if config.tie_word_embeddings:
            self.output_projection.weight = self.model.embedding.weight

    def forward(self, input_ids: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        hidden_states = self.model(input_ids, attention_mask=attention_mask)
        return cast(Tensor, self.output_projection(hidden_states))
