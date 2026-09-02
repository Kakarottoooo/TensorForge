"""Configuration for the small, inspectable Llama-style reference model."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Static model dimensions.

    Defaults deliberately describe a small model (~56M parameters) that fits on
    commodity GPUs. Tests use smaller configurations to keep CPU CI fast.
    """

    vocab_size: int = 32_000
    hidden_size: int = 512
    intermediate_size: int = 1_376
    num_hidden_layers: int = 8
    num_attention_heads: int = 8
    num_key_value_heads: int = 4
    max_position_embeddings: int = 4_096
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10_000.0
    tie_word_embeddings: bool = False

    def __post_init__(self) -> None:
        positive_ints = {
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "max_position_embeddings": self.max_position_embeddings,
        }
        for name, value in positive_ints.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
        if self.head_dim % 2 != 0:
            raise ValueError("attention head dimension must be even for rotary embeddings")
        if self.rms_norm_eps <= 0:
            raise ValueError("rms_norm_eps must be positive")
        if self.rope_theta <= 0:
            raise ValueError("rope_theta must be positive")

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads


def tiny_config(**overrides: int | float | bool) -> ModelConfig:
    """Return a fast configuration for tests and smoke runs."""

    values: dict[str, int | float | bool] = {
        "vocab_size": 128,
        "hidden_size": 64,
        "intermediate_size": 160,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "max_position_embeddings": 128,
    }
    values.update(overrides)
    return ModelConfig(**values)  # type: ignore[arg-type]
