from __future__ import annotations

import pytest

from tensorforge.model.config import ModelConfig


@pytest.mark.parametrize(
    ("field", "value"),
    [("vocab_size", 0), ("num_hidden_layers", -1), ("max_position_embeddings", 0)],
)
def test_positive_dimensions_are_enforced(field: str, value: int) -> None:
    with pytest.raises(ValueError, match=field):
        ModelConfig(**{field: value})


def test_attention_dimension_contracts_are_enforced() -> None:
    with pytest.raises(ValueError, match="hidden_size"):
        ModelConfig(hidden_size=63, num_attention_heads=4)
    with pytest.raises(ValueError, match="num_attention_heads"):
        ModelConfig(num_attention_heads=8, num_key_value_heads=3)
