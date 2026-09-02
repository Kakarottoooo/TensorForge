from __future__ import annotations

import pytest
import torch

from tensorforge.model.config import ModelConfig, tiny_config


@pytest.fixture
def model_config() -> ModelConfig:
    return tiny_config()


@pytest.fixture(autouse=True)
def deterministic_seed() -> None:
    torch.manual_seed(1234)
