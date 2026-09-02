"""Inference execution paths."""

from tensorforge.runtime.generation import GenerationConfig, greedy_generate
from tensorforge.runtime.triton_model import TritonModelExecutor

__all__ = ["GenerationConfig", "TritonModelExecutor", "greedy_generate"]
