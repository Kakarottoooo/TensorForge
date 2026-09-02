"""Strict Hugging Face Llama checkpoint import without a Transformers runtime dependency."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from tensorforge.model.config import ModelConfig
from tensorforge.model.llama import LlamaForCausalLM


def model_config_from_huggingface(checkpoint_dir: str | Path) -> ModelConfig:
    """Read the supported Llama configuration subset from a local checkpoint."""

    directory = Path(checkpoint_dir)
    raw = _read_json_object(directory / "config.json")
    if raw.get("model_type") != "llama":
        raise ValueError("checkpoint model_type must be llama")
    architectures = raw.get("architectures", [])
    if architectures and "LlamaForCausalLM" not in architectures:
        raise ValueError("checkpoint architectures must include LlamaForCausalLM")
    if raw.get("hidden_act", "silu") != "silu":
        raise ValueError("only SwiGLU checkpoints with hidden_act=silu are supported")
    if raw.get("attention_bias", False):
        raise ValueError("attention projection bias is not supported")
    if raw.get("mlp_bias", False):
        raise ValueError("MLP projection bias is not supported")
    if raw.get("rope_scaling") is not None:
        raise ValueError("rope_scaling checkpoints are not supported")

    required = (
        "vocab_size",
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "max_position_embeddings",
    )
    missing = [name for name in required if name not in raw]
    if missing:
        raise ValueError(f"checkpoint config is missing required fields: {missing}")
    return ModelConfig(
        vocab_size=_integer(raw, "vocab_size"),
        hidden_size=_integer(raw, "hidden_size"),
        intermediate_size=_integer(raw, "intermediate_size"),
        num_hidden_layers=_integer(raw, "num_hidden_layers"),
        num_attention_heads=_integer(raw, "num_attention_heads"),
        num_key_value_heads=_integer(raw, "num_key_value_heads"),
        max_position_embeddings=_integer(raw, "max_position_embeddings"),
        rms_norm_eps=float(raw.get("rms_norm_eps", 1e-6)),
        rope_theta=float(raw.get("rope_theta", 10_000.0)),
        tie_word_embeddings=bool(raw.get("tie_word_embeddings", False)),
    )


def load_huggingface_checkpoint(
    checkpoint_dir: str | Path,
    *,
    device: str | torch.device,
    dtype: torch.dtype,
) -> LlamaForCausalLM:
    """Load one unsharded safetensors Llama checkpoint through an audited key map."""

    try:
        from safetensors.torch import load_file
    except ImportError as error:  # pragma: no cover - exercised by dependency boundaries
        raise RuntimeError(
            "checkpoint loading requires the 'models' optional dependencies"
        ) from error

    directory = Path(checkpoint_dir)
    config = model_config_from_huggingface(directory)
    checkpoint_path = directory / "model.safetensors"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint tensor file does not exist: {checkpoint_path}")

    with torch.device("meta"):
        model = LlamaForCausalLM(config)
    source = load_file(str(checkpoint_path), device="cpu")
    expected = model.state_dict()
    mapped: dict[str, torch.Tensor] = {}
    consumed: set[str] = set()
    for target_key in expected:
        source_key = _huggingface_key(target_key)
        if (
            source_key not in source
            and target_key == "output_projection.weight"
            and config.tie_word_embeddings
        ):
            source_key = "model.embed_tokens.weight"
        try:
            mapped[target_key] = source[source_key]
        except KeyError as error:
            raise ValueError(f"checkpoint is missing tensor: {source_key}") from error
        consumed.add(source_key)

    unexpected = sorted(set(source) - consumed)
    if unexpected:
        raise ValueError(f"checkpoint has unsupported tensors: {unexpected}")
    model.load_state_dict(mapped, strict=True, assign=True)
    inv_freq = 1.0 / (
        config.rope_theta
        ** (torch.arange(0, config.head_dim, 2, dtype=torch.float32) / config.head_dim)
    )
    for layer in model.model.layers:
        layer.attention.rotary.inv_freq = inv_freq.clone()
    model.to(device=torch.device(device), dtype=dtype)
    model.eval()
    return model


def _huggingface_key(key: str) -> str:
    key = key.replace("model.embedding", "model.embed_tokens")
    key = key.replace(".attention.", ".self_attn.")
    key = key.replace(".input_norm.", ".input_layernorm.")
    key = key.replace(".post_attention_norm.", ".post_attention_layernorm.")
    key = key.replace("model.final_norm", "model.norm")
    return key.replace("output_projection", "lm_head")


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(f"checkpoint config does not exist: {path}") from None
    if not isinstance(value, dict):
        raise ValueError("checkpoint config must contain a JSON object")
    return value


def _integer(raw: dict[str, Any], name: str) -> int:
    value = raw[name]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"checkpoint field {name} must be an integer")
    return value
