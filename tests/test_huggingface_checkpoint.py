from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch

from tensorforge.model.config import tiny_config
from tensorforge.model.huggingface import load_huggingface_checkpoint
from tensorforge.model.llama import LlamaForCausalLM

safetensors = pytest.importorskip("safetensors.torch")


def _hf_key(key: str) -> str:
    key = key.replace("model.embedding", "model.embed_tokens")
    key = key.replace(".attention.", ".self_attn.")
    key = key.replace(".input_norm.", ".input_layernorm.")
    key = key.replace(".post_attention_norm.", ".post_attention_layernorm.")
    key = key.replace("model.final_norm", "model.norm")
    key = key.replace("output_projection", "lm_head")
    return key


def _write_checkpoint(path: Path, model: LlamaForCausalLM) -> None:
    config = model.config
    (path / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["LlamaForCausalLM"],
                "model_type": "llama",
                "hidden_act": "silu",
                "attention_bias": False,
                "vocab_size": config.vocab_size,
                "hidden_size": config.hidden_size,
                "intermediate_size": config.intermediate_size,
                "num_hidden_layers": config.num_hidden_layers,
                "num_attention_heads": config.num_attention_heads,
                "num_key_value_heads": config.num_key_value_heads,
                "max_position_embeddings": config.max_position_embeddings,
                "rms_norm_eps": config.rms_norm_eps,
                "rope_theta": config.rope_theta,
                "tie_word_embeddings": config.tie_word_embeddings,
            }
        ),
        encoding="utf-8",
    )
    tensors = {_hf_key(key): value.detach().clone() for key, value in model.state_dict().items()}
    safetensors.save_file(tensors, path / "model.safetensors")


def test_load_huggingface_checkpoint_preserves_full_prefix_logits(tmp_path: Path) -> None:
    torch.manual_seed(20260902)
    reference = LlamaForCausalLM(tiny_config()).eval()
    _write_checkpoint(tmp_path, reference)

    loaded = load_huggingface_checkpoint(tmp_path, device="cpu", dtype=torch.float32)

    input_ids = torch.tensor([[1, 7, 3, 11]], dtype=torch.long)
    torch.testing.assert_close(loaded(input_ids), reference(input_ids), rtol=0, atol=0)
    assert loaded.config == reference.config
    assert not loaded.training


@pytest.mark.cuda
def test_real_tinyllama_checkpoint_matches_transformers_logits() -> None:
    checkpoint_value = os.environ.get("TENSORFORGE_REAL_MODEL_DIR")
    if checkpoint_value is None:
        pytest.skip("set TENSORFORGE_REAL_MODEL_DIR to run the real-checkpoint gate")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the real-checkpoint gate")
    transformers = pytest.importorskip("transformers")
    checkpoint = Path(checkpoint_value)
    dtype = torch.bfloat16
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        checkpoint, local_files_only=True
    )
    input_ids = tokenizer("The capital of France is", return_tensors="pt").input_ids.cuda()
    reference = transformers.AutoModelForCausalLM.from_pretrained(
        checkpoint,
        local_files_only=True,
        dtype=dtype,
        attn_implementation="eager",
    ).cuda().eval()
    loaded = load_huggingface_checkpoint(checkpoint, device="cuda", dtype=dtype)

    with torch.inference_mode():
        expected = reference(input_ids).logits
        actual = loaded(input_ids)

    torch.testing.assert_close(actual, expected, rtol=3e-2, atol=2.5e-1)
    assert actual[:, -1].argmax(dim=-1).item() == expected[:, -1].argmax(dim=-1).item()
