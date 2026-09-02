from __future__ import annotations

import os
import random
from pathlib import Path

import pytest
import torch

from tensorforge.cache import PagedKVCache, PagedKVCacheConfig
from tensorforge.model.config import tiny_config
from tensorforge.model.huggingface import load_huggingface_checkpoint
from tensorforge.model.llama import LlamaForCausalLM
from tensorforge.runtime.batched_decode import BatchedPagedDecodeExecutor
from tensorforge.runtime.paged_decode import PagedDecodeExecutor
from tensorforge.runtime.prefill import prefill_request, prefill_requests


def test_parallel_prefill_matches_full_prefix_and_populates_paged_cache() -> None:
    torch.manual_seed(20260903)
    config = tiny_config(max_position_embeddings=32)
    model = LlamaForCausalLM(config).eval()
    cache = PagedKVCache(
        PagedKVCacheConfig(
            num_layers=config.num_hidden_layers,
            num_blocks=8,
            block_size=4,
            num_kv_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            max_sequences=2,
            max_sequence_length=config.max_position_embeddings,
            dtype=torch.float32,
            device=torch.device("cpu"),
        )
    )
    cache.create_sequence("request")
    prompt = (1, 7, 3, 11, 5, 9)

    actual = prefill_request(model, cache, "request", prompt)

    expected = model(torch.tensor([prompt], dtype=torch.long))[:, -1]
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    assert cache.sequence_layout("request").context_length == len(prompt)
    keys, values = cache.materialize("request", layer_index=0)
    assert keys.shape == (len(prompt), config.num_key_value_heads, config.head_dim)
    assert values.shape == keys.shape


def test_variable_length_batched_prefill_matches_each_full_prefix() -> None:
    torch.manual_seed(20260904)
    config = tiny_config(max_position_embeddings=32)
    model = LlamaForCausalLM(config).eval()
    cache = PagedKVCache(
        PagedKVCacheConfig(
            num_layers=config.num_hidden_layers,
            num_blocks=12,
            block_size=4,
            num_kv_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            max_sequences=3,
            max_sequence_length=config.max_position_embeddings,
            dtype=torch.float32,
            device=torch.device("cpu"),
        )
    )
    prompts = {"short": (1, 2, 3), "long": (4, 5, 6, 7, 8, 9)}
    for request_id in prompts:
        cache.create_sequence(request_id)

    result = prefill_requests(model, cache, prompts)

    assert result.errors == {}
    assert set(result.logits) == set(prompts)
    for request_id, prompt in prompts.items():
        expected = model(torch.tensor([prompt], dtype=torch.long))[:, -1]
        torch.testing.assert_close(
            result.logits[request_id].unsqueeze(0), expected, rtol=1e-5, atol=1e-6
        )
        assert cache.sequence_layout(request_id).context_length == len(prompt)


@pytest.mark.cuda
@pytest.mark.triton
def test_real_checkpoint_prefill_then_paged_decode_matches_full_prefix() -> None:
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
    prompt = tuple(
        tokenizer("The capital of France is", return_tensors="pt").input_ids[0].tolist()
    )
    model = load_huggingface_checkpoint(checkpoint, device="cuda", dtype=dtype)
    config = model.config
    cache = PagedKVCache(
        PagedKVCacheConfig(
            num_layers=config.num_hidden_layers,
            num_blocks=128,
            block_size=16,
            num_kv_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            max_sequences=1,
            max_sequence_length=config.max_position_embeddings,
            dtype=dtype,
            device=torch.device("cuda"),
        )
    )
    cache.create_sequence("real")

    prefill_logits = prefill_request(model, cache, "real", prompt)
    prompt_tensor = torch.tensor([prompt], dtype=torch.long, device="cuda")
    expected_prefill = model(prompt_tensor)[:, -1]
    torch.testing.assert_close(prefill_logits, expected_prefill, rtol=3e-2, atol=2.5e-1)

    next_token = int(prefill_logits.argmax(dim=-1).item())
    decode_logits = PagedDecodeExecutor(model, cache).append_token("real", next_token)
    expected_decode = model(
        torch.cat(
            (prompt_tensor, torch.tensor([[next_token]], device="cuda")), dim=1
        )
    )[:, -1]
    torch.testing.assert_close(decode_logits, expected_decode, rtol=5e-2, atol=5e-1)
    assert cache.sequence_layout("real").context_length == len(prompt) + 1


@pytest.mark.cuda
@pytest.mark.triton
def test_real_checkpoint_teacher_forced_long_decode_matches_transformers() -> None:
    checkpoint_value = os.environ.get("TENSORFORGE_REAL_MODEL_DIR")
    if checkpoint_value is None:
        pytest.skip("set TENSORFORGE_REAL_MODEL_DIR to run the real-checkpoint gate")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the real-checkpoint gate")
    transformers = pytest.importorskip("transformers")
    checkpoint = Path(checkpoint_value)
    dtype = torch.bfloat16
    model = load_huggingface_checkpoint(checkpoint, device="cuda", dtype=dtype)
    reference = transformers.AutoModelForCausalLM.from_pretrained(
        checkpoint,
        local_files_only=True,
        dtype=dtype,
        attn_implementation="sdpa",
    ).cuda().eval()
    config = model.config
    cache = PagedKVCache(
        PagedKVCacheConfig(
            num_layers=config.num_hidden_layers,
            num_blocks=256,
            block_size=16,
            num_kv_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            max_sequences=1,
            max_sequence_length=config.max_position_embeddings,
            dtype=dtype,
            device=torch.device("cuda"),
        )
    )
    executor = BatchedPagedDecodeExecutor(model, cache)
    executor.create_request("teacher-forced")
    generator = random.Random(20260908)
    prompt = tuple(generator.randrange(config.vocab_size) for _ in range(128))
    prompt_tensor = torch.tensor([prompt], dtype=torch.long, device="cuda")

    with torch.inference_mode():
        actual = prefill_request(model, cache, "teacher-forced", prompt)[0]
        reference_output = reference(input_ids=prompt_tensor, use_cache=True)
        expected = reference_output.logits[0, -1]
        past_key_values = reference_output.past_key_values
        for _ in range(8):
            torch.testing.assert_close(actual, expected, rtol=3e-2, atol=2.5e-1)
            expected_token = int(expected.argmax())
            if int(actual.argmax()) != expected_token:
                top_two = torch.topk(expected.float(), 2).values
                assert float(top_two[0] - top_two[1]) <= 0.125
            actual = executor.append_tokens({"teacher-forced": expected_token}).logits[
                "teacher-forced"
            ]
            reference_output = reference(
                input_ids=torch.tensor([[expected_token]], device="cuda"),
                past_key_values=past_key_values,
                use_cache=True,
            )
            expected = reference_output.logits[0, -1]
            past_key_values = reference_output.past_key_values

    assert cache.sequence_layout("teacher-forced").context_length == 136
