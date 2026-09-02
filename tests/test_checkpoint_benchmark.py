from __future__ import annotations

import json
from pathlib import Path

import pytest

from tensorforge.benchmark.checkpoint_backends import generated_token_digest
from tensorforge.benchmark.checkpoint_comparison import compare_checkpoint_reports
from tensorforge.benchmark.checkpoint_suite import load_checkpoint_manifest


def _manifest() -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "checkpoint": {
            "repository": "org/model",
            "revision": "a" * 40,
        },
        "runtime": {
            "block_size": 16,
            "num_blocks": 64,
            "mode": "eager",
            "fusion_level": "none",
            "batch_buckets": [1, 4],
            "context_buckets": [128, 256],
        },
        "cases": [
            {
                "name": "burst-b4",
                "model": {
                    "vocab_size": 128,
                    "hidden_size": 64,
                    "intermediate_size": 128,
                    "num_hidden_layers": 2,
                    "num_attention_heads": 4,
                    "num_key_value_heads": 2,
                    "max_position_embeddings": 256,
                },
                "prompt_length": 32,
                "generation_length": 8,
                "batch_size": 4,
                "concurrency": 4,
                "precision": "bf16",
                "warmup_repetitions": 1,
                "measured_repetitions": 2,
            }
        ],
    }


def test_load_checkpoint_manifest_requires_immutable_revision(tmp_path: Path) -> None:
    payload = _manifest()
    checkpoint = payload["checkpoint"]
    assert isinstance(checkpoint, dict)
    checkpoint["revision"] = "main"
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="40-character commit"):
        load_checkpoint_manifest(path)


def test_load_checkpoint_manifest_preserves_runtime_and_workload(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(_manifest()), encoding="utf-8")

    manifest = load_checkpoint_manifest(path)

    assert manifest.checkpoint.repository == "org/model"
    assert manifest.runtime.batch_buckets == (1, 4)
    assert manifest.runtime.context_buckets == (128, 256)
    assert manifest.workloads[0].batch_size == 4
    assert manifest.workloads[0].execution.backend == "tensorforge"


def test_generated_token_digest_is_request_order_independent_and_content_sensitive() -> None:
    first = generated_token_digest({"b": (3, 4), "a": (1, 2)})
    reordered = generated_token_digest({"a": (1, 2), "b": (3, 4)})
    changed = generated_token_digest({"a": (1, 2), "b": (3, 5)})

    assert first == reordered
    assert first != changed


def test_comparison_rejects_different_checkpoint_revisions(tmp_path: Path) -> None:
    base = {
        "hardware": {
            "benchmark_backend": "tensorforge",
            "checkpoint": {
                "repository": "org/model",
                "revision": "a" * 40,
                "config_sha256": "1" * 64,
                "weights_sha256": "2" * 64,
                "weights_size_bytes": 10,
            },
            "gpus": [{"name": "GPU", "total_memory_bytes": 1}],
        },
        "cases": [],
    }
    changed = json.loads(json.dumps(base))
    changed["hardware"]["benchmark_backend"] = "vllm"
    changed["hardware"]["checkpoint"]["revision"] = "b" * 40
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first.write_text(json.dumps(base), encoding="utf-8")
    second.write_text(json.dumps(changed), encoding="utf-8")

    with pytest.raises(ValueError, match="checkpoint identity"):
        compare_checkpoint_reports((first, second))
