from __future__ import annotations

import json
from pathlib import Path

import pytest

from tensorforge.benchmark.execution_benchmark import load_execution_manifest
from tensorforge.runtime.decode_bucket import DecodeExecutionMode, DecodeFusionLevel


def test_phase6_manifest_covers_graph_hits_and_shape_fallback() -> None:
    repository = Path(__file__).resolve().parents[1]
    manifest = load_execution_manifest(repository / "benchmarks" / "phase6-execution.json")

    assert {case.name for case in manifest.cases} == {
        "eligible-b1",
        "eligible-b4",
        "eligible-b8",
        "batch-shape-fallback",
    }
    fallback = next(case for case in manifest.cases if case.name == "batch-shape-fallback")
    assert fallback.batch_size > max(fallback.batch_buckets)


def test_execution_manifest_rejects_unknown_case_fields(tmp_path: Path) -> None:
    payload = {
        "schema_version": "1.0",
        "model": {
            "vocab_size": 128,
            "hidden_size": 64,
            "intermediate_size": 160,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "max_position_embeddings": 64,
        },
        "cases": [
            {
                "name": "invalid",
                "batch_size": 1,
                "initial_context": 4,
                "decode_steps": 4,
                "batch_buckets": [1],
                "context_buckets": [16],
                "uncontrolled": True,
            }
        ],
    }
    path = tmp_path / "execution.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="unknown execution case keys"):
        load_execution_manifest(path)


def test_phase7a_manifest_defines_ordered_cumulative_variants_and_long_contexts() -> None:
    repository = Path(__file__).resolve().parents[1]
    manifest = load_execution_manifest(
        repository / "benchmarks" / "phase7a-cumulative.json"
    )

    assert [variant.name for variant in manifest.variants] == [
        "paged_dynamic_eager",
        "stable_bucket_eager",
        "triton_rms_norm",
        "triton_residual_rms_norm",
        "triton_all_fusions",
        "triton_all_compile",
        "triton_all_cuda_graph",
    ]
    assert manifest.variants[-1].mode is DecodeExecutionMode.CUDA_GRAPH
    assert manifest.variants[-1].fusion_level is DecodeFusionLevel.ALL
    assert manifest.variants[-1].parent == "triton_all_fusions"
    assert manifest.variants[0].stable_bucket is False
    assert {case.context_buckets[-1] for case in manifest.cases} == {
        32,
        128,
        512,
        2048,
    }
    assert all(case.setup_method == "reference_prefill" for case in manifest.cases)
