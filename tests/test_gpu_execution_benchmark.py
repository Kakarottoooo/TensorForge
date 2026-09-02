from __future__ import annotations

from pathlib import Path

import pytest
import torch

from tensorforge.benchmark.execution_benchmark import (
    ExecutionCase,
    ExecutionManifest,
    ExecutionVariant,
    run_execution_suite,
    write_execution_reports,
)
from tensorforge.benchmark.schema import ModelSpec
from tensorforge.runtime.decode_bucket import DecodeExecutionMode, DecodeFusionLevel

pytestmark = [pytest.mark.cuda, pytest.mark.triton]


def test_cumulative_suite_compares_dynamic_fused_and_graph_paths(tmp_path: Path) -> None:
    pytest.importorskip("triton")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    manifest = ExecutionManifest(
        model=ModelSpec(
            vocab_size=128,
            hidden_size=64,
            intermediate_size=160,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=8,
        ),
        cases=(
            ExecutionCase(
                name="smoke",
                batch_size=1,
                initial_context=4,
                decode_steps=2,
                batch_buckets=(1,),
                context_buckets=(8,),
                setup_method="reference_prefill",
            ),
        ),
        num_blocks=4,
        warmup_repetitions=1,
        measured_repetitions=1,
        variants=(
            ExecutionVariant("dynamic", DecodeExecutionMode.EAGER, stable_bucket=False),
            ExecutionVariant(
                "fused",
                DecodeExecutionMode.EAGER,
                DecodeFusionLevel.ALL,
                parent="dynamic",
            ),
            ExecutionVariant(
                "graph",
                DecodeExecutionMode.CUDA_GRAPH,
                DecodeFusionLevel.ALL,
                parent="fused",
            ),
        ),
    )

    result = run_execution_suite(
        manifest,
        device=torch.device("cuda"),
        repository=Path(__file__).resolve().parents[1],
    )
    variants = result["cases"][0]["variants"]

    assert [item["variant"]["name"] for item in variants] == [
        "dynamic",
        "fused",
        "graph",
    ]
    assert variants[0]["aggregate"]["address_stable_all_runs"] is None
    assert variants[1]["aggregate"]["address_stable_all_runs"] is True
    assert variants[2]["aggregate"]["measured_metric_totals"]["graph_hits"] == 2
    paths = write_execution_reports(result, tmp_path)
    assert all(path.is_file() for path in paths)
