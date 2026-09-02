from __future__ import annotations

import json
from pathlib import Path

import pytest

from tensorforge.benchmark.schema import Precision
from tensorforge.benchmark.suite import load_manifest


def test_manifest_is_strict_and_typed(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "cases": [
                    {
                        "name": "case",
                        "model": {"vocab_size": 128, "hidden_size": 64},
                        "precision": "bf16",
                    }
                ],
            }
        )
    )

    workloads = load_manifest(path)

    assert workloads[0].precision == Precision.BF16
    assert workloads[0].model.vocab_size == 128


def test_manifest_rejects_unknown_keys(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "cases": [{"name": "case", "model": {}, "mystery": True}],
            }
        )
    )

    with pytest.raises(ValueError, match="unknown workload keys"):
        load_manifest(path)
