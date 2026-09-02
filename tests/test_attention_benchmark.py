from __future__ import annotations

import json
from pathlib import Path

import pytest

from tensorforge.benchmark.attention_benchmark import load_attention_manifest


def test_attention_manifest_rejects_unknown_case_fields(tmp_path: Path) -> None:
    path = tmp_path / "attention.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "cases": [
                    {
                        "name": "decode",
                        "batch": 1,
                        "query_heads": 8,
                        "kv_heads": 4,
                        "head_dim": 64,
                        "context_length": 128,
                        "block_size": 16,
                        "precision": "fp16",
                        "uncontrolled": True,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unknown attention case keys"):
        load_attention_manifest(path)
