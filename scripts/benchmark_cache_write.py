"""Measure vectorized paged-cache prefill writes against the former scalar loop."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from tensorforge.benchmark.hardware import collect_hardware_metadata
from tensorforge.cache.paged import AppendReservation, PagedKVCache, PagedKVCacheConfig


def _cache(max_tokens: int, device: torch.device) -> PagedKVCache:
    return PagedKVCache(
        PagedKVCacheConfig(
            num_layers=1,
            num_blocks=(max_tokens + 15) // 16,
            block_size=16,
            num_kv_heads=4,
            head_dim=64,
            max_sequences=1,
            max_sequence_length=max_tokens,
            dtype=torch.float16,
            device=device,
        )
    )


def _scalar_reference(
    cache: PagedKVCache,
    reservation: AppendReservation,
    keys: Tensor,
    values: Tensor,
) -> None:
    layout = cache.sequence_layout(reservation.request_id, reservation)
    for source_index in range(reservation.token_count):
        position = reservation.start_position + source_index
        logical_block, block_offset = divmod(position, cache.config.block_size)
        physical_block = layout.physical_blocks[logical_block]
        cache.key_cache[0, physical_block, block_offset].copy_(keys[source_index])
        cache.value_cache[0, physical_block, block_offset].copy_(values[source_index])


def _time(operation: Callable[[], None], repetitions: int, device: torch.device) -> float:
    for _ in range(10):
        operation()
    torch.cuda.synchronize(device)
    event_factory: Any = torch.cuda.Event
    start = event_factory(enable_timing=True)
    end = event_factory(enable_timing=True)
    start.record()
    for _ in range(repetitions):
        operation()
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end)) / repetitions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--token-counts", type=int, nargs="+", default=(1, 32, 128, 512))
    parser.add_argument("--repetitions", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("cache-write benchmark requires CUDA")
    if args.repetitions <= 0 or any(count <= 0 for count in args.token_counts):
        raise ValueError("token counts and repetitions must be positive")

    rows = []
    for token_count in args.token_counts:
        scalar_cache = _cache(token_count, device)
        optimized_cache = _cache(token_count, device)
        scalar_cache.create_sequence("request")
        optimized_cache.create_sequence("request")
        scalar_reservation = scalar_cache.begin_append("request", token_count)
        optimized_reservation = optimized_cache.begin_append("request", token_count)
        keys = torch.randn(
            token_count, 4, 64, dtype=torch.float16, device=device
        )
        values = torch.randn_like(keys)
        scalar_ms = _time(
            partial(_scalar_reference, scalar_cache, scalar_reservation, keys, values),
            args.repetitions,
            device,
        )
        optimized_ms = _time(
            partial(
                optimized_cache.write_layer,
                optimized_reservation,
                0,
                keys,
                values,
            ),
            args.repetitions,
            device,
        )
        _scalar_reference(scalar_cache, scalar_reservation, keys, values)
        optimized_cache.write_layer(optimized_reservation, 0, keys, values)
        scalar_cache.record_layer_write(scalar_reservation, 0)
        scalar_cache.commit(scalar_reservation)
        optimized_cache.commit(optimized_reservation)
        scalar_materialized = scalar_cache.materialize("request", 0)
        optimized_materialized = optimized_cache.materialize("request", 0)
        torch.testing.assert_close(scalar_materialized[0], optimized_materialized[0])
        torch.testing.assert_close(scalar_materialized[1], optimized_materialized[1])
        rows.append(
            {
                "token_count": token_count,
                "scalar_reference_ms": scalar_ms,
                "optimized_ms": optimized_ms,
                "speedup": scalar_ms / optimized_ms,
            }
        )
    repository = Path(__file__).resolve().parents[1]
    report = {
        "schema_version": "1.0",
        "hardware": collect_hardware_metadata(repository).to_dict(),
        "repetitions": args.repetitions,
        "cache_shape": {
            "block_size": 16,
            "num_kv_heads": 4,
            "head_dim": 64,
            "dtype": "fp16",
        },
        "rows": rows,
        "correctness": "materialized key/value tensors match exactly",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    print(args.output.resolve())


if __name__ == "__main__":
    main()
