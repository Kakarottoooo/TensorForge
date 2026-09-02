from __future__ import annotations

import random

import pytest
import torch

from tensorforge.cache.paged import CacheExhaustedError, PagedKVCache, PagedKVCacheConfig


def test_append_crosses_logical_blocks_and_preserves_token_order() -> None:
    config = PagedKVCacheConfig(
        num_layers=1,
        num_blocks=4,
        block_size=2,
        num_kv_heads=2,
        head_dim=4,
        max_sequences=2,
        max_sequence_length=8,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    cache = PagedKVCache(config)
    cache.create_sequence("request-a")

    reservation = cache.begin_append("request-a", token_count=3)
    keys = torch.arange(24, dtype=torch.float32).view(3, 2, 4)
    values = keys + 100
    cache.write_layer(reservation, layer_index=0, keys=keys, values=values)
    cache.commit(reservation)

    view = cache.layer_view("request-a", layer_index=0)
    assert view.context_length == 3
    assert view.block_table.tolist()[:2] == [0, 1]
    actual_keys, actual_values = cache.materialize("request-a", layer_index=0)
    torch.testing.assert_close(actual_keys, keys)
    torch.testing.assert_close(actual_values, values)


def test_exhausted_append_is_atomic() -> None:
    config = PagedKVCacheConfig(
        num_layers=1,
        num_blocks=2,
        block_size=2,
        num_kv_heads=1,
        head_dim=4,
        max_sequences=2,
        max_sequence_length=8,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    cache = PagedKVCache(config)
    cache.create_sequence("owner")
    reservation = cache.begin_append("owner", 3)
    values = torch.randn(3, 1, 4)
    cache.write_layer(reservation, 0, values, values)
    cache.commit(reservation)
    before = cache.stats()
    cache.create_sequence("waiting")

    with pytest.raises(CacheExhaustedError, match="only 0"):
        cache.begin_append("waiting", 1)

    after = cache.stats()
    assert after.used_blocks == before.used_blocks
    assert after.reserved_tokens == 0
    assert cache.layer_view("owner", 0).context_length == 3
    assert cache.layer_view("waiting", 0).context_length == 0


def test_rollback_returns_new_pages_for_deterministic_reuse() -> None:
    config = PagedKVCacheConfig(
        num_layers=1,
        num_blocks=3,
        block_size=2,
        num_kv_heads=1,
        head_dim=4,
        max_sequences=2,
        max_sequence_length=6,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    cache = PagedKVCache(config)
    cache.create_sequence("cancelled")
    cancelled = cache.begin_append("cancelled", 3)
    assert cache.layer_view("cancelled", 0, cancelled).block_table.tolist()[:2] == [0, 1]

    cache.rollback(cancelled)
    assert cache.stats().free_blocks == 3
    assert cache.layer_view("cancelled", 0).block_table.tolist()[:2] == [-1, -1]

    cache.create_sequence("replacement")
    replacement = cache.begin_append("replacement", 3)
    assert cache.layer_view("replacement", 0, replacement).block_table.tolist()[:2] == [0, 1]


def test_partial_commit_keeps_verified_prefix_and_reclaims_tail_pages() -> None:
    config = PagedKVCacheConfig(
        num_layers=1,
        num_blocks=3,
        block_size=2,
        num_kv_heads=1,
        head_dim=2,
        max_sequences=1,
        max_sequence_length=6,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    cache = PagedKVCache(config)
    cache.create_sequence("speculative")
    reservation = cache.begin_append("speculative", 3)
    keys = torch.arange(6, dtype=torch.float32).view(3, 1, 2)
    cache.write_layer(reservation, 0, keys, -keys)

    cache.commit(reservation, accepted_tokens=1)

    assert cache.layer_view("speculative", 0).context_length == 1
    assert cache.stats().used_blocks == 1
    assert cache.layer_view("speculative", 0).block_table.tolist()[:2] == [0, -1]
    actual_keys, _ = cache.materialize("speculative", 0)
    torch.testing.assert_close(actual_keys, keys[:1])


def test_release_cancels_pending_append_and_returns_sequence_slot() -> None:
    config = PagedKVCacheConfig(
        num_layers=1,
        num_blocks=3,
        block_size=2,
        num_kv_heads=1,
        head_dim=2,
        max_sequences=1,
        max_sequence_length=6,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    cache = PagedKVCache(config)
    cache.create_sequence("cancelled")
    committed = cache.begin_append("cancelled", 2)
    values = torch.randn(2, 1, 2)
    cache.write_layer(committed, 0, values, values)
    cache.commit(committed)
    cache.begin_append("cancelled", 2)

    cache.release("cancelled")

    assert cache.stats().free_blocks == 3
    assert cache.stats().active_sequences == 0
    assert cache.block_owners() == (None, None, None)
    cache.create_sequence("replacement")
    assert cache.layer_view("replacement", 0).block_table.tolist() == [-1, -1, -1]


def test_optimized_writer_metadata_resolves_mapping_and_allows_commit() -> None:
    config = PagedKVCacheConfig(
        num_layers=2,
        num_blocks=3,
        block_size=2,
        num_kv_heads=1,
        head_dim=2,
        max_sequences=1,
        max_sequence_length=6,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    cache = PagedKVCache(config)
    cache.create_sequence("optimized")
    reservation = cache.begin_append("optimized", 3)

    assert cache.sequence_layout("optimized", reservation).physical_blocks == (0, 1)
    assert cache.sequence_layout("optimized", reservation).context_length == 3
    assert cache.append_location(reservation, 0).physical_block == 0
    assert cache.append_location(reservation, 2).physical_block == 1
    assert cache.append_location(reservation, 2).block_offset == 0
    for layer_index in range(config.num_layers):
        cache.record_layer_write(reservation, layer_index)
    cache.commit(reservation)

    assert cache.sequence_layout("optimized").context_length == 3
    assert cache.stats().reserved_tokens == 0


def test_seeded_request_churn_preserves_allocator_accounting() -> None:
    config = PagedKVCacheConfig(
        num_layers=2,
        num_blocks=8,
        block_size=4,
        num_kv_heads=2,
        head_dim=4,
        max_sequences=4,
        max_sequence_length=16,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    cache = PagedKVCache(config)
    randomizer = random.Random(20260901)
    lengths: dict[str, int] = {}
    next_request = 0

    for _ in range(500):
        if len(lengths) < config.max_sequences and (not lengths or randomizer.random() < 0.3):
            request_id = f"request-{next_request}"
            next_request += 1
            cache.create_sequence(request_id)
            lengths[request_id] = 0
        else:
            request_id = randomizer.choice(list(lengths))
            if randomizer.random() < 0.25:
                cache.release(request_id)
                del lengths[request_id]
            else:
                remaining = config.max_sequence_length - lengths[request_id]
                if remaining:
                    token_count = randomizer.randint(1, min(5, remaining))
                    try:
                        reservation = cache.begin_append(request_id, token_count)
                    except CacheExhaustedError:
                        pass
                    else:
                        values = torch.randn(token_count, config.num_kv_heads, config.head_dim)
                        for layer_index in range(config.num_layers):
                            cache.write_layer(reservation, layer_index, values, values)
                        accepted = randomizer.randint(0, token_count)
                        cache.commit(reservation, accepted_tokens=accepted)
                        lengths[request_id] += accepted

        stats = cache.stats()
        owners = cache.block_owners()
        assert stats.used_blocks == sum(owner is not None for owner in owners)
        assert stats.free_blocks + stats.used_blocks == stats.total_blocks
        assert stats.active_sequences == len(lengths)
        assert stats.committed_tokens == sum(lengths.values())
        assert stats.reserved_tokens == 0

    for request_id in list(lengths):
        cache.release(request_id)
    assert cache.stats().free_blocks == config.num_blocks
