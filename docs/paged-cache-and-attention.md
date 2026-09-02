# Phase 4 paged cache and GQA decode-attention design

## Ownership boundary

`PagedKVCache` is the sole owner of physical KV pages, request slots, block tables, committed
lengths, free lists, and append transactions. The scheduler may decide which request runs, but it
must not mutate page ownership. The attention kernel consumes only tensor views: a query, one
layer's physical K/V pages, logical block-table rows, and context lengths. It never sees request IDs
or allocator objects.

The cache storage layout is:

```text
key/value: [layer, physical_block, token_offset, kv_head, head_dim]
block table: [sequence_slot, logical_block] -> physical_block
lengths: [sequence_slot] -> committed token count
```

For logical token position `p`, the control plane computes logical block `p // block_size` and
offset `p % block_size`. The block table supplies the physical block. Pages owned by one request may
be arbitrarily noncontiguous; neither materialization nor the Triton kernel assumes adjacency.

## Transaction protocol

The public lifecycle is:

1. `create_sequence(request_id)` reserves a stable sequence slot;
2. `begin_append(request_id, token_count)` atomically reserves every additional page or changes
   nothing and raises `CacheExhaustedError`;
3. `write_layer(reservation, layer, keys, values)` writes the reserved positions for each layer;
4. `layer_view(..., reservation)` exposes the pending end position so the current token can attend
   to itself without prematurely committing the sequence;
5. `commit(reservation)` publishes the new length only after every layer is written;
6. `rollback(reservation)` restores the old logical boundary and returns pages acquired by the
   failed transaction;
7. `release(request_id)` first cancels pending work, then returns every page and the sequence slot.

`commit(..., accepted_tokens=k)` retains only a verified prefix of a multi-token reservation and
immediately frees tail pages. This is cache mechanism, not speculative-decoding policy. Phase 7 can
use it without adding a second cache implementation.

Physical contents are not zeroed on rollback or release because block tables and lengths make stale
positions unreachable. Ownership metadata is changed before a page is returned to the free heap;
new owners overwrite positions before committing them. Tests verify deterministic reuse, atomic
exhaustion, active-reservation cancellation, partial commit, and 500 steps of seeded request churn.

## One-pass paged GQA kernel

The common path launches one Triton program per `(batch, query_head)`. A query head maps to KV head
`query_head // (num_query_heads // num_kv_heads)`, so K/V are never physically repeated for GQA.
Each program follows the logical block table, loads masked token/head tiles, evaluates QK dot
products in FP32, and maintains an online-softmax triple:

```text
running_max
running_sum = sum(exp(score - running_max))
accumulator = sum(exp(score - running_max) * value)
```

Combining each token tile rescales the previous sum and accumulator when the maximum increases.
Working memory is `O(head_dim)` per program rather than `O(context_length)`. Tail tokens, partial
pages, non-power-of-two head dimensions, and context buckets are masked. Output is converted back to
FP32, FP16, or BF16 storage only at the final store.

Autotuning searches token tiles 16/32/64 and 4/8 warps, keyed by query/KV head counts, head dimension,
and context bucket. Context lengths remain device tensors. Supplying a scheduler-owned
`max_context_length` bucket avoids a GPU-to-host `.item()` synchronization; omitting it uses a safe
dynamic fallback that is not graph-capture eligible.

## Split-KV long-context path

Measurements showed that one-pass attention regressed for batch 1, long context, and only 8–32
query heads: too few programs serially scanned 2,048 tokens. The shape-aware path therefore splits
contexts of at least 1,024 tokens into 256-token partitions when `batch * query_heads <= 32`.

The first kernel produces a partial maximum, exponential sum, and FP32 value accumulator for every
`(batch, query_head, split)`. A second kernel combines partitions using the same log-sum-exp
correction. This increases parallelism without materializing attention scores. The dispatch decision
and selected split count are serialized in benchmark results.

## Stable-buffer interface

`PagedAttentionWorkspace.allocate(...)` creates caller-owned FP32 split buffers. The attention API
also accepts a caller-owned output tensor. Passing the workspace, output, and explicit context bucket
removes wrapper allocations and host synchronization while keeping tensor addresses stable. Phase 6
may capture this canonical path in CUDA Graphs; Phase 4 only verifies storage reuse and does not make
a graph-speedup claim.

The kernel already accepts multiple block-table rows and unequal context lengths. Phase 5 continuous
batching can gather scheduler-selected rows into fixed bucket buffers while retaining one physical
cache and one attention implementation.

## Correctness and benchmark boundary

The readable oracle reconstructs logical K/V order from arbitrary physical pages, maps GQA heads,
computes QK and softmax in FP32, and accumulates values in FP32. GPU tests cover FP32/FP16/BF16,
head dimensions 64/80/128, GQA ratios from 2:1 through 8:1, contexts 1–2,048, unequal batch lengths,
partial pages, and noncontiguous block tables.

`PagedDecodeExecutor` processes one request token transactionally through every Llama layer. Tests
compare its complete logits at every token against the Phase 1 model recomputing the full prefix,
including RoPE positions and the current token's cache visibility.

The Phase 4 microbenchmark compares preallocated Triton execution with the Phase 1-equivalent
PyTorch path over contiguous logical K/V. PyTorch head expansion, score matmul, FP32 softmax, and
value aggregation are inside its timed region. This is a semantic baseline, not a claim of parity
with PyTorch SDPA, FlashAttention, vLLM, or another paged kernel. Logical bytes are a work model, not
Nsight Compute DRAM transactions.

## Explicit limitations

- The integrated executor is intentionally single-request; scheduler lifecycle belongs to Phase 5.
- Prompt ingestion currently uses repeated decode steps, not a dedicated prefill-attention kernel.
- Page allocation is a host-serialized control-plane operation; kernels do not allocate pages.
- Prefix sharing, copy-on-write pages, quantized KV, sliding windows, and multi-GPU cache sharding
  are not implemented.
- Split-KV workspace memory is explicit and must be budgeted by a future executor.
