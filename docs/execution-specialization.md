# Address-stable decode execution

Phase 6 specializes the existing `BatchTokenExecutor` boundary; it does not introduce a second
cache or scheduler. `BucketedPagedDecodeExecutor` owns persistent buffers for each selected
`(batch_capacity, context_capacity)` pair, while `PagedKVCache` remains the only authority for
logical sequence length, physical page ownership, append transactions, rollback, and release.

## Stable bucket contract

Each bucket allocates its token IDs, positions, block table, context lengths, append destinations,
active mask, logits, attention outputs, and split-KV workspaces once. CPU cache metadata is staged
through two persistent pinned slabs. Tensor addresses therefore remain stable across scheduler
refill and physical-page reuse; the values and logical-to-physical mapping may change on every
replay.

Fusion levels extend the same bucket with caller-owned input-norm, residual, post-attention norm,
hidden-state, SwiGLU activation, and final-norm outputs. `rms_norm` replaces every norm while keeping
the residual add separate; `residual_rms_norm` additionally fuses the attention residual and MLP
input norm; `all` also replaces the SwiGLU activation. Linear projections remain PyTorch/cuBLAS.

Inactive lanes have context length one for a legal attention launch but an `active_mask` of zero.
The masked Triton KV writer performs no store for those lanes. Only active logit views are returned.
Returned views are ephemeral and are overwritten by the next execution; callers retaining logits
must clone them.

## Three execution modes

- `eager` executes the same fixed-shape bucket directly.
- `torch_compile` applies `torch.compile(dynamic=False)` to PyTorch model segments. Inductor's
  internal CUDA Graph option is disabled so this row does not silently become a graph row. Custom
  Triton KV-write and paged-attention launches remain explicit graph boundaries.
- `cuda_graph` performs one side-stream warmup and one explicit `torch.cuda.CUDAGraph` capture per
  bucket. The first call is a miss/capture; later calls replay and count as hits.

Compile and CUDA Graph are alternative children of fully fused eager execution. They are not
stacked or compared as if one causally contains the other.

Cache reservation/commit/rollback, pinned-host staging, request selection, and bucket lookup remain
outside capture. Captured work begins after the asynchronous staging copies and includes model
operators, masked KV stores, paged attention, and the stable logit copy.

## Shape fallback and observability

The smallest fitting batch and context capacities are selected independently. If either dimension
exceeds all configured buckets, execution delegates to the existing dynamic
`BatchedPagedDecodeExecutor`. The metrics retain `batch_overflow`, `context_overflow`, or
`batch_and_context_overflow`; CUDA Graph overflow calls are misses, never hits. Failures roll back
all active append reservations.

Metrics expose eager/compiled/graph calls, graph hit/miss, capture count and cost, warmup cost,
compile count and cost, padded lanes, dynamic eager fallbacks, and reason counts. Buffer addresses
are exposed for regression tests and benchmark evidence.

## Measurement boundary

`benchmarks/phase6-execution.json` is immutable input for the Phase 6 comparison. Setup fills the
initial prefix and absorbs compilation or capture. CUDA events then bracket every steady-state
append, while host throughput includes cache transactions, two pinned-control copies, dispatch,
and final synchronization. Mode order is seed-shuffled per repetition. Every run compares its first
measured logit with the independent full-prefix model before it is accepted.

The fallback control deliberately submits batch three to batch buckets one and two. It proves and
measures fallback behavior; it is not evidence for captured or compiled execution. The compile row
is a hybrid segmented path, so results must not be described as full-model compilation. Setup cost
is reported separately and every slowdown is retained.
