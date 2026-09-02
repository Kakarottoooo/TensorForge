# Phase 6 validation record

Date: 2026-09-01 (America/Los_Angeles) / 2026-09-02 UTC

## Delivered boundary

Phase 6 adds one `BucketedPagedDecodeExecutor` behind the existing scheduler executor contract.
Each `(batch_capacity, context_capacity)` bucket owns persistent pinned-host controls, device
controls, block tables, context lengths, append destinations, active masks, logits, attention
outputs, and split-KV workspaces. The scheduler still owns lifecycle and selection;
`PagedKVCache` still owns every logical sequence, physical page, reservation, commit, rollback, and
release.

The cache exposes CPU-owned sequence layout and append-location metadata so staging does not read a
GPU scalar. A capture-safe Triton writer consumes device-side physical blocks and offsets. Padded
lanes are store-masked. Eager, segmented `torch.compile`, and explicit CUDA Graph execution all use
that path. Oversized batch or context dimensions delegate to the pre-existing dynamic eager
executor and retain the exact fallback reason.

`torch.compile(dynamic=False)` specializes PyTorch model segments while the custom Triton KV writer
and paged attention remain explicit graph boundaries. Inductor CUDA Graphs are disabled for that
row. The CUDA Graph row performs side-stream warmup then one explicit capture per bucket. Cache
transactions and two pinned-host control copies remain outside capture.

## Correctness and lifecycle evidence

FP32 and FP16 tests compare repeated eager, compile, capture, and replay logits with independent
full-prefix model execution. Address maps must remain identical across appends. A padded two-lane
graph with only one active request proves the inactive writer lane is masked and that release
returns every page. Batch overflow proves dynamic eager fallback and records `batch_overflow`.

An end-to-end continuous scheduler test runs three heterogeneous requests through a two-lane graph
bucket. Requests finish and refill the bucket while logical-to-physical mappings change. Generated
tokens match independent full-prefix greedy generation; one capture is reused, graph hits are
nonzero, and cache state returns to zero. The CPU cache test independently checks layout,
cross-block append locations, optimized layer-write accounting, commit, and reserved-token cleanup.

## Curated experiment

The immutable input is `benchmarks/phase6-execution.json`. Suite
`5cdd5b28-5937-46cb-9753-d7077672c0ce` measured clean commit
`9160667c34f2e3b6299649e52c4dcf89917db838`. Hardware/software identity is retained in JSON: RTX
3080 Ti (compute capability 8.6), PyTorch 2.5.1+cu121, CUDA runtime 12.1, Triton 3.1.0, and WSL2.
The hardware fingerprint is
`77278e66c8912b496682306deb107db0d1c325287febc4504b8bf666e266ea0c`.

The model has four layers, width 256, eight query heads, four KV heads, FP16 weights/cache,
16-token pages, and 64 physical blocks. Each case uses one excluded warmup and five seed-shuffled
measured repetitions of 16 decode steps after an eight-token setup prefix. CUDA events bracket each
append. Host throughput includes cache transactions, two pinned-control copies, dispatch, and the
final synchronization. Every run passes an independent full-prefix logit gate.

| Case | Mode | CUDA P50/P95/P99 ms | Mean tok/s | Mean latency change vs eager | Measured graph hit/miss | Shape fallbacks | Cold setup/capture/compile ms |
|---|---|---:|---:|---:|---:|---:|---:|
| B1 / C32 | Eager | 16.227/17.835/18.021 | 112.91 | control | 0/0 | 0 | 1374.68/0/0 |
| B1 / C32 | `torch.compile` | 9.280/10.185/10.767 | 156.43 | -38.1% | 0/0 | 0 | 5139.09/0/5105.76 |
| B1 / C32 | CUDA Graph | 1.125/1.748/1.930 | 886.51 | -90.4% | 80/0 | 0 | 164.05/146.17/0 |
| B4 / C32 | Eager | 9.726/18.193/18.615 | 390.58 | control | 0/0 | 0 | 584.18/0/0 |
| B4 / C32 | `torch.compile` | 6.604/10.483/10.843 | 592.78 | -35.8% | 0/0 | 0 | 2851.31/0/2776.64 |
| B4 / C32 | CUDA Graph | 1.482/2.325/2.445 | 2624.08 | -85.8% | 80/0 | 0 | 189.99/159.13/0 |
| B8 / C32 | Eager | 18.910/21.364/22.291 | 588.60 | control | 0/0 | 0 | 60.83/0/0 |
| B8 / C32 | `torch.compile` | 11.125/14.717/15.339 | 1007.65 | -42.8% | 0/0 | 0 | 1941.21/0/1905.09 |
| B8 / C32 | CUDA Graph | 3.418/4.547/4.980 | 2679.36 | -80.0% | 80/0 | 0 | 173.20/153.64/0 |
| B3 overflow | Eager fallback | 21.006/26.671/27.504 | 188.19 | control | 0/0 | 80 | 2356.91/0/0 |
| B3 overflow | Compile-labelled fallback | 19.686/25.910/29.976 | 250.80 | not attributable | 0/0 | 80 | 47.58/0/0 |
| B3 overflow | Graph-labelled fallback | 19.858/23.016/24.147 | 236.12 | not attributable | 0/80 | 80 | 55.66/0/0 |

For eligible shapes, explicit CUDA Graph reduced mean step latency by 80.0–90.4% and increased mean
host throughput by 4.55–7.85x versus eager in this suite. Segmented compile reduced mean step
latency by 35.8–42.8%, but its first process-local compilation cost was 1.91–5.11 seconds. CUDA
Graph capture cost was 146–159 ms, with total cold setup of 164–190 ms. Each measured eligible
graph row recorded 80 hits and zero misses; each setup repetition separately recorded one capture
miss followed by seven prefix replays.

The shape-fallback rows all execute the exact same dynamic eager implementation. Their different
latencies and throughput are order/interference variance, not mode speedups. The graph-labelled row
correctly recorded zero hits, 80 misses, and 80 batch-overflow fallbacks. No stable-bucket address
claim is made for fallback rows because no bucket exists.

## Variance and setup boundary

This RTX 3080 Ti is a shared Windows display GPU. Step distributions were broad: for example B1
eager ranged from 4.84 to 18.08 ms, and the identical-path overflow controls diverged materially.
The report retains all 80 per-step samples per mode and does not substitute the best run. GPU state
was 55 C, about 125 W, and 1,755 MHz SM before the suite; it ended at 55 C, about 124 W, and 1,755
MHz, with memory clock changing from 9,501 to 9,251 MHz.

Cold setup includes first-use kernel work as well as capture or compile. Cross-case compiler and
kernel caches remain process-local, so cold numbers are order-dependent and are not process-start
SLOs for every row. Steady-state excludes setup. Returned logit tensors are stable views and are
overwritten by the next call; consumers that retain them must clone.

## Verification

```text
Windows control environment:
python -m ruff check .
All checks passed

python -m mypy tensorforge scripts
Success: no issues found in 48 source files

python -m pytest -q
68 passed, 91 skipped

WSL CUDA/Triton environment:
python -m pytest -q
159 passed
```

Windows skips Triton-only tests. WSL runs the complete CPU, CUDA, custom-kernel, cache, scheduler,
bucket, compile, and explicit graph suite. The `torch.compile` test clears a pytest-only structured
trace handler because PyTorch 2.5 otherwise invokes unavailable `nvcc --version` while rendering an
optional debug artifact; runtime errors are not suppressed.

## Claim boundary and next gate

Phase 6 does not claim full-model `torch.compile`, graph-captured cache allocation or scheduler
logic, asynchronous request selection, isolated-datacenter variance, production prefill, vLLM
parity, speculative decoding, or multi-GPU scaling. It establishes an address-stable decode
substrate with explicit, measured eligibility and fallback semantics.

Phase 7 should implement speculative draft/target verification as a multi-token transaction on the
same `PagedKVCache`: proposal append, accepted-prefix commit, rejected-tail rollback, replacement
token append, and cache/full-prefix equivalence must precede performance claims.
