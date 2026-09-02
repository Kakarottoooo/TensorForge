# Phase 5 validation record

Date: 2026-09-01 (America/Los_Angeles) / 2026-09-02 UTC

## Delivered boundary

Phase 5 adds an explicit `queued` → `prefill` → `decode` → `completed`/`failed` lifecycle plus a
distinct `cancelled` terminal state. Admission reserves prompt plus maximum output tokens against
request and global budgets before mutation. Completion, cancellation, executor failure, and cache
exhaustion all release request budget, sequence slot, active reservation, and committed physical
pages through the one canonical `PagedKVCache`.

`BatchedPagedDecodeExecutor` performs heterogeneous one-token GPU batches over shared model weights
and cache tensors. Each request reserves independently, so one request that needs an unavailable
page fails without poisoning a peer that fits. Model/kernel failure rolls back the batch's active
reservations. Per-request absolute RoPE positions use a batch-aware broadcast shape.

The scheduler exposes no/static/continuous policies without duplicating runtime or cache paths.
No batching admits one request; static batching waits for a cohort to drain; continuous batching
refills freed slots on every step.

## Correctness and stress evidence

CPU lifecycle tests require the exact state history, request/global budget rejection before
mutation, immediate cancellation reclamation, replacement admission, per-request failure isolation,
and distinct policy batch sequences. A deterministic 500-operation submit/cancel/execute churn run
checks global budget and maximum batch invariants after every operation, then requires zero queued,
active, executor, or budget state.

CUDA tests compare heterogeneous batched logits with independent full-prefix recomputation in
FP32, FP16, and BF16. A stricter FP32 comparison caught a cross-request RoPE broadcast defect that
low-precision tolerance could conceal; the final helper reshapes one position per batch item.
End-to-end continuous generation for three different request lengths matches independent
full-prefix greedy tokens and terminates with zero used pages.

A real one-block cache-exhaustion test submits two requests in one batch: the request that can append
inside its existing page commits, while the request requiring a new page receives
`CacheExhaustedError` and retains length zero. This establishes failure isolation with the actual
allocator and GPU executor rather than a mock.

## Curated experiment

The immutable input is `benchmarks/phase5-scheduler.json`. Suite
`d5b3a69c-a9b8-41d9-ac17-80583c261b1c` measured clean commit
`f3e5b4c1942a47246880c15fb240ce1ff6c0b482`. Hardware/software identity is retained in JSON: RTX
3080 Ti (12 GiB, compute capability 8.6), driver 591.86, PyTorch 2.5.1+cu121, CUDA runtime 12.1,
Triton 3.1.0, Python 3.11.13, and WSL2 Linux 6.6.87.2. The hardware fingerprint is
`77278e66c8912b496682306deb107db0d1c325287febc4504b8bf666e266ea0c`.

The 4-layer model has width 256, 8 query heads, 4 KV heads, FP16 weights/cache, 16-token pages, 64
physical blocks, and an 8-request/8-token step limit. Each policy receives one excluded warmup and
five measured repetitions. Measured policy order is seed-shuffled per repetition. The burst plan
has 16 heterogeneous requests. The churn plan has 24 requests, seeded logical arrivals over steps
0–12, and six planned cancellations after a three-step delay. Every run retains request traces and
asserts leak-free cache/budget termination.

| Workload | Policy | Output tok/s mean | Output tok/s P50 | Min–max | Latency P50/P95/P99 ms | TTFT P50 ms | TPOT P50 ms |
|---|---|---:|---:|---:|---:|---:|---:|
| Heterogeneous burst | No batching | 62.71 | 62.34 | 60.76–64.95 | 885.25/1663.35/1767.11 | 835.31 | 5.58 |
| Heterogeneous burst | Static | 243.97 | 244.61 | 238.25–248.41 | 226.58/434.42/448.92 | 204.25 | 7.33 |
| Heterogeneous burst | Continuous | 273.21 | 277.91 | 260.94–281.64 | 206.82/388.81/408.44 | 184.79 | 8.17 |
| Seeded arrival/churn | No batching | 44.33 | 41.46 | 24.90–66.66 | 778.27/2570.39/2655.42 | 749.34 | 6.88 |
| Seeded arrival/churn | Static | 160.11 | 179.44 | 78.41–220.18 | 215.66/578.18/701.74 | 174.39 | 7.91 |
| Seeded arrival/churn | Continuous | 204.00 | 229.33 | 101.38–290.30 | 139.05/379.35/466.82 | 105.14 | 8.66 |

On the burst workload, continuous batching improved mean output throughput 1.120x over static and
4.357x over no batching. Its request-latency P50/P95/P99 improved 8.7%/10.5%/9.0% versus static.
On churn, mean throughput improved 1.274x versus static and 4.602x versus no batching; median
throughput ratios were 1.278x and 5.531x. Churn request-latency P50/P95/P99 improved
35.5%/34.4%/33.5% versus static.

The throughput gain is not a universal latency win. Continuous batching regressed median TPOT by
11.4% on burst and 9.4% on churn versus static because a larger GPU batch makes each token step
longer. These regressions are retained, not filtered.

## Variance and cancellation boundary

Burst throughput was stable within each policy. Churn throughput was broad across the five runs:
24.90–66.66, 78.41–220.18, and 101.38–290.30 output tokens/s for no/static/continuous. The RTX 3080
Ti was a shared Windows display GPU with other graphics processes, not an isolated persistence-mode
accelerator. The raw repetitions, mean, median, tail distributions, GPU state, and policy order are
therefore all retained; the report does not substitute the best run.

Arrivals and cancellations are logical scheduler steps, not wall-clock Poisson events. Continuous
batching completed one planned cancellation target before its cancellation step in each repetition,
producing 19 completed/5 cancelled requests versus 18/6 for the other policies. Completed-output
throughput therefore has a different terminal-output denominator for churn. Executed-input
throughput and missed-cancellation counts remain in JSON. The burst workload is the cleaner exact
terminal-count comparison.

GPU state was 53 C, 122.12 W, 1,755 MHz SM before the suite and 52 C, 120.65 W, 1,755 MHz after;
the power limit was 350 W. Host and CUDA elapsed times closely agree because the scheduler performs
one batched GPU-to-host argmax synchronization before recording token timestamps.

## Verification

```text
Windows control environment:
python -m ruff check .
All checks passed

python -m mypy tensorforge scripts
Success: no issues found in 44 source files

python -m pytest -q
65 passed, 83 skipped

WSL CUDA/Triton environment:
python -m pytest -q
148 passed
```

Windows skips Triton-only tests; WSL executes the complete CPU, CUDA, paged-attention, integrated
decode, scheduler, and prior kernel grid.

## Claim boundary and next gate

Phase 5 does not claim a parallel prefill kernel, wall-clock arrival service model, request fairness
or priority policy, prefix sharing, CUDA Graph speedup, `torch.compile` speedup, speculative decode,
vLLM parity, multi-GPU execution, isolated-datacenter variance, or production SLOs. Prefill currently
walks the one-token paged decode path, deliberately preserving one cache/executor contract.

Phase 6 should first make active-batch buffers and control metadata address-stable, then compare
eager, `torch.compile`, and CUDA Graph decode with capture misses and shape/batch buckets reported.
Graph work must depend on this lifecycle and cache interface rather than create a second scheduler.
