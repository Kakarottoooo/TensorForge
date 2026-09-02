# Phase 4 validation record

Date: 2026-09-01 (America/Los_Angeles) / 2026-09-02 UTC

## Delivered boundary

Phase 4 adds a transactional logical-to-physical paged KV allocator, arbitrary physical-page
mapping, atomic exhaustion, deterministic reuse, cancellation/release, full and partial append
commit, rollback, allocator accounting, a readable paged GQA oracle, one-pass online-softmax Triton
decode attention, a two-kernel split-KV long-context path, fixed-address output/workspace support,
and a single-request incremental Llama executor.

The cache owns tensors and page lifecycle; the kernel owns attention math; the runtime owns the
per-token transaction. Scheduler state and CUDA Graph policy remain outside these modules.

## Correctness and failure evidence

CPU allocator tests cross page boundaries, verify logical token order, require exhaustion to be
atomic, reclaim only transaction-owned pages on rollback, retain only accepted speculative prefixes,
cancel active reservations on release, and execute 500 deterministic request-churn operations while
checking block ownership, free/used accounting, sequence counts, committed tokens, and final leak-free
reclamation.

CUDA tests cover FP32/FP16/BF16, query head counts 4/8/32, KV head counts 1/2/4/8, head dimensions
64/80/128, contexts 1–2,048, batch-specific unequal lengths, partial pages, and noncontiguous physical
tables. Both one-pass and split-KV outputs pass dtype-specific `torch.testing.assert_close` gates.
The complete incremental executor compares every token's logits against the Phase 1 full-prefix
recomputation for FP32/FP16/BF16.

Caller-provided output and split workspace retain their original storage address. This establishes a
capture-compatible buffer contract but is not a CUDA Graph execution or performance claim.

## Curated experiment

The immutable input is `benchmarks/phase4-attention.json`. Suite
`0b888e8f-4ad3-4cc2-8a02-2febfb25b132` measured clean commit
`42e4782fc71656cb27af082f0509db5c5f7098f7`. Hardware fingerprint and software identity are retained
in the JSON: RTX 3080 Ti, compute capability 8.6, driver 591.86, PyTorch 2.5.1+cu121, CUDA runtime
12.1, Triton 3.1.0, Python 3.11.13, and WSL2 Linux 6.6.87.2.

The fixed matrix has 14 cases and 30 implementation rows. Each implementation receives 50 ms warmup
and a 500 ms measurement window. P05/P50/P95 use CUDA-event timing. First-call JIT/autotune is kept
separate. The empirical 256 MiB `torch.copy_` ceiling was 802.89 GB/s. GPU state moved from 54 C,
124.32 W, 1,755 MHz SM before the suite to 59 C, 205.43 W, 1,950 MHz SM after it; power limit was
350 W.

| Case | PyTorch P50 ms | One-pass P50 ms | Auto P50 ms | Auto vs PyTorch | Split vs one-pass |
|---|---:|---:|---:|---:|---:|
| Default, B1, context 1 | 0.024576 | — | 0.006144 | 4.000x | — |
| Default, B1, context 512 | 0.028672 | — | 0.022528 | 1.273x | — |
| Default, B1, context 2,048 | 0.136336 | 0.072704 | 0.020480 | 6.657x | 3.550x |
| Default, B8, context 512 | 0.301152 | — | 0.023552 | 12.787x | — |
| 7B-style, B1, context 512 | 0.051200 | — | 0.037888 | 1.351x | — |
| 7B-style, B1, context 2,048 | 0.219392 | 0.133120 | 0.037888 | 5.791x | 3.514x |
| 7B-style, B8, context 512 | 0.532480 | — | 0.072704 | 7.324x | — |
| 7B-style, B1, context 512, BF16 | 0.056320 | — | 0.036864 | 1.528x | — |

All 14 selected auto-path rows improved over the timed expanded-GQA PyTorch baseline in this exact
matrix; the observed range was 1.273x–25.033x. The upper ratios are not promoted as general speedup
claims: several PyTorch rows have broad P95 tails and small-shape dispatch costs. Split-versus-one-pass
is the cleaner one-factor ablation for the long-context optimization and measured 3.55x/3.51x.

First-call wall time ranged from 0.161 ms for an already tuned shape key to 686.703 ms for a new
JIT/autotune key. Production executors must warm or persist compiled configurations. The largest
recorded absolute error was 0.0001221. The largest raw relative error was 0.05882 near zero; the
combined absolute/relative dtype tolerance passed for every element.

## Timed comparison boundary

The PyTorch baseline receives contiguous logical K/V and performs `repeat_interleave` GQA expansion,
score matmul, FP32 softmax, and value aggregation inside timing. Triton receives physical pages and a
block table; output and split workspace are preallocated outside timing. This is a comparison with
TensorForge's preserved explicit PyTorch semantics, not PyTorch SDPA, FlashAttention, vLLM, or
TensorRT-LLM.

Reported effective bandwidth uses unique semantic query/output/K/V bytes plus block-table metadata.
It is not measured DRAM traffic; cache hits, transaction amplification, occupancy, and stalls require
Nsight Compute. The report does not infer them from logical GB/s.

## Verification

```text
Windows control environment:
python -m ruff check .
All checks passed

python -m mypy tensorforge scripts
Success: no issues found in 39 source files

python -m pytest -q
54 passed, 78 skipped

WSL CUDA/Triton environment:
python -m pytest -q
132 passed
```

Windows capability skips cover Triton-only tests. WSL executes the GPU kernel, split-KV, integrated
decode, and existing Phase 3 grid.

## Claim boundary and next gate

Phase 4 does not claim a prefill kernel, multi-request scheduler, end-to-end throughput improvement,
CUDA Graph speedup, speculative policy, prefix sharing, sliding-window attention, quantized KV,
multi-GPU cache sharding, vLLM parity, or Nsight Compute counter evidence.

Phase 5 may now depend on the canonical cache and attention interfaces. Its gate is an explicit
queued/prefill/decode/completed/failed request state machine, global and per-request token budgets,
cancellation/failure reclamation, seeded arrival/churn stress, and P50/P95/P99 latency plus throughput
against static and no-batching controls.
