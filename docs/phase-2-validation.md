# Phase 2 validation record

Date: 2026-09-01 (America/Los_Angeles) / 2026-09-02 UTC

## Delivered measurement boundary

Phase 2 adds schema-versioned workload and execution identities, request-level token timelines,
host TTFT/TPOT/request latency, CUDA Event time, throughput, PyTorch allocator measurements,
out-of-process GPU sampling, strict hardware/software/Git metadata, explicit OOM batch reduction,
machine-readable JSON/CSV, Markdown reports, and summarized `torch.profiler` captures. The backend
protocol and execution identity reserve explicit semantics for compile, CUDA Graph, paged cache,
continuous batching, speculative decoding, external runtimes, and tensor-parallel execution.

## Curated experiment

The immutable input is `benchmarks/phase2-reference.json`. The measured code commit is
`14534447b392f76d2f69073ea65b2c48a43a2a39`, recorded with `git_dirty=false`. Hardware fingerprint
`b43cb88537a4c0abefefdd19b186820b9f8d79b5f65f5ab0de2890b62e5fb626` identifies an RTX 3080 Ti
(12,288 MiB reported by `nvidia-smi`, compute capability 8.6), driver 591.86 (driver CUDA capability
13.1), PyTorch 2.5.1+cu118 with CUDA runtime 11.8 and cuDNN 9.1.0, Python 3.12.2, and Windows 11.
Triton and Nsight Systems were unavailable in this environment.

The random-weight reference model has approximately 56 million parameters and uses explicit
PyTorch attention plus full-prefix recomputation. Results are not checkpoint-quality or optimized
runtime claims.

| Prompt / output | Batch | Repetitions | TTFT P50 / P95 / P99 ms | TPOT P50 / P95 / P99 ms | Mean output tok/s | Peak allocated MiB |
|---:|---:|---:|---:|---:|---:|---:|
| 128 / 32 | 1 | 10 | 17.850 / 31.563 / 31.672 | 13.778 / 17.577 / 17.938 | 70.199 | 136.675 |
| 512 / 32 | 4 | 10 | 14.976 / 29.727 / 29.727 | 15.840 / 21.032 / 21.032 | 246.574 | 383.442 |
| 2048 / 32 | 1 | 5 | 23.997 / 24.561 / 24.668 | 25.193 / 25.513 / 25.525 | 39.783 | 522.945 |

All requested batch sizes completed; there was no OOM adjustment. Each measured repetition obtained
4–8 utilization samples depending on duration. The raw JSON retains every request trace, CUDA time,
allocator value, utilization sample summary, and backend counter. Batched requests complete tokens
together in this synchronous baseline, so requests in a repetition legitimately share timestamps.

These numbers are one reference suite, not a statistically complete performance study. The spread
between median and tail latency in the shorter cases is visible and must be investigated with
randomized repeated suites before optimization ratios are claimed.

## Profiler evidence

The curated profile uses prompt 512, batch 1, FP16, one prefill region and four full-prefix decode
steps. `torch.profiler` gives a **kernel-launch/latency-dominant candidate** classification with
provisional, nonexclusive self-device-time shares: matrix multiply 25.5%, memory/pointwise 39.6%,
and repeated kernels averaging at most 25 microseconds 39.6%. The launch share overlaps operator
categories and must not be added to them.

High self-device-time rows include repeated allocation/layout paths (`empty_strided`, `_to_copy`,
`cat`, `transpose`) and multiple shape-specific `bmm`/`mm` rows. This supports investigating buffer
reuse, fusion, KV caching, and launch reduction, but does **not** establish DRAM or compute roofline
position. Nsight Compute counters and arithmetic-intensity models remain required.

## Verification

```text
python -m ruff check .
All checks passed

python -m mypy tensorforge scripts
Success: no issues found in 29 source files

python -m pytest -q
42 passed
```

The benchmark smoke manifest, full curated manifest, three report formats, profiler JSON/Markdown,
and local Chrome trace were also generated successfully. The 6.3 MB Chrome trace remains ignored;
it can be regenerated from the recorded profile command.

## Claim boundary and next gate

Phase 2 does not contain Triton, paged KV cache, continuous batching, CUDA Graphs, compile ablations,
speculative decoding, vLLM, or multi-GPU claims. Phase 3 must first move to a supported Linux Triton
environment, implement exact operator oracles and autotuning keys, and establish per-shape numerical
and performance evidence for RMSNorm, fused residual/RMSNorm, and SwiGLU.
