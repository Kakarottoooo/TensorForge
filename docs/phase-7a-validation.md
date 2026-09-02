# Phase 7A validation record

Date: 2026-09-01 (America/Los_Angeles) / 2026-09-02 UTC

## Delivered boundary

Phase 7A removes the last split between the scalar Triton experiments and the production-shaped
paged decode path. `BucketedPagedDecodeExecutor` now owns one cumulative `DecodeFusionLevel`:
PyTorch controls, Triton RMSNorm, Triton RMSNorm plus fused attention residual/RMSNorm, or all of
those plus Triton SwiGLU. Every level uses the same transactional `PagedKVCache`, paged GQA
attention, caller-owned bucket buffers, scheduler-facing executor contract, and logits boundary.

RMSNorm, residual/RMSNorm, and SwiGLU accept optional caller-owned output tensors. The bucket
allocates those tensors once per layer and keeps their addresses stable across eager execution,
segmented `torch.compile`, CUDA Graph capture, and replay. GEMMs remain PyTorch/cuBLAS calls. Cache
allocation, request lifecycle, transaction commit/rollback, and pinned-host control staging remain
outside the graph.

The implementation preserves the unfused dynamic paged executor as the correctness-equivalent
baseline. The seven-row experiment then changes one named boundary at a time:

```text
paged dynamic eager
  -> address-stable bucket eager
  -> Triton RMSNorm
  -> fused residual/RMSNorm
  -> Triton SwiGLU (all fusions)
       -> segmented torch.compile
       -> explicit CUDA Graph
```

Compile and CUDA Graph are siblings with `triton_all_fusions` as their common causal parent. Their
results therefore must not be multiplied or interpreted as a cumulative compile-then-graph path.

## Correctness and lifecycle evidence

Kernel tests compare caller-owned-output RMSNorm, residual/RMSNorm, and SwiGLU against independent
PyTorch formulas and assert object identity as well as stable addresses. Decode tests compare all
four fusion levels against independent full-prefix model execution, including repeated cache
appends, segmented compile, capture, and replay. The graph test checks that inactive padded lanes do
not write KV state. Every request is released and the allocator returns to zero used and reserved
blocks.

The public benchmark performs a full-prefix numerical gate before accepting each measured run. Its
setup-only reference prefill writes a multi-token cache transaction using readable model math; the
last setup token then exercises the selected runtime so compilation or capture is excluded from
steady-state timing. This mechanism is an oracle and cache primer, not a production prefill-kernel
claim.

## Curated cumulative experiment

The immutable input is `benchmarks/phase7a-cumulative.json`. Suite
`caa3d1c6-0dd1-4b45-bbc3-35342e8eea9c` measured clean commit
`b268f16bcfa7c7adea45b9d3fc3ca9c92f557a38`. The machine was an RTX 3080 Ti (compute capability
8.6) under WSL2 with PyTorch 2.5.1+cu121, CUDA runtime 12.1, Triton 3.1.0, and driver 591.86. The
hardware fingerprint is
`77278e66c8912b496682306deb107db0d1c325287febc4504b8bf666e266ea0c`.

The model has four layers, width 256, intermediate width 688, eight query heads, four KV heads,
FP16 weights/cache, 16-token pages, and 256 physical blocks. The matrix covers B1/B4/B8 at context
32 and B1 at contexts 128, 512, and 2,048. Every row contains one excluded warmup and five
seed-shuffled repetitions of 16 decode steps: 80 measured samples per row and 3,360 samples in
total. CUDA events measure each append. Host throughput includes cache transactions, two
pinned-control copies, dispatch, and final synchronization.

The table below reports CUDA P50 and mean-latency change. A negative change is an improvement. The
baseline column always compares with dynamic eager; the parent column compares only with the
immediately preceding causal row.

| Case | Dynamic P50 ms | Stable bucket change | RMSNorm change | Residual/RMSNorm change | All-fusion change | Compile change | Graph P50 ms / change |
|---|---:|---:|---:|---:|---:|---:|---:|
| B1 / C32 | 15.871 | -16.9% | -38.2% | -3.0% | **+6.6%** | -13.1% | 0.818 / -87.8% |
| B4 / C32 | 6.728 | **+7.2%** | -20.3% | -2.2% | -18.2% | **+1.9%** | 2.009 / -71.0% |
| B8 / C32 | 10.196 | -43.9% | -11.0% | -7.6% | **+60.4%** | -24.1% | 1.754 / -80.2% |
| B1 / C128 | 5.797 | -26.2% | **+27.2%** | -36.1% | **+28.8%** | -15.6% | 0.886 / -83.1% |
| B1 / C512 | 6.916 | **+26.3%** | -4.8% | -1.2% | -15.2% | **+1.5%** | 2.288 / -69.2% |
| B1 / C2048 | 11.909 | -24.2% | **+59.8%** | -29.2% | **+1.7%** | -27.0% | 6.224 / -52.9% |

The complete P50/P95/P99, throughput, setup costs, per-run samples, variant order, and hardware
state are retained in `results/reference/phase7a-rtx3080ti-wsl/execution.{json,csv,md}`.

## Findings

- Explicit CUDA Graph is the only uniformly positive final specialization in this matrix. Relative
  to fully fused eager, it reduced mean step latency by 52.9–87.8%; relative to dynamic eager, by
  59.0–93.5%. Every shape recorded 80 graph hits, zero misses, and zero fallbacks. Capture cost was
  137–152 ms and is excluded from the steady-state numbers.
- Fused residual/RMSNorm improved its RMSNorm-only parent in all six cases, by 1.2–36.1% in mean
  latency. This is the most consistent scalar fusion in the cumulative eager path.
- Triton SwiGLU was not uniformly beneficial. Adding it improved B4/C32 and B1/C512 by 18.2% and
  15.2%, but regressed the other four parent comparisons by 1.7–60.4%. The result rejects a blanket
  "custom kernel is faster" claim for this small model and these shapes.
- Segmented `torch.compile` improved four of six comparisons with fully fused eager by 13.1–27.0%,
  but regressed B4/C32 by 1.9% and B1/C512 by 1.5%. Its first process-local compile cost ranged from
  7 ms after cache reuse to 3.03 s for the first encountered shape.
- Address-stable eager and standalone RMSNorm also had shape-dependent regressions. Stable eager
  regressed B4/C32 and B1/C512; RMSNorm regressed B1/C128 and B1/C2048 versus its parent. Address
  stability is therefore an enabling invariant for graph replay, not an independent speedup claim.

The causal direction is fixed by the manifest and randomized run order, but the exact mechanism
behind each regression is not proven by this suite. Plausible contributors include extra launch
overhead, very small scalar workloads relative to GEMMs, process-local kernel caches, and display-GPU
interference. Those are hypotheses, not measured conclusions.

## Variance and setup boundary

This RTX 3080 Ti is also a Windows display GPU. The distributions are broad enough that P50 and
mean-latency direction sometimes differ; parent percentages above deliberately use the registered
mean-latency metric, while the table separately exposes P50. All 80 samples are retained rather
than selecting the best repetition.

GPU state was 54 C, about 124 W, and 1,755 MHz SM before the suite; it ended at 55 C, about 125 W,
and 1,755 MHz. Memory clock changed from 9,501 to 9,251 MHz. Cold setup includes first-use Triton,
cuBLAS, and compiler work. Cross-case caches are process-local, so these cold values are
order-dependent and are not fresh-process startup SLOs for every row.

## Verification

```text
Windows control environment:
python -m ruff check .
All checks passed

python -m mypy tensorforge scripts
Success: no issues found in 48 source files

python -m pytest -q
69 passed, 98 skipped

WSL CUDA/Triton environment:
python -m pytest -q
167 passed
```

Windows skips Triton-only tests. WSL runs CPU, CUDA, custom-kernel, paged cache, scheduler, bucket,
compile, graph, and end-to-end benchmark smoke coverage. Phase 7A's public contracts were developed
red-green: caller-owned output rejection preceded the kernel API, fusion-level import and
full-prefix failures preceded the bucket implementation, and unknown manifest variants preceded
the schema 1.1 runner.

## Claim boundary

Phase 7A establishes one correctness-equivalent, address-stable, cumulatively measurable paged
decode path. It does not claim production prefill, language quality, FlashAttention or vLLM parity,
isolated-datacenter latency, optimal Triton configurations for every shape, speculative decoding,
or multi-GPU scaling. Results from Phases 3–7A use different workloads and must not be multiplied.
