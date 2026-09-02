# TensorForge implementation plan

## Invariants

Every optimized path must preserve a separate, readable PyTorch oracle. Each result is tied to
hardware metadata and a workload definition. No optimization is accepted without correctness
tests and an ablation against the immediately preceding configuration. Missing hardware or an
unsupported feature is reported as such; it is never replaced by synthetic numbers.

## Repository architecture

```mermaid
flowchart LR
    W[Benchmark workload] --> Q[Request scheduler]
    Q --> R[Inference runtime]
    R --> M[Llama model boundary]
    R --> C[KV-cache manager]
    M --> K[PyTorch or Triton operators]
    R --> X[Metrics]
    M --> P[Profiler]
    C --> X
    Q --> X
    P --> A[Hotspot analysis]
    X --> B[JSON / CSV / Markdown reports]
```

The model package owns mathematical semantics. The runtime owns execution policy. The cache owns
GPU memory lifecycle. The scheduler owns request state transitions, not tensors. Benchmark,
profiling, and metrics consume public boundaries so instrumentation does not contaminate kernels.

## Phased delivery

| Phase | Concrete deliverable | Acceptance gate |
|---|---|---|
| 1 — baseline | Explicit grouped-query Llama model, RoPE, RMSNorm, SwiGLU, greedy decode, FP32/FP16/BF16 selection | CPU unit suite; CUDA smoke for supported dtypes; causal and decode equivalence tests |
| 2 — measure | Workload schema, hardware capture, CUDA-event latency measurement, torch.profiler traces, JSON/CSV/Markdown reports | Repeated warmup/measured runs; percentile tests; real hardware metadata in every row |
| 3 — kernels | Triton RMSNorm, residual+RMSNorm, and fused SwiGLU elementwise path | Shape/dtype correctness grid and per-shape benchmark versus PyTorch |
| 4 — cache | Naive cache oracle plus preallocated paged cache with ownership and accounting | Lifecycle, exhaustion, reuse, fragmentation, and decode-equivalence stress tests |
| 5 — batching | Request state machine with no/static/continuous policies | Seeded arrival workload; latency percentiles and throughput comparison |
| 6 — execution | Separate ablations for compile, graph capture, reusable buffers, and transfer strategy | Eligibility/fallback tests plus measured ablation table |
| 7 — multi-GPU | Optional replicated workers first; tensor parallelism only if model scale justifies it | Single-versus-multi GPU throughput and scaling efficiency; NCCL overhead |
| 8 — release | Re-run benchmark matrix, hotspot analysis, failure notes, final report, resume claims | Clean checkout reproduction and all claims traceable to raw result IDs |

## Benchmark experiment design

Phase 2 will define one immutable workload record containing model config, prompt/generation length,
batch/concurrency, precision, seed, warmups, repetitions, optimization flags, and hardware fingerprint.
CUDA events will measure device execution; monotonic host clocks will measure TTFT, TPOT, and
request latency. Synchronization points will be explicit. Percentiles are computed from per-request
samples, not averages of batch timings.

The required matrix is prompt lengths 128/512/2048, generation lengths 32/128/256, batch sizes
1/4/8/16, and concurrency 1/4/16/32. An OOM-aware planner may skip or reduce a case, but the raw
record must retain requested dimensions, executed dimensions, and the reason for the adjustment.

## Optimization order

The initial ablation order is: explicit FP32 baseline, mixed precision, optimized PyTorch operators,
Triton fusions, paged KV cache, continuous batching, reusable buffers, then eligible CUDA graphs.
Changing one factor per row preserves attribution. CUDA graphs use bucketed static shapes and fall
back to eager execution for ineligible requests; they are not assumed to help prefill-heavy work.

## Known risks

- Triton is supported primarily on Linux; Windows development requires Linux/WSL or a rented GPU.
- Explicit attention materializes the score matrix and cannot make 2048-token high-batch cases fit
  by assumption. Phase 2 records capacity failures before optimized attention is considered.
- Randomly initialized weights validate runtime mathematics, not language quality. A compatible
  checkpoint adapter is future work and must not blur the baseline/optimized comparison.
- FP16/BF16 tolerances will be derived per operator; exact token equality alone can hide logit drift.

