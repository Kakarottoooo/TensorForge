# Unified decode path and cumulative ablation

Phase 7A removes the split between the Phase 3 scalar-fusion executor and the Phase 4–6 paged
runtime. `BucketedPagedDecodeExecutor` now accepts a cumulative `DecodeFusionLevel` while retaining
the same `BatchTokenExecutor`, scheduler, and `PagedKVCache` contracts.

## Per-layer dataflow

The fully fused path executes:

```text
hidden state
  -> Triton RMSNorm
  -> PyTorch/cuBLAS QKV projections and RoPE
  -> masked Triton paged KV write
  -> Triton paged GQA decode attention
  -> PyTorch/cuBLAS output projection
  -> Triton residual add + RMSNorm
       |-> materialized residual state
       `-> normalized MLP input
  -> PyTorch/cuBLAS gate/up projections
  -> Triton SwiGLU activation
  -> PyTorch/cuBLAS down projection
  -> caller-owned hidden-state output
```

Final RMSNorm also uses Triton. The fusion claim excludes every matrix multiplication: projection
GEMMs remain PyTorch/cuBLAS.

## Address ownership

The public Triton RMSNorm, residual/RMSNorm, and SwiGLU wrappers accept optional caller-owned output
tensors while preserving their allocation-returning compatibility path. The bucket owns one set of
norm, residual, activation, hidden-state, attention, workspace, and logit buffers per layer and
shape. These addresses remain stable across eager calls and CUDA Graph replay.

The cache still owns physical pages. Each append resolves the current logical-to-physical mapping
into the stable device controls, then commits only after every layer writer completes. Fusion does
not weaken rollback or release semantics.

## Causal ablation graph

```text
paged_dynamic_eager
  -> stable_bucket_eager
     -> triton_rms_norm
        -> triton_residual_rms_norm
           -> triton_all_fusions
              |-> triton_all_compile
              `-> triton_all_cuda_graph
```

Every row records both change versus the dynamic baseline and change versus its named parent.
Compile and CUDA Graph are siblings because neither causally contains the other. Inductor CUDA
Graphs remain disabled in the compile row, and custom Triton calls remain explicit compile graph
boundaries.

## Long-context setup

The cumulative manifest covers B1/B4/B8 at context 32 and B1 at contexts 128, 512, and 2,048.
Repeating the one-token decode path thousands of times would make excluded setup dominate the
experiment. For these cases a benchmark-only readable full-prefix routine writes a multi-token
`PagedKVCache` transaction, then the final setup token initializes the selected execution variant.
The measured region contains only the following 16 decode steps.

This reference priming is numerically checked but is not a production parallel-prefill kernel,
throughput result, or new cache implementation. Each run still compares its first measured logit
with independent full-prefix execution and requires leak-free cache release.

## Acceptance boundary

The implementation is accepted only if caller-owned kernel outputs match their PyTorch oracles,
all fusion levels preserve full-prefix logits, compile and graph paths remain correct, padded lanes
cannot write KV, scheduler churn reuses captures, every eligible bucket retains addresses, and all
reported rows retain cold setup, P50/P95/P99, throughput, graph counters, and regressions.
