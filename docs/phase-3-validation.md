# Phase 3 validation record

Date: 2026-09-01 (America/Los_Angeles) / 2026-09-02 UTC

## Delivered kernel boundary

Phase 3 adds production-callable Triton kernels for RMSNorm, fused residual add plus RMSNorm, and
SwiGLU activation. Each kernel has an explicit PyTorch oracle, FP32-sensitive arithmetic, masked
odd-width handling, shape-aware autotuning, numerical tests across FP32/FP16/BF16, and a model-level
Llama executor that replaces only the corresponding operators while preserving the Phase 1
weights and attention path.

The benchmark runner correctness-gates every case before timing. It records first-call JIT/autotune
cost separately from steady state; emits P05/P50/P95 CUDA-event timings; retains the selected
configuration and numerical error; and reports logical bytes, modeled FLOPs, arithmetic intensity,
effective logical bandwidth, and empirical roofline efficiency. A slower Triton result would remain
in the report as a regression.

## Curated experiment

The immutable input is `benchmarks/phase3-kernels.json`. The measured code commit is
`0bd1d9eb77d6cdf1ae12627a205ebc1b43d61826`, recorded with `git_dirty=false`. Hardware fingerprint
`77278e66c8912b496682306deb107db0d1c325287febc4504b8bf666e266ea0c` identifies an RTX 3080 Ti
(12,288 MiB reported by `nvidia-smi`, compute capability 8.6), driver 591.86, PyTorch 2.5.1+cu121,
CUDA runtime 12.1, Triton 3.1.0, Python 3.11.13, and WSL2 Linux kernel 6.6.87.2. The GPU moved from
54 C / 122.68 W before the suite to 60 C / 271.40 W after it; the raw report also preserves the
power limit and SM/memory clocks.

The suite contains 22 shape/precision cases and 44 implementation rows. It covers odd hidden
widths, decode rows, batched decode, default-model prefill, wide prefill, a 7B-style SwiGLU width,
and FP32/FP16/BF16. Steady-state timing uses a 25 ms warmup and 100 ms repetition window. The
curated outputs are `results/reference/phase3-rtx3080ti-wsl/kernels.{json,csv,md}`.

| Case | PyTorch P50 ms | Triton P50 ms | Speedup | Triton logical GB/s | Copy ceiling use |
|---|---:|---:|---:|---:|---:|
| RMSNorm, 1 x 4096, FP16 | 0.019456 | 0.005120 | 3.800x | 4.80 | 0.6% |
| RMSNorm, 2048 x 512, FP16 | 0.033792 | 0.009216 | 3.667x | 455.22 | 55.9% |
| RMSNorm, 16 x 4096, BF16 | 0.025600 | 0.005120 | 5.000x | 52.80 | 6.5% |
| Residual/RMSNorm, 1 x 4096, FP16 | 0.021504 | 0.005120 | 4.200x | 8.00 | 1.0% |
| Residual/RMSNorm, 2048 x 512, FP16 | 0.040960 | 0.014336 | 2.857x | 585.21 | 71.9% |
| SwiGLU, 2048 x 1376, FP16 | 0.043008 | 0.026624 | 1.615x | 635.08 | 78.0% |
| SwiGLU, 128 x 11008, FP16 | 0.021504 | 0.015360 | 1.400x | 550.40 | 67.6% |

Every measured Triton row improved in this matrix: the observed range was 1.400x to 5.000x, with
zero recorded regressions. This is a result for these exact shapes and environment, not a claim
that the kernels win for all devices or shapes. First-call JIT/autotune wall time ranged from
106.387 ms to 710.493 ms and is intentionally excluded from steady-state latency; deployments must
warm or persist caches before serving latency-sensitive traffic.

Autotuning selected RMSNorm/residual configurations spanning 1, 2, 4, and 8 warps and SwiGLU block
sizes 128, 256, and 512, rather than collapsing to one universal configuration. All 22 cases passed
their dtype-specific `torch.testing.assert_close` gate. The largest recorded absolute error was
0.0625 in BF16 residual/RMSNorm; the largest raw relative error was 0.111111 in an FP16 SwiGLU case
near zero, whose absolute error was 0.0078125. Raw maximum-relative-error alone is not the acceptance
criterion near zero; the combined per-element absolute/relative tolerance is.

## Roofline interpretation

An empirical 256 MiB `torch.copy_` benchmark measured 814.11 GB/s effective logical bandwidth.
The FP32 compute ceiling is a sourced specification-derived 34,201.6 GFLOP/s, calculated as 10,240
CUDA cores times 1.67 GHz boost times two FLOPs per cycle. Kernel arithmetic intensity counts rsqrt
or sigmoid as one special-function operation, while logical byte counts assume unique tensor traffic
and count a reused weight once.

These are deliberately transparent models, not hardware-counter claims. The 635.08 GB/s SwiGLU
row corresponds to 78.0% of the empirical copy ceiling, but neither number is an Nsight Compute DRAM
counter. Cache effects, instruction mix, transaction amplification, and boost behavior are not
resolved by this Phase 3 experiment.

## Verification

```text
Windows control environment:
python -m ruff check .
All checks passed

python -m mypy tensorforge scripts
Success: no issues found in 34 source files

python -m pytest -q
46 passed, 61 skipped

WSL CUDA/Triton environment:
python -m pytest -q
107 passed in 12.32s
```

Windows skips the capability-gated Triton tests because upstream Triton is unavailable there. The
WSL run executes the complete GPU kernel and integrated-model grid.

## Claim boundary and next gate

Phase 3 does not claim custom attention, KV-cache speedup, end-to-end generation speedup, DRAM
counter bandwidth, CUDA Graph capture, continuous batching, speculative decoding, vLLM parity, or
multi-GPU scaling. Attention and GEMMs in the integrated executor remain PyTorch operations.

Phase 4 must introduce a real GQA decode-attention kernel and a block-based KV allocator together:
logical block tables, physical block ownership, bounds and exhaustion behavior, append/rollback
semantics, and equivalence against full-prefix attention must be established before scheduler or
CUDA Graph work can safely depend on the cache interface.
