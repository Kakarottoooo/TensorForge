# Phase 1 validation record

Date: 2026-09-01

## Scope delivered

Phase 1 establishes the semantic reference only: model configuration, token embedding, explicit
grouped-query causal attention, rotary embeddings, RMSNorm, SwiGLU, residual paths, final
normalization, output projection, and full-prefix greedy generation. It does not include a KV cache,
custom Triton kernel, scheduler, profiler report, benchmark harness, or performance result.

## Environment

| Component | Observed value |
|---|---|
| GPU | NVIDIA GeForce RTX 3080 Ti, 12,288 MiB |
| NVIDIA driver | 591.86 |
| Driver-reported CUDA capability | CUDA 13.1 |
| PyTorch build | 2.5.1+cu118 |
| PyTorch CUDA runtime | 11.8 |
| cuDNN | 9.1.0 |
| CPU | AMD Ryzen 9 5900XT 16-Core Processor |
| OS | Microsoft Windows 11 Pro 10.0.26200 |
| Python | 3.12.2 |
| Triton | Not installed; Phase 3 requires a supported Linux CUDA environment |

The driver-reported CUDA version and the CUDA runtime embedded in the PyTorch wheel are different
values by design; both are recorded to avoid an ambiguous “CUDA version” claim.

## Verification evidence

```text
python -m ruff check .
All checks passed!

python -m pytest -q
26 passed

python -m scripts.smoke_generate --device cuda --dtype fp32
output shape [2, 20], checksum 2747

python -m scripts.smoke_generate --device cuda --dtype fp16
output shape [2, 20], checksum 2747

python -m scripts.smoke_generate --device cuda --dtype bf16
output shape [2, 20], checksum 2747
```

The checksum equality is a smoke-level signal for this seed and tiny random model. It is not a
general numerical-equivalence proof and is not a latency or throughput measurement. Operator-level
tests provide the primary numerical contracts; Phase 3 will add explicit dtype-specific tolerances
over a larger tensor-shape grid.

## Failure found and corrected

The first smoke invocation used `python scripts/smoke_generate.py`, which does not place the
repository root on Python's import path before an editable install. The supported checkout command
is now `python -m scripts.smoke_generate`; an editable install also makes the direct script import
valid. No model computation failed during that attempt.

## Claim boundary

Verified: tensor shapes, finite outputs, strict causal isolation, explicit RMSNorm and SwiGLU
formulas, deterministic greedy-decode equivalence, EOS termination, input validation, and supported
FP32/FP16/BF16 execution on the GPU above.

Unproven: language quality, checkpoint compatibility, performance, long-context capacity, memory
efficiency, Triton correctness, KV-cache equivalence, scheduler behavior, CUDA Graph eligibility,
and multi-GPU scaling.
