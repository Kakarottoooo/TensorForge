# Benchmark workloads

Manifests use benchmark schema `1.0`:

- `phase2-smoke.json` checks the measurement path quickly with a tiny model.
- `phase2-reference.json` measures the unoptimized ~56M-parameter PyTorch reference at three prompt
  and batch regimes. It is deliberately not the final required matrix.

The final matrix expands prompt lengths 128/512/2048, output lengths 32/128/256, batch sizes
1/4/8/16, and concurrency 1/4/16/32. Cases that exceed memory retain requested dimensions and an
explicit adjustment/failure record.

`phase3-kernels.json` defines the correctness-gated RMSNorm, residual/RMSNorm, and SwiGLU
microbenchmark matrix. It includes odd widths, decode and prefill row counts, default-model and
7B-style widths, FP16/BF16/FP32, timing duration, copy-ceiling size, and the sourced FP32 compute
ceiling derivation.
