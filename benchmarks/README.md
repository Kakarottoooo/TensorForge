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

`phase4-attention.json` fixes the paged GQA decode matrix before measurement. It covers context 1,
17, 128, 512, and 2,048; batch 1 and 8; default and 7B-style head layouts; FP16/BF16/FP32; and
retains every measured improvement or regression.

`phase5-scheduler.json` fixes model/cache capacity, token budgets, batch limits, repetitions, and
two seeded request plans. All policies receive the same prompts, generation limits, logical
arrival/cancellation steps, model weights, paged cache, and Triton execution path; only the
no/static/continuous refill policy changes.

`phase6-execution.json` fixes batch/context buckets, eligible shapes, one deliberate batch-overflow
shape, setup length, decode steps, and repetition count. Eager, segmented `torch.compile`, and
explicit CUDA Graph rows share model weights, cache ownership, custom kernels, and bucket buffers.
