# Phase 8R validation: real checkpoint, prefill, and external references

## Scope and immutable identity

Phase 8R moves the runtime from random-weight shape studies to a real Llama checkpoint without
changing the canonical cache or scheduler ownership boundaries. The published suite uses
`TinyLlama/TinyLlama-1.1B-Chat-v1.0` at commit
`fe8a4ea1ffedaf415f4da2f062534de366a451e6`:

- `config.json` SHA256: `486bedda3a6988332e60d9638a09ca4b260d34ebcf1b19e22cf3b140b63d8fe9`
- `model.safetensors` SHA256: `6e6001da2106d4757498752a021df6c2bdc332c650aae4bae6b0c004dcf14933`
- weight file size: 2,200,119,864 bytes
- measured TensorForge commit: `5b22e5e620d0e16b743dafc90fb3eafb79cccbaa`
- repository state during all backend runs: clean

The checkpoint was downloaded and checksum-verified outside the repository. The benchmark hashes
the local files again before execution. Model load and tokenization are excluded; every backend
receives the same deterministic token IDs, BF16 precision, greedy rule, output length, warmup, and
five measured repetitions.

## Implementation

The importer reads a deliberately narrow Llama configuration contract, creates TensorForge's model
on the meta device, maps every expected safetensors key, rejects missing or unexpected tensors,
materializes RoPE buffers, and then moves weights to the requested device/dtype. It currently
supports one unsharded safetensors file, standard RoPE, bias-free Llama attention/MLP, and SwiGLU.

Parallel prefill reserves each full prompt in `PagedKVCache`, packs unequal prompt lengths, applies
a causal-and-valid SDPA mask, writes per-layer K/V through the request block table, and commits only
after all layers and final logits succeed. A failure rolls every still-active reservation back. The
scheduler supports both the original one-token reference prefill and the new parallel mode; prompt
tokens count against the same request/global budgets.

Paged cache writes retain the direct two-copy one-token path. Multi-token writes derive logical
blocks and offsets as tensors, gather physical blocks from the sequence's device block table, and
scatter K/V without a Python token loop.

For wall-clock workloads, `RequestInput` may carry an external monotonic arrival timestamp. Late
admission therefore does not erase time already spent waiting while a prior GPU step ran.

## Correctness evidence

- The synthetic safetensors round trip is bitwise exact against the original TensorForge model.
- Real TinyLlama full-prefix logits match Transformers under the documented BF16 tolerance.
- CPU single and unequal-batch prefill match independent full-prefix logits and populate the paged
  cache in logical token order.
- Real prefill followed by Triton paged decode matches full-prefix execution.
- The 128-token, eight-step teacher-forced path matches Transformers with `rtol=3e-2` and
  `atol=2.5e-1`; observed maximum absolute differences during diagnosis were 0.09–0.19.
- The full WSL suite passes 180 tests; the Windows control suite passes 78 tests, with Linux-only
  Triton/real-GPU gates skipped there.
- Every completed benchmark run ends with zero active sequences, used blocks, reservations, or
  outstanding scheduler token budget.

The P128 free-running token hash differs from Transformers/vLLM. Diagnosis found that the first
two decode tokens match; at the third choice Transformers BF16 has a top-two margin of exactly zero,
while a 0.0625 TensorForge logit rounding difference chooses the other tied token. Feeding the same
oracle token to both paths keeps subsequent logits inside tolerance and restores matching argmaxes.
This is classified as tied-argmax numerical divergence, not cache corruption.

## Vectorized cache-write ablation

The microbenchmark uses one FP16 layer with four KV heads, head dimension 64, 16-token pages, ten
warmups, and 100 CUDA-event repetitions. Materialized K/V tensors must match exactly.

| Tokens written | Scalar loop ms | Optimized ms | Speedup |
|---:|---:|---:|---:|
| 1 | 0.045 | 0.038 | 1.19x |
| 32 | 1.111 | 0.177 | 6.28x |
| 128 | 4.560 | 0.149 | 30.67x |
| 512 | 17.708 | 0.149 | 119.12x |

The one-token fast path is intentionally separate. An intermediate implementation sent one token
through vectorized index construction and measured about 0.160 ms; restoring direct copies reduced
that exploratory result to about 0.043 ms. That rejected implementation is not used by the final
runtime.

## Real-checkpoint results

All rows ran on the same RTX 3080 Ti under WSL2. TensorForge and Transformers used PyTorch
2.5.1+cu121, CUDA runtime 12.1, Triton 3.1.0, and Transformers 4.57.6. The isolated vLLM reference
used vLLM 0.6.4.post1, PyTorch 2.5.1+cu124, CUDA runtime 12.4, Triton 3.1.0, and Transformers 4.46.3.
The differing embedded CUDA runtimes are another reason to treat vLLM as an external reference,
not a one-factor ablation.

The table reports per-run output-throughput P50 with the arithmetic mean in parentheses:

| Workload | TensorForge tok/s | Transformers SDPA tok/s | vLLM tok/s |
|---|---:|---:|---:|
| B1, prompt 32, output 8 | 26.42 (27.47) | 24.30 (28.25) | 129.90 (124.32) |
| B4 burst, prompt 32, output 8 | 114.77 (105.59) | 113.65 (123.62) | 516.30 (516.02) |
| B1, prompt 128, output 16 | 18.16 (19.99) | 35.35 (35.95) | 149.60 (150.74) |

On P32, TensorForge's median throughput is 8.7% above Transformers at B1 and 1.0% above it at B4;
these small gaps are not treated as decisive because both implementations show large run-to-run
variance. At P128, Transformers is 1.95x TensorForge. vLLM is 4.50–8.24x TensorForge across the
comparable cases and has much tighter B4 variance.

TensorForge's B4/P32 burst throughput spans 55.78–125.35 tok/s across five runs; Transformers spans
68.30–185.87 tok/s. The raw distributions are retained. Runtime snapshots started at 55 C and
1,755 MHz SM clock for every backend; TensorForge/Transformers ended at 56 C and vLLM at 59 C, so
the variance cannot be attributed to obvious thermal throttling from these coarse snapshots.

Only TensorForge executes the 0/5/10/15 ms staggered-arrival case in this offline suite. Its request
latency P50/P95/P99 is 603.77/881.92/885.54 ms, TTFT is 136.34/313.15/316.95 ms, and TPOT is
53.12/101.35/103.33 ms. Transformers and offline vLLM reject that row instead of silently converting
it into simultaneous arrivals.

## Conclusions and next bottleneck

Phase 8R validates that the allocator, parallel prefill, paged attention, scheduler lifecycle, and
real checkpoint interoperate correctly. It also removes the misleading possibility that strong
small-model kernel microbenchmarks imply a competitive full runtime. vLLM's 4.50–8.24x median
throughput advantage is the correct headline external result.

The next highest-leverage work is an adaptive final execution policy backed by real-model profiles:
separate prefill/decode timing, eliminate host synchronization and Python control overhead where
request semantics allow it, select only shape-positive fusions/graphs, and validate that policy on
held-out prompt/batch shapes. Speculative decoding or multi-GPU work should not precede this because
they would compound an already slower single-GPU path.
