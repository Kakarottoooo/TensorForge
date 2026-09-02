# TensorForge real-checkpoint benchmark

- Backend: `vllm`
- Checkpoint: `TinyLlama/TinyLlama-1.1B-Chat-v1.0@fe8a4ea1ffedaf415f4da2f062534de366a451e6`
- Config SHA256: `486bedda3a6988332e60d9638a09ca4b260d34ebcf1b19e22cf3b140b63d8fe9`
- GPU: NVIDIA GeForce RTX 3080 Ti (compute capability 8.6)
- Git: `5b22e5e620d0e16b743dafc90fb3eafb79cccbaa`; dirty: `False`

| Case | Batch | Prompt/output | Latency P50/P95/P99 ms | TTFT P50/P95/P99 ms | TPOT P50/P95/P99 ms | Output tok/s | Status |
|---|---:|---:|---:|---:|---:|---:|---|
| real-b1-p32-o8 | 1 | 32/8 | 61.587/80.611/83.770 | 21.043/33.154/35.490 | 6.172/6.780/6.897 | 124.316 | completed |
| real-b4-p32-o8-burst | 4 | 32/8 | 61.771/64.385/64.502 | 16.308/18.850/18.974 | 6.506/6.739/6.739 | 516.016 | completed |
| real-b4-p32-o8-staggered | n/a | 32/8 | n/a/n/a/n/a | n/a/n/a/n/a | n/a/n/a/n/a | n/a | failed |

Failure for `real-b4-p32-o8-staggered`: `ValueError: vLLM offline comparison supports simultaneous arrivals only`

| real-b1-p128-o16 | 1 | 128/16 | 106.953/120.325/122.605 | 17.140/22.883/23.035 | 5.670/6.613/6.712 | 150.741 | completed |

## Measurement and claim boundary

offline LLM.generate token-ID batch through completed outputs; model loading and tokenization excluded; TTFT/finish use vLLM metrics and intermediate token timestamps are interpolated

Only rows with the same pinned checkpoint, input-token plan, precision, GPU, generation semantics, and synchronization boundary are comparable. TensorForge uses parallel SDPA prefill followed by its transactional paged decode path; the Transformers reference uses SDPA and its standard contiguous KV cache. Loading and tokenization are excluded. Output-token SHA256 values in JSON expose semantic drift.
