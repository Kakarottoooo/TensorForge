# TensorForge real-checkpoint benchmark

- Backend: `transformers_sdpa`
- Checkpoint: `TinyLlama/TinyLlama-1.1B-Chat-v1.0@fe8a4ea1ffedaf415f4da2f062534de366a451e6`
- Config SHA256: `486bedda3a6988332e60d9638a09ca4b260d34ebcf1b19e22cf3b140b63d8fe9`
- GPU: NVIDIA GeForce RTX 3080 Ti (compute capability 8.6)
- Git: `5b22e5e620d0e16b743dafc90fb3eafb79cccbaa`; dirty: `False`

| Case | Batch | Prompt/output | Latency P50/P95/P99 ms | TTFT P50/P95/P99 ms | TPOT P50/P95/P99 ms | Output tok/s | Status |
|---|---:|---:|---:|---:|---:|---:|---|
| real-b1-p32-o8 | 1 | 32/8 | 329.152/367.612/371.087 | 26.457/65.333/65.879 | 40.553/46.164/46.558 | 28.253 | completed |
| real-b4-p32-o8-burst | 4 | 32/8 | 281.572/468.537/468.537 | 23.120/72.137/72.137 | 29.919/62.356/62.356 | 123.615 | completed |
| real-b4-p32-o8-staggered | n/a | 32/8 | n/a/n/a/n/a | n/a/n/a/n/a | n/a/n/a/n/a | n/a | failed |

Failure for `real-b4-p32-o8-staggered`: `ValueError: Transformers comparison supports simultaneous arrivals only`

| real-b1-p128-o16 | 1 | 128/16 | 452.580/803.744/870.217 | 52.336/66.389/68.449 | 27.686/49.202/53.460 | 35.952 | completed |

## Measurement and claim boundary

simultaneous host token-ID batch through final greedy token; model loading and tokenization excluded; CUDA explicitly synchronized at each emitted token

Only rows with the same pinned checkpoint, input-token plan, precision, GPU, generation semantics, and synchronization boundary are comparable. TensorForge uses parallel SDPA prefill followed by its transactional paged decode path; the Transformers reference uses SDPA and its standard contiguous KV cache. Loading and tokenization are excluded. Output-token SHA256 values in JSON expose semantic drift.
