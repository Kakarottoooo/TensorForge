# TensorForge real-checkpoint benchmark

- Backend: `tensorforge`
- Checkpoint: `TinyLlama/TinyLlama-1.1B-Chat-v1.0@fe8a4ea1ffedaf415f4da2f062534de366a451e6`
- Config SHA256: `486bedda3a6988332e60d9638a09ca4b260d34ebcf1b19e22cf3b140b63d8fe9`
- GPU: NVIDIA GeForce RTX 3080 Ti (compute capability 8.6)
- Git: `5b22e5e620d0e16b743dafc90fb3eafb79cccbaa`; dirty: `False`

| Case | Batch | Prompt/output | Latency P50/P95/P99 ms | TTFT P50/P95/P99 ms | TPOT P50/P95/P99 ms | Output tok/s | Status |
|---|---:|---:|---:|---:|---:|---:|---|
| real-b1-p32-o8 | 1 | 32/8 | 302.744/315.148/317.007 | 55.154/82.918/87.907 | 35.814/39.177/39.785 | 27.475 | completed |
| real-b4-p32-o8-burst | 4 | 32/8 | 278.808/573.659/573.659 | 66.266/167.501/167.501 | 29.853/58.022/58.022 | 105.593 | completed |
| real-b4-p32-o8-staggered | 4 | 32/8 | 603.767/881.915/885.541 | 136.335/313.153/316.953 | 53.119/101.346/103.332 | 56.575 | completed |
| real-b1-p128-o16 | 1 | 128/16 | 881.270/1149.538/1183.955 | 35.477/101.194/113.734 | 56.405/74.110/76.372 | 19.994 | completed |

## Measurement and claim boundary

scheduled host token IDs through terminal request release; model loading and tokenization excluded; scheduler token selection synchronizes each decode step

Only rows with the same pinned checkpoint, input-token plan, precision, GPU, generation semantics, and synchronization boundary are comparable. TensorForge uses parallel SDPA prefill followed by its transactional paged decode path; the Transformers reference uses SDPA and its standard contiguous KV cache. Loading and tokenization are excluded. Output-token SHA256 values in JSON expose semantic drift.
