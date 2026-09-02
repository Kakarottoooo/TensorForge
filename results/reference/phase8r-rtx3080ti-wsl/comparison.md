# Real-checkpoint backend comparison

- Checkpoint: `TinyLlama/TinyLlama-1.1B-Chat-v1.0@fe8a4ea1ffedaf415f4da2f062534de366a451e6`
- GPU: NVIDIA GeForce RTX 3080 Ti

| Case | Backend | Output tok/s P50 (mean) | P50 relative to TensorForge | Mean latency ms | Mean TTFT ms | Mean TPOT ms | Greedy tokens match | Status |
|---|---|---:|---:|---:|---:|---:|---:|---|
| real-b1-p32-o8 | tensorforge | 26.425 (27.475) | 1.000x | 293.095 | 55.007 | 34.013 | yes | completed |
| real-b4-p32-o8-burst | tensorforge | 114.774 (105.593) | 1.000x | 332.570 | 86.437 | 35.162 | yes | completed |
| real-b4-p32-o8-staggered | tensorforge | 51.926 (56.575) | 1.000x | 623.155 | 184.769 | 62.627 | yes | completed |
| real-b1-p128-o16 | tensorforge | 18.156 (19.994) | 1.000x | 857.309 | 52.218 | 53.673 | yes | completed |
| real-b1-p32-o8 | transformers_sdpa | 24.305 (28.253) | 0.920x | 306.249 | 40.674 | 37.939 | yes | completed |
| real-b4-p32-o8-burst | transformers_sdpa | 113.647 (123.615) | 0.990x | 294.373 | 34.253 | 37.160 | yes | completed |
| real-b4-p32-o8-staggered | transformers_sdpa | n/a (n/a) | n/a | n/a | n/a | n/a | n/a | failed |
| real-b1-p128-o16 | transformers_sdpa | 35.353 (35.952) | 1.947x | 501.123 | 46.780 | 30.290 | no | completed |
| real-b1-p32-o8 | vllm | 129.898 (124.316) | 4.916x | 65.598 | 22.175 | 6.203 | yes | completed |
| real-b4-p32-o8-burst | vllm | 516.303 (516.016) | 4.498x | 61.859 | 16.398 | 6.494 | yes | completed |
| real-b4-p32-o8-staggered | vllm | n/a (n/a) | n/a | n/a | n/a | n/a | n/a | failed |
| real-b1-p128-o16 | vllm | 149.598 (150.741) | 8.240x | 106.954 | 18.630 | 5.888 | no | completed |

## Interpretation

This is an external reference comparison, not a one-factor kernel ablation. vLLM uses its production-oriented FlashAttention, scheduler, paged cache, sampling path, and CUDA Graph defaults; Transformers uses SDPA and a contiguous cache; TensorForge uses parallel SDPA prefill and its custom paged Triton decode path. Model loading and tokenization are excluded for all backends. A greedy-token mismatch can result from BF16 argmax ties and must be interpreted with the separate teacher-forced logit gate.
