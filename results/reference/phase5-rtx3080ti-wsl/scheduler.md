# TensorForge Phase 5 continuous-batching report

- Suite: `d5b3a69c-a9b8-41d9-ac17-80583c261b1c`
- GPU: NVIDIA GeForce RTX 3080 Ti (compute capability 8.6)
- PyTorch / CUDA / Triton: `2.5.1+cu121` / `12.1` / `3.1.0`
- Git: `f3e5b4c1942a47246880c15fb240ce1ff6c0b482`; dirty: `False`

All rows use identical model weights, the same transactional PagedKVCache, and the same Triton GQA decode path. Only admission/refill policy changes. Arrival offsets are seeded logical scheduler steps, not wall-clock Poisson arrivals.

| Workload | Policy | Latency P50/P95/P99 ms | TTFT P50/P95/P99 ms | TPOT P50/P95/P99 ms | Output tok/s | Max batch | Completed/Cancelled/Failed |
|---|---|---:|---:|---:|---:|---:|---:|
| heterogeneous-burst | no_batching | 885.247/1663.347/1767.111 | 835.306/1603.379/1685.010 | 5.583/6.728/8.475 | 62.71 | 1 | 80/0/0 |
| heterogeneous-burst | static_batching | 226.580/434.420/448.920 | 204.250/375.221/385.274 | 7.332/8.375/8.450 | 243.97 | 8 | 80/0/0 |
| heterogeneous-burst | continuous_batching | 206.819/388.815/408.439 | 184.789/359.329/380.562 | 8.170/9.017/9.275 | 273.21 | 8 | 80/0/0 |
| seeded-arrival-churn | no_batching | 778.272/2570.395/2655.417 | 749.344/2549.215/2618.344 | 6.876/17.390/18.145 | 44.33 | 1 | 90/30/0 |
| seeded-arrival-churn | static_batching | 215.665/578.178/701.742 | 174.393/512.196/666.335 | 7.912/21.436/22.118 | 160.11 | 8 | 90/30/0 |
| seeded-arrival-churn | continuous_batching | 139.054/379.349/466.816 | 105.136/326.656/399.641 | 8.657/24.081/24.819 | 204.00 | 8 | 95/25/0 |

## Interpretation boundary

This is a scheduler-policy ablation, not a comparison with the Phase 1 full-prefix oracle, vLLM, FlashAttention, or a production prefill kernel. Prefill currently walks the same one-token paged decode executor. Completed-token throughput excludes work discarded by cancellation; executed-input throughput remains available in JSON.
