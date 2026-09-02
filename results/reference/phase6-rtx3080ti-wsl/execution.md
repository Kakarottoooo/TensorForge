# TensorForge Phase 6 execution-specialization report

- Suite: `5cdd5b28-5937-46cb-9753-d7077672c0ce`
- GPU: NVIDIA GeForce RTX 3080 Ti (compute capability 8.6)
- PyTorch / CUDA / Triton: `2.5.1+cu121` / `12.1` / `3.1.0`
- Git: `9160667c34f2e3b6299649e52c4dcf89917db838`; dirty: `False`

Setup/capture/compile costs are excluded from steady-state latency and reported separately. Every run passes an independent full-prefix numerical gate.

| Case | Mode | CUDA latency P50/P95/P99 ms | Mean tok/s | Mean change vs eager | Graph hit/miss | Shape fallback | Cold setup/capture/compile ms |
|---|---|---:|---:|---:|---:|---:|---:|
| eligible-b1 | eager | 16.227/17.835/18.021 | 112.91 | +0.0% | 0/0 | 0 | 1374.68/0.00/0.00 |
| eligible-b1 | torch_compile | 9.280/10.185/10.767 | 156.43 | -38.1% | 0/0 | 0 | 5139.09/0.00/5105.76 |
| eligible-b1 | cuda_graph | 1.125/1.748/1.930 | 886.51 | -90.4% | 80/0 | 0 | 164.05/146.17/0.00 |
| eligible-b4 | eager | 9.726/18.193/18.615 | 390.58 | +0.0% | 0/0 | 0 | 584.18/0.00/0.00 |
| eligible-b4 | torch_compile | 6.604/10.483/10.843 | 592.78 | -35.8% | 0/0 | 0 | 2851.31/0.00/2776.64 |
| eligible-b4 | cuda_graph | 1.482/2.325/2.445 | 2624.08 | -85.8% | 80/0 | 0 | 189.99/159.13/0.00 |
| eligible-b8 | eager | 18.910/21.364/22.291 | 588.60 | +0.0% | 0/0 | 0 | 60.83/0.00/0.00 |
| eligible-b8 | torch_compile | 11.125/14.717/15.339 | 1007.65 | -42.8% | 0/0 | 0 | 1941.21/0.00/1905.09 |
| eligible-b8 | cuda_graph | 3.418/4.547/4.980 | 2679.36 | -80.0% | 80/0 | 0 | 173.20/153.64/0.00 |
| batch-shape-fallback | eager | 21.006/26.671/27.504 | 188.19 | +0.0% | 0/0 | 80 | 2356.91/0.00/0.00 |
| batch-shape-fallback | torch_compile | 19.686/25.910/29.976 | 250.80 | -16.1% | 0/0 | 80 | 47.58/0.00/0.00 |
| batch-shape-fallback | cuda_graph | 19.858/23.016/24.147 | 236.12 | -16.7% | 0/80 | 80 | 55.66/0.00/0.00 |

## Interpretation boundary

This isolates execution specialization over the same address-stable bucket, model weights, transactional PagedKVCache, Triton KV writer, and Triton GQA attention. `torch.compile` is intentionally a hybrid segmented path because custom Triton calls are explicit graph boundaries. The fallback case measures dynamic eager delegation, not a captured or compiled shape. Regressions are retained.
