# TensorForge Phase 4 paged GQA decode-attention report

- Suite: `0b888e8f-4ad3-4cc2-8a02-2febfb25b132`
- GPU: NVIDIA GeForce RTX 3080 Ti (compute capability 8.6)
- PyTorch / CUDA / Triton: `2.5.1+cu121` / `12.1` / `3.1.0`
- Git: `42e4782fc71656cb27af082f0509db5c5f7098f7`; dirty: `False`
- Empirical copy ceiling: 802.89 GB/s
- Selected auto-path regressions versus PyTorch expanded GQA: **0**
- Forced one-pass ablation regressions: **0**

Steady-state timings exclude first-call JIT/autotune. The PyTorch baseline performs GQA head expansion inside the timed region; Triton output and split workspace are preallocated. Effective bandwidth uses semantic logical work and is not a DRAM-counter measurement.

| Case | Dtype | B | Q/KV heads | D | Context | Implementation | P50 ms | P95 ms | Speedup | Logical GB/s | Outcome |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---|
| default-c1-b1 | fp16 | 1 | 8/4 | 64 | 1 | torch_expanded_gqa | 0.024576 | 0.027648 | 1.000x | 0.13 | baseline |
| default-c1-b1 | fp16 | 1 | 8/4 | 64 | 1 | triton_auto_preallocated | 0.006144 | 0.006144 | 4.000x | 0.50 | improvement |
| default-c17-b1 | fp16 | 1 | 8/4 | 64 | 17 | torch_expanded_gqa | 0.024576 | 0.221184 | 1.000x | 0.79 | baseline |
| default-c17-b1 | fp16 | 1 | 8/4 | 64 | 17 | triton_auto_preallocated | 0.006144 | 0.006144 | 4.000x | 3.17 | improvement |
| default-c128-b1 | fp16 | 1 | 8/4 | 64 | 128 | torch_expanded_gqa | 0.230704 | 0.544717 | 1.000x | 0.58 | baseline |
| default-c128-b1 | fp16 | 1 | 8/4 | 64 | 128 | triton_auto_preallocated | 0.009216 | 0.009216 | 25.033x | 14.45 | improvement |
| default-c512-b1 | fp16 | 1 | 8/4 | 64 | 512 | torch_expanded_gqa | 0.028672 | 0.031744 | 1.000x | 18.36 | baseline |
| default-c512-b1 | fp16 | 1 | 8/4 | 64 | 512 | triton_auto_preallocated | 0.022528 | 0.023552 | 1.273x | 23.37 | improvement |
| default-c2048-b1 | fp16 | 1 | 8/4 | 64 | 2048 | torch_expanded_gqa | 0.136336 | 0.461824 | 1.000x | 15.40 | baseline |
| default-c2048-b1 | fp16 | 1 | 8/4 | 64 | 2048 | triton_one_pass_ablation | 0.072704 | 0.074752 | 1.875x | 28.88 | improvement |
| default-c2048-b1 | fp16 | 1 | 8/4 | 64 | 2048 | triton_auto_preallocated | 0.020480 | 0.021504 | 6.657x | 102.53 | improvement |
| default-c128-b8 | fp16 | 8 | 8/4 | 64 | 128 | torch_expanded_gqa | 0.045056 | 0.046080 | 1.000x | 23.64 | baseline |
| default-c128-b8 | fp16 | 8 | 8/4 | 64 | 128 | triton_auto_preallocated | 0.010240 | 0.011264 | 4.400x | 104.03 | improvement |
| default-c512-b8 | fp16 | 8 | 8/4 | 64 | 512 | torch_expanded_gqa | 0.301152 | 0.583014 | 1.000x | 13.99 | baseline |
| default-c512-b8 | fp16 | 8 | 8/4 | 64 | 512 | triton_auto_preallocated | 0.023552 | 0.023584 | 12.787x | 178.83 | improvement |
| 7b-c1-b1 | fp16 | 1 | 32/8 | 128 | 1 | torch_expanded_gqa | 0.022528 | 0.026384 | 1.000x | 0.91 | baseline |
| 7b-c1-b1 | fp16 | 1 | 32/8 | 128 | 1 | triton_auto_preallocated | 0.006144 | 0.007168 | 3.667x | 3.33 | improvement |
| 7b-c128-b1 | fp16 | 1 | 32/8 | 128 | 128 | torch_expanded_gqa | 0.030720 | 0.034816 | 1.000x | 17.60 | baseline |
| 7b-c128-b1 | fp16 | 1 | 32/8 | 128 | 128 | triton_auto_preallocated | 0.012288 | 0.012288 | 2.500x | 44.00 | improvement |
| 7b-c512-b1 | fp16 | 1 | 32/8 | 128 | 512 | torch_expanded_gqa | 0.051200 | 0.055296 | 1.000x | 41.28 | baseline |
| 7b-c512-b1 | fp16 | 1 | 32/8 | 128 | 512 | triton_auto_preallocated | 0.037888 | 0.038912 | 1.351x | 55.79 | improvement |
| 7b-c2048-b1 | fp16 | 1 | 32/8 | 128 | 2048 | torch_expanded_gqa | 0.219392 | 0.523674 | 1.000x | 38.31 | baseline |
| 7b-c2048-b1 | fp16 | 1 | 32/8 | 128 | 2048 | triton_one_pass_ablation | 0.133120 | 0.357376 | 1.648x | 63.14 | improvement |
| 7b-c2048-b1 | fp16 | 1 | 32/8 | 128 | 2048 | triton_auto_preallocated | 0.037888 | 0.038912 | 5.791x | 221.85 | improvement |
| 7b-c512-b8 | fp16 | 8 | 32/8 | 128 | 512 | torch_expanded_gqa | 0.532480 | 1.078886 | 1.000x | 31.76 | baseline |
| 7b-c512-b8 | fp16 | 8 | 32/8 | 128 | 512 | triton_auto_preallocated | 0.072704 | 0.075776 | 7.324x | 232.58 | improvement |
| 7b-c512-b1-bf16 | bf16 | 1 | 32/8 | 128 | 512 | torch_expanded_gqa | 0.056320 | 0.398336 | 1.000x | 37.53 | baseline |
| 7b-c512-b1-bf16 | bf16 | 1 | 32/8 | 128 | 512 | triton_auto_preallocated | 0.036864 | 0.037888 | 1.528x | 57.34 | improvement |
| default-c128-b1-fp32 | fp32 | 1 | 8/4 | 64 | 128 | torch_expanded_gqa | 0.115712 | 0.370586 | 1.000x | 2.30 | baseline |
| default-c128-b1-fp32 | fp32 | 1 | 8/4 | 64 | 128 | triton_auto_preallocated | 0.013312 | 0.014336 | 8.692x | 20.00 | improvement |

## Interpretation boundary

Logical bytes are not measured DRAM transactions. The PyTorch baseline expands GQA heads in the timed region, so semantic-work bandwidth is not physical traffic.
