# TensorForge Phase 3 kernel report

- Suite: `7a928e29-6ee4-4f2f-95ab-f473bf54f158`
- GPU: NVIDIA GeForce RTX 3080 Ti (compute capability 8.6)
- PyTorch / CUDA / Triton: `2.5.1+cu121` / `12.1` / `3.1.0`
- Git: `0bd1d9eb77d6cdf1ae12627a205ebc1b43d61826`; dirty: `False`
- Empirical copy ceiling: 814.11 GB/s
- FP32 compute ceiling: 34201.6 GFLOP/s (10240 CUDA cores * 1.67 GHz boost * 2 FLOP/cycle)
- GPU state before: `{'gpu_selector': '0', 'temperature_c': 54.0, 'power_draw_w': 122.68, 'power_limit_w': 350.0, 'sm_clock_mhz': 1755, 'memory_clock_mhz': 9501}`
- GPU state after: `{'gpu_selector': '0', 'temperature_c': 60.0, 'power_draw_w': 271.4, 'power_limit_w': 350.0, 'sm_clock_mhz': 1950, 'memory_clock_mhz': 9251}`
- Triton regressions versus PyTorch: **0**

Steady-state timings exclude first-call JIT/autotune. P05/P50/P95 are CUDA-event microbenchmark quantiles. Roofline uses logical unique bytes and the empirical copy ceiling; it is not a DRAM-counter measurement.

| Case | Op | Dtype | Shape | Implementation | P50 ms | P95 ms | Speedup | GB/s | Copy ceiling % | Outcome |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---|
| rms-odd | rms_norm | fp16 | 17x513 | torch_standard | 0.020480 | 0.024757 | 1.000x | 1.75 | 0.2% | baseline |
| rms-odd | rms_norm | fp16 | 17x513 | triton_autotuned | 0.005120 | 0.005120 | 4.000x | 7.01 | 0.9% | improvement |
| rms-decode-1 | rms_norm | fp16 | 1x4096 | torch_standard | 0.019456 | 0.022592 | 1.000x | 1.26 | 0.2% | baseline |
| rms-decode-1 | rms_norm | fp16 | 1x4096 | triton_autotuned | 0.005120 | 0.006144 | 3.800x | 4.80 | 0.6% | improvement |
| rms-decode-16 | rms_norm | fp16 | 16x4096 | torch_standard | 0.025600 | 0.030472 | 1.000x | 10.56 | 1.3% | baseline |
| rms-decode-16 | rms_norm | fp16 | 16x4096 | triton_autotuned | 0.006144 | 0.007168 | 4.167x | 44.00 | 5.4% | improvement |
| rms-prefill-default | rms_norm | fp16 | 2048x512 | torch_standard | 0.033792 | 0.037888 | 1.000x | 124.15 | 15.2% | baseline |
| rms-prefill-default | rms_norm | fp16 | 2048x512 | triton_autotuned | 0.009216 | 0.010240 | 3.667x | 455.22 | 55.9% | improvement |
| rms-prefill-wide | rms_norm | fp16 | 128x4096 | torch_standard | 0.030720 | 0.033792 | 1.000x | 68.53 | 8.4% | baseline |
| rms-prefill-wide | rms_norm | fp16 | 128x4096 | triton_autotuned | 0.007168 | 0.008192 | 4.286x | 293.71 | 36.1% | improvement |
| rms-decode-bf16 | rms_norm | bf16 | 16x4096 | torch_standard | 0.025600 | 0.027648 | 1.000x | 10.56 | 1.3% | baseline |
| rms-decode-bf16 | rms_norm | bf16 | 16x4096 | triton_autotuned | 0.005120 | 0.006144 | 5.000x | 52.80 | 6.5% | improvement |
| rms-decode-fp32 | rms_norm | fp32 | 16x4096 | torch_standard | 0.025600 | 0.026624 | 1.000x | 21.12 | 2.6% | baseline |
| rms-decode-fp32 | rms_norm | fp32 | 16x4096 | triton_autotuned | 0.006144 | 0.007168 | 4.167x | 88.00 | 10.8% | improvement |
| resnorm-odd | residual_rms_norm | fp16 | 17x513 | torch_standard | 0.023552 | 0.026470 | 1.000x | 3.01 | 0.4% | baseline |
| resnorm-odd | residual_rms_norm | fp16 | 17x513 | triton_autotuned | 0.006144 | 0.006144 | 3.833x | 11.52 | 1.4% | improvement |
| resnorm-decode-1 | residual_rms_norm | fp16 | 1x4096 | torch_standard | 0.021504 | 0.024115 | 1.000x | 1.90 | 0.2% | baseline |
| resnorm-decode-1 | residual_rms_norm | fp16 | 1x4096 | triton_autotuned | 0.005120 | 0.006144 | 4.200x | 8.00 | 1.0% | improvement |
| resnorm-decode-16 | residual_rms_norm | fp16 | 16x4096 | torch_standard | 0.028672 | 0.033792 | 1.000x | 18.57 | 2.3% | baseline |
| resnorm-decode-16 | residual_rms_norm | fp16 | 16x4096 | triton_autotuned | 0.006144 | 0.007168 | 4.667x | 86.67 | 10.6% | improvement |
| resnorm-prefill-default | residual_rms_norm | fp16 | 2048x512 | torch_standard | 0.040960 | 0.043008 | 1.000x | 204.83 | 25.2% | baseline |
| resnorm-prefill-default | residual_rms_norm | fp16 | 2048x512 | triton_autotuned | 0.014336 | 0.015360 | 2.857x | 585.21 | 71.9% | improvement |
| resnorm-prefill-wide | residual_rms_norm | fp16 | 128x4096 | torch_standard | 0.035840 | 0.039936 | 1.000x | 117.26 | 14.4% | baseline |
| resnorm-prefill-wide | residual_rms_norm | fp16 | 128x4096 | triton_autotuned | 0.010208 | 0.011264 | 3.511x | 411.69 | 50.6% | improvement |
| resnorm-decode-bf16 | residual_rms_norm | bf16 | 16x4096 | torch_standard | 0.028672 | 0.032778 | 1.000x | 18.57 | 2.3% | baseline |
| resnorm-decode-bf16 | residual_rms_norm | bf16 | 16x4096 | triton_autotuned | 0.006144 | 0.007168 | 4.667x | 86.67 | 10.6% | improvement |
| resnorm-decode-fp32 | residual_rms_norm | fp32 | 16x4096 | torch_standard | 0.029664 | 0.030720 | 1.000x | 35.90 | 4.4% | baseline |
| resnorm-decode-fp32 | residual_rms_norm | fp32 | 16x4096 | triton_autotuned | 0.007168 | 0.008192 | 4.138x | 148.57 | 18.2% | improvement |
| swiglu-odd | swiglu | fp16 | 17x513 | torch_standard | 0.007168 | 0.008192 | 1.000x | 7.30 | 0.9% | baseline |
| swiglu-odd | swiglu | fp16 | 17x513 | triton_autotuned | 0.004096 | 0.005120 | 1.750x | 12.77 | 1.6% | improvement |
| swiglu-decode-default | swiglu | fp16 | 1x1376 | torch_standard | 0.007168 | 0.008192 | 1.000x | 1.15 | 0.1% | baseline |
| swiglu-decode-default | swiglu | fp16 | 1x1376 | triton_autotuned | 0.004096 | 0.005120 | 1.750x | 2.02 | 0.2% | improvement |
| swiglu-batch-default | swiglu | fp16 | 16x1376 | torch_standard | 0.007264 | 0.008656 | 1.000x | 18.19 | 2.2% | baseline |
| swiglu-batch-default | swiglu | fp16 | 16x1376 | triton_autotuned | 0.005120 | 0.006144 | 1.419x | 25.80 | 3.2% | improvement |
| swiglu-prefill-default | swiglu | fp16 | 2048x1376 | torch_standard | 0.043008 | 0.046806 | 1.000x | 393.14 | 48.3% | baseline |
| swiglu-prefill-default | swiglu | fp16 | 2048x1376 | triton_autotuned | 0.026624 | 0.026624 | 1.615x | 635.08 | 78.0% | improvement |
| swiglu-decode-7b | swiglu | fp16 | 1x11008 | torch_standard | 0.007168 | 0.008192 | 1.000x | 9.21 | 1.1% | baseline |
| swiglu-decode-7b | swiglu | fp16 | 1x11008 | triton_autotuned | 0.005120 | 0.006144 | 1.400x | 12.90 | 1.6% | improvement |
| swiglu-prefill-7b | swiglu | fp16 | 128x11008 | torch_standard | 0.021504 | 0.022528 | 1.000x | 393.14 | 48.3% | baseline |
| swiglu-prefill-7b | swiglu | fp16 | 128x11008 | triton_autotuned | 0.015360 | 0.016384 | 1.400x | 550.40 | 67.6% | improvement |
| swiglu-batch-bf16 | swiglu | bf16 | 16x1376 | torch_standard | 0.007184 | 0.008192 | 1.000x | 18.39 | 2.3% | baseline |
| swiglu-batch-bf16 | swiglu | bf16 | 16x1376 | triton_autotuned | 0.005024 | 0.005120 | 1.430x | 26.29 | 3.2% | improvement |
| swiglu-batch-fp32 | swiglu | fp32 | 16x1376 | torch_standard | 0.008192 | 0.009216 | 1.000x | 32.25 | 4.0% | baseline |
| swiglu-batch-fp32 | swiglu | fp32 | 16x1376 | triton_autotuned | 0.005120 | 0.005120 | 1.600x | 51.60 | 6.3% | improvement |

## Interpretation boundary

Arithmetic intensity counts rsqrt/sigmoid as one special-function operation. Logical bytes count unique tensor traffic and shared weights once; actual DRAM transactions require Nsight Compute. Low arithmetic intensity makes these memory/launch candidates, but the report does not promote modeled traffic to measured bandwidth.
