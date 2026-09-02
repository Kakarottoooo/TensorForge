# TensorForge Phase 7A cumulative decode-ablation report

- Suite: `caa3d1c6-0dd1-4b45-bbc3-35342e8eea9c`
- GPU: NVIDIA GeForce RTX 3080 Ti (compute capability 8.6)
- PyTorch / CUDA / Triton: `2.5.1+cu121` / `12.1` / `3.1.0`
- Git: `b268f16bcfa7c7adea45b9d3fc3ca9c92f557a38`; dirty: `False`

Setup/capture/compile costs are excluded from steady-state latency and reported separately. Every run passes an independent full-prefix numerical gate.

| Case | Variant | Mode/fusion | CUDA latency P50/P95/P99 ms | Mean tok/s | Change vs baseline/parent | Graph hit/miss | Cold setup/capture/compile ms |
|---|---|---|---:|---:|---:|---:|---:|
| b1-c32 | paged_dynamic_eager | eager/none | 15.871/17.319/17.831 | 100.48 | +0.0%/+0.0% | 0/0 | 1748.01/0.00/0.00 |
| b1-c32 | stable_bucket_eager | eager/none | 8.122/15.956/16.134 | 118.74 | -16.9%/-16.9% | 0/0 | 24.91/0.00/0.00 |
| b1-c32 | triton_rms_norm | eager/rms_norm | 4.826/11.376/11.846 | 180.41 | -48.7%/-38.2% | 0/0 | 411.64/0.00/0.00 |
| b1-c32 | triton_residual_rms_norm | eager/residual_rms_norm | 4.308/11.230/11.384 | 188.67 | -50.2%/-3.0% | 0/0 | 416.86/0.00/0.00 |
| b1-c32 | triton_all_fusions | eager/all | 5.910/11.349/11.616 | 171.91 | -47.0%/+6.6% | 0/0 | 386.67/0.00/0.00 |
| b1-c32 | triton_all_compile | torch_compile/all | 5.139/8.892/9.400 | 190.02 | -53.9%/-13.1% | 0/0 | 3035.84/0.00/3025.97 |
| b1-c32 | triton_all_cuda_graph | cuda_graph/all | 0.818/1.190/1.399 | 1297.71 | -93.5%/-87.8% | 80/0 | 157.08/137.18/0.00 |
| b4-c32 | paged_dynamic_eager | eager/none | 6.728/19.661/20.126 | 505.80 | +0.0%/+0.0% | 0/0 | 144.81/0.00/0.00 |
| b4-c32 | stable_bucket_eager | eager/none | 5.649/17.427/17.915 | 528.73 | +7.2%/+7.2% | 0/0 | 648.73/0.00/0.00 |
| b4-c32 | triton_rms_norm | eager/rms_norm | 5.672/12.992/13.543 | 598.63 | -14.6%/-20.3% | 0/0 | 810.48/0.00/0.00 |
| b4-c32 | triton_residual_rms_norm | eager/residual_rms_norm | 4.993/12.884/13.282 | 623.65 | -16.4%/-2.2% | 0/0 | 921.68/0.00/0.00 |
| b4-c32 | triton_all_fusions | eager/all | 4.863/13.174/13.716 | 723.87 | -31.6%/-18.2% | 0/0 | 802.70/0.00/0.00 |
| b4-c32 | triton_all_compile | torch_compile/all | 5.129/10.545/10.702 | 663.58 | -30.3%/+1.9% | 0/0 | 1065.97/0.00/1006.53 |
| b4-c32 | triton_all_cuda_graph | cuda_graph/all | 2.009/2.702/2.996 | 2222.47 | -80.2%/-71.0% | 80/0 | 208.05/138.83/0.00 |
| b8-c32 | paged_dynamic_eager | eager/none | 10.196/23.985/24.983 | 726.69 | +0.0%/+0.0% | 0/0 | 80.06/0.00/0.00 |
| b8-c32 | stable_bucket_eager | eager/none | 6.309/10.013/10.329 | 1156.57 | -43.9%/-43.9% | 0/0 | 75.63/0.00/0.00 |
| b8-c32 | triton_rms_norm | eager/rms_norm | 5.756/8.247/8.692 | 1293.07 | -50.0%/-11.0% | 0/0 | 494.14/0.00/0.00 |
| b8-c32 | triton_residual_rms_norm | eager/residual_rms_norm | 5.429/7.867/8.275 | 1386.01 | -53.8%/-7.6% | 0/0 | 473.33/0.00/0.00 |
| b8-c32 | triton_all_fusions | eager/all | 7.828/14.726/14.987 | 1019.42 | -26.0%/+60.4% | 0/0 | 603.30/0.00/0.00 |
| b8-c32 | triton_all_compile | torch_compile/all | 5.578/12.176/13.080 | 1228.58 | -43.8%/-24.1% | 0/0 | 1253.98/0.00/1169.55 |
| b8-c32 | triton_all_cuda_graph | cuda_graph/all | 1.754/2.334/2.764 | 4324.02 | -85.3%/-80.2% | 80/0 | 231.42/152.03/0.00 |
| b1-c128 | paged_dynamic_eager | eager/none | 5.797/17.357/17.849 | 150.02 | +0.0%/+0.0% | 0/0 | 954.42/0.00/0.00 |
| b1-c128 | stable_bucket_eager | eager/none | 5.421/9.569/11.735 | 169.79 | -26.2%/-26.2% | 0/0 | 27.01/0.00/0.00 |
| b1-c128 | triton_rms_norm | eager/rms_norm | 4.841/12.879/13.242 | 163.51 | -6.1%/+27.2% | 0/0 | 25.73/0.00/0.00 |
| b1-c128 | triton_residual_rms_norm | eager/residual_rms_norm | 4.515/6.703/7.745 | 207.60 | -40.0%/-36.1% | 0/0 | 26.35/0.00/0.00 |
| b1-c128 | triton_all_fusions | eager/all | 4.766/12.275/12.474 | 184.06 | -22.7%/+28.8% | 0/0 | 26.55/0.00/0.00 |
| b1-c128 | triton_all_compile | torch_compile/all | 4.834/7.037/7.541 | 191.27 | -34.7%/-15.6% | 0/0 | 34.51/0.00/10.88 |
| b1-c128 | triton_all_cuda_graph | cuda_graph/all | 0.886/1.805/2.121 | 1015.51 | -86.9%/-83.1% | 80/0 | 173.54/143.70/0.00 |
| b1-c512 | paged_dynamic_eager | eager/none | 6.916/20.580/21.132 | 127.54 | +0.0%/+0.0% | 0/0 | 597.29/0.00/0.00 |
| b1-c512 | stable_bucket_eager | eager/none | 9.315/20.332/20.829 | 107.87 | +26.3%/+26.3% | 0/0 | 96.37/0.00/0.00 |
| b1-c512 | triton_rms_norm | eager/rms_norm | 14.660/16.486/16.638 | 109.65 | +20.2%/-4.8% | 0/0 | 96.21/0.00/0.00 |
| b1-c512 | triton_residual_rms_norm | eager/residual_rms_norm | 14.130/15.894/16.052 | 107.55 | +18.7%/-1.2% | 0/0 | 111.87/0.00/0.00 |
| b1-c512 | triton_all_fusions | eager/all | 6.700/15.508/15.642 | 128.76 | +0.6%/-15.2% | 0/0 | 80.42/0.00/0.00 |
| b1-c512 | triton_all_compile | torch_compile/all | 11.957/13.546/14.402 | 119.17 | +2.2%/+1.5% | 0/0 | 87.36/0.00/10.96 |
| b1-c512 | triton_all_cuda_graph | cuda_graph/all | 2.288/5.108/5.246 | 386.75 | -69.0%/-69.2% | 80/0 | 443.61/145.26/0.00 |
| b1-c2048 | paged_dynamic_eager | eager/none | 11.909/33.271/33.743 | 67.14 | +0.0%/+0.0% | 0/0 | 1374.52/0.00/0.00 |
| b1-c2048 | stable_bucket_eager | eager/none | 10.553/31.816/32.367 | 81.90 | -24.2%/-24.2% | 0/0 | 318.80/0.00/0.00 |
| b1-c2048 | triton_rms_norm | eager/rms_norm | 26.512/28.365/28.862 | 49.01 | +21.2%/+59.8% | 0/0 | 322.07/0.00/0.00 |
| b1-c2048 | triton_residual_rms_norm | eager/residual_rms_norm | 11.191/27.774/28.340 | 75.05 | -14.3%/-29.2% | 0/0 | 297.53/0.00/0.00 |
| b1-c2048 | triton_all_fusions | eager/all | 11.557/27.680/28.016 | 72.95 | -12.8%/+1.7% | 0/0 | 349.37/0.00/0.00 |
| b1-c2048 | triton_all_compile | torch_compile/all | 10.403/24.312/25.199 | 88.42 | -36.4%/-27.0% | 0/0 | 301.61/0.00/7.34 |
| b1-c2048 | triton_all_cuda_graph | cuda_graph/all | 6.224/15.843/16.744 | 143.55 | -59.0%/-52.9% | 80/0 | 499.83/148.44/0.00 |

## Interpretation boundary

Rows share model weights, transactional PagedKVCache, and paged GQA attention. Each variant names its causal parent; compile and CUDA Graph both compare with the fully fused eager parent rather than with each other. `torch.compile` remains segmented because custom Triton calls are explicit graph boundaries. Reference prefill is an excluded setup mechanism, not a production prefill claim. Regressions are retained.
