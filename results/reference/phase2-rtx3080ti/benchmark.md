# TensorForge benchmark report

- Suite ID: `7eba7f8e-99ed-4805-800b-63bb6f73a3cd`
- Schema: `1.0`
- Started: `2026-09-02T01:54:27.358061+00:00`
- Finished: `2026-09-02T01:54:50.315505+00:00`
- GPU(s): NVIDIA GeForce RTX 3080 Ti
- PyTorch: `2.5.1+cu118`; CUDA runtime: `11.8`; driver: `591.86`
- Hardware fingerprint: `98ff48931a4550afe452c95d04112eda140d4d2cc15b9a54ee33234002f12e7b`

Host percentiles are computed from request-level samples using linear interpolation. CUDA elapsed time is separately measured with device events. GPU utilization is sampled out-of-process and must be interpreted with its sample count in the raw JSON.

| Case | Precision | Prompt / output | Batch requested / run | TTFT P50 / P95 / P99 (ms) | TPOT P50 / P95 / P99 (ms) | Output tok/s | Peak allocated MiB | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| reference-p128-g32-b1-fp16 | fp16 | 128 / 32 | 1 / 1 | 18.204 / 34.001 / 35.090 | 13.575 / 17.524 / 18.172 | 71.885 | 136.675 | completed |
| reference-p512-g32-b4-fp16 | fp16 | 512 / 32 | 4 / 4 | 16.131 / 27.495 / 27.495 | 19.045 / 27.096 / 27.096 | 215.246 | 383.442 | completed |
| reference-p2048-g32-b1-fp16 | fp16 | 2048 / 32 | 1 / 1 | 23.957 / 24.533 / 24.575 | 25.138 / 25.330 / 25.362 | 39.884 | 522.945 | completed |

## Claim boundary

These measurements apply only to the exact random-weight model dimensions, workload, software stack, commit, and hardware fingerprint recorded above. They are not language-quality results and should not be compared with another runtime unless model semantics, precision, request set, synchronization, warmup, and measurement boundaries match.
