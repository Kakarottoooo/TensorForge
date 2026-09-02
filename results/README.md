# Results

`reference/` contains curated, commit-associated measurements. `local/` is ignored and is the
default location for exploratory results. Chrome traces are ignored because they are large; the
curated profile JSON/Markdown summaries retain operator totals and the claim boundary.

Never compare rows whose hardware fingerprint, workload identity, model semantics, precision, or
measurement boundary differs without explicitly accounting for that difference.

Phase 3 kernel reports include every tested shape, PyTorch and Triton rows, selected autotune config,
first-call tuning/JIT wall time, numerical error, logical byte/FLOP model, empirical copy ceiling,
and explicit improvement/regression labels.

Curated reports:

- `reference/phase8r-rtx3080ti-wsl/`: pinned TinyLlama 1.1B BF16 TensorForge,
  Transformers SDPA, and vLLM reference reports plus cross-backend comparison and the vectorized
  paged-cache write ablation, measured on clean commit `5b22e5e`.
- `reference/phase7a-rtx3080ti-wsl/execution.{json,csv,md}`: cumulative dynamic paged eager,
  address-stable bucket, Triton RMSNorm, fused residual/RMSNorm, Triton SwiGLU, segmented compile,
  and explicit CUDA Graph ablation on clean commit `b268f16`.
- `reference/phase6-rtx3080ti-wsl/execution.{json,csv,md}`: address-stable eager,
  segmented `torch.compile`, explicit CUDA Graph, and forced shape-fallback ablation with cold setup
  cost and steady-state P50/P95/P99 on clean commit `9160667`.
- `reference/phase5-rtx3080ti-wsl/scheduler.{json,csv,md}`: no/static/continuous request-policy
  ablation with full traces, seeded churn, terminal counts, and P50/P95/P99 on clean commit
  `f3e5b4c`.
- `reference/phase4-rtx3080ti-wsl/attention.{json,csv,md}`: paged GQA decode matrix plus forced
  one-pass versus auto split-KV long-context ablation on clean commit `42e4782`.
- `reference/phase3-rtx3080ti-wsl/kernels.{json,csv,md}`: 22 correctness-gated Phase 3 kernel cases
  measured on clean commit `0bd1d9e` under WSL2, PyTorch 2.5.1+cu121, and Triton 3.1.0.
- `reference/phase2-rtx3080ti/`: Phase 2 eager baseline and profiler summaries on Windows.
