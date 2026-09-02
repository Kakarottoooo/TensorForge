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

- `reference/phase3-rtx3080ti-wsl/kernels.{json,csv,md}`: 22 correctness-gated Phase 3 kernel cases
  measured on clean commit `0bd1d9e` under WSL2, PyTorch 2.5.1+cu121, and Triton 3.1.0.
- `reference/phase2-rtx3080ti/`: Phase 2 eager baseline and profiler summaries on Windows.
