# Results

`reference/` contains curated, commit-associated measurements. `local/` is ignored and is the
default location for exploratory results. Chrome traces are ignored because they are large; the
curated profile JSON/Markdown summaries retain operator totals and the claim boundary.

Never compare rows whose hardware fingerprint, workload identity, model semantics, precision, or
measurement boundary differs without explicitly accounting for that difference.

Phase 3 kernel reports include every tested shape, PyTorch and Triton rows, selected autotune config,
first-call tuning/JIT wall time, numerical error, logical byte/FLOP model, empirical copy ceiling,
and explicit improvement/regression labels.
