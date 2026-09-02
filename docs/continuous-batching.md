# Continuous batching contract

Phase 5 adds policy-parametric request scheduling above the one canonical `PagedKVCache`. The cache
continues to own physical blocks and append transactions; the scheduler owns admission, lifecycle,
fairness policy, and token budgets; the batched runtime owns one GPU token step for every selected
request. None of these layers duplicates cache state.

## Request lifecycle

Every admitted request starts in `queued`. Activation creates its cache sequence and moves it to
`prefill`; consuming the final prompt token moves it to `decode`. A length or EOS decision moves it
to `completed`. Executor/cache errors move only the affected request to `failed`. Explicit
cancellation uses the terminal `cancelled` state so cancellation is distinguishable from model or
allocator failure.

Terminal transition is the reclamation boundary. It removes the request from scheduler queues,
releases any cache reservation plus every committed physical page, returns the sequence slot, and
subtracts the request's declared token budget. Queued cancellation never creates cache state.

## Admission and execution budgets

Admission reserves `prompt_tokens + max_new_tokens` against both `max_request_tokens` and the global
`max_active_token_budget`. Rejection happens before scheduler or cache mutation. The global budget
therefore bounds accepted outstanding work, including queued requests; it is not a proxy for current
GPU memory.

`max_batch_requests` bounds resident request slots. The current decoder executes one input token per
selected request, so `max_batch_tokens` independently bounds the token count in a GPU step and
cannot exceed the request bound. KV capacity is separately bounded by physical pages and maximum
sequence length.

## Policy ablation

- `no_batching` activates and executes at most one request.
- `static_batching` forms a cohort only after the prior cohort drains.
- `continuous_batching` admits queued work into every newly available slot.

All three policies call the same `BatchedPagedDecodeExecutor`, model weights, transactional cache,
and paged GQA Triton kernel. The comparison changes only admission/refill behavior. Per-request RoPE
positions are applied with an explicit batch-aware shape; a shared-sequence RoPE helper would
cross-broadcast heterogeneous request positions.

The executor reserves one token independently for each request. An exhaustion error excludes that
request while peers with valid reservations continue. A later model/kernel error rolls back every
reservation from that call. Successful requests commit only after all layers finish.

## Benchmark and interpretation boundary

`benchmarks/phase5-scheduler.json` defines immutable seeded request plans. The burst workload tests
heterogeneous prompt/generation lengths. The arrival/churn workload adds logical-step arrivals and
cancellations. Policy order is seed-shuffled per repetition to reduce fixed thermal-order bias.
Warmups are excluded.

The JSON retains every request plan and trace plus request latency, TTFT, TPOT, queue time,
completed-output throughput, executed-input throughput, host/CUDA elapsed time, incremental peak
allocation, scheduler steps, maximum batch, terminal counts, and P50/P95/P99 distributions. Token
timestamps are recorded only after a batched GPU-to-host argmax synchronization. Every run asserts
zero active sequences, used pages, reservations, and outstanding token budget at termination.

Arrival and cancellation offsets are deterministic scheduler steps, not wall-clock Poisson events.
A faster policy can finish a cancellation target before its cancellation step; reports retain both
planned and missed cancellation counts. Completed-output throughput excludes discarded work, while
executed-input throughput exposes it.

This phase does not implement a parallel prefill kernel: prompt ingestion intentionally walks the
one-token paged path. It also does not claim CUDA Graph capture, `torch.compile`, speculative decode,
vLLM parity, or production service-level arrival modeling.
