# Benchmark and profiling methodology

## Measurement contract

TensorForge separates host-observed service metrics from CUDA device execution time. A request trace
contains its arrival, runtime admission, every completed output token, and completion timestamp.
TTFT is arrival to first completed token; TPOT is first-to-last token time divided by the remaining
token intervals; request latency includes queueing. Suite throughput is total completed output tokens
divided by the interval from earliest arrival to latest completion. P50/P95/P99 use Hyndman-Fan type
7 linear interpolation over request-level samples.

CUDA Events report device elapsed time for the same execution region. Host metrics include Python,
launch, explicit synchronization, and runtime overhead and therefore must not be substituted with
CUDA Event time. The eager baseline synchronizes after each generated token because a token is not
host-observable until its producing work completes. Future asynchronous schedulers retain CUDA event
dependencies and emit completion events without forcing a global device synchronization.

## Experiment identity

Every case hashes the full immutable workload into a stable case ID. The workload includes model
dimensions, precision, prompt/output lengths, requested batch and concurrency, arrival offsets,
warmups, repetitions, seed, sampling interval, and a structured execution identity:

- runtime backend and eager/compile/CUDA Graph/external mode;
- attention, RMSNorm, and SwiGLU implementation names;
- no/contiguous/paged cache policy;
- no/static/continuous scheduling policy;
- standard/speculative decode and draft model identity;
- tensor-parallel world size and explicit extra tags.

The suite records GPU identity, compute capability, VRAM, driver, PyTorch CUDA runtime, cuDNN,
Triton, CPU, OS, Python, visible GPU mapping, Git commit and dirty state. NVIDIA driver capability
and PyTorch's embedded CUDA runtime are intentionally separate fields.

## Warmup, synchronization, and memory

Model construction and deterministic prompt generation occur outside timed regions. Warmups execute
the identical backend and request set. Each measured repetition resets PyTorch peak-memory counters
after warmup, records allocated memory before and after execution, and records the absolute peak.
It does not infer cache usage from reserved memory. Later paged-cache accounting will be emitted as
backend counters and reconciled against allocator measurements.

GPU utilization and device-wide used memory are sampled by a persistent `nvidia-smi` process at a
recorded interval. These samples are coarse and device-wide; the raw JSON includes sample counts and
an unavailable reason for runs too short to sample. They are supporting observations, not kernel
utilization counters.

## Capacity handling

Only `torch.OutOfMemoryError` triggers automatic reduction. Batch size is halved, model and cached
allocations are released, and the exact requested/executed dimensions plus reason are retained.
Other failures abort instead of being mislabeled as capacity limits. A batch-one OOM is recorded as
a failed case. Prompt length, generation length, precision, and model dimensions never change
silently.

## Profiling and bottleneck claims

`torch.profiler` captures CPU/CUDA activity, shapes, memory, and named prefill/decode regions. The
analysis ranks self-device time and forms a **provisional candidate** classification from matrix-
multiply share, pointwise/layout share, and repeated short-kernel share. It cannot observe achieved
FLOP/s, DRAM bytes, occupancy, cache hit rate, or stall reasons, so it does not make a roofline claim.
The short-kernel share is nonexclusive and overlaps the operator categories; the shares must not be
summed as a partition of runtime.
Phase 3 must confirm important kernels with Nsight Compute counters and explicit arithmetic-
intensity models. Nsight Systems capture is supported through `scripts.nsys_profile` when `nsys` is
installed.

## Fair comparisons and ablations

An ablation changes one execution-identity field at a time and keeps model weights, prompts,
precision, warmup, repetitions, and measurement boundary fixed. Compile setup and CUDA Graph capture
are reported separately from steady-state inference. vLLM is an external reference only when model
weights, attention semantics, sampling policy, token counts, request arrivals, precision, and metric
boundaries can be matched; otherwise the difference is documented and the rows are not presented as
an optimization ratio.

Power management, clock state, competing GPU processes, and thermal state can affect results. Final
release experiments will add repeated randomized case order and between-run variance rather than
over-interpreting one suite.

## Real-checkpoint external references

Real-checkpoint reports pin the upstream repository and immutable commit, then hash the local
`config.json` and `model.safetensors`. Model load and tokenization are outside the timing boundary;
every backend consumes the same host token IDs, precision, greedy sampling rule, output budget,
warmups, and repetitions. Backends execute in separate dependency environments when their
Torch/Triton requirements conflict, and each report retains that software identity.

The comparison is deliberately not an ablation. vLLM uses its own FlashAttention, paged cache,
scheduler, sampling path, and CUDA Graph defaults; Transformers uses SDPA and a contiguous cache;
TensorForge uses parallel SDPA prefill followed by custom paged Triton decode. Offline vLLM exposes
first-token and finish times but not every intermediate token timestamp, so the report interpolates
intermediate timestamps while TPOT continues to use the observed endpoints.

Greedy-token hashes expose free-running divergence but do not replace a logit gate. In BF16, two
backends can satisfy teacher-forced logit tolerances yet choose different tokens when top logits are
exactly or nearly tied. TensorForge therefore tests long teacher-forced prefill/decode against
Transformers and reports tied-argmax divergence separately from cache or attention corruption.
