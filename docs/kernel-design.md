# Phase 3 Triton kernel design

## Contract and scope

The custom kernels are inference-only, require contiguous CUDA tensors, and accept FP32, FP16, or
BF16 storage. Reductions and nonlinear arithmetic are evaluated in FP32 before conversion to the
output dtype. The readable PyTorch functions in `tensorforge/kernels/reference.py` remain the
numerical oracle. Triton is optional at import time so CPU/Windows correctness CI remains usable.

Phase 3 implements:

1. RMSNorm;
2. fused residual addition plus RMSNorm, returning both the materialized residual sum and normalized
   output;
3. the elementwise SwiGLU activation after gate/up projections.

The two SwiGLU GEMMs and down projection remain PyTorch/cuBLAS. Calling this path “fused SwiGLU” does
not imply that matrix multiplications are inside the Triton kernel.

## RMSNorm program

Each Triton program ID owns one logical row. `BLOCK_SIZE` is the next power of two at least as large
as the hidden width, capped by the public wrapper at 16,384 columns. Column offsets form contiguous,
coalesced loads. A `columns < N_COLS` mask supplies zeros for non-power-of-two widths such as 513 or
1,376 and suppresses invalid stores.

Inputs are converted to FP32, squared, and reduced with `tl.sum`. The program computes mean square,
epsilon addition, reciprocal square root, and two multiplications (input scaling and learned weight)
in FP32. A single program avoids inter-program synchronization and intermediate buffers, at the cost
of register pressure for very wide rows. Widths beyond 16,384 need a hierarchical reduction and are
rejected rather than silently running an unsafe configuration.

## Fused residual plus RMSNorm

The residual kernel uses the same row ownership and reduction. It loads the input and residual once,
forms their FP32 sum, materializes that sum in the storage dtype for the next residual path, and uses
the FP32 sum for normalization. The unfused PyTorch path may round the materialized sum before
normalization, so equivalence is numerical rather than bitwise; dtype-specific tolerances and full
model logits tests guard this boundary.

Fusion removes the normalized path's second read of the residual sum and one launch. It does not
eliminate the residual-output store because subsequent blocks require that state.

## SwiGLU activation

Programs cover a flattened contiguous gate/up tensor in autotuned blocks of 128, 256, 512, or 1,024
elements. Loads are coalesced and masked at the tail. Gate and up values are promoted to FP32;
`gate * sigmoid(gate) * up` is stored in the input dtype. Tests include negative and positive sigmoid
saturation, odd shapes, and common 1,376/11,008 intermediate widths.

## Autotuning

RMSNorm and residual/RMSNorm search 1, 2, 4, and 8 warps with one stage. Their key includes both
`N_ROWS` and `N_COLS`: width determines reduction/register behavior, while row count determines
available parallelism and occupancy. SwiGLU searches four block/warp combinations keyed by total
element count. The selected configuration and first-call JIT/autotune wall time are serialized per
case. Steady-state timing starts only after correctness and tuning complete.

Autotune is intentionally shape-specific. A configuration selected for one decode row must not be
presented as optimal for a 2,048-token prefill.

## Integrated model path

`TritonModelExecutor` reuses an existing `LlamaForCausalLM` and its exact weights. Attention and
linear projections stay on the reference PyTorch path; input/final RMSNorm, attention residual plus
post-attention RMSNorm, and SwiGLU activation use the custom kernels. Tests compare complete logits
for multiple shapes in FP16/BF16. Token equality is required when the reference top-1/top-2 margin is
larger than the numerical error bound; near-tie logits remain governed by the numerical tolerance.

## Arithmetic intensity and roofline boundary

The benchmark records a transparent logical-work model:

| Kernel | Unique logical bytes | Modeled operations |
|---|---:|---:|
| RMSNorm | `(2R + 1) × N × element_size` | `R × (4N + 2)` |
| Residual + RMSNorm | `(4R + 1) × N × element_size` | `R × (5N + 2)` |
| SwiGLU activation | `3RN × element_size` | `3RN` |

`R` is row count and `N` is width. Rsqrt and sigmoid each count as one special-function operation;
this convention is explicit because hardware instruction costs are not one FLOP. Weight bytes are
counted once per invocation, representing unique traffic rather than guaranteed DRAM reads.

The empirical memory ceiling is a 256 MiB `torch.copy_` read+write benchmark in the same process.
The scalar FP32 compute ceiling is derived from the RTX 3080 Ti reference core count and boost clock
recorded in the manifest. Roofline efficiency uses the lower ceiling, but it remains a model: actual
DRAM transactions, cache hits, occupancy, and stalls require Nsight Compute counters. Modeled GB/s is
never labeled measured DRAM bandwidth.

The suite records GPU temperature, power draw/limit, and SM/memory clocks before and after the matrix
so thermal or power-state drift is visible rather than hidden inside an unexplained variance claim.

## Reproduction

From WSL2/Linux:

```bash
bash scripts/setup_phase3_wsl.sh
source ~/.venvs/tensorforge-py311/bin/activate
pytest -q
python -m scripts.benchmark_kernels \
  --manifest benchmarks/phase3-kernels.json \
  --output-dir results/local/phase3-kernels
```

The committed report is generated only from a clean Git commit. Cases where Triton loses to PyTorch
are labeled `regression` and counted at the top of the report; no result is filtered by outcome.
