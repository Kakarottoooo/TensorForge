"""Turn torch.profiler operator rows into bounded hotspot conclusions."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class OperatorProfile:
    name: str
    category: str
    calls: int
    self_device_time_us: float
    total_device_time_us: float
    self_cpu_time_us: float
    input_shapes: str

    @property
    def average_self_device_time_us(self) -> float:
        return self.self_device_time_us / self.calls if self.calls else 0.0


@dataclass(frozen=True, slots=True)
class BottleneckAssessment:
    classification: str
    confidence: str
    compute_share: float
    memory_or_pointwise_share: float
    launch_candidate_share: float
    evidence: str
    required_confirmation: str


def categorize_operator(name: str) -> str:
    lowered = name.lower()
    if lowered.startswith("tensorforge::"):
        return "region"
    if any(token in lowered for token in ("matmul", "mm", "bmm", "gemm")):
        return "matrix_multiply"
    if any(token in lowered for token in ("softmax", "attention")):
        return "attention"
    if any(token in lowered for token in ("norm", "rsqrt", "mean")):
        return "normalization"
    if any(token in lowered for token in ("silu", "sigmoid", "mul", "add")):
        return "pointwise"
    if any(token in lowered for token in ("copy", "to", "cat", "contiguous", "transpose")):
        return "memory_or_layout"
    return "other"


def assess_bottleneck(operators: Iterable[OperatorProfile]) -> BottleneckAssessment:
    """Produce a profiler-only hypothesis, never a roofline claim.

    torch.profiler supplies time and call counts, not DRAM bytes or achieved
    FLOP/s. The classification therefore directs the next measurement and is
    explicitly lower confidence than the Phase 3 Nsight/roofline analysis.
    """

    rows = [operator for operator in operators if operator.category != "region"]
    total = sum(row.self_device_time_us for row in rows)
    if total <= 0:
        return BottleneckAssessment(
            classification="unresolved",
            confidence="none",
            compute_share=0.0,
            memory_or_pointwise_share=0.0,
            launch_candidate_share=0.0,
            evidence="No device self-time was recorded.",
            required_confirmation="Capture a CUDA profile on a supported GPU.",
        )
    compute = sum(row.self_device_time_us for row in rows if row.category == "matrix_multiply")
    memory = sum(
        row.self_device_time_us
        for row in rows
        if row.category in {"normalization", "pointwise", "memory_or_layout"}
    )
    launch = sum(
        row.self_device_time_us
        for row in rows
        if row.calls >= 4 and row.average_self_device_time_us <= 25.0
    )
    compute_share = compute / total
    memory_share = memory / total
    launch_share = launch / total
    if compute_share >= 0.5:
        classification = "compute-dominant candidate"
    elif launch_share >= 0.35:
        classification = "kernel-launch/latency-dominant candidate"
    elif memory_share >= 0.35:
        classification = "memory/pointwise-dominant candidate"
    else:
        classification = "mixed candidate"
    return BottleneckAssessment(
        classification=classification,
        confidence="provisional",
        compute_share=compute_share,
        memory_or_pointwise_share=memory_share,
        launch_candidate_share=launch_share,
        evidence=(
            f"torch.profiler nonexclusive self-device-time shares: matrix multiply "
            f"{compute_share:.1%}, memory/pointwise {memory_share:.1%}, short repeated kernels "
            f"{launch_share:.1%}. Launch candidates overlap operator categories."
        ),
        required_confirmation=(
            "Use Nsight Compute counters and arithmetic-intensity/roofline analysis; profiler "
            "timings alone cannot distinguish DRAM bandwidth from dependency or launch stalls."
        ),
    )


def analysis_to_dict(
    operators: tuple[OperatorProfile, ...], assessment: BottleneckAssessment
) -> dict[str, object]:
    return {
        "operators": [asdict(operator) for operator in operators],
        "assessment": asdict(assessment),
    }
