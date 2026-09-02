from __future__ import annotations

from tensorforge.profiling.analysis import (
    OperatorProfile,
    assess_bottleneck,
    categorize_operator,
)


def _operator(name: str, category: str, time_us: float, calls: int = 1) -> OperatorProfile:
    return OperatorProfile(name, category, calls, time_us, time_us, 0.0, "[]")


def test_operator_categorization_uses_specific_categories_first() -> None:
    assert categorize_operator("tensorforge::decode") == "region"
    assert categorize_operator("aten::mm") == "matrix_multiply"
    assert categorize_operator("aten::_softmax") == "attention"
    assert categorize_operator("aten::rsqrt") == "normalization"
    assert categorize_operator("aten::copy_") == "memory_or_layout"


def test_compute_assessment_is_explicitly_provisional() -> None:
    assessment = assess_bottleneck(
        (_operator("mm", "matrix_multiply", 70), _operator("copy", "memory_or_layout", 30))
    )

    assert assessment.classification == "compute-dominant candidate"
    assert assessment.confidence == "provisional"
    assert "roofline" in assessment.required_confirmation
