"""Reproducible torch.profiler capture for prefill and baseline decode."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch.profiler import ProfilerActivity, profile, record_function

from tensorforge.benchmark.hardware import collect_hardware_metadata
from tensorforge.benchmark.schema import ModelSpec, Precision
from tensorforge.model.llama import LlamaForCausalLM
from tensorforge.profiling.analysis import (
    OperatorProfile,
    analysis_to_dict,
    assess_bottleneck,
    categorize_operator,
)
from tensorforge.runtime.dtypes import parse_dtype, validate_dtype_support


@dataclass(frozen=True, slots=True)
class ProfileSpec:
    model: ModelSpec
    prompt_length: int
    decode_steps: int
    batch_size: int
    precision: Precision
    seed: int = 7
    row_limit: int = 40

    def __post_init__(self) -> None:
        values = {
            "prompt_length": self.prompt_length,
            "decode_steps": self.decode_steps,
            "batch_size": self.batch_size,
            "row_limit": self.row_limit,
        }
        for name, value in values.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.prompt_length + self.decode_steps + 1 > self.model.max_position_embeddings:
            raise ValueError("profile workload exceeds model context capacity")


def _device_time(event: Any, prefix: str) -> float:
    for name in (f"{prefix}_device_time_total", f"{prefix}_cuda_time_total"):
        value = getattr(event, name, None)
        if value is not None:
            return float(value)
    return 0.0


def _operator_rows(events: Any, row_limit: int) -> tuple[OperatorProfile, ...]:
    rows = []
    for event in events:
        self_device = _device_time(event, "self")
        total_device = _device_time(event, "")
        if total_device == 0.0:
            total_device = float(getattr(event, "device_time_total", 0.0))
        rows.append(
            OperatorProfile(
                name=str(event.key),
                category=categorize_operator(str(event.key)),
                calls=int(event.count),
                self_device_time_us=self_device,
                total_device_time_us=total_device,
                self_cpu_time_us=float(event.self_cpu_time_total),
                input_shapes=str(event.input_shapes),
            )
        )
    rows.sort(key=lambda row: row.self_device_time_us, reverse=True)
    return tuple(rows[:row_limit])


def capture_profile(
    spec: ProfileSpec,
    *,
    device: torch.device,
    output_directory: Path,
    repository: Path | None = None,
) -> tuple[Path, Path, Path]:
    dtype = parse_dtype(spec.precision.value)
    validate_dtype_support(dtype, device)
    torch.manual_seed(spec.seed)
    model = LlamaForCausalLM(spec.model.to_model_config()).to(device=device, dtype=dtype).eval()
    prompt = torch.randint(
        spec.model.vocab_size,
        (spec.batch_size, spec.prompt_length),
        device=device,
    )

    with torch.inference_mode():
        _ = model(prompt)
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)
    output_directory.mkdir(parents=True, exist_ok=True)
    trace_path = output_directory / "trace.json"
    with profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as captured:
        with torch.inference_mode(), record_function("tensorforge::prefill"):
            logits = model(prompt)
            next_token = logits[:, -1].argmax(-1, keepdim=True)
            generated = torch.cat((prompt, next_token), dim=1)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
        with torch.inference_mode(), record_function("tensorforge::decode"):
            for _ in range(spec.decode_steps):
                logits = model(generated)
                next_token = logits[:, -1].argmax(-1, keepdim=True)
                generated = torch.cat((generated, next_token), dim=1)
            if device.type == "cuda":
                torch.cuda.synchronize(device)

    captured.export_chrome_trace(str(trace_path))
    operators = _operator_rows(captured.key_averages(group_by_input_shape=True), spec.row_limit)
    assessment = assess_bottleneck(operators)
    payload = {
        "schema_version": "1.0",
        "spec": asdict(spec),
        "hardware": collect_hardware_metadata(repository).to_dict(),
        **analysis_to_dict(operators, assessment),
        "trace_file": trace_path.name,
    }
    json_path = output_directory / "profile-summary.json"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    markdown_path = output_directory / "profile-summary.md"
    lines = [
        "# TensorForge profiler summary",
        "",
        f"Classification: **{assessment.classification}** ({assessment.confidence}).",
        "",
        assessment.evidence,
        "",
        f"Required confirmation: {assessment.required_confirmation}",
        "",
        "| Operator | Category | Calls | Self device (us) | Avg self device (us) | Self CPU (us) |",
        "|---|---|---:|---:|---:|---:|",
    ]
    lines.extend(
        f"| `{row.name}` | {row.category} | {row.calls} | {row.self_device_time_us:.3f} | "
        f"{row.average_self_device_time_us:.3f} | {row.self_cpu_time_us:.3f} |"
        for row in operators
    )
    lines.extend(
        [
            "",
            "The classification is a timing-based hypothesis, not a roofline result. The raw "
            "Chrome trace is generated locally; operator rows and exact profile dimensions are "
            "preserved in the JSON summary.",
            "",
        ]
    )
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    return trace_path, json_path, markdown_path
