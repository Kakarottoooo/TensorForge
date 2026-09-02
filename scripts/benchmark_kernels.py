"""Run correctness-gated Triton microbenchmarks from a versioned manifest."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from tensorforge.benchmark.kernel_benchmark import (
    load_kernel_manifest,
    run_kernel_suite,
    write_kernel_reports,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Triton kernel benchmarks require an available CUDA device")
    repository = Path(__file__).resolve().parents[1]
    manifest = load_kernel_manifest(args.manifest)
    result = run_kernel_suite(manifest, device=device, repository=repository)
    for path in write_kernel_reports(result, args.output_dir):
        print(path.resolve())


if __name__ == "__main__":
    main()
