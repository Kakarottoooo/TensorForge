"""Execute a versioned TensorForge benchmark manifest."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from tensorforge.benchmark.reporting import write_suite_reports
from tensorforge.benchmark.suite import load_manifest, run_suite


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    repository = Path(__file__).resolve().parents[1]
    workloads = load_manifest(args.manifest)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA benchmark requested but CUDA is unavailable")
    result = run_suite(workloads, device=device, repository=repository)
    paths = write_suite_reports(result, args.output_dir)
    for path in paths:
        print(path.resolve())
    failures = [case for case in result.cases if case.status != "completed"]
    if failures:
        raise SystemExit(
            f"{len(failures)} benchmark case(s) did not complete; reports were written"
        )


if __name__ == "__main__":
    main()
