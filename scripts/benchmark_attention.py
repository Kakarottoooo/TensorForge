"""Run correctness-gated paged GQA decode-attention benchmarks."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from tensorforge.benchmark.attention_benchmark import (
    load_attention_manifest,
    run_attention_suite,
    write_attention_reports,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("paged attention benchmarks require an available CUDA device")
    repository = Path(__file__).resolve().parents[1]
    manifest = load_attention_manifest(args.manifest)
    result = run_attention_suite(manifest, device=device, repository=repository)
    for path in write_attention_reports(result, args.output_dir):
        print(path.resolve())


if __name__ == "__main__":
    main()
