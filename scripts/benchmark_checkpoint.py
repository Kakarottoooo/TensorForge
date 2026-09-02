"""Benchmark one real-checkpoint backend under the shared TensorForge contract."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from tensorforge.benchmark.checkpoint_suite import (
    load_checkpoint_manifest,
    run_checkpoint_suite,
    write_checkpoint_reports,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument(
        "--backend",
        choices=("tensorforge", "transformers_sdpa", "vllm"),
        required=True,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    device = torch.device(args.device)
    repository = Path(__file__).resolve().parents[1]
    manifest = load_checkpoint_manifest(args.manifest)
    result = run_checkpoint_suite(
        manifest,
        backend=args.backend,
        checkpoint_dir=args.checkpoint_dir,
        device=device,
        repository=repository,
    )
    for path in write_checkpoint_reports(result, args.output_dir):
        print(path.resolve())


if __name__ == "__main__":
    main()
