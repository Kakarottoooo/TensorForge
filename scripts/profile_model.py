"""Capture a torch.profiler trace and summarized hotspot report."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from tensorforge.benchmark.schema import ModelSpec, Precision
from tensorforge.profiling.capture import ProfileSpec, capture_profile


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--precision", choices=[item.value for item in Precision], default="fp16")
    parser.add_argument("--prompt-length", type=int, default=512)
    parser.add_argument("--decode-steps", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--tiny", action="store_true")
    args = parser.parse_args()

    model = (
        ModelSpec(
            vocab_size=128,
            hidden_size=64,
            intermediate_size=160,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=max(128, args.prompt_length + args.decode_steps + 1),
        )
        if args.tiny
        else ModelSpec(
            max_position_embeddings=max(2_048, args.prompt_length + args.decode_steps + 1)
        )
    )
    paths = capture_profile(
        ProfileSpec(
            model=model,
            prompt_length=args.prompt_length,
            decode_steps=args.decode_steps,
            batch_size=args.batch_size,
            precision=Precision(args.precision),
        ),
        device=torch.device(args.device),
        output_directory=args.output_dir,
        repository=Path(__file__).resolve().parents[1],
    )
    for path in paths:
        print(path.resolve())


if __name__ == "__main__":
    main()
