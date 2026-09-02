"""Run the profiler workload under Nsight Systems when `nsys` is installed."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt-length", type=int, default=512)
    parser.add_argument("--decode-steps", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--precision", default="fp16")
    args = parser.parse_args()

    nsys = shutil.which("nsys")
    if nsys is None:
        raise RuntimeError("Nsight Systems `nsys` executable was not found on PATH")
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    profile_output = output.parent / f"{output.name}-torch-profiler"
    command = [
        nsys,
        "profile",
        "--trace=cuda,nvtx,osrt",
        "--sample=none",
        "--force-overwrite=true",
        f"--output={output}",
        sys.executable,
        "-m",
        "scripts.profile_model",
        "--output-dir",
        str(profile_output),
        "--prompt-length",
        str(args.prompt_length),
        "--decode-steps",
        str(args.decode_steps),
        "--batch-size",
        str(args.batch_size),
        "--precision",
        args.precision,
    ]
    repository = Path(__file__).resolve().parents[1]
    subprocess.run(command, check=True, cwd=repository)


if __name__ == "__main__":
    main()
