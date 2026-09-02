"""Merge compatible real-checkpoint backend reports into one comparison."""

from __future__ import annotations

import argparse
from pathlib import Path

from tensorforge.benchmark.checkpoint_comparison import (
    compare_checkpoint_reports,
    write_checkpoint_comparison,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = compare_checkpoint_reports(tuple(args.reports))
    for path in write_checkpoint_comparison(result, args.output_dir):
        print(path.resolve())


if __name__ == "__main__":
    main()
