"""Build a single-process DReME-generated ML dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-samples", type=int, default=100, help="Number of geometry candidates to attempt.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data") / "dreme_dataset_100.npz",
        help="Output NPZ path for successful samples.",
    )
    parser.add_argument("--seed", type=int, default=20260504, help="Base random seed.")
    parser.add_argument("--max-failures", type=int, default=20, help="Stop after this many failed evaluations.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))
    sys.path.insert(0, str(repo_root))

    from dreme_inverse.dataset_builder import build_dataset, default_geometry_spec, default_run_config

    output_path = args.output
    if not output_path.is_absolute():
        output_path = repo_root / output_path

    summary = build_dataset(
        args.num_samples,
        output_path,
        spec=default_geometry_spec(),
        run_config=default_run_config(repo_root),
        seed=args.seed,
        max_failures=args.max_failures,
        progress_interval=50,
    )
    return 0 if summary["failure_count"] <= args.max_failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
