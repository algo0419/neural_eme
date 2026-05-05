"""Build a target-conditioned bend dataset for 50:50 and 70:30 splitting."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Bend splitting YAML config.")
    parser.add_argument("--num-random", type=int, default=1000, help="Number of random bend designs to evaluate.")
    parser.add_argument(
        "--include-pso",
        type=Path,
        nargs="*",
        default=[],
        help="PSO best-design or all_evaluations NPZ files to merge.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data") / "bend_multitarget_dataset.npz",
        help="Output target-conditioned dataset NPZ.",
    )
    parser.add_argument("--seed", type=int, default=20260505, help="Random seed.")
    parser.add_argument("--progress-interval", type=int, default=50, help="Print progress every N random designs.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))
    sys.path.insert(0, str(repo_root))

    from dreme_inverse.bend_dataset import build_bend_multitarget_dataset, config_from_yaml

    config_path = args.config if args.config.is_absolute() else repo_root / args.config
    output_path = args.output if args.output.is_absolute() else repo_root / args.output
    include_pso = [path if path.is_absolute() else repo_root / path for path in args.include_pso]
    config = config_from_yaml(config_path, repo_root)

    print(f"config: {config_path}")
    print(f"num_random: {args.num_random}")
    print(f"include_pso: {[str(path) for path in include_pso]}")
    print(f"output: {output_path}")
    summary = build_bend_multitarget_dataset(
        config,
        output_path,
        num_random=args.num_random,
        include_pso_paths=include_pso,
        seed=args.seed,
        progress_interval=args.progress_interval,
    )
    print(f"rows: {summary['num_rows']}")
    print(f"raw_geometries: {summary['num_raw_geometries']}")
    print(f"sources: {summary['sources']}")
    print(f"targets: {summary['targets']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
