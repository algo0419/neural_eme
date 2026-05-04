"""Analyze a DReME-generated dataset and save diagnostic plots."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="Dataset .npz file containing Z and Y arrays.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs") / "dataset_analysis",
        help="Directory where <dataset_name>/ analysis outputs will be written.",
    )
    parser.add_argument("--seed", type=int, default=20260504, help="Random seed for geometry plot selection.")
    parser.add_argument("--num-geometries", type=int, default=6, help="Number of random geometries to plot.")
    parser.add_argument(
        "--total-intensity-threshold",
        type=float,
        default=1.05,
        help="Flag total transmitted intensity values above this threshold.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))

    from dreme_inverse.analysis import analyze_dataset

    dataset_path = args.dataset
    if not dataset_path.is_absolute():
        dataset_path = repo_root / dataset_path

    output_root = args.output_root
    if not output_root.is_absolute():
        output_root = repo_root / output_root
    output_dir = output_root / dataset_path.stem

    summary = analyze_dataset(
        dataset_path,
        output_dir,
        seed=args.seed,
        num_geometries=args.num_geometries,
        total_intensity_threshold=args.total_intensity_threshold,
    )

    print(f"dataset: {summary['dataset_path']}")
    print(f"size: {summary['num_samples']}")
    print(f"z_dim: {summary['z_dim']}")
    print(f"y_dim: {summary['y_dim']}")
    print(f"output_dir: {summary['output_dir']}")
    print("suspicious:")
    print(json.dumps(summary["suspicious"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
