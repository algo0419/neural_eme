"""Train a PyTorch MLP forward surrogate for DReME datasets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True, help="Dataset .npz file containing Z and Y arrays.")
    parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=64, help="Training batch size.")
    parser.add_argument("--hidden-dim", type=int, default=128, help="MLP hidden width.")
    parser.add_argument("--num-layers", type=int, default=3, help="Number of hidden layers.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs") / "forward_mlp",
        help="Directory where <dataset_name>/ artifacts will be written.",
    )
    parser.add_argument("--lr", type=float, default=1.0e-3, help="AdamW learning rate.")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="AdamW weight decay.")
    parser.add_argument("--seed", type=int, default=20260504, help="Random seed for splits and training.")
    parser.add_argument("--val-fraction", type=float, default=0.1, help="Validation split fraction.")
    parser.add_argument("--test-fraction", type=float, default=0.1, help="Test split fraction.")
    parser.add_argument("--activation", type=str, default="silu", help="Activation: silu, gelu, relu, or tanh.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))

    from dreme_inverse.train_utils import TrainConfig, train_forward_surrogate

    dataset_path = args.dataset
    if not dataset_path.is_absolute():
        dataset_path = repo_root / dataset_path

    output_root = args.output_root
    if not output_root.is_absolute():
        output_root = repo_root / output_root
    output_dir = output_root / dataset_path.stem

    config = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        activation=args.activation,
    )
    summary = train_forward_surrogate(dataset_path, output_dir, config=config)

    print(f"dataset: {summary['dataset_path']}")
    print(f"output_dir: {summary['output_dir']}")
    print(f"device: {summary['device']}")
    print(f"size: {summary['num_samples']}")
    print(f"z_dim: {summary['z_dim']}")
    print(f"y_dim: {summary['y_dim']}")
    print(f"split_sizes: {summary['split_sizes']}")
    print(f"parameters: {summary['num_parameters']}")
    print(f"final_train_mse: {summary['final_train_mse']:.6g}")
    print(f"final_val_mse: {summary['final_val_mse']:.6g}")
    print(f"test_mse_normalized: {summary['test_mse_normalized']:.6g}")
    print(f"test_mse_physical: {summary['test_mse_physical']:.6g}")
    print("artifacts:")
    print(json.dumps(summary["artifacts"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
