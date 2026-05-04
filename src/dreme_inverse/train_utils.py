"""Training utilities for forward DReME surrogate models."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .analysis import load_dataset_arrays, response_labels
from .models import ForwardMLP, count_parameters, require_torch


torch = require_torch()


@dataclass(frozen=True)
class TrainConfig:
    """Configuration for forward MLP training."""

    epochs: int = 200
    batch_size: int = 64
    hidden_dim: int = 128
    num_layers: int = 3
    learning_rate: float = 1.0e-3
    weight_decay: float = 0.0
    seed: int = 20260504
    val_fraction: float = 0.1
    test_fraction: float = 0.1
    activation: str = "silu"


@dataclass(frozen=True)
class NormalizerStats:
    """Mean/std normalization fitted on the training split only."""

    z_mean: np.ndarray
    z_std: np.ndarray
    y_mean: np.ndarray
    y_std: np.ndarray

    def normalize_z(self, Z: np.ndarray) -> np.ndarray:
        return (np.asarray(Z, dtype=np.float32) - self.z_mean) / self.z_std

    def normalize_y(self, Y: np.ndarray) -> np.ndarray:
        return (np.asarray(Y, dtype=np.float32) - self.y_mean) / self.y_std

    def denormalize_y(self, Y_norm: np.ndarray) -> np.ndarray:
        return np.asarray(Y_norm, dtype=np.float32) * self.y_std + self.y_mean


def train_forward_surrogate(
    dataset_path: str | Path,
    output_dir: str | Path,
    *,
    config: Optional[TrainConfig] = None,
) -> Dict[str, Any]:
    """Train a forward MLP surrogate and save artifacts."""
    config = config or TrainConfig()
    _validate_config(config)
    dataset_path = Path(dataset_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    Z, Y, dataset_metadata = load_dataset_arrays(dataset_path)
    _validate_arrays(Z, Y)
    splits = split_indices(Z.shape[0], config.val_fraction, config.test_fraction, config.seed)

    stats = fit_normalizer(Z[splits["train"]], Y[splits["train"]])
    loaders = make_dataloaders(Z, Y, splits, stats, config.batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _seed_everything(config.seed)
    model = ForwardMLP(
        Z.shape[1],
        Y.shape[1],
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        activation=config.activation,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    loss_fn = torch.nn.MSELoss()

    history = {"epoch": [], "train_mse": [], "val_mse": []}
    for epoch in range(1, config.epochs + 1):
        train_mse = _train_one_epoch(model, loaders["train"], optimizer, loss_fn, device)
        val_mse = _evaluate_loss(model, loaders["val"], loss_fn, device) if loaders["val"] is not None else math.nan
        history["epoch"].append(epoch)
        history["train_mse"].append(train_mse)
        history["val_mse"].append(val_mse)

    test_mse_norm = _evaluate_loss(model, loaders["test"], loss_fn, device) if loaders["test"] is not None else math.nan
    predictions = predict_normalized(model, stats.normalize_z(Z), device=device)
    Y_pred = stats.denormalize_y(predictions)
    test_indices = splits["test"]
    test_mse_physical = _mse(Y[test_indices], Y_pred[test_indices]) if test_indices.size else math.nan

    labels = response_labels(Y.shape[1])
    artifact_paths = save_training_artifacts(
        output_dir,
        model,
        stats,
        history,
        Y[test_indices],
        Y_pred[test_indices],
        labels,
        {
            "dataset_path": str(dataset_path),
            "dataset_metadata_keys": sorted(dataset_metadata.keys()),
            "config": asdict(config),
            "device": str(device),
            "num_parameters": count_parameters(model),
            "num_samples": int(Z.shape[0]),
            "z_dim": int(Z.shape[1]),
            "y_dim": int(Y.shape[1]),
            "split_sizes": {name: int(indices.size) for name, indices in splits.items()},
            "test_mse_normalized": float(test_mse_norm),
            "test_mse_physical": float(test_mse_physical),
        },
    )

    summary = {
        "dataset_path": str(dataset_path),
        "output_dir": str(output_dir),
        "device": str(device),
        "num_samples": int(Z.shape[0]),
        "z_dim": int(Z.shape[1]),
        "y_dim": int(Y.shape[1]),
        "split_sizes": {name: int(indices.size) for name, indices in splits.items()},
        "num_parameters": count_parameters(model),
        "final_train_mse": float(history["train_mse"][-1]),
        "final_val_mse": float(history["val_mse"][-1]),
        "test_mse_normalized": float(test_mse_norm),
        "test_mse_physical": float(test_mse_physical),
        "artifacts": artifact_paths,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def split_indices(
    sample_count: int,
    val_fraction: float,
    test_fraction: float,
    seed: int,
) -> Dict[str, np.ndarray]:
    """Create reproducible train/validation/test index splits."""
    if sample_count < 2:
        raise ValueError("At least two samples are required for train/test splitting.")
    if val_fraction < 0.0 or test_fraction < 0.0 or val_fraction + test_fraction >= 1.0:
        raise ValueError("Split fractions must be non-negative and sum to less than 1.")

    rng = np.random.default_rng(seed)
    indices = rng.permutation(sample_count)
    test_count = max(1, int(round(sample_count * test_fraction))) if test_fraction > 0 else 0
    val_count = max(1, int(round(sample_count * val_fraction))) if val_fraction > 0 and sample_count >= 3 else 0
    if sample_count - test_count - val_count < 1:
        val_count = max(0, val_count - 1)
    if sample_count - test_count - val_count < 1:
        test_count = max(0, test_count - 1)

    test = np.sort(indices[:test_count])
    val = np.sort(indices[test_count : test_count + val_count])
    train = np.sort(indices[test_count + val_count :])
    if train.size == 0:
        raise ValueError("Train split is empty.")
    return {"train": train, "val": val, "test": test}


def fit_normalizer(Z_train: np.ndarray, Y_train: np.ndarray, eps: float = 1.0e-8) -> NormalizerStats:
    """Fit mean/std statistics on training arrays."""
    z_mean = np.mean(Z_train, axis=0, dtype=np.float64).astype(np.float32)
    y_mean = np.mean(Y_train, axis=0, dtype=np.float64).astype(np.float32)
    z_std = np.std(Z_train, axis=0, dtype=np.float64).astype(np.float32)
    y_std = np.std(Y_train, axis=0, dtype=np.float64).astype(np.float32)
    z_std = np.where(z_std < eps, 1.0, z_std).astype(np.float32)
    y_std = np.where(y_std < eps, 1.0, y_std).astype(np.float32)
    return NormalizerStats(z_mean=z_mean, z_std=z_std, y_mean=y_mean, y_std=y_std)


def make_dataloaders(
    Z: np.ndarray,
    Y: np.ndarray,
    splits: Dict[str, np.ndarray],
    stats: NormalizerStats,
    batch_size: int,
) -> Dict[str, Optional[Any]]:
    """Create PyTorch dataloaders for each split."""
    loaders: Dict[str, Optional[Any]] = {}
    Z_norm = stats.normalize_z(Z)
    Y_norm = stats.normalize_y(Y)
    for split_name, indices in splits.items():
        if indices.size == 0:
            loaders[split_name] = None
            continue
        z_tensor = torch.as_tensor(Z_norm[indices], dtype=torch.float32)
        y_tensor = torch.as_tensor(Y_norm[indices], dtype=torch.float32)
        dataset = torch.utils.data.TensorDataset(z_tensor, y_tensor)
        loaders[split_name] = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split_name == "train"),
            drop_last=False,
        )
    return loaders


def predict_normalized(model: Any, Z_norm: np.ndarray, *, device: Any) -> np.ndarray:
    """Run model predictions for normalized geometry vectors."""
    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, Z_norm.shape[0], 4096):
            batch = torch.as_tensor(Z_norm[start : start + 4096], dtype=torch.float32, device=device)
            outputs.append(model(batch).detach().cpu().numpy())
    return np.concatenate(outputs, axis=0)


def save_training_artifacts(
    output_dir: Path,
    model: Any,
    stats: NormalizerStats,
    history: Dict[str, List[float]],
    Y_true_test: np.ndarray,
    Y_pred_test: np.ndarray,
    labels: List[str],
    checkpoint_metadata: Dict[str, Any],
) -> Dict[str, Any]:
    """Save checkpoint, normalizer stats, curves, and prediction plots."""
    checkpoint_path = output_dir / "model_checkpoint.pt"
    normalizer_path = output_dir / "normalizer_stats.npz"
    history_path = output_dir / "training_history.json"
    curves_path = output_dir / "training_curves.png"
    scatter_dir = output_dir / "predicted_vs_true"

    np.savez(
        normalizer_path,
        z_mean=stats.z_mean,
        z_std=stats.z_std,
        y_mean=stats.y_mean,
        y_std=stats.y_std,
    )
    history_path.write_text(json.dumps(history, indent=2, sort_keys=True), encoding="utf-8")
    plot_training_curves(history, curves_path)
    scatter_paths = plot_predicted_vs_true(Y_true_test, Y_pred_test, labels, scatter_dir)
    torch.save(
        {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "model_class": "ForwardMLP",
            "model_config": {
                "input_dim": int(checkpoint_metadata["z_dim"]),
                "output_dim": int(checkpoint_metadata["y_dim"]),
                "hidden_dim": int(checkpoint_metadata["config"]["hidden_dim"]),
                "num_layers": int(checkpoint_metadata["config"]["num_layers"]),
                "activation": checkpoint_metadata["config"]["activation"],
            },
            "state_dict": model.state_dict(),
            "normalizer": {
                "z_mean": stats.z_mean,
                "z_std": stats.z_std,
                "y_mean": stats.y_mean,
                "y_std": stats.y_std,
            },
            "history": history,
            "metadata": checkpoint_metadata,
        },
        checkpoint_path,
    )
    return {
        "checkpoint": str(checkpoint_path),
        "normalizer": str(normalizer_path),
        "history": str(history_path),
        "training_curves": str(curves_path),
        "predicted_vs_true": scatter_paths,
    }


def plot_training_curves(history: Dict[str, List[float]], output_path: str | Path) -> Path:
    """Save train/validation MSE curves."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    epochs = history["epoch"]
    fig, axis = plt.subplots(figsize=(6.4, 4.0))
    axis.plot(epochs, history["train_mse"], label="train", linewidth=1.8)
    if any(np.isfinite(history["val_mse"])):
        axis.plot(epochs, history["val_mse"], label="validation", linewidth=1.8)
    axis.set_xlabel("epoch")
    axis.set_ylabel("MSE on normalized response")
    axis.set_title("Forward Surrogate Training")
    axis.set_yscale("log")
    axis.grid(True, alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_predicted_vs_true(
    Y_true: np.ndarray,
    Y_pred: np.ndarray,
    labels: List[str],
    output_dir: str | Path,
) -> List[str]:
    """Save one predicted-vs-true scatter plot for each response component."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: List[str] = []
    for idx, label in enumerate(labels):
        path = output_dir / f"{_safe_filename(label)}.png"
        fig, axis = plt.subplots(figsize=(4.2, 4.0))
        if Y_true.size:
            true_values = Y_true[:, idx]
            pred_values = Y_pred[:, idx]
            mask = np.isfinite(true_values) & np.isfinite(pred_values)
            axis.scatter(true_values[mask], pred_values[mask], s=16, alpha=0.75, color="tab:blue")
            if np.count_nonzero(mask):
                min_value = float(min(np.min(true_values[mask]), np.min(pred_values[mask])))
                max_value = float(max(np.max(true_values[mask]), np.max(pred_values[mask])))
                if math.isclose(min_value, max_value):
                    min_value -= 0.5
                    max_value += 0.5
                axis.plot([min_value, max_value], [min_value, max_value], color="black", linestyle="--", linewidth=1.0)
        axis.set_title(label)
        axis.set_xlabel("true")
        axis.set_ylabel("predicted")
        axis.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        paths.append(str(path))
    return paths


def _train_one_epoch(model: Any, loader: Any, optimizer: Any, loss_fn: Any, device: Any) -> float:
    model.train()
    total_loss = 0.0
    total_count = 0
    for z_batch, y_batch in loader:
        z_batch = z_batch.to(device)
        y_batch = y_batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        prediction = model(z_batch)
        loss = loss_fn(prediction, y_batch)
        loss.backward()
        optimizer.step()
        batch_count = int(z_batch.shape[0])
        total_loss += float(loss.detach().cpu()) * batch_count
        total_count += batch_count
    return total_loss / max(total_count, 1)


def _evaluate_loss(model: Any, loader: Any, loss_fn: Any, device: Any) -> float:
    model.eval()
    total_loss = 0.0
    total_count = 0
    with torch.no_grad():
        for z_batch, y_batch in loader:
            z_batch = z_batch.to(device)
            y_batch = y_batch.to(device)
            prediction = model(z_batch)
            loss = loss_fn(prediction, y_batch)
            batch_count = int(z_batch.shape[0])
            total_loss += float(loss.detach().cpu()) * batch_count
            total_count += batch_count
    return total_loss / max(total_count, 1)


def _validate_config(config: TrainConfig) -> None:
    if config.epochs <= 0:
        raise ValueError("epochs must be positive.")
    if config.batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if config.hidden_dim <= 0:
        raise ValueError("hidden_dim must be positive.")
    if config.num_layers <= 0:
        raise ValueError("num_layers must be positive.")
    if config.learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive.")
    if config.weight_decay < 0.0:
        raise ValueError("weight_decay cannot be negative.")


def _validate_arrays(Z: np.ndarray, Y: np.ndarray) -> None:
    if Z.ndim != 2 or Y.ndim != 2:
        raise ValueError(f"Expected 2D Z and Y arrays; got Z={Z.shape}, Y={Y.shape}.")
    if Z.shape[0] != Y.shape[0]:
        raise ValueError(f"Z and Y sample counts differ: {Z.shape[0]} vs {Y.shape[0]}.")
    if Z.shape[0] < 2:
        raise ValueError("At least two samples are required.")
    if not np.all(np.isfinite(Z)):
        raise ValueError("Z contains NaN or Inf.")
    if not np.all(np.isfinite(Y)):
        raise ValueError("Y contains NaN or Inf.")


def _seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size == 0:
        return math.nan
    return float(np.mean((np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)) ** 2))


def _safe_filename(label: str) -> str:
    return "".join(char if char.isalnum() or char in ("-", "_") else "_" for char in label).strip("_") or "response"
