"""Analysis utilities for DReME-generated ML datasets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .geometry_sampler import GeometrySpec, z_to_geometry_parameters


RESPONSE_LABELS = ("T0", "T1", "T_total", "loss", "relative_phase")


def analyze_dataset(
    dataset_path: str | Path,
    output_dir: str | Path,
    *,
    seed: int = 20260504,
    num_geometries: int = 6,
    total_intensity_threshold: float = 1.05,
) -> Dict[str, Any]:
    """Analyze a ``.npz`` dataset containing ``Z`` and ``Y`` arrays.

    Plots and a text summary are written into ``output_dir``. The returned
    dictionary is JSON-compatible and suitable for scripts/tests.
    """
    dataset_path = Path(dataset_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    Z, Y, metadata = load_dataset_arrays(dataset_path)
    if Z.ndim != 2:
        raise ValueError(f"Z must be 2D; got shape {Z.shape}.")
    if Y.ndim != 2:
        raise ValueError(f"Y must be 2D; got shape {Y.shape}.")
    if Z.shape[0] != Y.shape[0]:
        raise ValueError(f"Z and Y row counts differ: {Z.shape[0]} vs {Y.shape[0]}.")

    labels = response_labels(Y.shape[1])
    suspicious = identify_suspicious_values(Y, total_intensity_threshold=total_intensity_threshold)
    plot_paths = {
        "histograms": str(plot_response_histograms(Y, labels, output_dir / "response_histograms.png")),
        "scatter": str(plot_response_scatter(Y, labels, output_dir / "response_scatter.png")),
    }

    geometry_paths: List[str] = []
    spec = geometry_spec_from_metadata(metadata)
    if spec is not None and Z.shape[0] > 0:
        geometry_paths = plot_random_geometries(
            Z,
            Y,
            spec,
            labels,
            output_dir / "geometry_samples",
            seed=seed,
            count=num_geometries,
        )
    elif Z.shape[0] > 0:
        suspicious["notes"].append("Geometry plots skipped because metadata_json does not contain a GeometrySpec.")

    summary = {
        "dataset_path": str(dataset_path),
        "output_dir": str(output_dir),
        "num_samples": int(Z.shape[0]),
        "z_dim": int(Z.shape[1]),
        "y_dim": int(Y.shape[1]),
        "response_labels": labels,
        "suspicious": suspicious,
        "plots": plot_paths,
        "geometry_plots": geometry_paths,
        "metadata_keys": sorted(metadata.keys()),
    }
    write_summary(summary, output_dir / "summary.txt")
    return summary


def load_dataset_arrays(dataset_path: str | Path) -> tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Load ``Z``, ``Y``, and optional ``metadata_json`` from a dataset file."""
    dataset_path = Path(dataset_path).expanduser().resolve()
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file does not exist: {dataset_path}")

    with np.load(dataset_path, allow_pickle=False) as data:
        if "Z" not in data.files or "Y" not in data.files:
            raise KeyError(f"{dataset_path} must contain arrays named 'Z' and 'Y'.")
        Z = np.asarray(data["Z"], dtype=float)
        Y = np.asarray(data["Y"], dtype=float)
        metadata: Dict[str, Any] = {}
        if "metadata_json" in data.files:
            metadata = json.loads(str(data["metadata_json"].item()))
    return Z, Y, metadata


def response_labels(y_dim: int) -> List[str]:
    """Return names for known response components, with generic fallbacks."""
    labels = list(RESPONSE_LABELS[: min(y_dim, len(RESPONSE_LABELS))])
    labels.extend(f"y_{idx}" for idx in range(len(labels), y_dim))
    return labels


def identify_suspicious_values(
    Y: np.ndarray,
    *,
    total_intensity_threshold: float = 1.05,
    max_examples: int = 12,
) -> Dict[str, Any]:
    """Find common numerical and physical outliers in response data."""
    Y = np.asarray(Y, dtype=float)
    labels = response_labels(Y.shape[1] if Y.ndim == 2 else 0)
    suspicious: Dict[str, Any] = {
        "nan_count": int(np.count_nonzero(np.isnan(Y))),
        "inf_count": int(np.count_nonzero(np.isinf(Y))),
        "negative_intensity_count": 0,
        "total_intensity_above_threshold_count": 0,
        "examples": {},
        "notes": [],
    }
    if Y.ndim != 2:
        suspicious["notes"].append(f"Suspicious value scan expected 2D Y; got shape {Y.shape}.")
        return suspicious

    suspicious["examples"]["nan"] = _coordinate_examples(np.argwhere(np.isnan(Y)), labels, max_examples)
    suspicious["examples"]["inf"] = _coordinate_examples(np.argwhere(np.isinf(Y)), labels, max_examples)

    intensity_columns = list(range(min(3, Y.shape[1])))
    if intensity_columns:
        negative_mask = np.zeros(Y.shape, dtype=bool)
        negative_mask[:, intensity_columns] = Y[:, intensity_columns] < -1e-12
        suspicious["negative_intensity_count"] = int(np.count_nonzero(negative_mask))
        suspicious["examples"]["negative_intensity"] = _coordinate_examples(
            np.argwhere(negative_mask),
            labels,
            max_examples,
        )

    if Y.shape[1] >= 3:
        total = Y[:, 2]
        high_total_mask = np.isfinite(total) & (total > total_intensity_threshold)
        suspicious["total_intensity_above_threshold_count"] = int(np.count_nonzero(high_total_mask))
        suspicious["examples"]["total_intensity_above_threshold"] = [
            {"sample": int(idx), "value": float(total[idx])}
            for idx in np.flatnonzero(high_total_mask)[:max_examples]
        ]
    else:
        suspicious["notes"].append("Total intensity scan skipped because Y has fewer than three columns.")

    return suspicious


def geometry_spec_from_metadata(metadata: Dict[str, Any]) -> Optional[GeometrySpec]:
    """Reconstruct a ``GeometrySpec`` from dataset metadata, if present."""
    spec_data = metadata.get("spec")
    if not isinstance(spec_data, dict):
        return None
    try:
        return GeometrySpec(**spec_data)
    except TypeError:
        return None


def plot_response_histograms(Y: np.ndarray, labels: List[str], output_path: str | Path) -> Path:
    """Save histograms for every response component."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    y_dim = Y.shape[1]
    cols = min(3, max(1, y_dim))
    rows = int(np.ceil(y_dim / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.1 * cols, 3.0 * rows), squeeze=False)
    for idx, axis in enumerate(axes.ravel()):
        if idx >= y_dim:
            axis.axis("off")
            continue
        values = Y[:, idx]
        finite_values = values[np.isfinite(values)]
        axis.hist(finite_values, bins=40, color="tab:blue", alpha=0.82, edgecolor="white")
        axis.set_title(labels[idx])
        axis.set_xlabel("value")
        axis.set_ylabel("count")
        if finite_values.size:
            axis.axvline(float(np.median(finite_values)), color="black", linewidth=1.0, linestyle="--")
    fig.suptitle("Response Histograms", y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_response_scatter(Y: np.ndarray, labels: List[str], output_path: str | Path) -> Path:
    """Save pairwise scatter plots for the main response components."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plots = []
    if Y.shape[1] >= 2:
        plots.append(("T0 vs T1", Y[:, 0], Y[:, 1], labels[0], labels[1]))
    if Y.shape[1] >= 4:
        plots.append(("T_total vs loss", Y[:, 2], Y[:, 3], labels[2], labels[3]))
    if Y.shape[1] >= 2:
        ratio = _safe_ratio(Y[:, 0], Y[:, 1])
        if np.count_nonzero(np.isfinite(ratio)) > 0:
            x = Y[:, 2] if Y.shape[1] >= 3 else np.arange(Y.shape[0], dtype=float)
            x_label = labels[2] if Y.shape[1] >= 3 else "sample index"
            plots.append(("T0/T1 ratio", x, ratio, x_label, "T0 / T1"))

    if not plots:
        plots.append(("response vs sample", np.arange(Y.shape[0], dtype=float), Y[:, 0], "sample index", labels[0]))

    fig, axes = plt.subplots(1, len(plots), figsize=(4.4 * len(plots), 3.6), squeeze=False)
    for axis, (title, x, y, x_label, y_label) in zip(axes.ravel(), plots):
        mask = np.isfinite(x) & np.isfinite(y)
        axis.scatter(np.asarray(x)[mask], np.asarray(y)[mask], s=14, alpha=0.72, color="tab:green")
        axis.set_title(title)
        axis.set_xlabel(x_label)
        axis.set_ylabel(y_label)
        axis.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_random_geometries(
    Z: np.ndarray,
    Y: np.ndarray,
    spec: GeometrySpec,
    labels: List[str],
    output_dir: str | Path,
    *,
    seed: int,
    count: int,
) -> List[str]:
    """Plot random geometries and annotate their response vector."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if count <= 0 or Z.shape[0] == 0:
        return []

    rng = np.random.default_rng(seed)
    sample_count = min(count, Z.shape[0])
    indices = np.sort(rng.choice(Z.shape[0], size=sample_count, replace=False))
    paths: List[str] = []
    for idx in indices:
        output_path = output_dir / f"geometry_{idx:05d}.png"
        plot_geometry_with_response(Z[idx], Y[idx], spec, labels, output_path, sample_index=int(idx))
        paths.append(str(output_path))
    return paths


def plot_geometry_with_response(
    z: np.ndarray,
    y: np.ndarray,
    spec: GeometrySpec,
    labels: List[str],
    output_path: str | Path,
    *,
    sample_index: int,
) -> Path:
    """Save one annotated geometry plot."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    params = z_to_geometry_parameters(z, spec)
    prop = params["prop_lengths"]
    widths = params["widths"]
    curvatures = params["curvatures"]

    dense_s = np.linspace(0.0, spec.total_length, 500)
    dense_widths = np.interp(dense_s, prop, widths)
    dense_curvatures = np.interp(dense_s, prop, curvatures)
    x, center_y = _centerline_from_curvature(dense_s, dense_curvatures)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.5))
    axes[0].plot(x * 1e6, center_y * 1e6, color="tab:blue", linewidth=1.8)
    axes[0].scatter(x[0] * 1e6, center_y[0] * 1e6, color="tab:green", s=18, label="in")
    axes[0].scatter(x[-1] * 1e6, center_y[-1] * 1e6, color="tab:red", s=18, label="out")
    axes[0].set_title("Centerline")
    axes[0].set_xlabel("x (um)")
    axes[0].set_ylabel("y (um)")
    axes[0].set_aspect("equal", adjustable="box")
    axes[0].legend(loc="best", fontsize=8)

    axes[1].plot(dense_s * 1e6, dense_widths * 1e6, color="tab:orange")
    axes[1].scatter(prop * 1e6, widths * 1e6, color="black", s=10)
    axes[1].set_title("Width")
    axes[1].set_xlabel("s (um)")
    axes[1].set_ylabel("width (um)")

    axes[2].plot(dense_s * 1e6, dense_curvatures, color="tab:purple")
    axes[2].scatter(prop * 1e6, curvatures, color="black", s=10)
    axes[2].set_title("Curvature")
    axes[2].set_xlabel("s (um)")
    axes[2].set_ylabel("1/m")

    response_text = ", ".join(f"{label}={float(value):.4g}" for label, value in zip(labels, y))
    fig.suptitle(f"Sample {sample_index}: {response_text}", fontsize=10)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def write_summary(summary: Dict[str, Any], output_path: str | Path) -> Path:
    """Write a compact text summary next to the plots."""
    output_path = Path(output_path)
    suspicious = summary["suspicious"]
    lines = [
        f"dataset: {summary['dataset_path']}",
        f"num_samples: {summary['num_samples']}",
        f"z_dim: {summary['z_dim']}",
        f"y_dim: {summary['y_dim']}",
        f"response_labels: {', '.join(summary['response_labels'])}",
        "",
        "suspicious values:",
        f"  NaN entries: {suspicious['nan_count']}",
        f"  Inf entries: {suspicious['inf_count']}",
        f"  negative intensity entries: {suspicious['negative_intensity_count']}",
        "  total intensity above threshold entries: "
        f"{suspicious['total_intensity_above_threshold_count']}",
    ]
    notes = suspicious.get("notes", [])
    if notes:
        lines.append("")
        lines.append("notes:")
        lines.extend(f"  {note}" for note in notes)
    lines.append("")
    lines.append("plots:")
    for name, path in summary["plots"].items():
        lines.append(f"  {name}: {path}")
    for path in summary["geometry_plots"]:
        lines.append(f"  geometry: {path}")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def _safe_ratio(numerator: np.ndarray, denominator: np.ndarray, eps: float = 1e-15) -> np.ndarray:
    ratio = np.full_like(numerator, np.nan, dtype=float)
    mask = np.isfinite(numerator) & np.isfinite(denominator) & (np.abs(denominator) > eps)
    ratio[mask] = numerator[mask] / denominator[mask]
    return ratio


def _coordinate_examples(indices: np.ndarray, labels: List[str], max_examples: int) -> List[Dict[str, Any]]:
    examples = []
    for row, col in indices[:max_examples]:
        examples.append({"sample": int(row), "component": labels[int(col)], "column": int(col)})
    return examples


def _centerline_from_curvature(prop_lengths: np.ndarray, curvatures: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ds = np.diff(prop_lengths)
    avg_curvature = 0.5 * (curvatures[:-1] + curvatures[1:])
    angles = np.zeros_like(prop_lengths)
    angles[1:] = np.cumsum(avg_curvature * ds)

    avg_angles = 0.5 * (angles[:-1] + angles[1:])
    x = np.zeros_like(prop_lengths)
    y = np.zeros_like(prop_lengths)
    x[1:] = np.cumsum(np.cos(avg_angles) * ds)
    y[1:] = np.cumsum(np.sin(avg_angles) * ds)
    return x, y
