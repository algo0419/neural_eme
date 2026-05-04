"""Geometry sampling utilities for DReME inverse-design datasets."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


@dataclass(frozen=True)
class GeometrySpec:
    """Physical bounds and smoothness limits for normalized geometry vectors.

    The normalized vector layout is compatible with ``run_dreme``:
    ``[width_z..., curvature_z...]`` when ``use_curvature`` is true, otherwise
    ``[width_z...]``.
    """

    num_sections: int
    width_min: float
    width_max: float
    total_length: float
    max_adjacent_width_change: float
    max_adjacent_curvature_change: float
    use_curvature: bool = True
    curvature_min: float = 0.0
    curvature_max: Optional[float] = 1.0e4
    zero_curvature_endpoints: bool = True

    @property
    def z_dim(self) -> int:
        return self.num_sections * (2 if self.use_curvature else 1)

    def to_run_dreme_config(self) -> Dict[str, Any]:
        """Return config keys consumed by ``run_dreme``."""
        config = {
            "num_sections": self.num_sections,
            "length": self.total_length,
            "width_min": self.width_min,
            "width_max": self.width_max,
            "zero_curvature_endpoints": self.zero_curvature_endpoints,
        }
        if self.use_curvature:
            config.update(
                {
                    "curvature_min": self.curvature_min,
                    "curvature_max": self.curvature_max,
                }
            )
        return config

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


def sample_random_z(spec: GeometrySpec, rng: np.random.Generator) -> np.ndarray:
    """Sample one normalized geometry vector satisfying ``spec``."""
    _validate_spec(spec)
    widths = _sample_bounded_profile(
        spec.num_sections,
        spec.width_min,
        spec.width_max,
        spec.max_adjacent_width_change,
        rng,
    )
    width_z = _physical_to_unit(widths, spec.width_min, spec.width_max)

    if not spec.use_curvature:
        z = width_z
    else:
        curvature_max = _curvature_max(spec)
        curvatures = _sample_curvature_profile(spec, rng)
        curvature_z = _physical_to_unit(curvatures, spec.curvature_min, curvature_max)
        if spec.zero_curvature_endpoints:
            curvature_z[0] = 0.0
            curvature_z[-1] = 0.0
        z = np.concatenate([width_z, curvature_z])

    if not is_valid_z(z, spec):
        raise RuntimeError("Internal sampler produced an invalid geometry.")
    return z.astype(float)


def is_valid_z(z: np.ndarray, spec: GeometrySpec) -> bool:
    """Validate normalized vector shape, bounds, and adjacent physical jumps."""
    try:
        params = z_to_geometry_parameters(z, spec)
    except (TypeError, ValueError):
        return False

    z_arr = np.asarray(z, dtype=float).reshape(-1)
    if z_arr.shape != (spec.z_dim,):
        return False
    if not np.all(np.isfinite(z_arr)):
        return False
    if np.any(z_arr < 0.0) or np.any(z_arr > 1.0):
        return False

    widths = params["widths"]
    if np.any(widths < spec.width_min) or np.any(widths > spec.width_max):
        return False
    if np.any(np.abs(np.diff(widths)) > spec.max_adjacent_width_change + 1e-15):
        return False

    if spec.use_curvature:
        curvatures = params["curvatures"]
        curvature_max = _curvature_max(spec)
        endpoint_mask = np.ones(spec.num_sections, dtype=bool)
        if spec.zero_curvature_endpoints:
            endpoint_mask[[0, -1]] = False
            if abs(curvatures[0]) > 1e-12 or abs(curvatures[-1]) > 1e-12:
                return False
        if np.any(curvatures[endpoint_mask] < spec.curvature_min):
            return False
        if np.any(curvatures > curvature_max):
            return False
        if np.any(np.abs(np.diff(curvatures)) > spec.max_adjacent_curvature_change + 1e-9):
            return False

    return True


def z_to_geometry_parameters(z: np.ndarray, spec: GeometrySpec) -> Dict[str, np.ndarray]:
    """Convert normalized ``z`` to physical geometry arrays."""
    _validate_spec(spec)
    z_arr = np.asarray(z, dtype=float).reshape(-1)
    if z_arr.shape != (spec.z_dim,):
        raise ValueError(f"Expected z length {spec.z_dim}, got {z_arr.size}.")
    if not np.all(np.isfinite(z_arr)):
        raise ValueError("z contains NaN or Inf.")
    if np.any(z_arr < 0.0) or np.any(z_arr > 1.0):
        raise ValueError("z values must be in [0, 1].")

    width_z = z_arr[: spec.num_sections]
    widths = spec.width_min + width_z * (spec.width_max - spec.width_min)
    prop_lengths = np.linspace(0.0, spec.total_length, spec.num_sections)

    params: Dict[str, np.ndarray] = {
        "prop_lengths": prop_lengths,
        "widths": widths,
    }
    if spec.use_curvature:
        curvature_max = _curvature_max(spec)
        curvature_z = z_arr[spec.num_sections :]
        curvatures = spec.curvature_min + curvature_z * (curvature_max - spec.curvature_min)
        if spec.zero_curvature_endpoints:
            curvatures = curvatures.copy()
            curvatures[0] = 0.0
            curvatures[-1] = 0.0
        params["curvatures"] = curvatures
    else:
        params["curvatures"] = np.zeros(spec.num_sections, dtype=float)

    return params


def plot_geometry_from_z(z: np.ndarray, spec: GeometrySpec, output_path: str | Path) -> None:
    """Save a compact plot of centerline, width, and curvature profiles."""
    params = z_to_geometry_parameters(z, spec)
    if not is_valid_z(z, spec):
        raise ValueError("Cannot plot invalid geometry.")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    prop = params["prop_lengths"]
    widths = params["widths"]
    curvatures = params["curvatures"]

    dense_s = np.linspace(0.0, spec.total_length, 500)
    dense_widths = np.interp(dense_s, prop, widths)
    dense_curvatures = np.interp(dense_s, prop, curvatures)
    x, y = _centerline_from_curvature(dense_s, dense_curvatures)

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
    axes[0].plot(x * 1e6, y * 1e6, color="tab:blue", linewidth=1.8)
    axes[0].scatter(x[0] * 1e6, y[0] * 1e6, color="tab:green", s=18, label="in")
    axes[0].scatter(x[-1] * 1e6, y[-1] * 1e6, color="tab:red", s=18, label="out")
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

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _validate_spec(spec: GeometrySpec) -> None:
    if spec.num_sections < 3:
        raise ValueError("num_sections must be at least 3.")
    if spec.width_min <= 0 or spec.width_min >= spec.width_max:
        raise ValueError("Width bounds must satisfy 0 < width_min < width_max.")
    if spec.total_length <= 0:
        raise ValueError("total_length must be positive.")
    if spec.max_adjacent_width_change <= 0:
        raise ValueError("max_adjacent_width_change must be positive.")
    if spec.max_adjacent_width_change > spec.width_max - spec.width_min:
        raise ValueError("max_adjacent_width_change should not exceed the width range.")
    if spec.use_curvature:
        curvature_max = _curvature_max(spec)
        if spec.curvature_min < 0 or spec.curvature_min > curvature_max:
            raise ValueError("Curvature bounds must satisfy 0 <= curvature_min <= curvature_max.")
        if spec.max_adjacent_curvature_change <= 0:
            raise ValueError("max_adjacent_curvature_change must be positive.")
    elif spec.max_adjacent_curvature_change < 0:
        raise ValueError("max_adjacent_curvature_change cannot be negative.")


def _curvature_max(spec: GeometrySpec) -> float:
    if spec.curvature_max is None:
        raise ValueError("curvature_max is required when use_curvature=True.")
    return float(spec.curvature_max)


def _physical_to_unit(values: np.ndarray, min_value: float, max_value: float) -> np.ndarray:
    if max_value == min_value:
        return np.zeros_like(values, dtype=float)
    return np.clip((values - min_value) / (max_value - min_value), 0.0, 1.0)


def _sample_bounded_profile(
    num_sections: int,
    min_value: float,
    max_value: float,
    max_delta: float,
    rng: np.random.Generator,
) -> np.ndarray:
    values = np.empty(num_sections, dtype=float)
    values[0] = rng.uniform(min_value, max_value)
    for idx in range(1, num_sections):
        low = max(min_value, values[idx - 1] - max_delta)
        high = min(max_value, values[idx - 1] + max_delta)
        values[idx] = rng.uniform(low, high)
    return values


def _sample_curvature_profile(spec: GeometrySpec, rng: np.random.Generator) -> np.ndarray:
    curvature_max = _curvature_max(spec)
    if not spec.zero_curvature_endpoints:
        return _sample_bounded_profile(
            spec.num_sections,
            spec.curvature_min,
            curvature_max,
            spec.max_adjacent_curvature_change,
            rng,
        )

    values = np.zeros(spec.num_sections, dtype=float)
    for idx in range(1, spec.num_sections - 1):
        remaining_steps_to_zero = spec.num_sections - 1 - idx
        low = max(
            spec.curvature_min,
            values[idx - 1] - spec.max_adjacent_curvature_change,
            -spec.max_adjacent_curvature_change * remaining_steps_to_zero,
        )
        high = min(
            curvature_max,
            values[idx - 1] + spec.max_adjacent_curvature_change,
            spec.max_adjacent_curvature_change * remaining_steps_to_zero,
        )
        if low > high:
            raise RuntimeError("Curvature constraints are infeasible.")
        values[idx] = rng.uniform(low, high)
    return values


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
