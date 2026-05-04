"""Generate random valid DReME geometry vectors and preview plots."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))

    from dreme_inverse.geometry_sampler import (
        GeometrySpec,
        plot_geometry_from_z,
        sample_random_z,
        z_to_geometry_parameters,
    )

    outputs_dir = repo_root / "outputs"
    plots_dir = outputs_dir / "geometry_plots"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Conservative ranges within the sample dataset:
    # top_width dataset: 1-3 um, curvature dataset: 0-330000 1/m.
    spec = GeometrySpec(
        num_sections=5,
        width_min=1.2e-6,
        width_max=2.0e-6,
        total_length=10.0e-6,
        max_adjacent_width_change=0.18e-6,
        use_curvature=True,
        curvature_min=0.0,
        curvature_max=1.0e4,
        max_adjacent_curvature_change=5.0e3,
        zero_curvature_endpoints=True,
    )

    rng = np.random.default_rng(20260504)
    sample_count = 100
    z_samples = np.array([sample_random_z(spec, rng) for _ in range(sample_count)])

    params = [z_to_geometry_parameters(z, spec) for z in z_samples]
    widths = np.stack([item["widths"] for item in params])
    curvatures = np.stack([item["curvatures"] for item in params])
    prop_lengths = params[0]["prop_lengths"]

    output_npz = outputs_dir / "random_geometries_100.npz"
    np.savez(
        output_npz,
        z=z_samples,
        widths=widths,
        curvatures=curvatures,
        prop_lengths=prop_lengths,
        spec_json=json.dumps(spec.as_dict(), sort_keys=True),
    )

    plot_count = 6
    for idx in range(plot_count):
        plot_geometry_from_z(z_samples[idx], spec, plots_dir / f"geometry_{idx:03d}.png")

    print(f"saved: {output_npz}")
    print(f"z shape: {z_samples.shape}")
    print(f"width range um: {widths.min() * 1e6:.3f} to {widths.max() * 1e6:.3f}")
    print(f"curvature range 1/m: {curvatures.min():.1f} to {curvatures.max():.1f}")
    print(f"plots: {plots_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
