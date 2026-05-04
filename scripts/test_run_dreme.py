"""Command-line smoke test for dreme_inverse.wrapper.run_dreme."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))
    sys.path.insert(0, str(repo_root))

    from dreme_inverse.wrapper import run_dreme

    num_sections = 5
    rng = np.random.default_rng(7)
    z = rng.uniform(0.0, 1.0, size=2 * num_sections)

    config = {
        "repo_root": str(repo_root),
        "dataset_path": str(repo_root / "sample_datasets" / "Si_rectangular_single_waveguide"),
        "num_sections": num_sections,
        "length": 10.0e-6,
        "width_min": 1.2e-6,
        "width_max": 2.0e-6,
        "curvature_min": 0.0,
        "curvature_max": 1.0e4,
        "resolution": 500,
        "force_unitary": False,
        "force_passive": False,
        "return_raw_smatrix": False,
        "verbose": False,
    }

    result = run_dreme(z, config)
    print("z:", np.array2string(z, precision=4))
    print("valid:", result["valid"])
    print("error:", result["error"])
    print("response:", np.array2string(result["response"], precision=8))
    print("geometry:", result["metadata"].get("geometry_type"))
    print("smatrix_shape:", result["metadata"].get("smatrix_shape"))
    if result["output_amplitudes"] is not None:
        print("output_amplitudes_shape:", result["output_amplitudes"].shape)

    return 0 if result["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
