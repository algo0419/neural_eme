"""Minimal end-to-end smoke test for the dataset-based EME/DReME path.

This follows the flow in examples/Si_linear_taper_simul.ipynb:
DataUpdater -> LinearTaper -> EME -> Runner.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable


REQUIRED_DATASET_FILES = (
    "dataset_info.py",
    "neff.pkl",
    "TE_pol.pkl",
    "overlap.pkl",
)


def fail(message: str, hint: str | None = None) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    if hint:
        print(f"HINT: {hint}", file=sys.stderr)
    raise SystemExit(1)


def validate_dataset(dataset_dir: Path) -> None:
    if not dataset_dir.exists():
        fail(
            f"Dataset path does not exist: {dataset_dir}",
            "Run this script from a clone of dataset-based-eme, or pass a repo root that contains sample_datasets.",
        )
    if not dataset_dir.is_dir():
        fail(f"Dataset path is not a directory: {dataset_dir}")

    missing = [name for name in REQUIRED_DATASET_FILES if not (dataset_dir / name).exists()]
    if missing:
        fail(
            f"Dataset is missing required files: {', '.join(missing)}",
            "If this is a fresh clone, run: git lfs install && git lfs pull",
        )

    for name in REQUIRED_DATASET_FILES:
        path = dataset_dir / name
        if path.suffix != ".pkl":
            continue
        header = path.read_bytes()[:80]
        if header.startswith(b"version https://git-lfs.github.com/spec"):
            fail(
                f"{path} is a Git LFS pointer, not the real dataset file.",
                "Run: git lfs install && git lfs pull",
            )
        if path.stat().st_size < 1024:
            fail(
                f"{path} is unexpectedly small ({path.stat().st_size} bytes).",
                "The sample dataset may not have downloaded correctly; run git lfs pull.",
            )


def preinitialize_ray(repo_root: Path) -> None:
    """Start Ray before em_simulation imports its matrix helpers.

    The core package initializes Ray with the repository root as the working
    directory. If this smoke test is run from an in-repo virtualenv, that can
    cause Ray to package hundreds of MB of dependencies. Initializing Ray here
    keeps the same behavior while excluding local environment and output dirs.
    """
    try:
        import ray
    except ImportError as exc:
        fail(
            f"Missing Python dependency: {exc.name}",
            "Install dependencies with: pip install -r requirements.txt",
        )

    if ray.is_initialized():
        return

    ray.init(
        ignore_reinit_error=True,
        include_dashboard=False,
        log_to_driver=False,
        runtime_env={
            "working_dir": str(repo_root),
            "excludes": [
                ".git/",
                ".venv/",
                ".venv312/",
                "sample_datasets/",
                "examples/",
                "outputs/",
                "__pycache__/",
            ],
        },
    )


def format_complex_vector(values: Iterable[complex], max_items: int = 12) -> str:
    values = list(values)
    shown = values[:max_items]
    lines = [f"  [{idx:02d}] {value.real:+.6e}{value.imag:+.6e}j" for idx, value in enumerate(shown)]
    if len(values) > max_items:
        lines.append(f"  ... ({len(values) - max_items} more)")
    return "\n".join(lines)


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    dataset_dir = repo_root / "sample_datasets" / "Si_rectangular_single_waveguide"
    outputs_dir = repo_root / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    validate_dataset(dataset_dir)

    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    try:
        import numpy as np
    except ImportError as exc:
        fail(
            f"Missing Python dependency: {exc.name}",
            "Install dependencies with: pip install -r requirements.txt",
        )

    preinitialize_ray(repo_root)

    try:
        import em_simulation as sim
    except ImportError as exc:
        fail(
            f"Could not import em_simulation or one of its dependencies: {exc}",
            "Install dependencies with: pip install -r requirements.txt",
        )

    try:
        dataset = sim.DataUpdater(str(dataset_dir), is_testmode=True)

        # Same simple geometry as examples/Si_linear_taper_simul.ipynb.
        geometry = sim.LinearTaper(
            dataset,
            input_width=1.0e-6,
            output_width=2.0e-6,
            length=10.0e-6,
        )

        eme = sim.EME(geometry)
        if hasattr(eme, "calc_Smatrix"):
            eme.calc_Smatrix()

        propagator = getattr(eme, "propagator", eme)
        smatrix = getattr(propagator, "smatrix", None)
        if smatrix is None:
            fail("EME propagator did not expose an S-matrix after calc_Smatrix().")

        runner = sim.Runner(eme)
        inner_runner = getattr(runner, "runner", runner)
        radiation_mask = getattr(inner_runner, "_radiation_mode_mask", None)
        if radiation_mask is None:
            fail("Runner does not expose a radiation-mode mask; cannot infer input vector length.")

        initial_non_radiation = np.where(np.asarray(radiation_mask[0]) == False)[0]
        input_amplitudes = np.zeros(len(initial_non_radiation), dtype=np.complex64)
        input_amplitudes[0] = 1.0 + 0.0j

        sectional_amplitudes = runner.propagate(input_amplitudes)
        output_amplitudes = runner.propagate_lumped_smatrix(input_amplitudes)
        if output_amplitudes is None and sectional_amplitudes is not None:
            output_amplitudes = sectional_amplitudes[-1]
        if output_amplitudes is None:
            fail("Propagation returned no output amplitudes.")

        output_intensities = np.abs(output_amplitudes) ** 2

        mode_count = output_amplitudes.shape[0] // 2
        mode_labels = np.array(
            [f"forward_mode_{idx + 1}" for idx in range(mode_count)]
            + [f"backward_mode_{idx + 1}" for idx in range(mode_count)]
        )
        tracking_mode_names = getattr(geometry, "_tracking_mode_names", None)

        npz_path = outputs_dir / "smoke_test_result.npz"
        txt_path = outputs_dir / "smoke_test_summary.txt"
        np.savez(
            npz_path,
            smatrix_shape=np.asarray(smatrix.shape, dtype=np.int64),
            output_amplitudes=output_amplitudes,
            output_intensities=output_intensities,
            input_amplitudes=input_amplitudes,
            mode_labels=mode_labels,
            tracking_mode_names=np.asarray(tracking_mode_names)
            if tracking_mode_names is not None
            else np.asarray([]),
        )

        summary = [
            "DReME smoke test summary",
            f"repo_root: {repo_root}",
            f"dataset_dir: {dataset_dir}",
            "geometry: LinearTaper(input_width=1e-6, output_width=2e-6, length=10e-6)",
            f"S-matrix shape: {tuple(smatrix.shape)}",
            f"input amplitudes length: {len(input_amplitudes)}",
            f"mode labels: {', '.join(mode_labels.tolist())}",
            "output amplitudes:",
            format_complex_vector(output_amplitudes),
            "output intensities:",
            "\n".join(f"  [{idx:02d}] {value:.6e}" for idx, value in enumerate(output_intensities)),
        ]
        if tracking_mode_names is not None:
            summary.extend(
                [
                    "tracking mode names:",
                    np.array2string(np.asarray(tracking_mode_names), threshold=80),
                ]
            )
        summary.extend([f"saved_npz: {npz_path}", f"saved_summary: {txt_path}"])
        summary_text = "\n".join(summary) + "\n"
        txt_path.write_text(summary_text, encoding="utf-8")

        print(summary_text)
    except KeyError as exc:
        fail(
            f"Dataset lookup failed for parameter point/key: {exc}",
            "The geometry may require sample points that are absent from the precomputed dataset.",
        )
    except ModuleNotFoundError as exc:
        fail(
            f"Missing Python dependency: {exc.name}",
            "Install dependencies with: pip install -r requirements.txt",
        )
    except MemoryError:
        fail(
            "The smoke test ran out of memory while loading the sample dataset.",
            "The sample overlap.pkl is large; close other processes or run on a larger-memory machine.",
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
