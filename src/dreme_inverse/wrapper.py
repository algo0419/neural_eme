"""Reusable DReME wrapper for ML dataset generation.

The wrapper uses the repository's public API:
DataUpdater -> Geometry -> EME -> Runner.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np


RESPONSE_SIZE = 5
REQUIRED_DATASET_FILES = ("dataset_info.py", "neff.pkl", "TE_pol.pkl", "overlap.pkl")
_DATASET_CACHE: Dict[Tuple[str, bool], Any] = {}


def run_dreme(z: np.ndarray, config: dict) -> dict:
    """Run one DReME simulation for a normalized geometry vector.

    For datasets with both ``top_width`` and ``curvature``, this uses
    ``SingleCustomBend``. For width-only datasets, it falls back to
    ``CustomTaper``.
    """
    metadata: Dict[str, Any] = {}
    try:
        config = dict(config or {})
        repo_root = _repo_root(config)
        dataset_path = _dataset_path(config, repo_root)
        _validate_dataset_path(dataset_path)
        _ensure_import_paths(repo_root)
        _preinitialize_ray(repo_root, config)

        import em_simulation as sim

        dataset = _load_dataset(sim, dataset_path, bool(config.get("is_testmode", True)))
        parameter_names = list(getattr(dataset, "parameter_names", dataset.get_parameter_names()))
        parameter_grid = getattr(dataset, "parameter_grid", dataset.get_parameter_grid())
        has_width = "top_width" in parameter_names
        has_curvature = "curvature" in parameter_names
        if not has_width:
            raise ValueError("Dataset does not expose a 'top_width' parameter.")

        z_norm = _normalized_vector(z)
        geometry, geometry_meta = _build_geometry(sim, dataset, z_norm, config, parameter_grid, has_curvature)
        metadata.update(geometry_meta)
        metadata["dataset_path"] = str(dataset_path)
        metadata["parameter_names"] = parameter_names

        eme = sim.EME(
            geometry,
            force_passive=bool(config.get("force_passive", False)),
            force_unitary=bool(config.get("force_unitary", False)),
        )
        eme.calc_Smatrix()
        propagator = getattr(eme, "propagator", eme)
        smatrix = getattr(propagator, "smatrix", None)
        if smatrix is None:
            raise RuntimeError("EME propagator did not expose smatrix after calc_Smatrix().")

        runner = sim.Runner(eme)
        input_amplitudes = _make_te0_like_input(runner, config)
        output_amplitudes = runner.propagate_lumped_smatrix(input_amplitudes)
        if output_amplitudes is None:
            sectional = runner.propagate(input_amplitudes)
            output_amplitudes = None if sectional is None else sectional[-1]
        if output_amplitudes is None:
            raise RuntimeError("Runner did not return output amplitudes.")

        response, response_meta = _make_response(output_amplitudes, runner)
        metadata.update(response_meta)
        metadata["input_amplitudes"] = input_amplitudes
        metadata["smatrix_shape"] = tuple(smatrix.shape)
        metadata["tracking_mode_names"] = getattr(geometry, "_tracking_mode_names", None)

        return {
            "response": response,
            "raw_smatrix": smatrix if bool(config.get("return_raw_smatrix", True)) else None,
            "output_amplitudes": output_amplitudes,
            "metadata": metadata,
            "valid": True,
            "error": None,
        }
    except Exception as exc:  # Keep batch dataset generation alive on bad z.
        metadata.setdefault("error_type", type(exc).__name__)
        return {
            "response": np.full(RESPONSE_SIZE, np.nan, dtype=float),
            "raw_smatrix": None,
            "output_amplitudes": None,
            "metadata": metadata,
            "valid": False,
            "error": str(exc),
        }


def _repo_root(config: dict) -> Path:
    if config.get("repo_root"):
        return Path(config["repo_root"]).expanduser().resolve()
    return Path(__file__).resolve().parents[2]


def _dataset_path(config: dict, repo_root: Path) -> Path:
    if config.get("dataset_path"):
        return Path(config["dataset_path"]).expanduser().resolve()
    return repo_root / "sample_datasets" / "Si_rectangular_single_waveguide"


def _validate_dataset_path(dataset_path: Path) -> None:
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset path does not exist: {dataset_path}. "
            "Run from the repository root or set config['dataset_path']."
        )
    if not dataset_path.is_dir():
        raise NotADirectoryError(f"Dataset path is not a directory: {dataset_path}")

    missing = [name for name in REQUIRED_DATASET_FILES if not (dataset_path / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"Dataset is missing required files: {missing}. "
            "If this is a fresh clone, run: git lfs install && git lfs pull"
        )

    for name in REQUIRED_DATASET_FILES:
        path = dataset_path / name
        if path.suffix != ".pkl":
            continue
        header = path.read_bytes()[:80]
        if header.startswith(b"version https://git-lfs.github.com/spec"):
            raise RuntimeError(f"{path} is a Git LFS pointer. Run: git lfs pull")
        if path.stat().st_size < 1024:
            raise RuntimeError(f"{path} is unexpectedly small. Run: git lfs pull")


def _ensure_import_paths(repo_root: Path) -> None:
    src_root = repo_root / "src"
    for path in (str(repo_root), str(src_root)):
        if path not in sys.path:
            sys.path.insert(0, path)


def _preinitialize_ray(repo_root: Path, config: dict) -> None:
    if not bool(config.get("initialize_ray", True)):
        return
    try:
        import ray
    except ImportError as exc:
        raise ImportError("Missing dependency 'ray'. Install with: pip install -r requirements.txt") from exc

    if ray.is_initialized():
        return

    excludes = list(
        config.get(
            "ray_excludes",
            [
                ".git/",
                ".venv/",
                ".venv312/",
                "sample_datasets/",
                "examples/",
                "outputs/",
                "__pycache__/",
            ],
        )
    )
    ray.init(
        ignore_reinit_error=True,
        include_dashboard=False,
        log_to_driver=False,
        runtime_env={"working_dir": str(repo_root), "excludes": excludes},
    )


def _load_dataset(sim: Any, dataset_path: Path, is_testmode: bool) -> Any:
    key = (str(dataset_path), is_testmode)
    if key not in _DATASET_CACHE:
        _DATASET_CACHE[key] = sim.DataUpdater(str(dataset_path), is_testmode=is_testmode)
    return _DATASET_CACHE[key]


def _normalized_vector(z: np.ndarray) -> np.ndarray:
    values = np.asarray(z, dtype=float).reshape(-1)
    if values.size == 0:
        raise ValueError("z must contain at least one parameter.")
    if not np.all(np.isfinite(values)):
        raise ValueError("z contains NaN or Inf.")
    return np.clip(values, 0.0, 1.0)


def _build_geometry(
    sim: Any,
    dataset: Any,
    z_norm: np.ndarray,
    config: dict,
    parameter_grid: dict,
    has_curvature: bool,
) -> Tuple[Any, Dict[str, Any]]:
    num_sections = int(config.get("num_sections", 5))
    if num_sections < 3:
        raise ValueError("config['num_sections'] must be at least 3.")

    width_grid = np.asarray(parameter_grid["top_width"], dtype=float)
    width_min = float(config.get("width_min", width_grid.min()))
    width_max = float(config.get("width_max", width_grid.max()))
    width_min = max(width_min, float(width_grid.min()))
    width_max = min(width_max, float(width_grid.max()))
    if width_min >= width_max:
        raise ValueError("Width bounds are invalid after clipping to the dataset grid.")

    length = float(config.get("length", 10.0e-6))
    if length <= 0:
        raise ValueError("config['length'] must be positive.")

    resolution = int(config.get("resolution", 500))
    limit_mode_number = int(config.get("limit_mode_number", 0))
    verbose = bool(config.get("verbose", False))

    if has_curvature:
        expected = 2 * num_sections
        if z_norm.size != expected:
            raise ValueError(
                f"SingleCustomBend expects z length {expected} for num_sections={num_sections} "
                f"(first half widths, second half curvatures); got {z_norm.size}."
            )
        width_z = z_norm[:num_sections]
        curvature_z = z_norm[num_sections:]
        curvature_grid = np.asarray(parameter_grid["curvature"], dtype=float)
        curvature_min = float(config.get("curvature_min", curvature_grid.min()))
        curvature_max = float(config.get("curvature_max", min(float(curvature_grid.max()), 2.0e4)))
        curvature_min = max(curvature_min, float(curvature_grid.min()))
        curvature_max = min(curvature_max, float(curvature_grid.max()))
        if curvature_min > curvature_max:
            raise ValueError("Curvature bounds are invalid after clipping to the dataset grid.")

        widths = width_min + width_z * (width_max - width_min)
        curvatures = curvature_min + curvature_z * (curvature_max - curvature_min)
        if bool(config.get("zero_curvature_endpoints", True)):
            curvatures[0] = 0.0
            curvatures[-1] = 0.0
        prop_lengths = np.linspace(0.0, length, num_sections)
        geometry = sim.SingleCustomBend(
            dataset,
            prop_lengths,
            widths,
            curvatures,
            input_angle=float(config.get("input_angle", 0.0)),
            resolution=resolution,
            limit_mode_number=limit_mode_number,
            verbose=verbose,
        )
        return geometry, {
            "geometry_type": "SingleCustomBend",
            "num_sections": num_sections,
            "length": length,
            "widths": widths,
            "curvatures": curvatures,
            "prop_lengths": prop_lengths,
            "z_clipped": z_norm,
        }

    expected = num_sections
    if z_norm.size != expected:
        raise ValueError(f"CustomTaper expects z length {expected} for num_sections={num_sections}; got {z_norm.size}.")
    widths = width_min + z_norm * (width_max - width_min)
    geometry = sim.CustomTaper(
        dataset,
        length=length,
        num_sections=num_sections,
        width_list=widths,
        prop_angle=float(config.get("prop_angle", 0.0)),
        resolution=resolution,
        limit_mode_number=limit_mode_number,
        verbose=verbose,
    )
    return geometry, {
        "geometry_type": "CustomTaper",
        "num_sections": num_sections,
        "length": length,
        "widths": widths,
        "z_clipped": z_norm,
    }


def _make_te0_like_input(runner: Any, config: dict) -> np.ndarray:
    inner_runner = getattr(runner, "runner", runner)
    radiation_mask = getattr(inner_runner, "_radiation_mode_mask", None)
    if radiation_mask is None:
        raise RuntimeError("Runner does not expose _radiation_mode_mask; cannot infer input dimension.")
    initial_mask = np.asarray(radiation_mask[0])
    non_radiation_count = int(np.count_nonzero(initial_mask == False))
    if non_radiation_count <= 0:
        raise RuntimeError("No non-radiation input modes are available.")

    configured = config.get("input_amplitudes")
    if configured is not None:
        amplitudes = np.asarray(configured, dtype=np.complex64).reshape(-1)
        if amplitudes.shape != (non_radiation_count,):
            raise ValueError(
                f"config['input_amplitudes'] must have length {non_radiation_count}; got {amplitudes.size}."
            )
        return amplitudes

    amplitudes = np.zeros(non_radiation_count, dtype=np.complex64)
    amplitudes[0] = 1.0 + 0.0j
    return amplitudes


def _make_response(output_amplitudes: np.ndarray, runner: Any) -> Tuple[np.ndarray, Dict[str, Any]]:
    output_amplitudes = np.asarray(output_amplitudes)
    if output_amplitudes.ndim != 1 or output_amplitudes.size < 2:
        raise RuntimeError("Output amplitudes must be a 1D forward/backward mode vector.")

    mode_count = output_amplitudes.size // 2
    forward = output_amplitudes[:mode_count]
    intensities = np.abs(forward) ** 2

    inner_runner = getattr(runner, "runner", runner)
    radiation_mask = np.asarray(getattr(inner_runner, "_radiation_mode_mask"))
    guided_forward_mask = radiation_mask[-1, :mode_count] == False
    guided_indices = np.where(guided_forward_mask)[0]
    guided_intensities = intensities[guided_forward_mask]

    first = float(guided_intensities[0]) if guided_intensities.size >= 1 else 0.0
    second = float(guided_intensities[1]) if guided_intensities.size >= 2 else 0.0
    total_transmitted = float(np.sum(guided_intensities)) if guided_intensities.size else 0.0
    loss = float(1.0 - total_transmitted)
    if guided_indices.size >= 2:
        a0 = forward[guided_indices[0]]
        a1 = forward[guided_indices[1]]
        relative_phase = float(np.angle(a1 * np.conjugate(a0)))
    else:
        relative_phase = 0.0

    response = np.asarray([first, second, total_transmitted, loss, relative_phase], dtype=float)
    mode_labels = np.asarray(
        [f"forward_mode_{idx + 1}" for idx in range(mode_count)]
        + [f"backward_mode_{idx + 1}" for idx in range(mode_count)]
    )
    return response, {
        "mode_count": mode_count,
        "mode_labels": mode_labels,
        "guided_forward_indices": guided_indices,
        "guided_forward_mask": guided_forward_mask,
    }
