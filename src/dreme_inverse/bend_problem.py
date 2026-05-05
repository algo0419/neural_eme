"""Bend mode-splitting wrapper for paper-style DReME evaluations."""

from __future__ import annotations

import contextlib
import io
import math
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

from .wrapper import (
    _configure_runtime_noise_filters,
    _ensure_import_paths,
    _preinitialize_ray,
    _validate_dataset_path,
)


TARGET_5050 = np.asarray([[0.5, 0.5], [0.5, 0.5]], dtype=float)
TARGET_7030 = np.asarray([[0.7, 0.3], [0.3, 0.7]], dtype=float)
_DATASET_CACHE: Dict[Tuple[str, bool], Any] = {}


def run_bend_dreme(z: np.ndarray, config: dict) -> dict:
    """Run one DReME bend mode-splitting evaluation.

    ``z`` is a normalized vector of profile control values. The default
    configuration keeps width fixed at 2 um and interprets ``z`` as interior
    Bezier control values for a positive curvature profile with zero curvature
    endpoints.
    """
    metadata: Dict[str, Any] = {}
    try:
        cfg = _normalized_config(config)
        runtime = dict(cfg.get("runtime", {}))
        _configure_runtime_noise_filters(runtime)

        repo_root = _repo_root(cfg)
        dataset_path = _dataset_path(cfg, repo_root)
        _validate_dataset_path(dataset_path)
        _ensure_import_paths(repo_root)
        _preinitialize_ray(repo_root, _ray_config(repo_root, runtime))

        suppressed_stdout = ""
        if bool(runtime.get("suppress_expected_warnings", True)):
            with contextlib.redirect_stdout(io.StringIO()) as buffer:
                import em_simulation as sim

                dataset = _load_dataset(sim, dataset_path, bool(cfg["dataset"].get("is_testmode", True)))
            suppressed_stdout = buffer.getvalue().strip()
        else:
            import em_simulation as sim

            dataset = _load_dataset(sim, dataset_path, bool(cfg["dataset"].get("is_testmode", True)))

        parameter_names = list(getattr(dataset, "parameter_names", dataset.get_parameter_names()))
        parameter_grid = getattr(dataset, "parameter_grid", dataset.get_parameter_grid())
        _validate_dataset_support(parameter_names, parameter_grid, cfg)

        geometry_params = z_to_bend_geometry_parameters(z, cfg)
        geometry = sim.SingleCustomBend(
            dataset,
            geometry_params["prop_len_list"],
            geometry_params["width_list"],
            geometry_params["curvature_list"],
            input_angle=float(cfg["geometry"].get("input_angle", 0.0)),
            resolution=int(cfg["geometry"].get("eme_resolution", 500)),
            limit_mode_number=int(cfg["geometry"].get("limit_mode_number", 0)),
            verbose=bool(cfg["geometry"].get("verbose", False)),
        )

        eme = sim.EME(
            geometry,
            force_passive=bool(cfg["eme"].get("force_passive", False)),
            force_unitary=bool(cfg["eme"].get("force_unitary", True)),
        )
        eme.calc_Smatrix()
        runner = sim.Runner(eme)

        extracted = extract_bend_response(runner)
        foms = compute_target_foms(extracted["T"], extracted, geometry_params["max_curvature"], cfg)

        propagator = getattr(eme, "propagator", eme)
        smatrix = getattr(propagator, "smatrix", None)
        inner_runner = getattr(runner, "runner", runner)
        lumped_smatrix = getattr(inner_runner, "_lumped_smatrix", None)

        metadata.update(
            {
                "dataset_path": str(dataset_path),
                "dataset_parameter_names": parameter_names,
                "dataset_parameter_ranges": _parameter_ranges(parameter_grid),
                "geometry_type": "SingleCustomBend",
                "geometry": _copy_geometry_metadata(geometry_params),
                "tracking_mode_names": getattr(geometry, "_tracking_mode_names", None),
                "mode_count": extracted["mode_count"],
                "input_guided_count": extracted["input_guided_count"],
                "guided_forward_indices": extracted["guided_forward_indices"],
                "guided_forward_mask": extracted["guided_forward_mask"],
                "lumapi_call_made": not bool(cfg["dataset"].get("is_testmode", True)),
                "force_unitary": bool(cfg["eme"].get("force_unitary", True)),
                "force_passive": bool(cfg["eme"].get("force_passive", False)),
                "z_clipped": geometry_params["z_clipped"],
            }
        )
        if smatrix is not None:
            metadata["smatrix_shape"] = tuple(smatrix.shape)
        if suppressed_stdout and bool(runtime.get("record_suppressed_stdout", False)):
            metadata["suppressed_stdout"] = suppressed_stdout

        response = np.asarray(
            [
                extracted["T"][0, 0],
                extracted["T"][0, 1],
                extracted["T"][1, 0],
                extracted["T"][1, 1],
                extracted["total_te0"],
                extracted["total_te1"],
                extracted["loss_te0"],
                extracted["loss_te1"],
                foms["50:50"]["fom"],
                foms["70:30"]["fom"],
            ],
            dtype=float,
        )
        diagnostics = {
            "total_te0": extracted["total_te0"],
            "total_te1": extracted["total_te1"],
            "loss_te0": extracted["loss_te0"],
            "loss_te1": extracted["loss_te1"],
            "reflection_te0": extracted["reflection_te0"],
            "reflection_te1": extracted["reflection_te1"],
            "power_error_te0": extracted["power_error_te0"],
            "power_error_te1": extracted["power_error_te1"],
            "max_curvature": geometry_params["max_curvature"],
        }

        return {
            "response": response,
            "T": extracted["T"],
            "diagnostics": diagnostics,
            "fom_5050": foms["50:50"]["fom"],
            "fom_7030": foms["70:30"]["fom"],
            "foms": foms,
            "targets": {"50:50": TARGET_5050.copy(), "70:30": TARGET_7030.copy()},
            "output_amplitudes": {
                "te0": extracted["out_te0"],
                "te1": extracted["out_te1"],
            },
            "raw_smatrix": smatrix if bool(cfg["eme"].get("return_raw_smatrix", False)) else None,
            "lumped_smatrix": lumped_smatrix if bool(cfg["eme"].get("return_lumped_smatrix", False)) else None,
            "metadata": metadata,
            "valid": True,
            "error": None,
        }
    except Exception as exc:
        metadata.setdefault("error_type", type(exc).__name__)
        return {
            "response": None,
            "T": None,
            "diagnostics": {},
            "fom_5050": math.nan,
            "fom_7030": math.nan,
            "foms": {},
            "targets": {"50:50": TARGET_5050.copy(), "70:30": TARGET_7030.copy()},
            "output_amplitudes": None,
            "raw_smatrix": None,
            "lumped_smatrix": None,
            "metadata": metadata,
            "valid": False,
            "error": str(exc),
        }


def z_to_bend_geometry_parameters(z: np.ndarray, config: dict) -> Dict[str, np.ndarray | float]:
    """Convert normalized profile controls into `SingleCustomBend` inputs."""
    cfg = _normalized_config(config)
    geom = cfg["geometry"]
    bounds = cfg["bounds"]
    z_arr = np.asarray(z, dtype=float).reshape(-1)
    if z_arr.size == 0:
        raise ValueError("z must contain at least one curvature control value.")
    if not np.all(np.isfinite(z_arr)):
        raise ValueError("z contains NaN or Inf.")

    expected = expected_z_dim(cfg)
    if z_arr.size != expected:
        raise ValueError(f"Expected z length {expected}, got {z_arr.size}.")
    z_clipped = np.clip(z_arr, 0.0, 1.0)

    variable_width = bool(geom.get("variable_width", False))
    width_control_count = int(geom.get("width_control_points", 0)) if variable_width else 0
    curvature_control_count = int(geom["curvature_control_points"])
    width_z = z_clipped[:width_control_count]
    curvature_z = z_clipped[width_control_count : width_control_count + curvature_control_count]

    profile_resolution = int(geom["profile_resolution"])
    prop_len_list = np.linspace(0.0, float(geom["total_length"]), profile_resolution)
    curvature_unit = _profile_from_controls(
        curvature_z,
        profile_resolution,
        endpoint_value=float(geom.get("curvature_endpoint_value", 0.0)),
    )
    curvature_min = float(bounds["curvature_min"])
    curvature_parameter_max = float(geom.get("curvature_parameter_max", bounds["curvature_max"]))
    curvature_parameter_max = min(curvature_parameter_max, float(bounds["curvature_max"]))
    if curvature_parameter_max < curvature_min:
        raise ValueError("curvature_parameter_max must be >= curvature_min.")
    curvature_list = curvature_min + curvature_unit * (curvature_parameter_max - curvature_min)
    curvature_list = np.clip(curvature_list, float(bounds["curvature_min"]), float(bounds["curvature_max"]))

    if variable_width:
        width_unit = _profile_from_controls(
            width_z,
            profile_resolution,
            endpoint_value=float(geom.get("width_endpoint_value", 0.5)),
        )
        width_list = float(bounds["width_min"]) + width_unit * (float(bounds["width_max"]) - float(bounds["width_min"]))
        if bool(geom.get("width_endpoints_fixed", True)):
            fixed_width = float(geom["fixed_width"])
            width_list[0] = fixed_width
            width_list[-1] = fixed_width
    else:
        width_list = np.ones(profile_resolution, dtype=float) * float(geom["fixed_width"])
    width_list = np.clip(width_list, float(bounds["width_min"]), float(bounds["width_max"]))

    x, y, angles = centerline_from_curvature(prop_len_list, curvature_list)
    return {
        "prop_len_list": prop_len_list,
        "width_list": width_list,
        "curvature_list": curvature_list,
        "centerline_x": x,
        "centerline_y": y,
        "prop_angle": angles,
        "max_curvature": float(np.max(np.abs(curvature_list))) if curvature_list.size else 0.0,
        "z_clipped": z_clipped,
    }


def expected_z_dim(config: dict) -> int:
    """Return the normalized vector dimension for the configured bend profile."""
    cfg = _normalized_config(config)
    geom = cfg["geometry"]
    width_dim = int(geom.get("width_control_points", 0)) if bool(geom.get("variable_width", False)) else 0
    return width_dim + int(geom["curvature_control_points"])


def extract_bend_response(runner: Any) -> Dict[str, Any]:
    """Propagate TE0-like and TE1-like inputs and extract first-two-mode powers."""
    inner_runner = getattr(runner, "runner", runner)
    radiation_mask = np.asarray(getattr(inner_runner, "_radiation_mode_mask"))
    if radiation_mask.ndim != 2:
        raise RuntimeError("Runner radiation mode mask has unexpected shape.")
    mode_count = int(getattr(inner_runner, "_mode_count"))
    initial_mask = radiation_mask[0]
    input_guided_count = int(np.count_nonzero(initial_mask == False))
    if input_guided_count < 2:
        raise RuntimeError("At least two guided input modes are required for bend splitting.")

    te0 = np.zeros(input_guided_count, dtype=np.complex64)
    te1 = np.zeros(input_guided_count, dtype=np.complex64)
    te0[0] = 1.0 + 0.0j
    te1[1] = 1.0 + 0.0j

    out_te0 = np.asarray(runner.propagate_lumped_smatrix(te0))
    out_te1 = np.asarray(runner.propagate_lumped_smatrix(te1))
    if out_te0.shape != out_te1.shape or out_te0.size < 2 * mode_count:
        raise RuntimeError("Runner returned inconsistent output amplitude shapes.")

    T = np.asarray(
        [
            [np.abs(out_te0[0]) ** 2, np.abs(out_te0[1]) ** 2],
            [np.abs(out_te1[0]) ** 2, np.abs(out_te1[1]) ** 2],
        ],
        dtype=float,
    )
    guided_forward_mask = radiation_mask[-1, :mode_count] == False
    guided_backward_mask = radiation_mask[-1, mode_count : 2 * mode_count] == False
    guided_forward_indices = np.where(guided_forward_mask)[0]

    forward_te0 = out_te0[:mode_count]
    forward_te1 = out_te1[:mode_count]
    backward_te0 = out_te0[mode_count : 2 * mode_count]
    backward_te1 = out_te1[mode_count : 2 * mode_count]
    total_te0 = float(np.sum(np.abs(forward_te0[guided_forward_mask]) ** 2))
    total_te1 = float(np.sum(np.abs(forward_te1[guided_forward_mask]) ** 2))
    reflection_te0 = float(np.sum(np.abs(backward_te0[guided_backward_mask]) ** 2))
    reflection_te1 = float(np.sum(np.abs(backward_te1[guided_backward_mask]) ** 2))

    return {
        "T": T,
        "out_te0": out_te0,
        "out_te1": out_te1,
        "total_te0": total_te0,
        "total_te1": total_te1,
        "loss_te0": float(1.0 - total_te0),
        "loss_te1": float(1.0 - total_te1),
        "reflection_te0": reflection_te0,
        "reflection_te1": reflection_te1,
        "power_error_te0": float(abs(1.0 - total_te0)),
        "power_error_te1": float(abs(1.0 - total_te1)),
        "mode_count": mode_count,
        "input_guided_count": input_guided_count,
        "guided_forward_indices": guided_forward_indices,
        "guided_forward_mask": guided_forward_mask,
    }


def compute_target_foms(
    T: np.ndarray,
    extracted: Dict[str, Any],
    max_curvature: float,
    config: dict,
) -> Dict[str, Dict[str, float]]:
    """Compute 50:50 and 70:30 FOMs with curvature and power penalties."""
    cfg = _normalized_config(config)
    fom_cfg = cfg["fom"]
    beta = float(fom_cfg.get("beta_power_error", 1.0))
    power_error = float(extracted["power_error_te0"] + extracted["power_error_te1"])
    curvature_penalty = compute_curvature_penalty(max_curvature, cfg)
    targets = {"50:50": TARGET_5050, "70:30": TARGET_7030}
    results: Dict[str, Dict[str, float]] = {}
    for name, target in targets.items():
        target_error = float(np.sum(np.abs(np.asarray(T, dtype=float) - target)))
        results[name] = {
            "target_error": target_error,
            "curvature_penalty": float(curvature_penalty),
            "power_error": power_error,
            "fom": float(target_error + curvature_penalty + beta * power_error),
        }
    return results


def compute_curvature_penalty(max_curvature: float, config: dict) -> float:
    """Paper-style soft penalty for exceeding a curvature limit.

    The default penalty is ``max(0, (max_curvature - limit) / scale)**power``.
    This keeps the penalty dimensionless and lets ``penalty_scale`` define the
    curvature excess that contributes order-one error to the FOM.
    """
    cfg = _normalized_config(config)
    fom_cfg = cfg["fom"]
    limit = float(fom_cfg.get("curvature_limit", 325000.0))
    scale = float(fom_cfg.get("penalty_scale", 10000.0))
    power = float(fom_cfg.get("penalty_power", 2.0))
    if scale <= 0:
        raise ValueError("fom.penalty_scale must be positive.")
    excess = max(0.0, float(max_curvature) - limit)
    return float((excess / scale) ** power)


def centerline_from_curvature(prop_len_list: np.ndarray, curvature_list: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Integrate curvature into a top-view centerline."""
    prop = np.asarray(prop_len_list, dtype=float)
    curvature = np.asarray(curvature_list, dtype=float)
    if prop.ndim != 1 or curvature.ndim != 1 or prop.shape != curvature.shape:
        raise ValueError("prop_len_list and curvature_list must be 1D arrays with the same shape.")
    if prop.size < 2:
        raise ValueError("At least two propagation samples are required.")
    ds = np.diff(prop)
    avg_curvature = 0.5 * (curvature[:-1] + curvature[1:])
    angles = np.zeros_like(prop)
    angles[1:] = np.cumsum(avg_curvature * ds)
    avg_angles = 0.5 * (angles[:-1] + angles[1:])
    x = np.zeros_like(prop)
    y = np.zeros_like(prop)
    x[1:] = np.cumsum(np.cos(avg_angles) * ds)
    y[1:] = np.cumsum(np.sin(avg_angles) * ds)
    return x, y, angles


def load_bend_config(path: str | Path, *, repo_root: str | Path | None = None) -> dict:
    """Load a YAML bend config and resolve repository-relative paths later."""
    import yaml

    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as file:
        loaded = yaml.safe_load(file) or {}
    cfg = _normalized_config(loaded)
    if repo_root is not None:
        cfg["repo_root"] = str(Path(repo_root).expanduser().resolve())
    return cfg


def _normalized_config(config: dict) -> dict:
    cfg = deepcopy(config or {})
    cfg.setdefault("repo_root", None)
    cfg.setdefault("dataset", {})
    cfg["dataset"].setdefault("path", "sample_datasets/Si_rectangular_single_waveguide")
    cfg["dataset"].setdefault("is_testmode", True)
    cfg.setdefault("bounds", {})
    cfg["bounds"].setdefault("width_min", 1.0e-6)
    cfg["bounds"].setdefault("width_max", 3.0e-6)
    cfg["bounds"].setdefault("curvature_min", 0.0)
    cfg["bounds"].setdefault("curvature_max", 330000.0)
    cfg.setdefault("geometry", {})
    cfg["geometry"].setdefault("fixed_width", 2.0e-6)
    cfg["geometry"].setdefault("variable_width", False)
    cfg["geometry"].setdefault("width_control_points", 4)
    cfg["geometry"].setdefault("width_endpoint_value", 0.5)
    cfg["geometry"].setdefault("width_endpoints_fixed", True)
    cfg["geometry"].setdefault("curvature_control_points", 6)
    cfg["geometry"].setdefault("curvature_endpoint_value", 0.0)
    cfg["geometry"].setdefault("curvature_parameter_max", 120000.0)
    cfg["geometry"].setdefault("profile_resolution", 100)
    cfg["geometry"].setdefault("eme_resolution", 500)
    cfg["geometry"].setdefault("total_length", 31.41592653589793e-6)
    cfg["geometry"].setdefault("input_angle", 0.0)
    cfg["geometry"].setdefault("limit_mode_number", 0)
    cfg["geometry"].setdefault("verbose", False)
    cfg.setdefault("eme", {})
    cfg["eme"].setdefault("force_unitary", True)
    cfg["eme"].setdefault("force_passive", False)
    cfg["eme"].setdefault("return_raw_smatrix", False)
    cfg["eme"].setdefault("return_lumped_smatrix", False)
    cfg.setdefault("fom", {})
    cfg["fom"].setdefault("curvature_limit", 325000.0)
    cfg["fom"].setdefault("penalty_scale", 10000.0)
    cfg["fom"].setdefault("penalty_power", 2.0)
    cfg["fom"].setdefault("beta_power_error", 1.0)
    cfg.setdefault("runtime", {})
    cfg["runtime"].setdefault("initialize_ray", True)
    cfg["runtime"].setdefault("ray_logging_level", "ERROR")
    cfg["runtime"].setdefault("suppress_expected_warnings", True)
    cfg["runtime"].setdefault(
        "ray_excludes",
        [".git/", ".venv/", ".venv312/", "sample_datasets/", "examples/", "outputs/", "data/", "__pycache__/"],
    )
    cfg.setdefault("test", {})
    cfg["test"].setdefault("simple_curvature_control_value", 0.16)
    cfg["test"].setdefault("random_low", 0.02)
    cfg["test"].setdefault("random_high", 0.45)
    cfg["test"].setdefault("seed", 20260505)
    return cfg


def _repo_root(config: dict) -> Path:
    if config.get("repo_root"):
        return Path(config["repo_root"]).expanduser().resolve()
    return Path(__file__).resolve().parents[2]


def _dataset_path(config: dict, repo_root: Path) -> Path:
    dataset_path = Path(config["dataset"]["path"]).expanduser()
    if not dataset_path.is_absolute():
        dataset_path = repo_root / dataset_path
    return dataset_path.resolve()


def _ray_config(repo_root: Path, runtime: dict) -> dict:
    cfg = dict(runtime)
    cfg["repo_root"] = str(repo_root)
    return cfg


def _load_dataset(sim: Any, dataset_path: Path, is_testmode: bool) -> Any:
    key = (str(dataset_path), bool(is_testmode))
    if key not in _DATASET_CACHE:
        _DATASET_CACHE[key] = sim.DataUpdater(str(dataset_path), is_testmode=is_testmode)
    return _DATASET_CACHE[key]


def _validate_dataset_support(parameter_names: list[str], parameter_grid: dict, config: dict) -> None:
    for name in ("top_width", "curvature"):
        if name not in parameter_names:
            raise ValueError(f"Dataset does not expose required parameter '{name}'.")
        if name not in parameter_grid:
            raise ValueError(f"Dataset parameter grid does not expose '{name}'.")
    bounds = config["bounds"]
    width_grid = np.asarray(parameter_grid["top_width"], dtype=float)
    curvature_grid = np.asarray(parameter_grid["curvature"], dtype=float)
    if float(bounds["width_min"]) < width_grid.min() or float(bounds["width_max"]) > width_grid.max():
        raise ValueError("Configured width bounds exceed the dataset top_width grid.")
    if float(bounds["curvature_min"]) < curvature_grid.min() or float(bounds["curvature_max"]) > curvature_grid.max():
        raise ValueError("Configured curvature bounds exceed the dataset curvature grid.")


def _profile_from_controls(control_values: np.ndarray, resolution: int, *, endpoint_value: float) -> np.ndarray:
    if resolution < 2:
        raise ValueError("profile_resolution must be at least 2.")
    controls = np.concatenate([[endpoint_value], np.asarray(control_values, dtype=float), [endpoint_value]])
    controls = np.clip(controls, 0.0, 1.0)
    n = controls.size - 1
    t = np.linspace(0.0, 1.0, resolution)
    profile = np.zeros_like(t, dtype=float)
    for idx, value in enumerate(controls):
        profile += math.comb(n, idx) * (t**idx) * ((1.0 - t) ** (n - idx)) * value
    return np.clip(profile, 0.0, 1.0)


def _parameter_ranges(parameter_grid: dict) -> dict[str, tuple[float, float]]:
    ranges = {}
    for name, values in parameter_grid.items():
        arr = np.asarray(values, dtype=float)
        ranges[name] = (float(arr.min()), float(arr.max()))
    return ranges


def _copy_geometry_metadata(geometry_params: Dict[str, Any]) -> Dict[str, Any]:
    keys = (
        "prop_len_list",
        "width_list",
        "curvature_list",
        "centerline_x",
        "centerline_y",
        "prop_angle",
        "max_curvature",
    )
    return {key: deepcopy(geometry_params[key]) for key in keys}
