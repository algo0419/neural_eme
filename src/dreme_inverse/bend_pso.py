"""PSO baseline for bend mode-splitting DReME optimization."""

from __future__ import annotations

import csv
import json
import math
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .bend_problem import (
    TARGET_5050,
    TARGET_7030,
    compute_curvature_penalty,
    expected_z_dim,
    run_bend_dreme,
)


TARGET_PRESERVE = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=float)


@dataclass(frozen=True)
class BendPSOConfig:
    """Hyperparameters for standard particle swarm optimization."""

    num_particles: int
    num_iterations: int
    inertia: float = 0.72
    cognitive: float = 1.49
    social: float = 1.49
    seed: int = 20260505
    invalid_penalty: float = 1.0e6
    initial_velocity_scale: float = 0.12


def run_bend_pso(
    config: dict,
    output_dir: str | Path,
    *,
    target: str = "split",
    ratio: float = 0.5,
    pso_config: BendPSOConfig,
    data_output_path: str | Path | None = None,
) -> Dict[str, Any]:
    """Optimize a bend design vector with PSO using ``run_bend_dreme``."""
    _validate_pso_config(pso_config)
    target_spec = make_target_spec(target, ratio)
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if data_output_path is not None:
        data_output_path = Path(data_output_path).expanduser().resolve()
        data_output_path.parent.mkdir(parents=True, exist_ok=True)

    cfg = deepcopy(config)
    z_dim = expected_z_dim(cfg)
    rng = np.random.default_rng(pso_config.seed)
    positions = rng.uniform(0.0, 1.0, size=(pso_config.num_particles, z_dim))
    velocities = rng.uniform(
        -pso_config.initial_velocity_scale,
        pso_config.initial_velocity_scale,
        size=(pso_config.num_particles, z_dim),
    )

    personal_best_positions = positions.copy()
    personal_best_scores = np.full(pso_config.num_particles, np.inf, dtype=float)
    personal_best_records: list[Optional[Dict[str, Any]]] = [None] * pso_config.num_particles
    global_best_position: Optional[np.ndarray] = None
    global_best_score = math.inf
    global_best_record: Optional[Dict[str, Any]] = None

    evaluations: List[Dict[str, Any]] = []
    history: List[Dict[str, Any]] = []

    for iteration in range(pso_config.num_iterations + 1):
        if iteration > 0:
            assert global_best_position is not None
            r1 = rng.random(size=positions.shape)
            r2 = rng.random(size=positions.shape)
            velocities = (
                pso_config.inertia * velocities
                + pso_config.cognitive * r1 * (personal_best_positions - positions)
                + pso_config.social * r2 * (global_best_position - positions)
            )
            positions = np.clip(positions + velocities, 0.0, 1.0)
            velocities = np.where((positions <= 0.0) | (positions >= 1.0), 0.0, velocities)

        for particle_index in range(pso_config.num_particles):
            record = evaluate_particle(
                positions[particle_index],
                cfg,
                target_spec,
                invalid_penalty=pso_config.invalid_penalty,
                iteration=iteration,
                particle_index=particle_index,
            )
            evaluations.append(record)
            score = float(record["fom"])
            if score < personal_best_scores[particle_index]:
                personal_best_scores[particle_index] = score
                personal_best_positions[particle_index] = positions[particle_index].copy()
                personal_best_records[particle_index] = record
            if score < global_best_score:
                global_best_score = score
                global_best_position = positions[particle_index].copy()
                global_best_record = record

        assert global_best_record is not None
        history.append(_history_row(iteration, global_best_record))

    assert global_best_position is not None
    assert global_best_record is not None
    arrays = _evaluation_arrays(evaluations, z_dim)
    artifact_paths = save_pso_outputs(
        output_dir,
        cfg,
        pso_config,
        target_spec,
        history,
        arrays,
        global_best_position,
        global_best_record,
        data_output_path=data_output_path,
    )

    first_best = float(history[0]["best_fom"])
    final_best = float(history[-1]["best_fom"])
    summary = {
        "target_name": target_spec["name"],
        "target_matrix": target_spec["matrix"].tolist(),
        "output_dir": str(output_dir),
        "data_output_path": str(data_output_path) if data_output_path is not None else None,
        "num_particles": pso_config.num_particles,
        "num_iterations": pso_config.num_iterations,
        "z_dim": z_dim,
        "first_best_fom": first_best,
        "best_fom": final_best,
        "improvement": float(first_best - final_best),
        "best_T": np.asarray(global_best_record["T"], dtype=float).tolist(),
        "best_max_curvature": float(global_best_record["max_curvature"]),
        "best_power_error": float(global_best_record["power_error"]),
        "lumapi_call_made": bool(any(record["lumapi_call_made"] for record in evaluations)),
        "valid_evaluations": int(np.count_nonzero(arrays["valid"])),
        "total_evaluations": int(len(evaluations)),
        "artifacts": artifact_paths,
    }
    return summary


def evaluate_particle(
    z: np.ndarray,
    config: dict,
    target_spec: Dict[str, Any],
    *,
    invalid_penalty: float,
    iteration: int,
    particle_index: int,
) -> Dict[str, Any]:
    """Evaluate one particle and convert failures to a finite penalty."""
    z = np.clip(np.asarray(z, dtype=float).reshape(-1), 0.0, 1.0)
    result = run_bend_dreme(z, config)
    if not result.get("valid", False):
        return {
            "iteration": int(iteration),
            "particle_index": int(particle_index),
            "z": z,
            "valid": False,
            "fom": float(invalid_penalty),
            "target_error": float(invalid_penalty),
            "curvature_penalty": 0.0,
            "power_error": 0.0,
            "T": np.full((2, 2), np.nan, dtype=float),
            "response": np.full(10, np.nan, dtype=float),
            "max_curvature": np.nan,
            "total_te0": np.nan,
            "total_te1": np.nan,
            "loss_te0": np.nan,
            "loss_te1": np.nan,
            "reflection_te0": np.nan,
            "reflection_te1": np.nan,
            "lumapi_call_made": False,
            "error": str(result.get("error")),
            "metadata": result.get("metadata", {}),
        }

    fitness = target_fitness_from_result(result, target_spec, config)
    diagnostics = result.get("diagnostics", {})
    return {
        "iteration": int(iteration),
        "particle_index": int(particle_index),
        "z": z,
        "valid": True,
        "fom": float(fitness["fom"]),
        "target_error": float(fitness["target_error"]),
        "curvature_penalty": float(fitness["curvature_penalty"]),
        "power_error": float(fitness["power_error"]),
        "T": np.asarray(result["T"], dtype=float),
        "response": np.asarray(result["response"], dtype=float),
        "max_curvature": float(diagnostics.get("max_curvature", np.nan)),
        "total_te0": float(diagnostics.get("total_te0", np.nan)),
        "total_te1": float(diagnostics.get("total_te1", np.nan)),
        "loss_te0": float(diagnostics.get("loss_te0", np.nan)),
        "loss_te1": float(diagnostics.get("loss_te1", np.nan)),
        "reflection_te0": float(diagnostics.get("reflection_te0", np.nan)),
        "reflection_te1": float(diagnostics.get("reflection_te1", np.nan)),
        "lumapi_call_made": bool(result.get("metadata", {}).get("lumapi_call_made", False)),
        "error": None,
        "metadata": result.get("metadata", {}),
    }


def target_fitness_from_result(result: Dict[str, Any], target_spec: Dict[str, Any], config: dict) -> Dict[str, float]:
    """Compute target-specific FOM using bend-problem penalties and targets."""
    target_key = target_spec["bend_problem_key"]
    if target_key is not None and target_key in result.get("foms", {}):
        return {key: float(value) for key, value in result["foms"][target_key].items()}

    diagnostics = result.get("diagnostics", {})
    T = np.asarray(result["T"], dtype=float)
    target_matrix = np.asarray(target_spec["matrix"], dtype=float)
    target_error = float(np.sum(np.abs(T - target_matrix)))
    curvature_penalty = compute_curvature_penalty(float(diagnostics["max_curvature"]), config)
    power_error = float(diagnostics["power_error_te0"] + diagnostics["power_error_te1"])
    beta = float(config.get("fom", {}).get("beta_power_error", 1.0))
    return {
        "target_error": target_error,
        "curvature_penalty": float(curvature_penalty),
        "power_error": power_error,
        "fom": float(target_error + curvature_penalty + beta * power_error),
    }


def make_target_spec(target: str, ratio: float) -> Dict[str, Any]:
    """Create a supported target matrix descriptor."""
    target = target.strip().lower()
    if target in {"preserve", "identity", "mode-preserving", "mode_preserving"}:
        return {
            "name": "preserve_TE0TE1",
            "matrix": TARGET_PRESERVE.copy(),
            "target": "preserve",
            "ratio": None,
            "bend_problem_key": None,
        }
    if target != "split":
        raise ValueError("target must be 'split' or 'preserve'.")

    if math.isclose(float(ratio), 0.5, rel_tol=0.0, abs_tol=1e-12):
        return {
            "name": "split_50_50_TE0TE1",
            "matrix": TARGET_5050.copy(),
            "target": "split",
            "ratio": 0.5,
            "bend_problem_key": "50:50",
        }
    if math.isclose(float(ratio), 0.7, rel_tol=0.0, abs_tol=1e-12):
        return {
            "name": "split_70_30_TE0TE1",
            "matrix": TARGET_7030.copy(),
            "target": "split",
            "ratio": 0.7,
            "bend_problem_key": "70:30",
        }
    if 0.0 <= ratio <= 1.0:
        matrix = np.asarray([[ratio, 1.0 - ratio], [1.0 - ratio, ratio]], dtype=float)
        ratio_label = f"{int(round(ratio * 100)):02d}_{int(round((1.0 - ratio) * 100)):02d}"
        return {
            "name": f"split_{ratio_label}_TE0TE1",
            "matrix": matrix,
            "target": "split",
            "ratio": float(ratio),
            "bend_problem_key": None,
        }
    raise ValueError("ratio must be in [0, 1].")


def save_pso_outputs(
    output_dir: Path,
    config: dict,
    pso_config: BendPSOConfig,
    target_spec: Dict[str, Any],
    history: List[Dict[str, Any]],
    arrays: Dict[str, np.ndarray],
    best_z: np.ndarray,
    best_record: Dict[str, Any],
    *,
    data_output_path: str | Path | None,
) -> Dict[str, str]:
    """Write run artifacts and plots."""
    paths = {
        "summary": str(output_dir / "summary.txt"),
        "config_used": str(output_dir / "config_used.yaml"),
        "history": str(output_dir / "pso_history.csv"),
        "all_evaluations": str(output_dir / "all_evaluations.npz"),
        "best_design": str(output_dir / "best_design.npz"),
        "best_fom_plot": str(output_dir / "best_fom_vs_iteration.png"),
        "best_split_error_plot": str(output_dir / "best_split_error_vs_iteration.png"),
        "best_t_heatmap": str(output_dir / "best_T_matrix_heatmap.png"),
        "best_width_profile": str(output_dir / "best_width_profile.png"),
        "best_curvature_profile": str(output_dir / "best_curvature_profile.png"),
        "best_bend_layout": str(output_dir / "best_bend_layout.png"),
    }
    if data_output_path is not None:
        paths["data_result"] = str(data_output_path)

    _write_config_snapshot(Path(paths["config_used"]), config, pso_config, target_spec)
    _write_history_csv(Path(paths["history"]), history)
    _write_summary(Path(paths["summary"]), config, pso_config, target_spec, history, best_z, best_record, paths)
    _save_all_evaluations(Path(paths["all_evaluations"]), arrays, target_spec)
    _save_best_design(Path(paths["best_design"]), best_z, best_record, target_spec)
    if data_output_path is not None:
        _save_best_design(Path(data_output_path), best_z, best_record, target_spec)
    _plot_history(history, Path(paths["best_fom_plot"]), "best_fom", "Best FOM", log_scale=True)
    _plot_history(history, Path(paths["best_split_error_plot"]), "best_split_error", "Best Split Error", log_scale=False)
    _plot_best_profiles(best_record, paths)
    return paths


def _evaluation_arrays(evaluations: List[Dict[str, Any]], z_dim: int) -> Dict[str, np.ndarray]:
    response_size = max(np.asarray(record["response"]).size for record in evaluations)
    return {
        "Z": np.stack([np.asarray(record["z"], dtype=float).reshape(z_dim) for record in evaluations]),
        "responses": np.stack([_fixed_size(record["response"], response_size) for record in evaluations]),
        "T": np.stack([np.asarray(record["T"], dtype=float).reshape(2, 2) for record in evaluations]),
        "FOM": np.asarray([record["fom"] for record in evaluations], dtype=float),
        "target_error": np.asarray([record["target_error"] for record in evaluations], dtype=float),
        "curvature_penalty": np.asarray([record["curvature_penalty"] for record in evaluations], dtype=float),
        "power_error": np.asarray([record["power_error"] for record in evaluations], dtype=float),
        "max_curvature": np.asarray([record["max_curvature"] for record in evaluations], dtype=float),
        "iteration_index": np.asarray([record["iteration"] for record in evaluations], dtype=int),
        "particle_index": np.asarray([record["particle_index"] for record in evaluations], dtype=int),
        "valid": np.asarray([record["valid"] for record in evaluations], dtype=bool),
    }


def _history_row(iteration: int, record: Dict[str, Any]) -> Dict[str, float]:
    T = np.asarray(record["T"], dtype=float)
    return {
        "iteration": int(iteration),
        "best_fom": float(record["fom"]),
        "best_split_error": float(record["target_error"]),
        "best_T00": float(T[0, 0]),
        "best_T01": float(T[0, 1]),
        "best_T10": float(T[1, 0]),
        "best_T11": float(T[1, 1]),
        "best_max_curvature": float(record["max_curvature"]),
        "best_power_error": float(record["power_error"]),
    }


def _fixed_size(values: Any, size: int) -> np.ndarray:
    result = np.full(size, np.nan, dtype=float)
    arr = np.asarray(values, dtype=float).reshape(-1)
    result[: min(size, arr.size)] = arr[: min(size, arr.size)]
    return result


def _save_all_evaluations(path: Path, arrays: Dict[str, np.ndarray], target_spec: Dict[str, Any]) -> None:
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "target_name": target_spec["name"],
        "target": target_spec["target"],
        "ratio": target_spec["ratio"],
        "target_matrix": np.asarray(target_spec["matrix"], dtype=float).tolist(),
    }
    np.savez(path, **arrays, target_metadata_json=json.dumps(metadata, sort_keys=True))


def _save_best_design(path: Path, best_z: np.ndarray, best_record: Dict[str, Any], target_spec: Dict[str, Any]) -> None:
    geometry = best_record.get("metadata", {}).get("geometry", {})
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "target_name": target_spec["name"],
        "target_matrix": np.asarray(target_spec["matrix"], dtype=float).tolist(),
        "fom": float(best_record["fom"]),
        "target_error": float(best_record["target_error"]),
        "power_error": float(best_record["power_error"]),
        "curvature_penalty": float(best_record["curvature_penalty"]),
        "lumapi_call_made": bool(best_record["lumapi_call_made"]),
    }
    np.savez(
        path,
        z=np.asarray(best_z, dtype=float),
        T=np.asarray(best_record["T"], dtype=float),
        response=np.asarray(best_record["response"], dtype=float),
        FOM=np.asarray(float(best_record["fom"]), dtype=float),
        target_matrix=np.asarray(target_spec["matrix"], dtype=float),
        prop_len_list=np.asarray(geometry.get("prop_len_list", []), dtype=float),
        width_list=np.asarray(geometry.get("width_list", []), dtype=float),
        curvature_list=np.asarray(geometry.get("curvature_list", []), dtype=float),
        centerline_x=np.asarray(geometry.get("centerline_x", []), dtype=float),
        centerline_y=np.asarray(geometry.get("centerline_y", []), dtype=float),
        metadata_json=json.dumps(metadata, sort_keys=True),
    )


def _write_config_snapshot(path: Path, config: dict, pso_config: BendPSOConfig, target_spec: Dict[str, Any]) -> None:
    import yaml

    snapshot = {
        "bend_config": _jsonable(config),
        "pso_config": asdict(pso_config),
        "target": {
            "name": target_spec["name"],
            "target": target_spec["target"],
            "ratio": target_spec["ratio"],
            "matrix": np.asarray(target_spec["matrix"], dtype=float).tolist(),
        },
    }
    path.write_text(yaml.safe_dump(snapshot, sort_keys=False), encoding="utf-8")


def _write_history_csv(path: Path, history: List[Dict[str, Any]]) -> None:
    fieldnames = [
        "iteration",
        "best_fom",
        "best_split_error",
        "best_T00",
        "best_T01",
        "best_T10",
        "best_T11",
        "best_max_curvature",
        "best_power_error",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


def _write_summary(
    path: Path,
    config: dict,
    pso_config: BendPSOConfig,
    target_spec: Dict[str, Any],
    history: List[Dict[str, Any]],
    best_z: np.ndarray,
    best_record: Dict[str, Any],
    paths: Dict[str, str],
) -> None:
    bounds = config.get("bounds", {})
    geometry = best_record.get("metadata", {}).get("geometry", {})
    widths = np.asarray(geometry.get("width_list", []), dtype=float)
    curvatures = np.asarray(geometry.get("curvature_list", []), dtype=float)
    T = np.asarray(best_record["T"], dtype=float)
    first_best = float(history[0]["best_fom"])
    final_best = float(history[-1]["best_fom"])
    lines = [
        f"target_name: {target_spec['name']}",
        f"target_matrix: {np.asarray(target_spec['matrix'], dtype=float).tolist()}",
        f"num_particles: {pso_config.num_particles}",
        f"num_iterations: {pso_config.num_iterations}",
        f"seed: {pso_config.seed}",
        f"z_dim: {best_z.size}",
        f"first_best_fom: {first_best:.10g}",
        f"best_fom: {final_best:.10g}",
        f"improvement: {first_best - final_best:.10g}",
        f"best_target_error: {float(best_record['target_error']):.10g}",
        f"best_power_error: {float(best_record['power_error']):.10g}",
        f"best_curvature_penalty: {float(best_record['curvature_penalty']):.10g}",
        f"best_max_curvature: {float(best_record['max_curvature']):.10g}",
        f"lumapi_or_lumerical_call_made: {bool(best_record['lumapi_call_made'])}",
        "",
        "T_matrix:",
        f"  {T[0, 0]:.10g} {T[0, 1]:.10g}",
        f"  {T[1, 0]:.10g} {T[1, 1]:.10g}",
        "",
        "bounds_check:",
        f"  width_min: {float(widths.min()) if widths.size else math.nan:.10g}",
        f"  width_max: {float(widths.max()) if widths.size else math.nan:.10g}",
        f"  curvature_min: {float(curvatures.min()) if curvatures.size else math.nan:.10g}",
        f"  curvature_max: {float(curvatures.max()) if curvatures.size else math.nan:.10g}",
        f"  configured_width_bounds: [{bounds.get('width_min')}, {bounds.get('width_max')}]",
        f"  configured_curvature_bounds: [{bounds.get('curvature_min')}, {bounds.get('curvature_max')}]",
        "",
        "artifacts:",
    ]
    lines.extend(f"  {name}: {artifact_path}" for name, artifact_path in paths.items())
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot_history(history: List[Dict[str, Any]], path: Path, key: str, ylabel: str, *, log_scale: bool) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    iterations = [row["iteration"] for row in history]
    values = [row[key] for row in history]
    fig, axis = plt.subplots(figsize=(6.2, 3.8))
    axis.plot(iterations, values, marker="o", linewidth=1.8)
    axis.set_xlabel("iteration")
    axis.set_ylabel(ylabel)
    axis.set_title(ylabel + " vs Iteration")
    if log_scale and all(value > 0 for value in values):
        axis.set_yscale("log")
    axis.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_best_profiles(best_record: Dict[str, Any], paths: Dict[str, str]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    geometry = best_record.get("metadata", {}).get("geometry", {})
    prop = np.asarray(geometry.get("prop_len_list", []), dtype=float)
    widths = np.asarray(geometry.get("width_list", []), dtype=float)
    curvatures = np.asarray(geometry.get("curvature_list", []), dtype=float)
    x = np.asarray(geometry.get("centerline_x", []), dtype=float)
    y = np.asarray(geometry.get("centerline_y", []), dtype=float)
    T = np.asarray(best_record["T"], dtype=float)

    fig, axis = plt.subplots(figsize=(4.4, 3.8))
    image = axis.imshow(T, vmin=0.0, vmax=max(1.0, float(np.nanmax(T))), cmap="viridis")
    axis.set_xticks([0, 1], labels=["out0", "out1"])
    axis.set_yticks([0, 1], labels=["TE0 in", "TE1 in"])
    axis.set_title("Best T Matrix")
    for row in range(2):
        for col in range(2):
            axis.text(col, row, f"{T[row, col]:.3f}", ha="center", va="center", color="white")
    fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(paths["best_t_heatmap"], dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(6.2, 3.8))
    axis.plot(prop * 1e6, widths * 1e6, linewidth=1.8, color="tab:blue")
    axis.set_xlabel("propagation length (um)")
    axis.set_ylabel("top width (um)")
    axis.set_title("Best Width Profile")
    axis.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(paths["best_width_profile"], dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(6.2, 3.8))
    axis.plot(prop * 1e6, curvatures, linewidth=1.8, color="tab:purple")
    axis.set_xlabel("propagation length (um)")
    axis.set_ylabel("curvature (1/m)")
    axis.set_title("Best Curvature Profile")
    axis.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(paths["best_curvature_profile"], dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(5.0, 4.5))
    axis.plot(x * 1e6, y * 1e6, linewidth=1.8, color="tab:green")
    axis.scatter(x[0] * 1e6, y[0] * 1e6, color="black", s=20, label="in")
    axis.scatter(x[-1] * 1e6, y[-1] * 1e6, color="tab:red", s=20, label="out")
    axis.set_xlabel("x (um)")
    axis.set_ylabel("y (um)")
    axis.set_title("Best Bend Layout")
    axis.set_aspect("equal", adjustable="box")
    axis.grid(True, alpha=0.25)
    axis.legend(loc="best")
    fig.tight_layout()
    fig.savefig(paths["best_bend_layout"], dpi=180, bbox_inches="tight")
    plt.close(fig)


def _validate_pso_config(config: BendPSOConfig) -> None:
    if config.num_particles <= 0:
        raise ValueError("num_particles must be positive.")
    if config.num_iterations <= 0:
        raise ValueError("num_iterations must be positive.")
    if config.initial_velocity_scale < 0:
        raise ValueError("initial_velocity_scale cannot be negative.")
    if config.invalid_penalty <= 0:
        raise ValueError("invalid_penalty must be positive.")


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)
