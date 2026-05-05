"""Target-conditioned dataset builder for bend mode-splitting responses."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

from .bend_problem import TARGET_5050, TARGET_7030, expected_z_dim, load_bend_config, run_bend_dreme
from .bend_pso import make_target_spec, target_fitness_from_result


RAW_RESPONSE_LABELS = (
    "T00",
    "T01",
    "T10",
    "T11",
    "total_te0",
    "total_te1",
    "loss_te0",
    "loss_te1",
    "reflection_te0",
    "reflection_te1",
    "power_error_te0",
    "power_error_te1",
    "max_curvature",
)
TARGET_SPECS = (
    make_target_spec("split", 0.5),
    make_target_spec("split", 0.7),
)
TARGET_TYPE_IDS = {"split_50_50_TE0TE1": 0, "split_70_30_TE0TE1": 1}
TARGET_CONVENTION_IDS = {"TE0TE1_mode_preserving_bias": 0}
SOURCE_IDS = {"random": 0, "pso_50_50": 1, "pso_70_30": 2, "pso_unknown": 3}


@dataclass
class RawBendSample:
    """One geometry evaluated once, independent of target conditioning."""

    z: np.ndarray
    raw_response: np.ndarray
    T: np.ndarray
    source_label: str
    raw_index: int
    valid: bool
    error: Optional[str] = None
    geometry: Optional[Dict[str, Any]] = None


def build_bend_multitarget_dataset(
    config: dict,
    output_path: str | Path,
    *,
    num_random: int,
    include_pso_paths: Optional[Iterable[str | Path]] = None,
    seed: int = 20260505,
    progress_interval: int = 25,
) -> Dict[str, Any]:
    """Build and save a target-conditioned bend dataset."""
    if num_random < 0:
        raise ValueError("num_random cannot be negative.")
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cfg = deepcopy(config)
    z_dim = expected_z_dim(cfg)

    raw_samples: List[RawBendSample] = []
    failures: List[Dict[str, Any]] = []
    rng = np.random.default_rng(seed)
    random_low, random_high = _random_sampling_bounds(cfg)
    for sample_index in range(num_random):
        z = rng.uniform(random_low, random_high, size=z_dim)
        result = run_bend_dreme(z, cfg)
        if result.get("valid", False):
            raw_samples.append(_raw_sample_from_dreme_result(z, result, "random", len(raw_samples)))
        else:
            failures.append({"sample_index": sample_index, "source": "random", "error": result.get("error")})
        if progress_interval > 0 and (sample_index + 1) % progress_interval == 0:
            print(
                f"random progress: evaluated={sample_index + 1}/{num_random}, "
                f"valid={len(raw_samples)}, failed={len(failures)}",
                flush=True,
            )

    for path in include_pso_paths or []:
        loaded = load_pso_samples(path, z_dim=z_dim)
        for sample in loaded:
            sample.raw_index = len(raw_samples)
            raw_samples.append(sample)

    dataset = expand_target_conditioned_rows(raw_samples, cfg, failures=failures, seed=seed)
    save_bend_multitarget_dataset(output_path, dataset)
    summary = {
        "output_path": str(output_path),
        "num_rows": int(dataset["Z"].shape[0]),
        "num_raw_geometries": int(len(raw_samples)),
        "num_random_requested": int(num_random),
        "num_failures": int(len(failures)),
        "targets": sorted(set(dataset["target_labels"].tolist())),
        "sources": sorted(set(dataset["source_labels"].tolist())),
    }
    print(
        "done: "
        f"raw_geometries={summary['num_raw_geometries']}, rows={summary['num_rows']}, "
        f"failures={summary['num_failures']}, output={output_path}",
        flush=True,
    )
    return summary


def load_bend_multitarget_dataset(path: str | Path) -> Dict[str, Any]:
    """Load a bend multitarget dataset into arrays plus parsed metadata."""
    path = Path(path).expanduser().resolve()
    with np.load(path, allow_pickle=False) as data:
        result = {name: np.asarray(data[name]) for name in data.files if name != "metadata_json"}
        result["metadata"] = json.loads(str(data["metadata_json"].item())) if "metadata_json" in data.files else {}
    return result


def load_pso_samples(path: str | Path, *, z_dim: int) -> List[RawBendSample]:
    """Load PSO best-design or all-evaluation NPZ files as raw samples."""
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"PSO file does not exist: {path}")
    with np.load(path, allow_pickle=False) as data:
        files = set(data.files)
        source_label = _infer_pso_source_label(path, data)
        if {"Z", "T"}.issubset(files):
            Z = np.asarray(data["Z"], dtype=float)
            T = np.asarray(data["T"], dtype=float)
            responses = np.asarray(data["responses"], dtype=float) if "responses" in files else None
            power_error = np.asarray(data["power_error"], dtype=float) if "power_error" in files else None
            max_curvature = np.asarray(data["max_curvature"], dtype=float) if "max_curvature" in files else None
            valid = np.asarray(data["valid"], dtype=bool) if "valid" in files else np.ones(Z.shape[0], dtype=bool)
            samples = []
            for idx in range(Z.shape[0]):
                if not valid[idx]:
                    continue
                raw_response = _raw_response_from_arrays(
                    T[idx],
                    responses[idx] if responses is not None else None,
                    power_error=float(power_error[idx]) if power_error is not None else None,
                    max_curvature=float(max_curvature[idx]) if max_curvature is not None else None,
                )
                samples.append(
                    RawBendSample(
                        z=np.asarray(Z[idx], dtype=float).reshape(z_dim),
                        raw_response=raw_response,
                        T=np.asarray(T[idx], dtype=float).reshape(2, 2),
                        source_label=source_label,
                        raw_index=-1,
                        valid=True,
                        geometry=None,
                    )
                )
            return samples

        if {"z", "T"}.issubset(files):
            z = np.asarray(data["z"], dtype=float).reshape(z_dim)
            T = np.asarray(data["T"], dtype=float).reshape(2, 2)
            response = np.asarray(data["response"], dtype=float) if "response" in files else None
            curvature_list = np.asarray(data["curvature_list"], dtype=float) if "curvature_list" in files else np.asarray([])
            raw_response = _raw_response_from_arrays(T, response)
            if curvature_list.size:
                raw_response[12] = float(np.max(np.abs(curvature_list)))
            geometry = {
                "prop_len_list": np.asarray(data["prop_len_list"], dtype=float) if "prop_len_list" in files else np.asarray([]),
                "width_list": np.asarray(data["width_list"], dtype=float) if "width_list" in files else np.asarray([]),
                "curvature_list": curvature_list,
                "centerline_x": np.asarray(data["centerline_x"], dtype=float) if "centerline_x" in files else np.asarray([]),
                "centerline_y": np.asarray(data["centerline_y"], dtype=float) if "centerline_y" in files else np.asarray([]),
            }
            return [
                RawBendSample(
                    z=z,
                    raw_response=raw_response,
                    T=T,
                    source_label=source_label,
                    raw_index=-1,
                    valid=True,
                    geometry=geometry,
                )
            ]

    raise KeyError(f"Unsupported PSO NPZ schema: {path}")


def expand_target_conditioned_rows(
    raw_samples: List[RawBendSample],
    config: dict,
    *,
    failures: Optional[List[Dict[str, Any]]] = None,
    seed: int,
) -> Dict[str, Any]:
    """Duplicate each raw geometry row across supported target conditions."""
    rows = []
    for sample in raw_samples:
        result_like = _result_like_from_raw_sample(sample)
        for spec in TARGET_SPECS:
            fitness = target_fitness_from_result(result_like, spec, config)
            rows.append(
                {
                    "z": sample.z,
                    "raw_response": sample.raw_response,
                    "T": sample.T,
                    "condition": _condition_vector(spec),
                    "target_label": spec["name"],
                    "target_type_id": TARGET_TYPE_IDS[spec["name"]],
                    "source_label": sample.source_label,
                    "source_id": SOURCE_IDS.get(sample.source_label, SOURCE_IDS["pso_unknown"]),
                    "raw_index": sample.raw_index,
                    "fom": float(fitness["fom"]),
                    "target_error": float(fitness["target_error"]),
                    "curvature_penalty": float(fitness["curvature_penalty"]),
                    "power_error": float(fitness["power_error"]),
                }
            )

    if rows:
        Z = np.stack([row["z"] for row in rows]).astype(float)
        R = np.stack([row["raw_response"] for row in rows]).astype(float)
        T = np.stack([row["T"] for row in rows]).astype(float)
        C = np.stack([row["condition"] for row in rows]).astype(float)
    else:
        z_dim = expected_z_dim(config)
        Z = np.empty((0, z_dim), dtype=float)
        R = np.empty((0, len(RAW_RESPONSE_LABELS)), dtype=float)
        T = np.empty((0, 2, 2), dtype=float)
        C = np.empty((0, 3), dtype=float)

    FOM = np.asarray([row["fom"] for row in rows], dtype=float)
    target_error = np.asarray([row["target_error"] for row in rows], dtype=float)
    curvature_penalty = np.asarray([row["curvature_penalty"] for row in rows], dtype=float)
    power_error = np.asarray([row["power_error"] for row in rows], dtype=float)
    source_labels = np.asarray([row["source_label"] for row in rows])
    source_ids = np.asarray([row["source_id"] for row in rows], dtype=int)
    target_labels = np.asarray([row["target_label"] for row in rows])
    target_type_ids = np.asarray([row["target_type_id"] for row in rows], dtype=int)
    raw_indices = np.asarray([row["raw_index"] for row in rows], dtype=int)
    elite = compute_elite_masks(FOM, target_labels)

    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seed": int(seed),
        "random_sampling_bounds": list(_random_sampling_bounds(config)),
        "num_raw_geometries": int(len(raw_samples)),
        "num_rows": int(Z.shape[0]),
        "z_dim": int(Z.shape[1]) if Z.ndim == 2 else 0,
        "raw_response_labels": list(RAW_RESPONSE_LABELS),
        "condition_labels": ["target_type_id", "split_ratio", "target_convention_id"],
        "target_type_ids": TARGET_TYPE_IDS,
        "target_convention_ids": TARGET_CONVENTION_IDS,
        "source_ids": SOURCE_IDS,
        "targets": {spec["name"]: np.asarray(spec["matrix"], dtype=float).tolist() for spec in TARGET_SPECS},
        "failures": failures or [],
        "lumapi_call_made": not bool(config.get("dataset", {}).get("is_testmode", True)),
    }
    return {
        "Z": Z,
        "R": R,
        "T": T,
        "C": C,
        "FOM": FOM,
        "target_error": target_error,
        "curvature_penalty": curvature_penalty,
        "power_error": power_error,
        "source_labels": source_labels,
        "source_ids": source_ids,
        "target_labels": target_labels,
        "target_type_ids": target_type_ids,
        "raw_indices": raw_indices,
        "elite_top_1pct": elite["elite_top_1pct"],
        "elite_top_5pct": elite["elite_top_5pct"],
        "elite_top_10pct": elite["elite_top_10pct"],
        "metadata": metadata,
    }


def compute_elite_masks(FOM: np.ndarray, target_labels: np.ndarray) -> Dict[str, np.ndarray]:
    """Compute top-k-percent elite masks independently for each target."""
    masks = {
        "elite_top_1pct": np.zeros(FOM.shape, dtype=bool),
        "elite_top_5pct": np.zeros(FOM.shape, dtype=bool),
        "elite_top_10pct": np.zeros(FOM.shape, dtype=bool),
    }
    for target in np.unique(target_labels):
        target_indices = np.where(target_labels == target)[0]
        if target_indices.size == 0:
            continue
        finite_indices = target_indices[np.isfinite(FOM[target_indices])]
        if finite_indices.size == 0:
            continue
        ordered = finite_indices[np.argsort(FOM[finite_indices])]
        for pct, name in ((0.01, "elite_top_1pct"), (0.05, "elite_top_5pct"), (0.10, "elite_top_10pct")):
            count = max(1, int(np.ceil(ordered.size * pct)))
            masks[name][ordered[:count]] = True
    return masks


def save_bend_multitarget_dataset(output_path: str | Path, dataset: Dict[str, Any]) -> None:
    """Save a target-conditioned dataset to NPZ."""
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {key: value for key, value in dataset.items() if key != "metadata"}
    np.savez(output_path, **arrays, metadata_json=json.dumps(dataset["metadata"], sort_keys=True))


def _raw_sample_from_dreme_result(z: np.ndarray, result: Dict[str, Any], source_label: str, raw_index: int) -> RawBendSample:
    diagnostics = result.get("diagnostics", {})
    raw_response = np.asarray(
        [
            result["T"][0, 0],
            result["T"][0, 1],
            result["T"][1, 0],
            result["T"][1, 1],
            diagnostics.get("total_te0", np.nan),
            diagnostics.get("total_te1", np.nan),
            diagnostics.get("loss_te0", np.nan),
            diagnostics.get("loss_te1", np.nan),
            diagnostics.get("reflection_te0", np.nan),
            diagnostics.get("reflection_te1", np.nan),
            diagnostics.get("power_error_te0", np.nan),
            diagnostics.get("power_error_te1", np.nan),
            diagnostics.get("max_curvature", np.nan),
        ],
        dtype=float,
    )
    return RawBendSample(
        z=np.asarray(z, dtype=float),
        raw_response=raw_response,
        T=np.asarray(result["T"], dtype=float),
        source_label=source_label,
        raw_index=raw_index,
        valid=True,
        geometry=result.get("metadata", {}).get("geometry"),
    )


def _raw_response_from_arrays(
    T: np.ndarray,
    response: Optional[np.ndarray],
    *,
    power_error: Optional[float] = None,
    max_curvature: Optional[float] = None,
) -> np.ndarray:
    T = np.asarray(T, dtype=float).reshape(2, 2)
    raw = np.full(len(RAW_RESPONSE_LABELS), np.nan, dtype=float)
    raw[:4] = T.reshape(-1)
    if response is not None:
        response = np.asarray(response, dtype=float).reshape(-1)
        if response.size >= 8:
            raw[4] = response[4]
            raw[5] = response[5]
            raw[6] = response[6]
            raw[7] = response[7]
        if response.size >= 4 and not np.isfinite(raw[4]):
            raw[4] = float(np.sum(T[0]))
            raw[5] = float(np.sum(T[1]))
            raw[6] = 1.0 - raw[4]
            raw[7] = 1.0 - raw[5]
    else:
        raw[4] = float(np.sum(T[0]))
        raw[5] = float(np.sum(T[1]))
        raw[6] = 1.0 - raw[4]
        raw[7] = 1.0 - raw[5]
    if power_error is not None and np.isfinite(power_error):
        raw[10] = float(power_error) / 2.0
        raw[11] = float(power_error) / 2.0
    if max_curvature is not None and np.isfinite(max_curvature):
        raw[12] = float(max_curvature)
    return raw


def _result_like_from_raw_sample(sample: RawBendSample) -> Dict[str, Any]:
    raw = sample.raw_response
    diagnostics = {
        "total_te0": float(raw[4]),
        "total_te1": float(raw[5]),
        "loss_te0": float(raw[6]),
        "loss_te1": float(raw[7]),
        "reflection_te0": float(raw[8]),
        "reflection_te1": float(raw[9]),
        "power_error_te0": float(raw[10]) if np.isfinite(raw[10]) else float(abs(1.0 - raw[4])),
        "power_error_te1": float(raw[11]) if np.isfinite(raw[11]) else float(abs(1.0 - raw[5])),
        "max_curvature": float(raw[12]) if np.isfinite(raw[12]) else 0.0,
    }
    return {"T": sample.T, "diagnostics": diagnostics, "foms": {}}


def _condition_vector(target_spec: Dict[str, Any]) -> np.ndarray:
    ratio = float(target_spec["ratio"])
    return np.asarray(
        [
            float(TARGET_TYPE_IDS[target_spec["name"]]),
            ratio,
            float(TARGET_CONVENTION_IDS["TE0TE1_mode_preserving_bias"]),
        ],
        dtype=float,
    )


def _infer_pso_source_label(path: Path, data: Any) -> str:
    name = path.stem.lower()
    metadata = {}
    if "metadata_json" in data.files:
        try:
            metadata = json.loads(str(data["metadata_json"].item()))
        except json.JSONDecodeError:
            metadata = {}
    target_name = str(metadata.get("target_name", name)).lower()
    combined = f"{name} {target_name}"
    if "50_50" in combined or "50:50" in combined:
        return "pso_50_50"
    if "70_30" in combined or "70:30" in combined:
        return "pso_70_30"
    return "pso_unknown"


def config_from_yaml(config_path: str | Path, repo_root: str | Path) -> dict:
    """Load bend config for CLI scripts."""
    return load_bend_config(config_path, repo_root=repo_root)


def _random_sampling_bounds(config: dict) -> tuple[float, float]:
    random_cfg = config.get("random_sampling", {})
    test_cfg = config.get("test", {})
    low = float(random_cfg.get("low", test_cfg.get("random_low", 0.02)))
    high = float(random_cfg.get("high", test_cfg.get("random_high", 0.45)))
    if not (0.0 <= low <= high <= 1.0):
        raise ValueError("Random sampling bounds must satisfy 0 <= low <= high <= 1.")
    return low, high
