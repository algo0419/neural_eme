"""Single-process DReME ML dataset builder."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .geometry_sampler import GeometrySpec, sample_random_z
from .wrapper import RESPONSE_SIZE, run_dreme


def build_dataset(
    num_samples: int,
    output_path: str | Path,
    *,
    spec: GeometrySpec,
    run_config: Optional[dict] = None,
    seed: int = 0,
    max_failures: int = 20,
    progress_interval: int = 50,
) -> Dict[str, Any]:
    """Generate and evaluate a DReME dataset.

    ``num_samples`` is the number of candidate geometries to attempt. Failed
    evaluations are logged separately and are not included in ``Z`` or ``Y``.
    If ``output_path`` already exists, the builder resumes from the
    ``attempted_count`` stored in its metadata.
    """
    if num_samples <= 0:
        raise ValueError("num_samples must be positive.")
    if max_failures < 0:
        raise ValueError("max_failures cannot be negative.")
    if progress_interval <= 0:
        raise ValueError("progress_interval must be positive.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    failures_path = _failures_path(output_path)
    run_config = dict(run_config or {})
    run_config.setdefault("return_raw_smatrix", False)

    state = _load_existing_state(output_path, failures_path, spec)
    attempted_count = int(state["attempted_count"])
    successes: List[np.ndarray] = list(state["Z"])
    responses: List[np.ndarray] = list(state["Y"])
    sample_metadata: List[Dict[str, Any]] = list(state["sample_metadata"])
    failure_count = int(state["failure_count"])

    if attempted_count >= num_samples:
        print(
            f"resume: output already has {attempted_count} attempted samples "
            f"for requested num_samples={num_samples}."
        )
        return _summary(output_path, failures_path, attempted_count, len(successes), failure_count)

    print(
        f"building dataset: start={attempted_count}, target={num_samples}, "
        f"existing_successes={len(successes)}, existing_failures={failure_count}"
    )

    for sample_index in range(attempted_count, num_samples):
        if failure_count >= max_failures:
            print(f"stopping: max_failures={max_failures} reached before sample {sample_index}.")
            break

        z = _sample_for_index(spec, seed, sample_index)
        result = run_dreme(z, run_config)

        if result["valid"]:
            response = np.asarray(result["response"], dtype=float).reshape(-1)
            if response.size != RESPONSE_SIZE:
                raise RuntimeError(f"Expected response size {RESPONSE_SIZE}, got {response.size}.")
            successes.append(z)
            responses.append(response)
            sample_metadata.append(
                {
                    "sample_index": sample_index,
                    "success_index": len(successes) - 1,
                    "run_metadata": _jsonable_metadata(result.get("metadata", {})),
                }
            )
        else:
            failure_count += 1
            _append_failure(
                failures_path,
                {
                    "sample_index": sample_index,
                    "z": z.tolist(),
                    "error": result.get("error"),
                    "metadata": _jsonable_metadata(result.get("metadata", {})),
                    "failure_count": failure_count,
                },
            )

        attempted_count = sample_index + 1
        _save_dataset(output_path, spec, run_config, seed, attempted_count, successes, responses, sample_metadata)

        if attempted_count % progress_interval == 0 or attempted_count == num_samples:
            print(
                f"progress: attempted={attempted_count}/{num_samples}, "
                f"success={len(successes)}, failed={failure_count}"
            )

    return _summary(output_path, failures_path, attempted_count, len(successes), failure_count)


def default_geometry_spec() -> GeometrySpec:
    """Conservative sample-dataset geometry spec."""
    return GeometrySpec(
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


def default_run_config(repo_root: str | Path) -> Dict[str, Any]:
    """Run configuration compatible with the repository sample dataset."""
    repo_root = Path(repo_root).resolve()
    config = default_geometry_spec().to_run_dreme_config()
    config.update(
        {
            "repo_root": str(repo_root),
            "dataset_path": str(repo_root / "sample_datasets" / "Si_rectangular_single_waveguide"),
            "resolution": 500,
            "return_raw_smatrix": False,
            "force_unitary": False,
            "force_passive": False,
            "verbose": False,
        }
    )
    return config


def _sample_for_index(spec: GeometrySpec, seed: int, sample_index: int) -> np.ndarray:
    seed_sequence = np.random.SeedSequence([int(seed), int(sample_index)])
    rng = np.random.default_rng(seed_sequence)
    return sample_random_z(spec, rng)


def _load_existing_state(output_path: Path, failures_path: Path, spec: GeometrySpec) -> Dict[str, Any]:
    if not output_path.exists():
        return {
            "attempted_count": 0,
            "Z": [],
            "Y": [],
            "sample_metadata": [],
            "failure_count": _count_jsonl(failures_path),
        }

    data = np.load(output_path, allow_pickle=False)
    z_array = np.asarray(data["Z"], dtype=float)
    y_array = np.asarray(data["Y"], dtype=float)
    if z_array.ndim != 2 or z_array.shape[1] != spec.z_dim:
        raise ValueError(f"Existing Z has shape {z_array.shape}; expected second dimension {spec.z_dim}.")
    if y_array.ndim != 2 or y_array.shape[1] != RESPONSE_SIZE:
        raise ValueError(f"Existing Y has shape {y_array.shape}; expected second dimension {RESPONSE_SIZE}.")

    metadata = json.loads(str(data["metadata_json"].item()))
    attempted_count = int(metadata.get("attempted_count", z_array.shape[0] + _count_jsonl(failures_path)))
    return {
        "attempted_count": attempted_count,
        "Z": [row for row in z_array],
        "Y": [row for row in y_array],
        "sample_metadata": list(metadata.get("samples", [])),
        "failure_count": _count_jsonl(failures_path),
    }


def _save_dataset(
    output_path: Path,
    spec: GeometrySpec,
    run_config: dict,
    seed: int,
    attempted_count: int,
    successes: List[np.ndarray],
    responses: List[np.ndarray],
    sample_metadata: List[Dict[str, Any]],
) -> None:
    z_array = np.stack(successes).astype(float) if successes else np.empty((0, spec.z_dim), dtype=float)
    y_array = np.stack(responses).astype(float) if responses else np.empty((0, RESPONSE_SIZE), dtype=float)
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "attempted_count": attempted_count,
        "num_success": int(z_array.shape[0]),
        "z_dim": spec.z_dim,
        "y_dim": RESPONSE_SIZE,
        "spec": spec.as_dict(),
        "run_config": _jsonable_metadata(run_config),
        "samples": sample_metadata,
    }
    np.savez(
        output_path,
        Z=z_array,
        Y=y_array,
        metadata_json=json.dumps(metadata, sort_keys=True),
    )


def _append_failure(failures_path: Path, record: Dict[str, Any]) -> None:
    failures_path.parent.mkdir(parents=True, exist_ok=True)
    record = dict(record)
    record["logged_utc"] = datetime.now(timezone.utc).isoformat()
    with failures_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, sort_keys=True) + "\n")


def _failures_path(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.stem}_failures.jsonl")


def _count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as file:
        return sum(1 for line in file if line.strip())


def _jsonable_metadata(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return _jsonable_metadata(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable_metadata(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, complex):
        return {"real": float(value.real), "imag": float(value.imag)}
    if isinstance(value, dict):
        return {str(key): _jsonable_metadata(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable_metadata(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _summary(
    output_path: Path,
    failures_path: Path,
    attempted_count: int,
    success_count: int,
    failure_count: int,
) -> Dict[str, Any]:
    summary = {
        "output_path": str(output_path),
        "failures_path": str(failures_path),
        "attempted_count": attempted_count,
        "success_count": success_count,
        "failure_count": failure_count,
    }
    print(
        "done: "
        f"attempted={attempted_count}, success={success_count}, failed={failure_count}, "
        f"output={output_path}"
    )
    return summary
