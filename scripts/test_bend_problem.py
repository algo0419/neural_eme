"""Smoke-test the bend mode-splitting DReME wrapper."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs") / "bend_splitting.yaml",
        help="YAML config for the bend splitting problem.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs") / "bend_problem",
        help="Directory for test plots and summary.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))
    sys.path.insert(0, str(repo_root))

    from dreme_inverse.bend_problem import expected_z_dim, load_bend_config, run_bend_dreme

    config_path = args.config if args.config.is_absolute() else repo_root / args.config
    output_dir = args.output_dir if args.output_dir.is_absolute() else repo_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_bend_config(config_path, repo_root=repo_root)
    z_dim = expected_z_dim(config)
    test_cfg = config.get("test", {})

    simple_z = np.ones(z_dim, dtype=float) * float(test_cfg.get("simple_curvature_control_value", 0.16))
    rng = np.random.default_rng(int(test_cfg.get("seed", 20260505)))
    random_z = rng.uniform(
        float(test_cfg.get("random_low", 0.02)),
        float(test_cfg.get("random_high", 0.45)),
        size=z_dim,
    )

    cases = {
        "simple": run_bend_dreme(simple_z, config),
        "random": run_bend_dreme(random_z, config),
    }

    summary_lines = [f"config: {config_path}", f"output_dir: {output_dir}", ""]
    any_lumapi_call = False
    for name, result in cases.items():
        summary_lines.extend(_case_summary(name, result))
        summary_lines.append("")
        if result["valid"]:
            any_lumapi_call = any_lumapi_call or bool(result["metadata"].get("lumapi_call_made", False))
            _save_case_plots(name, result, output_dir)

    summary_lines.append(f"any_lumapi_or_lumerical_call_made: {any_lumapi_call}")
    summary_path = output_dir / "test_bend_problem_summary.txt"
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print("\n".join(summary_lines))
    print(f"summary: {summary_path}")
    return 0 if all(result["valid"] for result in cases.values()) else 1


def _case_summary(name: str, result: Dict[str, Any]) -> list[str]:
    lines = [f"[{name}]"]
    if not result["valid"]:
        lines.append(f"valid: false")
        lines.append(f"error: {result['error']}")
        return lines

    metadata = result["metadata"]
    diagnostics = result["diagnostics"]
    lines.append("valid: true")
    lines.append(f"dataset_parameter_names: {metadata['dataset_parameter_names']}")
    lines.append(f"dataset_parameter_ranges: {metadata['dataset_parameter_ranges']}")
    lines.append(f"mode_count: {metadata['mode_count']}")
    lines.append("T_matrix:")
    lines.extend(_format_matrix(np.asarray(result["T"])))
    lines.append(f"FOM_5050: {float(result['fom_5050']):.8g}")
    lines.append(f"FOM_7030: {float(result['fom_7030']):.8g}")
    lines.append(f"total_te0: {float(diagnostics['total_te0']):.8g}")
    lines.append(f"total_te1: {float(diagnostics['total_te1']):.8g}")
    lines.append(f"loss_te0: {float(diagnostics['loss_te0']):.8g}")
    lines.append(f"loss_te1: {float(diagnostics['loss_te1']):.8g}")
    lines.append(f"reflection_te0: {float(diagnostics['reflection_te0']):.8g}")
    lines.append(f"reflection_te1: {float(diagnostics['reflection_te1']):.8g}")
    lines.append(f"max_curvature: {float(diagnostics['max_curvature']):.8g}")
    lines.append(f"lumapi_or_lumerical_call_made: {metadata.get('lumapi_call_made', False)}")
    return lines


def _format_matrix(matrix: np.ndarray) -> list[str]:
    return ["  " + " ".join(f"{value: .8f}" for value in row) for row in matrix]


def _save_case_plots(name: str, result: Dict[str, Any], output_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    geom = result["metadata"]["geometry"]
    prop = np.asarray(geom["prop_len_list"], dtype=float)
    widths = np.asarray(geom["width_list"], dtype=float)
    curvatures = np.asarray(geom["curvature_list"], dtype=float)
    x = np.asarray(geom["centerline_x"], dtype=float)
    y = np.asarray(geom["centerline_y"], dtype=float)
    T = np.asarray(result["T"], dtype=float)

    fig, axis = plt.subplots(figsize=(6.2, 3.8))
    axis.plot(prop * 1e6, widths * 1e6, color="tab:blue", linewidth=1.8)
    axis.set_xlabel("propagation length (um)")
    axis.set_ylabel("top width (um)")
    axis.set_title(f"{name}: width profile")
    axis.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / f"{name}_width_profile.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(6.2, 3.8))
    axis.plot(prop * 1e6, curvatures, color="tab:purple", linewidth=1.8)
    axis.set_xlabel("propagation length (um)")
    axis.set_ylabel("curvature (1/m)")
    axis.set_title(f"{name}: curvature profile")
    axis.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / f"{name}_curvature_profile.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(5.0, 4.5))
    axis.plot(x * 1e6, y * 1e6, color="tab:green", linewidth=1.8)
    axis.scatter(x[0] * 1e6, y[0] * 1e6, color="black", s=20, label="in")
    axis.scatter(x[-1] * 1e6, y[-1] * 1e6, color="tab:red", s=20, label="out")
    axis.set_xlabel("x (um)")
    axis.set_ylabel("y (um)")
    axis.set_title(f"{name}: top-view bend layout")
    axis.set_aspect("equal", adjustable="box")
    axis.grid(True, alpha=0.25)
    axis.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_dir / f"{name}_top_view_layout.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.6))
    image = axes[0].imshow(T, vmin=0.0, vmax=max(1.0, float(np.nanmax(T))), cmap="viridis")
    axes[0].set_xticks([0, 1], labels=["out0", "out1"])
    axes[0].set_yticks([0, 1], labels=["TE0 in", "TE1 in"])
    axes[0].set_title("T matrix")
    for row in range(2):
        for col in range(2):
            axes[0].text(col, row, f"{T[row, col]:.3f}", ha="center", va="center", color="white")
    fig.colorbar(image, ax=axes[0], fraction=0.046, pad=0.04)

    axes[1].bar(["T00", "T01", "T10", "T11"], T.reshape(-1), color=["tab:blue", "tab:orange", "tab:green", "tab:red"])
    axes[1].set_ylim(0.0, max(1.0, float(np.nanmax(T)) * 1.1))
    axes[1].set_ylabel("intensity")
    axes[1].set_title("first-two-mode powers")
    axes[1].grid(True, axis="y", alpha=0.25)
    fig.suptitle(f"{name}: output intensity matrix")
    fig.tight_layout()
    fig.savefig(output_dir / f"{name}_output_intensity_matrix.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
