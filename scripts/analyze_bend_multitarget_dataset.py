"""Analyze a bend multitarget dataset and save diagnostic plots."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="Bend multitarget dataset NPZ.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs") / "bend_splitting.yaml",
        help="Bend config used to reconstruct geometry profiles.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs") / "bend_multitarget_dataset_analysis",
        help="Directory for analysis plots.",
    )
    parser.add_argument("--top-k", type=int, default=3, help="Number of top candidates to plot per target.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))
    sys.path.insert(0, str(repo_root))

    from dreme_inverse.bend_dataset import RAW_RESPONSE_LABELS, load_bend_multitarget_dataset
    from dreme_inverse.bend_problem import load_bend_config, z_to_bend_geometry_parameters

    dataset_path = args.dataset if args.dataset.is_absolute() else repo_root / args.dataset
    config_path = args.config if args.config.is_absolute() else repo_root / args.config
    output_dir = args.output_dir if args.output_dir.is_absolute() else repo_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_bend_multitarget_dataset(dataset_path)
    config = load_bend_config(config_path, repo_root=repo_root)
    metadata = data["metadata"]
    labels = list(metadata.get("raw_response_labels", RAW_RESPONSE_LABELS))
    summary = analyze(data, config, labels, output_dir, top_k=args.top_k)
    summary_path = output_dir / "summary.txt"
    summary_path.write_text(_summary_text(dataset_path, output_dir, summary), encoding="utf-8")

    print(_summary_text(dataset_path, output_dir, summary))
    print(f"summary: {summary_path}")
    return 0


def analyze(data: dict, config: dict, labels: list[str], output_dir: Path, *, top_k: int) -> dict:
    target_labels = data["target_labels"].astype(str)
    source_labels = data["source_labels"].astype(str)
    FOM = np.asarray(data["FOM"], dtype=float)
    R = np.asarray(data["R"], dtype=float)
    T = np.asarray(data["T"], dtype=float)
    Z = np.asarray(data["Z"], dtype=float)
    raw_indices = np.asarray(data["raw_indices"], dtype=int)

    _plot_fom_histograms(FOM, target_labels, output_dir / "fom_histograms_per_target.png")
    _plot_source_fom_distributions(FOM, target_labels, source_labels, output_dir / "random_vs_pso_fom_distributions.png")
    _plot_split_error_histograms(T, target_labels, output_dir / "split_ratio_error_histograms.png")
    _plot_curvature_distribution(R[:, labels.index("max_curvature")], target_labels, output_dir / "curvature_distribution.png")
    _plot_source_breakdown(source_labels, output_dir / "source_breakdown.png")

    best_rows = []
    for target in np.unique(target_labels):
        indices = np.where(target_labels == target)[0]
        ordered = indices[np.argsort(FOM[indices])]
        for rank, row_idx in enumerate(ordered[: max(1, top_k)], start=1):
            best_rows.append(int(row_idx))
            _plot_t_heatmap(T[row_idx], output_dir / f"{target}_rank{rank}_T_heatmap.png", f"{target} rank {rank}")
            if rank == 1:
                _plot_best_geometry(
                    Z[row_idx],
                    config,
                    output_dir,
                    prefix=f"{target}_best",
                    title=f"{target} best geometry",
                )
    summary = {
        "num_rows": int(Z.shape[0]),
        "num_raw_geometries": int(np.unique(raw_indices).size) if raw_indices.size else 0,
        "targets": {},
        "sources": {},
        "best_rows": best_rows,
    }
    for target in np.unique(target_labels):
        mask = target_labels == target
        summary["targets"][target] = {
            "rows": int(np.count_nonzero(mask)),
            "best_fom": float(np.nanmin(FOM[mask])),
            "median_fom": float(np.nanmedian(FOM[mask])),
        }
    for source in np.unique(source_labels):
        summary["sources"][source] = int(np.count_nonzero(source_labels == source))
    return summary


def _plot_fom_histograms(FOM: np.ndarray, target_labels: np.ndarray, output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    targets = np.unique(target_labels)
    fig, axes = plt.subplots(1, len(targets), figsize=(5.0 * len(targets), 3.8), squeeze=False)
    for axis, target in zip(axes.ravel(), targets):
        values = FOM[target_labels == target]
        axis.hist(values[np.isfinite(values)], bins=40, color="tab:blue", alpha=0.82, edgecolor="white")
        axis.set_title(f"FOM: {target}")
        axis.set_xlabel("FOM")
        axis.set_ylabel("count")
        axis.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_source_fom_distributions(FOM: np.ndarray, target_labels: np.ndarray, source_labels: np.ndarray, output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    targets = np.unique(target_labels)
    sources = np.unique(source_labels)
    fig, axes = plt.subplots(1, len(targets), figsize=(5.2 * len(targets), 3.8), squeeze=False)
    for axis, target in zip(axes.ravel(), targets):
        values = [FOM[(target_labels == target) & (source_labels == source)] for source in sources]
        axis.boxplot(values, tick_labels=sources, showfliers=False)
        axis.set_title(f"FOM by source: {target}")
        axis.set_ylabel("FOM")
        axis.tick_params(axis="x", rotation=25)
        axis.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_split_error_histograms(T: np.ndarray, target_labels: np.ndarray, output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    targets = np.unique(target_labels)
    fig, axes = plt.subplots(1, len(targets), figsize=(5.0 * len(targets), 3.8), squeeze=False)
    for axis, target in zip(axes.ravel(), targets):
        ratio = 0.5 if "50_50" in target else 0.7
        target_matrix = np.asarray([[ratio, 1.0 - ratio], [1.0 - ratio, ratio]], dtype=float)
        errors = np.sum(np.abs(T[target_labels == target] - target_matrix), axis=(1, 2))
        axis.hist(errors[np.isfinite(errors)], bins=40, color="tab:orange", alpha=0.82, edgecolor="white")
        axis.set_title(f"split error: {target}")
        axis.set_xlabel("sum abs(T - target)")
        axis.set_ylabel("count")
        axis.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_curvature_distribution(max_curvature: np.ndarray, target_labels: np.ndarray, output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(6.2, 3.8))
    for target in np.unique(target_labels):
        values = max_curvature[target_labels == target]
        axis.hist(values[np.isfinite(values)], bins=40, alpha=0.55, label=target)
    axis.set_xlabel("max curvature (1/m)")
    axis.set_ylabel("count")
    axis.set_title("Curvature Distribution")
    axis.legend()
    axis.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_source_breakdown(source_labels: np.ndarray, output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sources, counts = np.unique(source_labels, return_counts=True)
    fig, axis = plt.subplots(figsize=(5.5, 3.8))
    axis.bar(sources, counts, color="tab:green")
    axis.set_ylabel("rows")
    axis.set_title("Source Breakdown")
    axis.tick_params(axis="x", rotation=25)
    axis.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_t_heatmap(T: np.ndarray, output_path: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(4.4, 3.8))
    image = axis.imshow(T, vmin=0.0, vmax=max(1.0, float(np.nanmax(T))), cmap="viridis")
    axis.set_xticks([0, 1], labels=["out0", "out1"])
    axis.set_yticks([0, 1], labels=["TE0 in", "TE1 in"])
    axis.set_title(title)
    for row in range(2):
        for col in range(2):
            axis.text(col, row, f"{T[row, col]:.3f}", ha="center", va="center", color="white")
    fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_best_geometry(z: np.ndarray, config: dict, output_dir: Path, *, prefix: str, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from dreme_inverse.bend_problem import z_to_bend_geometry_parameters

    params = z_to_bend_geometry_parameters(z, config)
    prop = np.asarray(params["prop_len_list"], dtype=float)
    widths = np.asarray(params["width_list"], dtype=float)
    curvatures = np.asarray(params["curvature_list"], dtype=float)
    x = np.asarray(params["centerline_x"], dtype=float)
    y = np.asarray(params["centerline_y"], dtype=float)

    fig, axis = plt.subplots(figsize=(6.2, 3.8))
    axis.plot(prop * 1e6, widths * 1e6, color="tab:blue", linewidth=1.8)
    axis.set_xlabel("propagation length (um)")
    axis.set_ylabel("top width (um)")
    axis.set_title(f"{title}: width")
    axis.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / f"{prefix}_width_profile.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(6.2, 3.8))
    axis.plot(prop * 1e6, curvatures, color="tab:purple", linewidth=1.8)
    axis.set_xlabel("propagation length (um)")
    axis.set_ylabel("curvature (1/m)")
    axis.set_title(f"{title}: curvature")
    axis.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / f"{prefix}_curvature_profile.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(5.0, 4.5))
    axis.plot(x * 1e6, y * 1e6, color="tab:green", linewidth=1.8)
    axis.scatter(x[0] * 1e6, y[0] * 1e6, color="black", s=20, label="in")
    axis.scatter(x[-1] * 1e6, y[-1] * 1e6, color="tab:red", s=20, label="out")
    axis.set_xlabel("x (um)")
    axis.set_ylabel("y (um)")
    axis.set_title(f"{title}: layout")
    axis.set_aspect("equal", adjustable="box")
    axis.grid(True, alpha=0.25)
    axis.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_dir / f"{prefix}_bend_layout.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def _summary_text(dataset_path: Path, output_dir: Path, summary: dict) -> str:
    lines = [
        f"dataset: {dataset_path}",
        f"output_dir: {output_dir}",
        f"num_rows: {summary['num_rows']}",
        f"num_raw_geometries: {summary['num_raw_geometries']}",
        "",
        "targets:",
    ]
    for target, values in summary["targets"].items():
        lines.append(
            f"  {target}: rows={values['rows']}, best_fom={values['best_fom']:.10g}, "
            f"median_fom={values['median_fom']:.10g}"
        )
    lines.append("")
    lines.append("sources:")
    for source, count in summary["sources"].items():
        lines.append(f"  {source}: {count}")
    lines.append("")
    lines.append("plots saved in output_dir")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
