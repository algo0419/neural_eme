"""Run PSO for paper-style bend mode-splitting targets."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Bend splitting YAML config.")
    parser.add_argument("--target", choices=["split", "preserve"], default="split", help="Optimization target family.")
    parser.add_argument("--ratio", type=float, default=0.5, help="Split ratio for --target split.")

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--quick", action="store_true", help="10 particles, 5 iterations.")
    mode.add_argument("--medium", action="store_true", help="30 particles, 20 iterations.")
    mode.add_argument("--paper-mode", action="store_true", help="100 particles, 80 iterations.")

    parser.add_argument("--output-root", type=Path, default=Path("outputs") / "bend_pso")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--seed", type=int, default=20260505)
    parser.add_argument("--inertia", type=float, default=0.72)
    parser.add_argument("--cognitive", type=float, default=1.49)
    parser.add_argument("--social", type=float, default=1.49)
    parser.add_argument("--invalid-penalty", type=float, default=1.0e6)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))
    sys.path.insert(0, str(repo_root))

    from dreme_inverse.bend_problem import load_bend_config
    from dreme_inverse.bend_pso import BendPSOConfig, make_target_spec, run_bend_pso

    config_path = args.config if args.config.is_absolute() else repo_root / args.config
    config = load_bend_config(config_path, repo_root=repo_root)
    target_spec = make_target_spec(args.target, args.ratio)
    preset = _preset(args)
    pso_config = BendPSOConfig(
        num_particles=preset["num_particles"],
        num_iterations=preset["num_iterations"],
        inertia=args.inertia,
        cognitive=args.cognitive,
        social=args.social,
        seed=args.seed,
        invalid_penalty=args.invalid_penalty,
    )

    output_root = args.output_root if args.output_root.is_absolute() else repo_root / args.output_root
    output_dir = output_root / target_spec["name"]
    data_dir = args.data_dir if args.data_dir.is_absolute() else repo_root / args.data_dir
    data_output_path = data_dir / f"bend_pso_{target_spec['name']}.npz"

    print(f"config: {config_path}")
    print(f"target: {target_spec['name']}")
    print(f"mode: {preset['name']}")
    print(f"particles: {pso_config.num_particles}")
    print(f"iterations: {pso_config.num_iterations}")
    print(f"output_dir: {output_dir}")
    print(f"data_output: {data_output_path}")

    summary = run_bend_pso(
        config,
        output_dir,
        target=args.target,
        ratio=args.ratio,
        pso_config=pso_config,
        data_output_path=data_output_path,
    )

    print("")
    print(f"best_fom: {summary['best_fom']:.10g}")
    print(f"first_best_fom: {summary['first_best_fom']:.10g}")
    print(f"improvement: {summary['improvement']:.10g}")
    print(f"improved: {summary['improvement'] > 0.0}")
    print(f"best_max_curvature: {summary['best_max_curvature']:.10g}")
    print(f"best_power_error: {summary['best_power_error']:.10g}")
    print(f"lumapi_or_lumerical_call_made: {summary['lumapi_call_made']}")
    print("T_matrix:")
    for row in summary["best_T"]:
        print("  " + " ".join(f"{value:.10g}" for value in row))
    print("")
    print(f"summary: {summary['artifacts']['summary']}")
    return 0


def _preset(args: argparse.Namespace) -> dict:
    if args.paper_mode:
        return {"name": "paper-mode", "num_particles": 100, "num_iterations": 80}
    if args.medium:
        return {"name": "medium", "num_particles": 30, "num_iterations": 20}
    return {"name": "quick", "num_particles": 10, "num_iterations": 5}


if __name__ == "__main__":
    raise SystemExit(main())
