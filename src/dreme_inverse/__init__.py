"""Reusable wrappers for dataset-based DReME inverse-design experiments."""

from .analysis import analyze_dataset, identify_suspicious_values, load_dataset_arrays
from .bend_dataset import build_bend_multitarget_dataset, load_bend_multitarget_dataset
from .bend_problem import run_bend_dreme
from .bend_pso import BendPSOConfig, run_bend_pso
from .dataset_builder import build_dataset, default_geometry_spec, default_run_config
from .geometry_sampler import GeometrySpec, is_valid_z, plot_geometry_from_z, sample_random_z, z_to_geometry_parameters
from .wrapper import run_dreme

__all__ = [
    "GeometrySpec",
    "BendPSOConfig",
    "analyze_dataset",
    "build_dataset",
    "build_bend_multitarget_dataset",
    "default_geometry_spec",
    "default_run_config",
    "identify_suspicious_values",
    "is_valid_z",
    "load_dataset_arrays",
    "load_bend_multitarget_dataset",
    "plot_geometry_from_z",
    "run_bend_dreme",
    "run_bend_pso",
    "run_dreme",
    "sample_random_z",
    "z_to_geometry_parameters",
]
