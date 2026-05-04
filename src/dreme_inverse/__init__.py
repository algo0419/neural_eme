"""Reusable wrappers for dataset-based DReME inverse-design experiments."""

from .dataset_builder import build_dataset, default_geometry_spec, default_run_config
from .geometry_sampler import GeometrySpec, is_valid_z, plot_geometry_from_z, sample_random_z, z_to_geometry_parameters
from .wrapper import run_dreme

__all__ = [
    "GeometrySpec",
    "build_dataset",
    "default_geometry_spec",
    "default_run_config",
    "is_valid_z",
    "plot_geometry_from_z",
    "run_dreme",
    "sample_random_z",
    "z_to_geometry_parameters",
]
