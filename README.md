# Neural EME / DReME Inverse Design Sandbox

This repository builds on `thdwotjd/dataset-based-eme`, a dataset-based Eigenmode Expansion (EME) framework for integrated photonics. The added code provides a small, reusable DReME-style wrapper for generating machine-learning responses from geometry parameter vectors.

The core upstream package remains in `em_simulation/`. The new wrapper code lives under `src/dreme_inverse/`.

## What Is Added

- `src/dreme_inverse/wrapper.py`: reusable `run_dreme(z, config) -> dict` API.
- `scripts/smoke_test_dreme.py`: end-to-end smoke test using the sample dataset.
- `scripts/test_run_dreme.py`: command-line test for one sampled design vector.
- `outputs/smoke_test_summary.txt`: current smoke-test summary.
- `outputs/smoke_test_result.npz`: saved smoke-test arrays.

## Environment

The upstream README lists Python 3.9-3.11, but the current dependency constraints resolved successfully here with Python 3.12. Python 3.14 did not work with the pinned `pillow==10.3.0`.

Recommended local setup:

```powershell
python -m venv .venv312
.\.venv312\Scripts\python.exe -m pip install --upgrade pip
.\.venv312\Scripts\python.exe -m pip install -r requirements.txt
```

If cloning fresh, make sure Git LFS data is present:

```powershell
git lfs install
git lfs pull
```

The sample dataset requires real pickle files under:

```text
sample_datasets/Si_rectangular_single_waveguide/
```

## Smoke Test

Run the existing end-to-end check:

```powershell
.\.venv312\Scripts\python.exe scripts\smoke_test_dreme.py
```

Expected outputs:

```text
outputs/smoke_test_result.npz
outputs/smoke_test_summary.txt
```

The script follows:

```text
DataUpdater -> LinearTaper -> EME -> Runner
```

It prints the S-matrix shape, output amplitudes, output intensities, and mode labels.

## DReME Wrapper

The ML-facing API is:

```python
from dreme_inverse.wrapper import run_dreme

result = run_dreme(z, config)
```

Return format:

```python
{
    "response": np.ndarray,
    "raw_smatrix": np.ndarray | None,
    "output_amplitudes": np.ndarray | None,
    "metadata": dict,
    "valid": bool,
    "error": str | None,
}
```

For the sample dataset, `dataset_info.py` exposes both `top_width` and `curvature`, so the wrapper uses the real upstream `SingleCustomBend` API. If a dataset only exposes width, it falls back to `CustomTaper`.

The current response vector is:

```text
[
  first output guided-mode intensity,
  second output guided-mode intensity, or 0 if unavailable,
  total transmitted guided-mode intensity,
  loss = 1 - total transmitted guided-mode intensity,
  relative phase between the first two guided output modes, or 0 if unavailable
]
```

Run one wrapper test:

```powershell
.\.venv312\Scripts\python.exe scripts\test_run_dreme.py
```

## Example Config

```python
config = {
    "dataset_path": "sample_datasets/Si_rectangular_single_waveguide",
    "num_sections": 5,
    "length": 10.0e-6,
    "width_min": 1.2e-6,
    "width_max": 2.0e-6,
    "curvature_min": 0.0,
    "curvature_max": 1.0e4,
    "resolution": 500,
    "return_raw_smatrix": False,
}
```

For `SingleCustomBend`, `z` length must be `2 * num_sections`: first half controls widths and second half controls curvatures. Values are clipped to `[0, 1]` and mapped into physical ranges.

## Notes

- The Lumerical API warning during tests is expected when using `DataUpdater(..., is_testmode=True)` with precomputed data.
- Ray is initialized by the EME matrix utilities. The wrapper pre-initializes Ray with exclusions so local virtual environments and datasets are not packaged into worker runtime archives.
- The upstream sample dataset is large and Git LFS-backed, especially `overlap.pkl`.

## Upstream Citation

If using the dataset-based EME framework or datasets in research, cite:

Song, J. and Sohn, Y.-I. "Ultra-fast and accurate multimode waveguide design based on a dataset-based eigenmode expansion method." Optics Express 33, 46815-46827 (2025). https://doi.org/10.1364/OE.567425

```bibtex
@article{10.1364/oe.567425,
  author  = {Song, Jaesung and Sohn, Young-Ik},
  title   = {Ultra-fast and accurate multimode waveguide design based on a dataset-based eigenmode expansion method},
  journal = {Optics Express},
  volume  = {33},
  number  = {22},
  pages   = {46815--46827},
  year    = {2025},
  doi     = {10.1364/OE.567425}
}
```

## License

The upstream project is licensed under the MIT License.
