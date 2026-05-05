# Bend Mode-Splitting Feasibility

## Verdict

Feasible with current precomputed dataset: **partial**.

The current `sample_datasets/Si_rectangular_single_waveguide` dataset contains the modal data needed to evaluate paper-style single-waveguide bend mode splitting with DReME/EME, including widths around 2 um and curvature. The missing pieces are not new modal data for the current parameter ranges; they are implementation pieces for the bend optimization pipeline: a paper-style bend geometry parameterizer, explicit 50:50 and 70:30 figures of merit, and validation scripts/reports.

New Lumerical/lumapi dataset acquisition is **not required** if the optimized geometries stay inside the existing single-waveguide parameter space:

- `top_width` in `[1e-6, 3e-6]` m
- `curvature` in `[0, 330000]` 1/m
- same material/cross-section/wavelength as the sample dataset

New Lumerical/lumapi acquisition would be required if the paper target needs parameters outside this dataset, such as widths outside 1-3 um, curvatures outside 0-330000 1/m, a different wavelength/material/cross-section, a multi-waveguide/gap parameter space, or independent rotation-angle modal data.

## Direct Answers

1. **Does the current sample dataset include curvature as a parameter?**

   Yes. `sample_datasets/Si_rectangular_single_waveguide/dataset_info.py` defines:

   - `parameter_names = ["top_width", "curvature"]`
   - `curvature = np.round(np.linspace(0, 330000, 166), 0)`

2. **Does it include width around 2 um?**

   Yes. `top_width = np.round(np.linspace(1e-6, 3e-6, 101), 9)`, so 2 um is on the grid.

3. **Does it support `SingleCustomBend` or `SingleBezierCurve` with width(s) and curvature(s)?**

   Yes for `SingleCustomBend`; partially for `SingleBezierCurve`.

   Recommended:

   - `sim.SingleCustomBend(dataset, prop_len_list, width_list, curvature_list, input_angle=0, resolution=...)`

   This is the best match for the paper-style optimization because it directly accepts sampled propagation lengths, width profile, and curvature profile. The existing `examples/optimization_examples/Si_5050_splitting_bend.ipynb` uses exactly this path.

   `SingleBezierCurve` exists, but it accepts a constant `top_width` plus Bezier centerline control points. It derives curvature from the centerline and does not directly support a varying width list. It is therefore not the right first geometry class for width-plus-curvature optimization. It also uses `np.math.factorial`, which is incompatible with the currently installed NumPy 2.x environment unless fixed.

4. **Can we compute the required mode-splitting response using the existing EME/Runner/S-matrix outputs?**

   Yes. The response can be computed from `Runner.propagate_lumped_smatrix(input_amplitudes)` or directly from the lumped scattering matrix after it is composed.

   For first two tracked forward output modes:

   - `a0 = output_amplitudes[0]`
   - `a1 = output_amplitudes[1]`
   - `T0 = abs(a0)**2`
   - `T1 = abs(a1)**2`
   - `T_total_guided = sum(abs(output_amplitudes[:mode_count][guided_forward_mask])**2)`
   - `loss = 1 - T_total_guided`
   - optional phase: `phase_01 = angle(a1 * conj(a0))`

   The optimization notebook's 50:50 FOM uses:

   - TE0 input: `|out[0]|^2` and `|out[1]|^2`
   - TE1 input: `|out[0]|^2` and `|out[1]|^2`
   - target all four values to 0.5

5. **Can we extract responses for both TE0 and TE1 input modes?**

   Yes. The runner accepts one input-amplitude vector over the non-radiation input modes. For the sample bend check, the input guided count was 10.

   Example launch vectors:

   - TE0-like input: `[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]`
   - TE1-like input: `[0, 1, 0, 0, 0, 0, 0, 0, 0, 0]`

   The output vector has length `2 * mode_count`: forward modes first, backward modes second. In the local check, `mode_count = 10`, so output shape was `(20,)`.

6. **Can one DReME simulation return enough S-matrix information to compute both 50:50 and 70:30 FOMs?**

   Yes. One geometry evaluation builds one EME S-matrix. That same S-matrix is enough to evaluate multiple input vectors and multiple target FOMs.

   Practical options:

   - Call `runner.propagate_lumped_smatrix(e0)` and `runner.propagate_lumped_smatrix(e1)` after one `sim.EME(geometry).calc_Smatrix()`.
   - Or compose/use the lumped S-matrix and extract the first two input columns.

   For a target matrix over first two forward modes:

   - 50:50 target can be `[[0.5, 0.5], [0.5, 0.5]]`.
   - 70:30 target should be explicitly defined by convention, for example `[[0.7, 0.3], [0.3, 0.7]]` for mode-preserving bias, or the crossed equivalent if that is the paper convention.

   No new EME simulation is needed to score both target matrices for the same geometry.

7. **Does any step require new Lumerical/lumapi dataset acquisition?**

   No, not for DReME/EME optimization inside the current sample dataset grid. `sim.DataUpdater(..., is_testmode=True)` loads the precomputed `neff.pkl`, `TE_pol.pkl`, and `overlap.pkl` data.

   New lumapi acquisition is needed only if the target geometry leaves the current dataset parameter space or if final validation requires fresh full-field/FDE/FDTD data beyond the cached modal dataset.

## Recommended Geometry Class

Use `SingleCustomBend`.

Reason:

- It directly matches the paper-style optimization notebook.
- It supports separate sampled width and curvature profiles.
- It works with the current dataset's `top_width` and `curvature` parameters.
- It avoids `SingleBezierCurve` limitations around constant width and NumPy 2.x `np.math` compatibility.

Recommended construction pattern:

```python
dataset = sim.DataUpdater(dataset_path, is_testmode=True)
geometry = sim.SingleCustomBend(
    dataset,
    prop_len_list,
    width_list,
    curvature_list,
    input_angle=0.0,
    resolution=5000,
    verbose=False,
)
eme = sim.EME(geometry, force_unitary=True)
runner = sim.Runner(eme)
```

The existing 50:50 notebook uses `force_unitary=1`. For final validation, decide whether the validation target should use raw DReME loss/passivity or the unitary-projected S-matrix.

## Exact Response Quantities To Extract

For each geometry, compute outputs for TE0 and TE1 inputs:

```python
out_te0 = runner.propagate_lumped_smatrix([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
out_te1 = runner.propagate_lumped_smatrix([0, 1, 0, 0, 0, 0, 0, 0, 0, 0])
```

Core intensity matrix:

```python
T00 = abs(out_te0[0])**2  # TE0 input -> output mode 0
T01 = abs(out_te0[1])**2  # TE0 input -> output mode 1
T10 = abs(out_te1[0])**2  # TE1 input -> output mode 0
T11 = abs(out_te1[1])**2  # TE1 input -> output mode 1
T = [[T00, T01], [T10, T11]]
```

Additional validation quantities:

- `total_te0 = sum(abs(out_te0[:mode_count][guided_forward_mask])**2)`
- `total_te1 = sum(abs(out_te1[:mode_count][guided_forward_mask])**2)`
- `loss_te0 = 1 - total_te0`
- `loss_te1 = 1 - total_te1`
- `reflection_te0 = sum(abs(out_te0[mode_count:])**2)` over relevant backward guided modes
- `reflection_te1 = sum(abs(out_te1[mode_count:])**2)` over relevant backward guided modes
- `phase_te0_01 = angle(out_te0[1] * conj(out_te0[0]))`
- `phase_te1_01 = angle(out_te1[1] * conj(out_te1[0]))`

Candidate scalar FOMs:

```python
fom_5050 = sum(abs(T - [[0.5, 0.5], [0.5, 0.5]]))
fom_7030 = sum(abs(T - target_7030))
```

where `target_7030` must be fixed to the paper convention before optimization.

## Exact Code Entry Points

Dataset metadata:

- `sample_datasets/Si_rectangular_single_waveguide/dataset_info.py`
- `em_simulation/data_updater/data_updater.py::DataUpdater`

Geometry:

- `em_simulation/__init__.py` exports `SingleCustomBend`, `SingleBezierCurve`, `EME`, and `Runner`
- `em_simulation/geometry/single_waveguide/single_tapered_bend/single_custom_bend.py::SingleCustomBend`
- `em_simulation/geometry/single_waveguide/single_bezier.py::SingleBezierCurve`
- `em_simulation/geometry/single_waveguide/single_waveguide.py::SingleWaveguide.calc_simulation_parameters`

Propagation and S-matrix:

- `em_simulation/propagator/eme.py::EME`
- `em_simulation/propagator/single_propagator/single_eme.py::SingleEME.calc_Smatrix`
- `em_simulation/runner/runner.py::Runner`
- `em_simulation/runner/single_runner.py::SingleRunner.propagate_lumped_smatrix`
- `em_simulation/runner/single_runner.py::SingleRunner._convert_input`

Existing paper-style example:

- `examples/optimization_examples/Si_5050_splitting_bend.ipynb`
  - `generate_smooth_bend(...)`
  - `evaluate(individual)`
  - uses `sim.SingleCustomBend(...)`
  - launches TE0-like and TE1-like inputs

Current DReME wrapper code to extend without deleting taper support:

- `src/dreme_inverse/wrapper.py::run_dreme`
- `src/dreme_inverse/wrapper.py::_build_geometry`
- `src/dreme_inverse/geometry_sampler.py::GeometrySpec`
- `src/dreme_inverse/dataset_builder.py::build_dataset`

## Local Verification Snapshot

A direct local smoke check with a simple `SingleCustomBend` at 2 um width and modest curvature produced:

- dataset parameters: `["top_width", "curvature"]`
- width range: `1e-6` to `3e-6` m
- includes 2 um: yes
- curvature range: `0` to `330000` 1/m
- geometry type: `SingleCustomBend`
- input guided count: `10`
- runner mode count: `10`
- S-matrix shape: `(16, 20, 20)` for the test discretization
- TE0-like output first two forward intensities: approximately `0.9783`, `0.0217`
- TE1-like output first two forward intensities: approximately `0.0217`, `0.9699`

This confirms the current precomputed dataset and EME/Runner stack can compute the raw quantities needed for both 50:50 and 70:30 bend mode-splitting FOMs.

## Missing Components For The Pivot

Because feasibility is partial, the missing implementation components are:

1. A bend-specific geometry parameterization module that maps an optimizer vector to:
   - `prop_len_list`
   - `width_list`
   - `curvature_list`
   - optional fixed `Rx`, `Ry`, and bend angle constraints from the paper notebook.

2. A bend-specific response extractor that returns the 2x2 first-two-mode intensity matrix for TE0 and TE1 inputs, plus loss/reflection/phase diagnostics.

3. Explicit 50:50 and 70:30 FOM definitions.

4. A validation script that runs selected optimized bends and saves:
   - geometry plots
   - intensity matrix
   - loss/reflection totals
   - target errors for 50:50 and 70:30
   - raw or unitary-projected S-matrix policy.

5. Optional: a small compatibility fix or avoidance plan for `SingleBezierCurve` if that class is later needed under NumPy 2.x.
