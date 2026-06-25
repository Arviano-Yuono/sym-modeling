# FEM Data Format

The discovery workflows consume EUCLID-compatible FEM CSV data. A dataset root contains numeric load-step folders:

```text
<dataset-root>/
  10/
    output_nodes.csv
    output_elements.csv
    output_integrator.csv
    output_reactions.csv
  20/
  30/
  ...
```

The current checked-in FEM dataset follows this layout:

```text
dataset/fem_data/plate_hole_fenics/
  NH2/
  NH4/
  IH/
  HW/
  GT/
```

## `output_nodes.csv`

Important columns:

- `x`, `y`: node coordinates
- `ux`, `uy`: nodal displacement
- `bcx`, `bcy`: boundary-condition/reaction labels

The loader uses nodal displacements and element gradients to reconstruct `F`.

## `output_elements.csv`

Important columns:

- `node1`, `node2`, `node3`: triangular element connectivity
- `Fxx`, `Fxy`, `Fyx`, `Fyy`: per-element deformation gradient components
- optional `Pxx`, `Pxy`, `Pyx`, `Pyy`: reference first Piola-Kirchhoff stress

`F` is the deformation gradient. For the 2D displacement field
`u = [ux, uy]`, the exported components follow:

```text
F = I + grad(u)

Fxx = 1 + dux/dx
Fxy =     dux/dy
Fyx =     duy/dx
Fyy = 1 + duy/dy
```

When stored as a flattened 2x2 tensor, the order is:

```text
[Fxx, Fxy, Fyx, Fyy]
```

The reference `P` columns are used for stress diagnostics and summaries.
Forward comparison can still run if `P` columns are missing; it skips only the
Piola metrics and plots for those steps and records that in
`comparison_summary.json`.

## `output_integrator.csv`

Important columns:

- `gradNa_node1_x`, `gradNa_node1_y`, etc.: shape-function gradients
- `qpWeight`: element quadrature weight/area

The loader assumes one quadrature point per linear triangular element.

## `output_reactions.csv`

Important columns:

- `forces`: global reaction force measurements for the labeled constrained DOF groups

EUCLID and SGEPPY weak-form runs use reaction forces in their weak-form fitting
paths. SGEPPY currently uses the JAX weak-form path.

## Loader Entry Point

```python
from sym_modeling.domains.fem.io.csv_loader import loadFemData

data = loadFemData("dataset/fem_data/plate_hole_fenics/NH2/10")
data.convertToNumpy()

print(data.F.shape)
print(data.I1.shape, data.I2.shape, data.I3.shape)
print(data.P.shape)
```

## Forward Comparison Outputs

Forward benchmark and generated forward runs use the same numeric step-folder
layout, so they can be compared directly:

```bash
uv run sym-fem-forward-compare \
  --forward-root output/forward_sgeppy/1e-4/ih \
  --reference-root dataset/fem_data/plate_hole_fenics/IH
```

The command compares shared numeric load-step folders and writes:

```text
<forward-root>/comparison/
  comparison_summary.json
  comparison_metrics.csv
  reaction_comparison.png
  step_<step>_displacement_umag.png
  step_<step>_displacement_ux.png
  step_<step>_displacement_uy.png
  step_<step>_Fnorm.png
  step_<step>_Fxx.png
  step_<step>_Fxy.png
  step_<step>_Fyx.png
  step_<step>_Fyy.png
  step_<step>_Pnorm.png
  step_<step>_Pxx.png
  ...
```

`comparison_metrics.csv` reports `rmse`, `mae`, `max_abs`, and `relative_l2`
for reactions, nodal displacement `u`, deformation gradient `F`, and first
Piola-Kirchhoff stress `P` when available.

Field plots use three panels: reference, forward result, and signed error
`forward - reference`. Displacement is plotted on nodes with mesh
triangulation. `F` and `P` are plotted per element.

## Generating Arruda-Boyce Data

The Arruda-Boyce plate-hole generator uses the checked-in tagged mesh and writes
the same EUCLID-compatible CSV layout:

```bash
docker compose run --rm fenicsx \
  python scripts/generate_arruda_boyce_dataset.py
```

By default, the generated dataset is written to
`dataset/fem_data/plate_hole_fenics/AB`. Run the script with `--help` to adjust
the load steps, output directory, energy scale, limiting chain stretch, or bulk
modulus.
