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
- optional `Pxx`, `Pxy`, `Pyx`, `Pyy`: reference first Piola-Kirchhoff stress

The reference `P` columns are used for stress diagnostics and summaries.

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
