# FEM Overview

The FEM domain code lives under `src/sym_modeling/domains/fem/`. It supports hyperelastic data generation/loading, kinematic preprocessing, and constitutive-law discovery methods.

## Package Shape

```text
src/sym_modeling/domains/fem/
  data.py                 # shared FEM data containers
  operators/kinematics.py # F, C, invariants, invariant derivatives
  io/csv_loader.py        # EUCLID-compatible CSV loader
  io/hyperelastic.py      # synthetic/tutorial hyperelastic CSV generation
  dolfinx.py              # DOLFINx simulation helper API
  forward_benchmark.py    # forward FEM benchmark utilities
  forward_comparison.py   # generated-vs-reference forward result comparison
  methods/euclid/         # fixed feature-library discovery
  methods/sgeppy/         # geppy-backed generated feature-library discovery
```

## Core Data Flow

The important FEM preprocessing flow is:

```text
CSV nodal displacement + mesh data
  -> loadFemData(...)
  -> reconstruct deformation gradient F per element
  -> compute C = F^T F
  -> compute I1, I2, I3 and J
  -> compute invariant derivatives dI/dF
  -> pass these quantities into EUCLID or SGEPPY
```

This is intentionally shared. EUCLID and SGEPPY should see the same `F`,
invariants, reactions, and reference stresses so comparison is fair.

## Important Objects

`FemDataset` / `FEMCaseData` in `data.py` stores:

- nodal coordinates and displacements
- Dirichlet boundary-condition labels
- reaction-force measurements
- element connectivity and shape-function gradients
- quadrature weights
- deformation gradient `F`
- invariants `I1`, `I2`, `I3`
- invariant derivatives `dI1dF`, `dI2dF`, `dI3dF`
- optional reference first Piola-Kirchhoff stress `P`

## Kinematics Utilities

`operators/kinematics.py` provides the shared tensor operations:

- `computeJacobian(F)`: computes `J = det(F)`.
- `computeCauchyGreenStrain(F)`: computes `C = F^T F`.
- `computeStrainInvariants(C)`: computes plane-strain `I1`, `I2`, `I3`.
- `computeStrainInvariantDerivatives(F, i)`: computes `dIi/dF`.

`F` and `C` use 2D Voigt-like flattened order:

```text
[F11, F12, F21, F22]
```

## Public FEM Imports

The top-level FEM package lazily exports the most common functions, for example:

```python
from sym_modeling.domains.fem import (
    ForwardComparisonConfig,
    loadFemData,
    compare_forward_results,
    computeCauchyGreenStrain,
    computeJacobian,
    computeStrainInvariants,
    computeStrainInvariantDerivatives,
)
```

## Forward Result Comparison

`forward_comparison.py` compares an already-generated forward FEM output root
against a reference FEM dataset. It does not rerun DOLFINx; it consumes the
shared CSV step folders and writes metrics, summaries, and field plots.

```bash
uv run sym-fem-forward-compare \
  --forward-root output/forward_sgeppy/1e-4/ih \
  --reference-root dataset/fem_data/plate_hole_fenics/IH
```

By default, outputs are written to `<forward-root>/comparison/`:

```text
comparison_summary.json
comparison_metrics.csv
reaction_comparison.png
step_10_displacement_umag.png
step_10_displacement_ux.png
step_10_Fnorm.png
step_10_Fxx.png
step_10_Pnorm.png
step_10_Pxx.png
...
```

The comparator automatically uses numeric load-step folders present in both
roots, sorted numerically. Use repeated `--load-step` arguments to restrict the
comparison:

```bash
uv run sym-fem-forward-compare \
  --forward-root output/forward_euclid/0/AB \
  --reference-root dataset/fem_data/plate_hole_fenics/AB \
  --load-step 10 \
  --load-step 20 \
  --plot-quantities ux,uy,umag,Fxx,Pxx
```
