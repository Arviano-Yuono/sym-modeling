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
  methods/euclid/         # fixed feature-library discovery
  methods/sgep/           # GEP-generated feature-library discovery
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
  -> pass these quantities into EUCLID or SGEP
```

This is intentionally shared. EUCLID and SGEP should see the same `F`, invariants, and reference stresses so comparison is fair.

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
    loadFemData,
    computeCauchyGreenStrain,
    computeJacobian,
    computeStrainInvariants,
    computeStrainInvariantDerivatives,
)
```
