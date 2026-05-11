# Documentation Index

This folder documents the main source-code paths used by the project, with extra focus on the FEM hyperelastic discovery workflow.

## Start Here

- [FEM Overview](fem-overview.md): how the FEM package is organized and how data moves through the code.
- [FEM Data Format](fem-data-format.md): the EUCLID-compatible CSV files expected by the loaders.
- [FEM Discovery Methods](fem-discovery-methods.md): how EUCLID and SGEP use the same FEM preprocessing but different model libraries.

## Most Important Source Areas

- `src/sym_modeling/domains/fem/io/`: CSV loading and hyperelastic data generation/export.
- `src/sym_modeling/domains/fem/operators/kinematics.py`: deformation-gradient, strain-invariant, and derivative utilities.
- `src/sym_modeling/domains/fem/methods/euclid/`: fixed-library sparse constitutive discovery.
- `src/sym_modeling/domains/fem/methods/sgep/`: generated-library GEP + sparse-regression discovery.
- `configs/sgep_config.json`: configurable SGEP experiment input.
