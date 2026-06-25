# Documentation Index

This folder documents the main source-code paths used by the project, with extra focus on the FEM hyperelastic discovery workflow.

## Start Here

- [FEM Overview](fem-overview.md): how the FEM package is organized and how data moves through the code.
- [FEM Data Format](fem-data-format.md): the EUCLID-compatible CSV files expected by the loaders, plus forward comparison outputs.
- [FEM Discovery Methods](fem-discovery-methods.md): how EUCLID and SGEPPY use the same FEM preprocessing but different model libraries.
- [CLI Reference](cli-reference.md): installed `sym-*` commands and common options.
- [ThermoCorr Formatting](thermocorr.md): commands for formatting ThermoCorr DIC data and plotting sampled meshes.

## Most Important Source Areas

- `src/sym_modeling/domains/fem/io/`: CSV loading and hyperelastic data generation/export.
- `src/sym_modeling/domains/fem/operators/kinematics.py`: deformation-gradient, strain-invariant, and derivative utilities.
- `src/sym_modeling/domains/fem/forward_comparison.py`: generated-vs-reference FEM forward comparison.
- `src/sym_modeling/domains/fem/methods/euclid/`: fixed-library sparse constitutive discovery.
- `src/sym_modeling/domains/fem/methods/sgeppy/`: geppy-backed generated-library discovery.
- `configs/sgeppy/`: reusable SGEPPY experiment configs.
