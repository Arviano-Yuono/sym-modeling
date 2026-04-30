# sym-modeling

`sym-modeling` is a new neutral repo for symbolic physics modeling, with separate domain packages for CFD and FEM and method packages layered on top of them.

The design goal is to stop coupling the core package shape to a single method or a single physics domain:

- `domains/cfd` contains CFD-specific data structures, operators, OpenFOAM I/O, and the `sparta` method package.
- `domains/fem` contains FEM-specific data structures, kinematics, CSV loading, and the `euclid` method package.
- `common` contains only genuinely reusable pieces such as sparse regression, logging, workflow interfaces, and lightweight mesh/metadata helpers.

## Layout

```text
src/sym_modeling/
  common/
  domains/
    cfd/
      io/
      methods/sparta/
    fem/
      io/
      operators/
      methods/euclid/
```

## What Was Ported

- SpaRTA-side CFD data handling, preprocessing, candidate libraries, and workflow orchestration were moved under `domains/cfd`.
- EUCLID-side FEM loading, kinematics, feature generation, weak-form assembly, constraints, and Lp solver were moved under `domains/fem`.
- The new repo keeps compatibility aliases like `FlowData` and `FemDataset`, but the architectural entry points are now `CFDCaseData` and `FEMCaseData`.

## Notes

- Large CFD/FEM datasets were not copied into this repo. Update method configs to point at your data location, or place data under this repo if you want a fully standalone setup.
- `foamlib` is an optional dependency because it is CFD-specific.
- Use Conda for any FEniCSx work in this repo. The PyPI `fenics` package is not used.

## Quick Start

```bash
pip install -e .[cfd,fem]
python -m unittest discover -s tests
```

## FEniCSx Setup

Create the Conda environment and register a notebook kernel:

```bash
conda env create -f environment.yml
conda activate sym-modeling-fenicsx
python -m ipykernel install --user --name sym-modeling-fenicsx --display-name "Python (sym-modeling-fenicsx)"
```

Then select the `Python (sym-modeling-fenicsx)` kernel in VS Code or Jupyter.

Quick import check:

```bash
python -c "import dolfinx, ufl; print(dolfinx.__version__)"
```

The environment installs the local package in editable mode with `pip install -e .`. If you also need optional project extras after activation:

```bash
pip install -e .[cfd]
pip install -e .[fem,viz]
```

Example imports:

```python
from sym_modeling.domains.cfd.methods.sparta import SpartaTrainer
from sym_modeling.domains.fem import generate_hyperelastic_data
from sym_modeling.domains.fem.methods.euclid import EuclidConfig, EuclidWorkflow
```

## Operating Sparta

SpaRTA is driven by a JSON config file that matches the dataclasses in
`src/sym_modeling/domains/cfd/methods/sparta/config.py`. Once you have a config,
the workflow is:

```python
from sym_modeling.domains.cfd.methods.sparta import SpartaTrainer

trainer = SpartaTrainer(config_path="path/to/sparta_config.json")
trainer.initialize()
trainer.train()
trainer.predict()
trainer.evaluate()
```

Artifacts are written under the run directory configured by `run.output_root`.

## Operating Euclid

The fastest end-to-end walkthrough is:

```bash
conda activate sym-modeling-fenicsx
python tutorial.py
```

That script generates tutorial FEM data, validates the CSV export, runs Euclid
discovery, and writes evaluation metrics plus Piola field comparison plots under
`output/tutorial_euclid/`.

By default the tutorial now generates a richer combined FEM suite with four
scenarios:

- right-boundary uniaxial tension
- top-boundary simple shear
- top-boundary vertical extension
- a mixed right-plus-top loading case

These are merged into one Euclid dataset so discovery sees more than one
deformation family. If you want the original single-case tutorial instead, run:

```bash
python tutorial.py --single-case
```

By default `tutorial.py` uses `material_model="euclid_neo_hookean_j2"` because
that matches Euclid's current feature basis. If you want the exact notebook-style
law from `hyperelasticity.ipynb`, run:

```bash
python tutorial.py --material-model notebook_neo_hookean_log
```

That notebook-style law uses `ln(J)` terms, so `NeoHookeanJ2` discovery will only
approximate it and the recovered coefficients can become very large.

Euclid expects FEM data laid out as:

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

If you want a local dataset generated from the notebook-style hyperelastic setup,
use the built-in generator first. It adapts `hyperelasticity.ipynb` into a 2D
plane-strain DOLFINx case and writes EUCLID-compatible CSVs.

```python
from sym_modeling.domains.fem import (
    HyperelasticGenerationConfig,
    generate_hyperelastic_data,
)

generate_hyperelastic_data(
    "generated/fem/hyperelastic_demo",
    HyperelasticGenerationConfig(),
)
```

You can verify that the exported CSV files reconstruct the same Piola stress field:

```python
from sym_modeling.domains.fem import validate_hyperelastic_data_export

report = validate_hyperelastic_data_export("generated/fem/hyperelastic_demo")
print(report.max_abs_error)
```

Then point Euclid at that directory:

```python
from sym_modeling.domains.fem.methods.euclid import EuclidConfig, EuclidWorkflow

config = EuclidConfig(femDataPathOverride="generated/fem/hyperelastic_demo")
workflow = EuclidWorkflow(config=config)
result = workflow.train()
theta = result.theta
```

If you generate a different set of load-step folder names, also set
`loadstepsOverride` in `EuclidConfig`.
