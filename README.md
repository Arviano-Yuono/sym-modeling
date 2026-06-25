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
      methods/sgeppy/
```

## What Was Ported

- SpaRTA-side CFD data handling, preprocessing, candidate libraries, and workflow orchestration were moved under `domains/cfd`.
- EUCLID-side FEM loading, kinematics, feature generation, weak-form assembly, constraints, and Lp solver were moved under `domains/fem`.
- The new repo keeps compatibility aliases like `FlowData` and `FemDataset`, but the architectural entry points are now `CFDCaseData` and `FEMCaseData`.

## Notes

- Large CFD/FEM datasets were not copied into this repo. Update method configs to point at your data location, or place data under this repo if you want a fully standalone setup.
- `foamlib` is an optional dependency because it is CFD-specific.
- `jax_fem` is an optional dependency for SGEPPY weak-form runs that use JAX-native arrays and autodiff.
- `torch_fem` is an optional dependency for the CUDA-only SGEPPY PyTorch backend.
- Use Docker for FEniCSx work on Linux when possible. The PyPI `fenics` package is not used.

## Quick Start

For non-FEniCSx work, use `uv` locally:

```bash
uv sync --extra cfd --extra fem --extra viz --extra dev
uv run pytest
```

This is useful for regular Python development, but it does not install the core
FEniCSx stack. Use Docker for DOLFINx/PETSc/MPI-backed workflows.

Installed CLI commands are documented in `docs/cli-reference.md`. The main
commands are `sym-fem-sgeppy`, `sym-fem-euclid`, `sym-fem-forward-compare`,
`sym-fem-denoise`, `sym-util-print`, and `sym-util-mesh`.

## FEniCSx Setup

The recommended Linux setup uses the repo Docker image. It installs the compiled
FEniCSx stack from conda-forge and uses `uv` for this repo's Python dependencies.
Do not install the old PyPI `fenics` package for this project.

Build the project image:

```bash
docker compose build fenicsx
```

Check FEniCSx imports:

```bash
docker compose run --rm fenicsx python -c "import dolfinx, ufl; print(dolfinx.__version__)"
```

Run tests:

```bash
docker compose run --rm fenicsx pytest
```

Run the Euclid tutorial in the container:

```bash
docker compose run --rm fenicsx python tutorial.py --single-case --skip-plots
```

Start Jupyter Lab at <http://localhost:8888>:

```bash
docker compose up lab
```

Generated files are written back to the host because the repo is mounted at
`/workspace` inside the container.

Open an interactive shell in the same environment:

```bash
docker compose run --rm fenicsx bash
```

Inside that shell, normal commands work as expected:

```bash
python tutorial.py
pytest
uv pip install --system -e ".[cfd,fem,viz,dev]"
```

## Dependency Management

Use `uv add` for normal Python libraries. Examples:

```bash
uv add numpy pandas scipy
uv add --optional fem gmsh seaborn
uv add --optional viz pyvista imageio
uv add --optional dev pytest jupyterlab
uv add --optional jax_fem jax jax-fem pyfiglet
```

Keep FEniCSx itself (`dolfinx`, PETSc, MPI, `petsc4py`) supplied by Docker or
Conda. Those packages depend on compiled system/MPI libraries and should not be
managed as ordinary PyPI dependencies here.

After changing Python dependencies, rebuild the Docker image:

```bash
docker compose build fenicsx
```

## Optional Conda Fallback

If you prefer Conda instead of Docker, use Conda only for the compiled FEniCSx
stack, then use `uv` for this repo's Python dependencies:

```bash
conda env create -f environment.yml
conda activate sym-modeling-fenicsx
uv pip install -e ".[cfd,fem,viz,dev]"
python -m ipykernel install --user --name sym-modeling-fenicsx --display-name "Python (sym-modeling-fenicsx)"
```

Quick import check:

```bash
python -c "import dolfinx, ufl; print(dolfinx.__version__)"
```

Then select the `Python (sym-modeling-fenicsx)` kernel in VS Code or Jupyter.

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

## Operating SGEPPY

SGEPPY is the geppy-backed FEM symbolic discovery runner. It generates candidate
strain-energy features, fits sparse coefficients, and writes run artifacts under
the configured `output_dir`.

Use the checked-in configs first:

```bash
uv sync --extra dev --extra torch_fem
uv run --extra torch_fem sym-fem-sgeppy --config configs/sgeppy/nh2.json
```

Available examples live under `configs/sgeppy/` and inherit shared defaults from
`configs/sgeppy/_base.json`.

SGEPPY uses `backend="torch"` by default: it reads FEM CSV data, stores
calculation arrays as backend arrays, computes `dQ/dF` with autodiff, and
assembles weak-form matrices with batched backend operations. The Torch path is
CUDA-only and requires `uv sync --extra torch_fem` plus a CUDA-capable PyTorch
build and NVIDIA driver. Use `--backend jax` with the `jax_fem` extra when you
want the JAX backend instead.

Small experiments are usually easiest as CLI overrides:

```bash
uv run --extra torch_fem sym-fem-sgeppy \
  --config configs/sgeppy/nh2.json \
  --loadsteps 10,20 \
  --noise-level 1e-4 \
  --generations 25 \
  --population-size 40 \
  --n-genes 3 \
  --output-dir output/sgeppy/nh2_noise_1e-4
```

Use `--noise-level` to add displacement noise while loading FEM data. This is
separate from `--epsilons`, which configures epsilon-constrained model fitness
ranking against `--fitness-metrics` values.

More detail is in `docs/fem-discovery-methods.md`.

To compare the JAX and Torch-CUDA backends on a small smoke run:

```bash
uv run --extra jax_fem --extra torch_fem python scripts/benchmark_sgeppy_backends.py \
  --config configs/sgeppy/nh2.json \
  --loadsteps 10 \
  --generations 0 \
  --population-size 3 \
  --output-json tmp/sgeppy_backend_ab/nh2.json
```

## Operating Euclid

The fastest end-to-end walkthrough is:

```bash
docker compose run --rm fenicsx python tutorial.py
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
docker compose run --rm fenicsx python tutorial.py --single-case
```

By default `tutorial.py` uses `material_model="euclid_neo_hookean_j2"` because
that matches Euclid's current feature basis. If you want the exact notebook-style
law from `hyperelasticity.ipynb`, run:

```bash
docker compose run --rm fenicsx python tutorial.py --material-model notebook_neo_hookean_log
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
