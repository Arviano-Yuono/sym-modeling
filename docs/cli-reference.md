# CLI Reference

The installed console commands are declared in `pyproject.toml` under
`[project.scripts]`. Run them through `uv run` from the repo root, or directly
inside an environment where `sym-modeling` is installed.

## FEM Commands

### `sym-fem-sgeppy`

Run geppy-backed SGEPPY FEM symbolic discovery.

```bash
uv run --extra jax_fem sym-fem-sgeppy \
  --config configs/sgeppy/nh2.json
```

Common options:

- `--config`: JSON config with a top-level `sgeppy` object.
- `--data-dir`: override the FEM dataset root.
- `--output-dir`: override the run artifact directory.
- `--loadsteps`: comma-separated numeric load-step folders, for example `10,20`.
- `--noise-level`: add displacement noise while loading FEM data.
- `--backend`: `jax` or CUDA-only `torch`.
- `--generations`, `--population-size`, `--n-genes`, `--seed`: search controls.
- `--fitness-metrics`, `--epsilons`: ranking and epsilon-constraint controls.

Use this when discovering a symbolic strain-energy expression from FEM CSV data.

### `sym-fem-euclid`

Run weak-form EUCLID discovery on FEM dataset folders.

```bash
uv run sym-fem-euclid \
  --data-root dataset/fem_data/plate_hole_fenics \
  --models NH2,NH4,IH \
  --output-root output/euclid
```

Common options:

- `--data-root`: root containing material-model dataset folders.
- `--models`: comma-separated model folders to run.
- `--output-root`: output directory for EUCLID artifacts.
- `--noise-level`: add displacement noise while loading FEM data.
- `--active-threshold`: coefficient threshold used when formatting expressions.

Use this for fixed-library sparse constitutive discovery.

### `sym-fem-forward-compare`

Compare a generated FEM forward result directory against a reference FEM dataset.

```bash
uv run sym-fem-forward-compare \
  --forward-root output/forward_sgeppy/1e-4/ih \
  --reference-root dataset/fem_data/plate_hole_fenics/IH
```

Common options:

- `--forward-root`: generated forward output root.
- `--reference-root`: reference FEM dataset root.
- `--output-dir`: comparison output directory; defaults to `<forward-root>/comparison`.
- `--load-step`: numeric load-step folder to compare; repeat to select multiple steps.
- `--plot-quantities`: comma-separated plot quantities, or `all`.

Use this after a discovered law has been run through the forward benchmark.

### `sym-fem-denoise`

Generate a denoised FEM CSV dataset from an existing clean FEM dataset.

```bash
uv run sym-fem-denoise \
  --data-dir dataset/fem_data/plate_hole_fenics/NH2 \
  --output-dir output/denoised/NH2 \
  --noise-level 1e-4
```

Common options:

- `--method`: `krr` or `mesh-laplacian`.
- `--data-dir`: clean FEM dataset root with numeric load-step folders.
- `--output-dir`: output dataset root.
- `--loadsteps`: comma-separated load steps; defaults to dataset discovery.
- `--noise-level`: artificial displacement noise level to add before denoising.
- `--objective`: metric minimized during parameter search.
- `--overwrite`: allow replacing an existing output directory.

Use this when testing discovery robustness against noisy displacement fields.

## Utility Commands

### `sym-util-print`

Export best SGEPPY symbolic expressions as readable text, Markdown, and LaTeX.

```bash
uv run sym-util-print \
  --input-root output/sgeppy \
  --models nh2,ih \
  --output-dir output/sgeppy/expression_report
```

Common options:

- `--input-root`: root containing model run folders.
- `--models`: comma-separated model folders to include.
- `--output-dir`: report directory; defaults to `<input-root>/expression_report`.
- `--title`: title used in Markdown and LaTeX reports.
- `--round-decimals`: round numeric coefficients in exported expressions.

Use this to inspect and share discovered expressions without reading raw JSON.

### `sym-util-mesh`

Render a DOLFINx/Gmsh mesh image from the command line.

```bash
uv run sym-util-mesh \
  --msh dataset/fem_data/plate_hole_fenics/IH/quarter_plate_hole.msh \
  --output output/mesh_IH.png \
  --show-markers
```

Common options:

- `--msh`: input Gmsh `.msh` file.
- `--output`: output image path.
- `--backend`: `matplotlib` or `pyvista`.
- `--gdim`: geometric dimension, `2` or `3`.
- `--show-markers`: color cells by mesh marker tags.
- `--line-width`, `--point-size`, `--window-size`, `--title`: rendering controls.

Use this to quickly verify mesh geometry and physical markers.

## Script Wrappers

Some commands also have script wrappers under `scripts/`. For example:

```bash
uv run python scripts/compare_forward_fem.py \
  --forward-root output/forward_sgeppy/1e-4/ih \
  --reference-root dataset/fem_data/plate_hole_fenics/IH
```

Prefer the `sym-*` command when available; use script wrappers when working from
an editable checkout or when a workflow has not yet been promoted to a console
command.
