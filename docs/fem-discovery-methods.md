# FEM Discovery Methods

This project currently has two FEM constitutive-discovery methods:

- EUCLID: fixed hand-written feature library.
- SGEPPY: generated symbolic feature library from a geppy-backed GEP loop.

Both methods reuse the same FEM preprocessing through deformation gradients and invariants.

## Shared Preprocessing

Both methods should start from the same data path:

```text
loadFemData(...)
  -> F
  -> C = F^T F
  -> I1, I2, I3, J
  -> dI1/dF, dI2/dF, dI3/dF
```

This shared preprocessing is the most important consistency rule in the FEM code. Method-specific logic should start after invariants and derivative quantities are available.

## EUCLID

Source path:

```text
src/sym_modeling/domains/fem/methods/euclid/
```

Main entry points:

- `EuclidConfig`
- `EuclidWorkflow`
- `feature_library.computeFeatures`
- `weak_form.computeFirstPiolaTheta`

EUCLID assumes a fixed strain-energy library:

```text
W(F) = sum_i theta_i * Q_i(F)
```

The feature library is built from reduced invariants such as:

```text
K1 = I1 * I3^(-1/3) - 3
K2 = (I1 + I3 - 1) * I3^(-2/3) - 3
J  = sqrt(I3)
```

It solves for sparse coefficients `theta` using weak-form equilibrium and reaction-force data.

## SGEPPY

Source path:

```text
src/sym_modeling/domains/fem/methods/sgeppy/
```

Main entry points:

- `SGEPConfig`
- `SGEPWorkflow`
- `run_gep_sparse.py`
- `configs/sgeppy/_base.json`
- `configs/sgeppy/*.json`

SGEPPY replaces the fixed EUCLID feature library with generated genes:

```text
W_n = sum_i theta_i * G_i
```

For FEM datasets, SGEPPY fits generated energy features with a backend-backed
weak-form/reaction-force target. The usual workflow is:

1. generates symbolic genes using configurable operators
2. evaluates genes on invariant variables such as `K1`, `K2`, `Jm1`
3. reference-normalizes each gene so `G(reference) = 0`
4. differentiates each gene through invariants to get the feature derivatives
5. fits sparse coefficients with weak-form equilibrium
6. scores candidates with RSS, RMSE, AIC, and AICc
7. evolves useful genes into the next generation

SGEPPY requires FEM data because the fitting path uses mesh, boundary
condition, and reaction-force measurements.

## SGEPPY Config

Run SGEPPY with:

```bash
uv run --extra jax_fem sym-fem-sgeppy --config configs/sgeppy/nh2.json
```

The per-model configs inherit shared defaults from `configs/sgeppy/_base.json`:

```text
configs/sgeppy/ab.json
configs/sgeppy/gt.json
configs/sgeppy/hw.json
configs/sgeppy/ih.json
configs/sgeppy/nh2.json
configs/sgeppy/nh4.json
```

### Minimal Config

The runner expects a top-level `"sgeppy"` object:

```json
{
  "sgeppy": {
    "data_dir": "dataset/fem_data/plate_hole_fenics/NH2",
    "loadsteps": [10, 20, 30],
    "output_dir": "output/sgeppy/nh2_test",
    "backend": "jax",
    "model": {
      "variable_names": ["K1", "K2", "Jm1"],
      "binary_operators": ["add", "sub", "mul", "protected_div"],
      "unary_operators": ["square", "cube", "protected_sqrt"],
      "n_genes": 3,
      "population_size": 20,
      "n_generations": 10,
      "fitness_metrics": ["aicc"]
    },
    "weak_form": {
      "penalty_lp": 1e-4,
      "num_iterations": 200,
      "threshold": 1e-2
    }
  }
}
```

Small experiment overrides are usually clearer from the CLI:

```bash
uv run --extra jax_fem sym-fem-sgeppy \
  --config configs/sgeppy/nh2.json \
  --loadsteps 10,20 \
  --noise-level 1e-4 \
  --generations 25 \
  --population-size 40 \
  --n-genes 3 \
  --output-dir output/sgeppy/nh2_noise_1e-4
```

### Dataset Inputs

For FEM-backed discovery, `data_dir` points to a dataset root containing numeric
load-step folders:

```text
dataset/fem_data/plate_hole_fenics/NH2/
  10/
  20/
  30/
```

Each load step is read with `loadFemData(...)`. SGEPPY's weak-form backend uses
nodal coordinates, displacements, elements, shape gradients, quadrature weights,
and reaction forces. The stress CSV columns `Pxx`, `Pxy`, `Pyx`, and `Pyy` are
still used to build diagnostic stress features and summaries.

### Important Config Fields

| Field | Meaning |
| --- | --- |
| `data_dir` | FEM dataset root. Required for SGEPPY backend fitting. |
| `loadsteps` | Load-step folders to use. If omitted, numeric folders are discovered. |
| `backend` | `jax` by default, or CUDA-only `torch`. |
| `precision` | `float64` or `float32`. |
| `cache_enabled` | Enable backend artifact and per-gene evaluator caches. |
| `cache_size` | Maximum weak-form artifact cache entries. |
| `cache_device_outputs` | Cache device-resident derivative artifacts when enabled. |
| `gene_cache_size` | Maximum per-gene evaluator cache entries. |
| `noise_level` | Additional displacement noise passed to the FEM CSV loader. |
| `max_elements_per_loadstep` | Optional element cap per load step. Use `null` for all elements. |
| `output_dir` | Directory for run artifacts. |
| `progress_log` | Print progress during evolution. |
| `generation_log` | Write per-generation best-candidate snapshots. |

The nested `model` object controls the symbolic search:

| Field | Meaning |
| --- | --- |
| `variable_names` | Scalar invariant variables available to genes. |
| `unary_operators` | Unary operators allowed in generated genes. |
| `binary_operators` | Binary operators allowed in generated genes. |
| `head_length` | GEP gene head length. Larger values allow larger expressions. |
| `n_genes` | Number of generated energy features per individual. |
| `population_size` | Number of individuals per generation. |
| `n_generations` | Number of evolutionary generations. |
| `random_seed` | Seed for reproducible runs. |
| `fitness_metrics` | Metrics used to rank candidates, for example `["rmse"]` or `["aicc"]`. |
| `epsilons` | Optional epsilon constraints paired with `fitness_metrics`. |

The nested `weak_form` object controls the Lp sparse solve used by weak-form
fitting.

### `backend`

SGEPPY supports two weak-form backends:

```json
"backend": "jax"
```

`jax` evaluates generated feature derivatives and weak-form assembly with JAX.
It is the default. Use `uv sync --extra jax_fem` before running this backend.

```json
"backend": "torch"
```

`torch` evaluates the same weak-form path with PyTorch tensors on CUDA. This
backend intentionally has no CPU fallback; use `uv sync --extra torch_fem` and
ensure an NVIDIA driver plus a CUDA-capable Torch build are available.

The same choice can be made from the CLI:

```bash
uv run --extra jax_fem sym-fem-sgeppy \
  --config configs/sgeppy/nh2.json \
  --backend jax
```

For a small JAX/PyTorch A/B smoke run:

```bash
uv run --extra jax_fem --extra torch_fem python scripts/benchmark_sgeppy_backends.py \
  --config configs/sgeppy/nh2.json \
  --loadsteps 10 \
  --generations 0 \
  --population-size 3 \
  --output-json tmp/sgeppy_backend_ab/nh2.json
```

### `epsilons`

`epsilons` is not a data-noise setting. It is an optional fitness constraint
mechanism. It must have the same length as `fitness_metrics`, and one entry must
be `null`/`none`; that entry is the metric to optimize after satisfying the
other metric limits.

```json
"fitness_metrics": ["rmse", "aicc"],
"epsilons": [0.05, null]
```

This means:

```text
minimize AICc
subject to RMSE <= 0.05
```

The same setting can be overridden from the CLI:

```bash
uv run --extra jax_fem sym-fem-sgeppy \
  --config configs/sgeppy/nh2.json \
  --fitness-metrics rmse,aicc \
  --epsilons 0.05,none
```

Use `--noise-level`, not `--epsilons`, to add displacement noise while loading
FEM data:

```bash
uv run --extra jax_fem sym-fem-sgeppy \
  --config configs/sgeppy/nh2.json \
  --noise-level 1e-4
```

### `variable_names`

`variable_names` controls which scalar invariant variables SGEPPY genes may use.
The recommended default is:

```json
"variable_names": ["K1", "K2", "Jm1"]
```

These are reduced/deviatoric variables plus a simple volumetric variable. They
also have a clean undeformed reference state:

```text
K1(reference)  = 0
K2(reference)  = 0
Jm1(reference) = 0
```

Available names include:

| Name | Meaning |
| --- | --- |
| `I1` | first invariant of `C = F^T F` |
| `I2` | second invariant of `C = F^T F` |
| `I3` | third invariant of `C = F^T F` |
| `J` | Jacobian, `det(F) = sqrt(I3)` |
| `Jm1` | volumetric offset, `J - 1` |
| `K1` | reduced invariant, `I1 * I3^(-1/3) - 3` |
| `K2` | reduced invariant, `(I1 + I3 - 1) * I3^(-2/3) - 3` |

### Operators

Prefer the protected operator names in configs:

```json
"binary_operators": ["add", "sub", "mul", "protected_div"],
"unary_operators": ["square", "cube", "protected_sqrt", "protected_log", "protected_exp"]
```

The protected operators avoid invalid values during evolution. Older short names
such as `div`, `sqrt`, `log`, and `exp` are not accepted by the current config
validation.

### Noise Sweeps

For a small noise sweep, keep each result in a separate output directory:

```bash
for noise in 0 1e-4 1e-3; do
  uv run --extra jax_fem sym-fem-sgeppy \
    --config configs/sgeppy/nh2.json \
    --noise-level "$noise" \
    --output-dir "output/sgeppy/nh2_noise_${noise}"
done
```

## Outputs

SGEPPY writes:

- `summary.json`: best expression, active genes, coefficients, metrics
- `history.csv`: generation-level fitness statistics
- `generation_log.csv`: best expression and coefficients after each generation
- `best_so_far.json`: latest durable best-candidate snapshot
- `expression_tree.png`: best individual expression tree when graph export works

Use `--disable-generation-log` when only final outputs are needed.
