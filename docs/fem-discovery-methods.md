# FEM Discovery Methods

This project currently has two FEM constitutive-discovery methods:

- EUCLID: fixed hand-written feature library.
- SGEP: generated symbolic feature library from a GEP-style population loop.

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

## SGEP

Source path:

```text
src/sym_modeling/domains/fem/methods/sgep/
```

Main entry points:

- `SGEPConfig`
- `SGEPWorkflow`
- `run_gep_sparse.py`
- `configs/sgep_config.json`
- `configs/sgep/*.json`

SGEP replaces the fixed EUCLID feature library with generated genes:

```text
W_n = sum_i theta_i * G_i
```

The current prototype:

1. generates symbolic genes using configurable operators
2. evaluates genes on invariant variables such as `K1`, `K2`, `Jm1`
3. reference-normalizes each gene so `G(reference) = 0`
4. converts each gene into a stress feature by differentiating through invariants
5. fits sparse coefficients using `stlsq`, `lasso`, `ridge`, `elasticnet`, or `ridge_threshold`
6. scores candidates with RSS, RMSE, AIC, and AICc
7. computes AICc-ablation gene fitness
8. evolves useful genes into the next generation

## SGEP Config

Run SGEP with:

```bash
uv run python -m sym_modeling.domains.fem.methods.sgep.run_gep_sparse \
  --config configs/sgep_config.json
```

or with the shorter console-script alias:

```bash
uv run sym-sgep --config configs/sgep_config.json
```

Per-model configs are also available:

```text
configs/sgep/nh2.json
configs/sgep/nh4.json
configs/sgep/ih.json
configs/sgep/hw.json
configs/sgep/gt.json
```

Important config fields:

- `data_dir`: EUCLID-compatible dataset root
- `loadsteps`: load-step folders to use
- `variable_names`: invariant variables available to genes
- `unary_operators`, `binary_operators`: generated expression operators
- `regression_method`: sparse fitting method
- `genes_per_model`, `num_models`, `generations`: evolutionary search size
- `output_dir`: where summaries, history, selected genes, and plots are saved

### `variable_names`

`variable_names` controls which scalar invariant variables SGEP genes are allowed
to use. For example, with:

```json
"variable_names": ["K1", "K2", "Jm1"]
```

generated genes may contain expressions like:

```text
K1
square(Jm1)
(K1 * K2)
log(K2)
```

The available variable names are:

| Name | Meaning |
| --- | --- |
| `I1` | first invariant of `C = F^T F` |
| `I2` | second invariant of `C = F^T F` |
| `I3` | third invariant of `C = F^T F` |
| `J` | Jacobian, `det(F) = sqrt(I3)` |
| `Jm1` | volumetric offset, `J - 1` |
| `K1` | reduced invariant, `I1 * I3^(-1/3) - 3` |
| `K2` | reduced invariant, `(I1 + I3 - 1) * I3^(-2/3) - 3` |

The recommended default is:

```json
"variable_names": ["K1", "K2", "Jm1"]
```

This is the most EUCLID-like SGEP input set. It gives generated genes
deviatoric/reduced invariant variables plus a simple volumetric variable.
It also has a clean undeformed reference state:

```text
K1(reference)  = 0
K2(reference)  = 0
Jm1(reference) = 0
```

That makes reference-state normalization straightforward:

```text
G <- G - G(reference)
```

so every gene contributes zero strain energy in the undeformed state.

Other valid choices include a raw-invariant search:

```json
"variable_names": ["I1", "I2", "J"]
```

or a larger search space:

```json
"variable_names": ["I1", "I2", "I3", "J", "K1", "K2", "Jm1"]
```

The larger set is more expressive, but it also creates more redundant ways to
represent similar physics, so the evolutionary search may need more generations
or a larger population.

Example:

```json
{
  "sgep": {
    "data_dir": "dataset/fem_data/plate_hole_fenics/NH2",
    "loadsteps": [10, 20, 30, 40],
    "variable_names": ["K1", "K2", "Jm1"],
    "unary_operators": ["neg", "square", "sqrt", "log", "exp"],
    "binary_operators": ["add", "sub", "mul", "div"],
    "regression_method": "stlsq"
  }
}
```

## Outputs

SGEP writes:

- `summary.json`: best expression, active genes, coefficients, metrics
- `history.csv`: best RMSE/RSS/AICc by generation
- `selected_genes.json`: selected/high-ranking genes per generation
- optional plots: history, predicted-vs-true stress, and per-loadstep Piola stress fields

For dataset-backed runs with `save_plots: true`, SGEP also writes EUCLID-style
Piola field comparisons:

```text
<output_dir>/piola_fields/loadstep_<step>/
  loadstep_<step>_Pxx.png
  loadstep_<step>_Pxy.png
  loadstep_<step>_Pyx.png
  loadstep_<step>_Pyy.png
  loadstep_<step>_Pnorm.png
```

The corresponding `summary.json` contains:

- `piola_field_plots`: plot paths grouped by load step
- `piola_field_metrics`: RMSE, MAE, and max absolute error per load step
- `piola_field_warnings`: skipped load steps or plotting-time warnings
