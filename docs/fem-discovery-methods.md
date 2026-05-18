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

For dataset-backed FEM runs, SGEP now uses the same weak-form/reaction-force
coefficient fitting target as EUCLID. The feature library is generated for each
candidate model, attached to the FEM data as `Q`, `dQ/dI1`, `dQ/dI2`, and
`dQ/dI3`, and then passed through the shared weak-form assembly.

The current workflow:

1. generates symbolic genes using configurable operators
2. evaluates genes on invariant variables such as `K1`, `K2`, `Jm1`
3. reference-normalizes each gene so `G(reference) = 0`
4. differentiates each gene through invariants to get the feature derivatives
5. fits sparse coefficients with the shared EUCLID-style weak-form Lp solver
6. scores candidates with weak-form RSS, RMSE, AIC, and AICc
7. computes AICc-ablation gene fitness
8. evolves useful genes into the next generation

Synthetic SGEP runs have no mesh, boundary conditions, or reaction forces, so
they keep the older direct Piola-stress sparse-regression fallback.

## SGEP Config

Run SGEP with:

```bash
uv run python -m sym_modeling.domains.fem.methods.sgep.run_gep_sparse \
  --config configs/sgep_config.json
```

or with the shorter console-script alias:

```bash
uv run sym-fem-sgep --config configs/sgep_config.json
```

Per-model configs are also available:

```text
configs/sgep/nh2.json
configs/sgep/nh4.json
configs/sgep/ih.json
configs/sgep/hw.json
configs/sgep/gt.json
```

### FEM-SGEP Inputs

The `sym-fem-sgep` runner takes one main input: an SGEP JSON config. The config
can either be a plain JSON object or wrapped under a top-level `"sgep"` key:

```json
{
  "sgep": {
    "data_dir": "dataset/fem_data/plate_hole_fenics/NH2",
    "loadsteps": [10, 20, 30, 40],
    "variable_names": ["K1", "K2", "Jm1"],
    "output_dir": "output/sgep/nh2"
  }
}
```

The most common command is:

```bash
uv run sym-fem-sgep --config configs/sgep/nh2.json
```

You can override the most frequently changed config values from the CLI:

```bash
uv run sym-fem-sgep \
  --config configs/sgep/nh2.json \
  --loadsteps 10,20 \
  --fitting-mode direct_stress \
  --generations 3 \
  --num-models 4 \
  --genes-per-model 3 \
  --output-dir /tmp/sgep_nh2_test \
  --skip-plots
```

#### Dataset Inputs

For FEM-backed discovery, `data_dir` should point to a dataset root containing
numeric load-step folders:

```text
dataset/fem_data/plate_hole_fenics/NH2/
  10/
  20/
  30/
  40/
```

Each load-step folder is read with `loadFemData(...)`. The important quantities
available after loading are:

| Quantity | Meaning | Used for |
| --- | --- | --- |
| nodal coordinates | reference mesh nodes | weak-form assembly and plotting |
| nodal displacements | measured or simulated displacement field | deformation-gradient recovery |
| element connectivity | triangular element node indices | weak-form assembly and plotting |
| shape gradients | element shape-function gradients | internal-force assembly |
| quadrature weights | element area/weight | weak-form integration |
| boundary reactions | measured reaction-force data | weak-form RHS target |
| `F` | deformation gradient per element | invariants and Piola stress |
| `P` | reference first Piola-Kirchhoff stress, if exported | stress plots and direct-stress diagnostics |

Dataset-backed SGEP fits coefficients using weak-form equilibrium and reaction
forces. Reference `P` columns are useful for plots and diagnostics, but the
weak-form fit itself is driven by the assembled reaction-force target.

If `data_dir` is omitted, SGEP runs on a synthetic stress dataset instead. That
mode is mainly useful for tests and quick algorithm checks because it has no
mesh, no boundary conditions, and no reaction-force measurements.

#### Config Input Groups

The SGEP config fields can be read in groups.

Dataset selection:

| Field | Meaning |
| --- | --- |
| `data_dir` | FEM dataset root. If `null`, use synthetic fallback data. |
| `loadsteps` | Load-step folders to use. If omitted, SGEP discovers numeric folders. |
| `fitting_mode` | `auto`, `weak_form`, or `direct_stress`. |
| `noise_level` | Displacement noise passed to the FEM CSV loader. |
| `max_elements_per_loadstep` | Optional direct-stress cap. Use `null` for all elements. |

Symbolic search space:

| Field | Meaning |
| --- | --- |
| `variable_names` | Scalar invariant variables available to genes. |
| `unary_operators` | Unary operators allowed in generated genes. |
| `binary_operators` | Binary operators allowed in generated genes. |
| `max_depth` | Maximum random expression-tree depth. |

Evolution settings:

| Field | Meaning |
| --- | --- |
| `genes_per_model` | Number of generated energy features in each candidate model. |
| `num_models` | Number of candidate models evaluated per generation. |
| `generations` | Number of evolutionary generations. |
| `random_seed` | Seed for reproducible population generation. |
| `mutation_rate` | Probability of subtree mutation when creating a child gene. |
| `crossover_rate` | Probability of subtree crossover between parent genes. |
| `random_fraction` | Fraction of each new population filled with fresh random genes. |
| `elite_model_count` | Number of best full candidate models whose genes survive. |
| `elite_gene_count` | Number of individually useful genes kept by ablation fitness. |

Sparse fitting and numerical filtering:

| Field | Meaning |
| --- | --- |
| `regression_method` | Direct-stress fallback solver name for synthetic/debug runs. |
| `regression_alpha` | Regularization strength for direct-stress fallback fitting. |
| `sparsity_threshold` | Small coefficient cutoff for active terms. |
| `selection_objective` | Candidate ranking objective, `aicc` or `epsilon_rmse`. |
| `active_terms_epsilon` | Active-term budget used by `epsilon_rmse`. |
| `refit_active_terms` | Refit after thresholding for direct-stress fallback fitting. |
| `derivative_step` | Finite-difference step for gene derivatives. |
| `invalid_value_limit` | Reject genes producing very large or invalid values. |
| `duplicate_correlation` | Reject nearly duplicate generated stress columns. |

Weak-form fitting:

| Field | Meaning |
| --- | --- |
| `weak_form_balance` | Relative balance applied during LHS/RHS normalization. |
| `weak_form_penalty_lp` | Lp sparse penalty strength. |
| `weak_form_p` | Lp exponent, usually less than 1 for sparsity. |
| `weak_form_num_increments` | Number of penalty-continuation increments. |
| `weak_form_factor_increments` | Penalty multiplier between increments. |
| `weak_form_num_guesses` | Number of solver starts. |
| `weak_form_num_iterations` | Maximum Lp iterations per solve. |
| `weak_form_threshold_iter` | Iteration convergence threshold. |
| `weak_form_threshold` | Final coefficient threshold. |
| `weak_form_energy_checks` | Reject sparse fits that violate simple energy checks. |

Output and runtime behavior:

| Field | Meaning |
| --- | --- |
| `output_dir` | Directory for `summary.json`, `history.csv`, selected genes, and plots. |
| `save_plots` | Enables history, scatter, and per-loadstep Piola field plots. |
| `progress_log` | Prints one-line progress updates per generation. |
| `compare_euclid` | Optionally run EUCLID comparison after SGEP. |
| `euclid_model` | EUCLID model label used by comparison mode. |

### `fitting_mode`

`fitting_mode` controls the regression target used after SGEP generates a
candidate energy library:

```json
"fitting_mode": "direct_stress"
```

The default is `"auto"`, which keeps the existing behavior: FEM dataset runs use
`weak_form`, and synthetic runs use `direct_stress`.

Use `weak_form` when you want the EUCLID-style target based on mesh equilibrium
and reaction forces. This is the most FEM-consistent option, but it is slower
because every candidate must assemble the weak form and run the Lp solver.

Use `direct_stress` for a faster dataset-backed run when the load-step CSV files
include reference Piola columns `Pxx`, `Pxy`, `Pyx`, and `Pyy`. SGEP still models
the strain energy `W = sum(theta_i * G_i)`, but fits coefficients by matching
`dW/dF` directly to exported Piola stress.

The same choice can be made from the CLI:

```bash
uv run sym-fem-sgep --config configs/sgep/nh2.json --fitting-mode direct_stress
```

### `selection_objective`

SGEP supports two candidate ranking objectives after each candidate model is
fitted:

```json
"selection_objective": "epsilon_rmse",
"active_terms_epsilon": 2
```

`aicc` is the original objective. It ranks candidates by corrected AIC, with
RMSE as the tie-breaker:

```text
rank = (AICc, RMSE)
```

`epsilon_rmse` is the recommended manual epsilon-constrained objective. It
minimizes RMSE under an active-term budget:

```text
minimize RMSE
subject to active_terms <= active_terms_epsilon
```

Here, `active_terms` is the number of nonzero fitted coefficients after sparse
regression. If no candidate satisfies the epsilon constraint in an early
generation, SGEP keeps infeasible candidates as a soft fallback and ranks them
after feasible candidates.

To manually sweep model complexity, run one epsilon per output directory:

```bash
for eps in 1 2 3 4; do
  uv run sym-fem-sgep \
    --config configs/sgep/nh2.json \
    --active-terms-epsilon "$eps" \
    --output-dir "output/sgep/nh2_eps${eps}"
done
```

### `max_elements_per_loadstep`

Dataset-backed weak-form SGEP uses the full load-step mesh because global
equilibrium and reaction-force assembly should not drop elements. The shipped
configs therefore disable the old direct-stress element cap:

```json
"max_elements_per_loadstep": null
```

From the CLI, either use:

```bash
uv run sym-fem-sgep --config configs/sgep/nh2.json --all-elements
```

or:

```bash
uv run sym-fem-sgep --config configs/sgep/nh2.json --max-elements-per-loadstep 0
```

For dataset-backed weak-form runs, these no-cap settings are the expected
choice. Per-loadstep Piola field plots also load the full mesh.

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
