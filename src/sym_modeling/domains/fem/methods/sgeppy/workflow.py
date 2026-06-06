from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Sequence

import geppy as gep
import numpy as np

from .sgep import SGEP, SGEPConfig
from sym_modeling.domains.fem.data import FeatureSet
from sym_modeling.domains.fem.io.csv_loader import loadFemData
from sym_modeling.domains.fem.methods.common.lp_solver import apply_penalty_lp_iteration
from sym_modeling.domains.fem.methods.common.regression import SparseFitResult, regression_metrics
from sym_modeling.domains.fem.methods.common.stress_data import (
    StressDataset,
    build_stress_dataset_from_fem_data,
    invariant_variables,
    load_stress_dataset_from_euclid_csv,
    reference_variables,
    resolve_loadsteps,
    synthetic_neo_hookean_dataset,
    variable_derivatives_wrt_F,
    variable_derivatives_wrt_invariants,
)
from sym_modeling.domains.fem.methods.common.weak_form import (
    assemble_B_matrix,
    zip_dofs,
)
from .backend import BACKENDS, PRECISIONS, backend_timing_keys, create_weak_form_backend, empty_backend_timing

JAX_TIMING_KEYS = backend_timing_keys("jax")


@dataclass
class WeakFormConfig:
    balance: float = 100.0
    penalty_lp: float = 1e-4
    p: float = 0.25
    num_increments: int = 5
    factor_increments: float = 5.0
    num_guesses: int = 1
    num_iterations: int = 200
    threshold_iter: float = 1e-6
    threshold: float = 1e-2


@dataclass
class SGEPWorkflowConfig:
    model: SGEPConfig = field(default_factory=SGEPConfig)
    backend: str = "jax"
    weak_form: WeakFormConfig = field(default_factory=WeakFormConfig)
    data_dir: str | None = None
    loadsteps: list[int] | None = None
    noise_level: float = 0.0
    max_elements_per_loadstep: int | None = 600
    synthetic_samples: int = 200
    synthetic_mu: float = 1.0
    synthetic_bulk: float = 10.0
    derivative_step: float = 1e-6
    invalid_value_limit: float = 1e8
    duplicate_correlation: float = 0.999999
    output_dir: str = "output/sgeppy_results"
    progress_log: bool = True
    generation_log: bool = True
    precision: str = "float64"
    cache_enabled: bool = True
    cache_size: int = 256
    cache_device_outputs: bool = True
    gene_cache_size: int = 1024

    def __post_init__(self) -> None:
        if self.backend not in BACKENDS:
            raise ValueError("backend must be one of: jax, torch.")
        if self.precision not in PRECISIONS:
            raise ValueError("precision must be one of: float64, float32.")
        self.cache_size = int(self.cache_size)
        if self.cache_size < 0:
            raise ValueError("cache_size must be non-negative.")
        self.gene_cache_size = int(self.gene_cache_size)
        if self.gene_cache_size < 0:
            raise ValueError("gene_cache_size must be non-negative.")


@dataclass
class SGEPResult:
    best_expression: str
    theta: np.ndarray
    metrics: dict
    history: list[dict]
    timing: dict[str, float]
    model: SGEP
    output_paths: dict[str, str] = field(default_factory=dict)


@dataclass
class _WeakFormDataCache:
    data: object
    dataset: StressDataset
    variables: dict[str, np.ndarray]
    X: np.ndarray
    dvar_dI: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]
    free_dofs: np.ndarray
    reactions: tuple[tuple[np.ndarray, float], ...]
    B_matrices: np.ndarray


class SGEPWorkflow:
    def __init__(self, config: SGEPWorkflowConfig | None = None):
        self.config = config or SGEPWorkflowConfig()
        self.dataset: StressDataset | None = None
        self.fem_datasets = None
        self.weak_form_cache: list[_WeakFormDataCache] | None = None
        self.model: SGEP | None = None
        self.result: SGEPResult | None = None
        self._backend_timing = empty_backend_timing(self.config.backend)
        self._backend = None
        self._generation_output_paths: dict[str, str] = {}

    def train(self) -> SGEPResult:
        wall_start = time.perf_counter()
        cpu_start = time.process_time()
        if self.config.data_dir is None:
            raise ValueError("backend='%s' requires data_dir." % self.config.backend)
        self._backend = create_weak_form_backend(self.config, self._backend_timing)
        self._backend.configure()
        self.dataset = self._load_dataset()
        self.fem_datasets = self._load_fem_datasets()
        self.weak_form_cache = self._build_weak_form_cache()
        variables = invariant_variables(self.dataset, self.config.model.variable_names)
        builder = self._backend.make_stress_feature_builder(
            self.dataset,
            self.config.model.variable_names,
            value_limit=self.config.invalid_value_limit,
            duplicate_correlation=self.config.duplicate_correlation,
        )
        evaluator = self._backend_evaluator(builder)
        self.model = SGEP(self.config.model)
        self._generation_output_paths = {}
        generation_callback = self._initialize_generation_log(self.model) if self.config.generation_log else None
        self.model.fit(
            variables,
            self.dataset.target_vector,
            feature_builder=builder,
            evaluator=evaluator,
            generation_callback=generation_callback,
        )
        fit = self.model.best_fit
        metrics = {
            "rss": fit.metrics.rss,
            "rmse": fit.metrics.rmse,
            "aic": fit.metrics.aic,
            "aicc": fit.metrics.aicc,
            "num_samples": fit.metrics.num_samples,
            "num_parameters": fit.metrics.num_parameters,
        }
        history = [dict(row) for row in self.model.logbook]
        timing = {
            "wall_seconds": time.perf_counter() - wall_start,
            "cpu_seconds": time.process_time() - cpu_start,
        }
        timing.update(self._backend_timing)
        self.result = SGEPResult(
            best_expression=_reference_normalized_expression(self.model),
            theta=self.model.best_individual.theta,
            metrics=metrics,
            history=history,
            timing=timing,
            model=self.model,
        )
        self.result.output_paths = self._save_outputs(self.result)
        return self.result

    def predict(self) -> np.ndarray:
        if self.result is None or self.dataset is None:
            raise RuntimeError("SGEPWorkflow.predict() requires a completed train() call.")
        variables = invariant_variables(self.dataset, self.config.model.variable_names)
        return self.result.model.predict(variables).reshape(-1, 4)

    def evaluate(self) -> SGEPResult:
        if self.result is None:
            raise RuntimeError("SGEPWorkflow.evaluate() requires a completed train() call.")
        return self.result

    def _load_dataset(self) -> StressDataset:
        if self.config.data_dir is not None:
            self._log("Loading FEM stress data from %s." % self.config.data_dir)
            return load_stress_dataset_from_euclid_csv(
                self.config.data_dir,
                loadsteps=self.config.loadsteps,
                max_elements_per_loadstep=self.config.max_elements_per_loadstep,
                noise_level=self.config.noise_level,
            )
        self._log("Using synthetic neo-Hookean data.")
        return synthetic_neo_hookean_dataset(
            num_samples=self.config.synthetic_samples,
            seed=self.config.model.random_seed,
            mu=self.config.synthetic_mu,
            bulk=self.config.synthetic_bulk,
        )

    def _load_fem_datasets(self):
        data_path = Path(self.config.data_dir)
        steps = list(self.config.loadsteps) if self.config.loadsteps is not None else resolve_loadsteps(data_path)
        datasets = []
        for step in steps:
            data = loadFemData(
                str(data_path / str(step)),
                AD=True,
                noiseLevel=self.config.noise_level,
                noiseType="displacement",
            )
            data.convertToNumpy()
            datasets.append(data)
        return datasets

    def _backend_evaluator(self, stress_builder):
        if self.weak_form_cache is None and self.fem_datasets is not None:
            self.weak_form_cache = self._build_weak_form_cache()
        weak_caches = self.weak_form_cache or []
        fem_datasets = [cache.data for cache in weak_caches]
        variable_names = self.config.model.variable_names

        if self._backend is None:
            self._backend = create_weak_form_backend(self.config, self._backend_timing)
            self._backend.configure()
        backend = self._backend
        eval_cache = backend.eval_cache
        backend_cases = [backend.prepare_case(cache) for cache in weak_caches]

        def evaluate(model: SGEP, individual, X: np.ndarray, y: np.ndarray):
            evaluation_start = time.perf_counter()
            stress_features, valid = stress_builder(model, individual, X)
            gene_valid = np.asarray(valid[: len(individual)], dtype=bool)
            valid_indices = np.flatnonzero(gene_valid)
            if valid_indices.size == 0:
                raise ValueError("No valid %s weak-form genes." % backend.name)

            residual_operators = []
            weak_config = self._weak_form_config()
            lhs = np.zeros((valid_indices.size, valid_indices.size), dtype=float)
            rhs = np.zeros(valid_indices.size, dtype=float)
            balance = float(getattr(weak_config, "balance", 100.0))
            for case_index, backend_case in enumerate(backend_cases):
                case_key = (
                    "loadstep",
                    case_index,
                    id(backend_case.data),
                    tuple(getattr(backend_case.F, "shape", ())),
                    str(getattr(backend_case.F, "dtype", "")),
                    str(getattr(backend_case.F, "device", "")),
                )
                artifact_key = eval_cache.weak_artifact_key(
                    case_key,
                    model,
                    individual,
                    valid_indices,
                    variable_names,
                    self.config.precision,
                    balance,
                )
                cached_artifact = eval_cache.get_artifact(artifact_key)
                if cached_artifact is not None:
                    step_lhs, step_rhs, residual_operator = cached_artifact
                    lhs += step_lhs
                    rhs += step_rhs
                    residual_operators.append(residual_operator)
                    continue

                start = time.perf_counter()
                _, dqdf = backend.feature_values_and_dqdf_device(
                    model,
                    individual,
                    backend_case.F,
                    variable_names,
                    gene_indices=valid_indices,
                    value_limit=self.config.invalid_value_limit,
                    data_key=(
                        "weak",
                        case_index,
                        id(backend_case.data),
                        tuple(getattr(backend_case.F, "shape", ())),
                        str(getattr(backend_case.F, "dtype", "")),
                        str(getattr(backend_case.F, "device", "")),
                    ),
                )
                backend.block_until_ready(dqdf)
                self._backend_timing[backend.timing_key("gene_derivative_seconds")] += time.perf_counter() - start

                start = time.perf_counter()
                weak_lhs = backend.compute_weak_lhs_device(backend_case, dqdf)
                step_lhs_device, step_rhs_device = backend.compute_reaction_balance_device(backend_case, weak_lhs, balance)
                residual_matrix_device, residual_target_device = backend.compute_residual_operator_device(
                    backend_case,
                    weak_lhs,
                    balance,
                )
                backend.block_until_ready((step_lhs_device, step_rhs_device, residual_matrix_device, residual_target_device))
                self._backend_timing[backend.timing_key("weak_lhs_seconds")] += time.perf_counter() - start

                start = time.perf_counter()
                step_lhs = _device_to_numpy(step_lhs_device)
                step_rhs = _device_to_numpy(step_rhs_device)
                residual_operators.append(
                    (
                        _device_to_numpy(residual_matrix_device),
                        _device_to_numpy(residual_target_device),
                    )
                )
                eval_cache.put_artifact(
                    artifact_key,
                    (
                        step_lhs,
                        step_rhs,
                        residual_operators[-1],
                    ),
                )
                self._backend_timing[backend.timing_key("transfer_seconds")] += time.perf_counter() - start
                lhs += step_lhs
                rhs += step_rhs

            def weak_cost(theta_candidate: np.ndarray) -> tuple[float, float, float]:
                residual = _residual_vector_from_operators(residual_operators, theta_candidate)
                weak_value = float(np.sum(np.square(residual)))
                penalty = float(getattr(weak_config, "penaltyLp", 0.0)) * float(
                    np.sum(np.power(np.abs(theta_candidate), float(getattr(weak_config, "p", 1.0))))
                )
                return weak_value, penalty, weak_value + penalty

            start = time.perf_counter()
            theta_valid = apply_penalty_lp_iteration(
                fem_datasets,
                lhs,
                rhs,
                weak_config,
                cost_fn=weak_cost,
                verbose=False,
            )
            self._backend_timing[backend.timing_key("lp_seconds")] += time.perf_counter() - start

            theta = np.zeros(stress_features.shape[1], dtype=float)
            theta[valid_indices] = theta_valid
            active = np.abs(theta) >= self.config.weak_form.threshold
            residual = _residual_vector_from_operators(residual_operators, theta_valid)
            metrics = regression_metrics(
                np.zeros_like(residual),
                residual,
                num_parameters=int(np.count_nonzero(active)),
            )
            prediction = stress_features @ theta if stress_features.shape[1] == theta.size else np.zeros_like(y)
            self._backend_timing[backend.timing_key("evaluation_seconds")] += time.perf_counter() - evaluation_start
            self._backend_timing[backend.timing_key("evaluations")] += 1.0
            return (
                SparseFitResult(
                    theta=theta,
                    prediction=prediction,
                    active_mask=active,
                    metrics=metrics,
                    column_scales=np.ones_like(theta),
                ),
                valid,
            )

        return evaluate

    def _jax_evaluator(self, stress_builder):
        return self._backend_evaluator(stress_builder)

    def _build_weak_form_cache(self) -> list[_WeakFormDataCache]:
        variable_names = self.config.model.variable_names
        weak_config = self._weak_form_config()
        caches = []
        for data in self.fem_datasets or []:
            dataset = build_stress_dataset_from_fem_data(data)
            variables = invariant_variables(dataset, variable_names)
            caches.append(
                _WeakFormDataCache(
                    data=data,
                    dataset=dataset,
                    variables=variables,
                    X=_variable_matrix(variables, variable_names),
                    dvar_dI=variable_derivatives_wrt_invariants(dataset, variable_names),
                    free_dofs=np.logical_not(zip_dofs(data.dirichlet_nodes)),
                    reactions=tuple((zip_dofs(reaction.dofs), float(reaction.force)) for reaction in data.reactions),
                    B_matrices=np.stack(
                        [assemble_B_matrix(data, element, weak_config) for element in range(data.numElements)]
                    ),
                )
            )
        return caches

    def _weak_form_config(self):
        weak = self.config.weak_form
        return SimpleNamespace(
            dim=2,
            numNodesPerElement=3,
            balance=float(weak.balance),
            penaltyLp=float(weak.penalty_lp),
            penaltyLp_init=float(weak.penalty_lp),
            p=float(weak.p),
            numIncrements=int(weak.num_increments),
            factorIncrements=float(weak.factor_increments),
            numGuesses=int(weak.num_guesses),
            numIterations=int(weak.num_iterations),
            lowestCost=-1.0,
            lowestCostGuessID=-1,
            threshold_iter=float(weak.threshold_iter),
            threshold=float(weak.threshold),
        )

    def _log(self, message: str) -> None:
        if self.config.progress_log:
            print("[SGEPPY] %s" % message, flush=True)

    def _initialize_generation_log(self, model: SGEP):
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        generation_log_path = output_dir / "generation_log.csv"
        best_so_far_path = output_dir / "best_so_far.json"
        generation_log_path.write_text("", encoding="utf-8")
        best_so_far_path.unlink(missing_ok=True)
        self._generation_output_paths = {
            "generation_log_csv": str(generation_log_path),
            "best_so_far_json": str(best_so_far_path),
        }

        def record_generation(row: dict, best_individual) -> None:
            best_fitness = _finite_values_or_none(best_individual.fitness.values)
            best_theta = _finite_values_or_none(getattr(best_individual, "theta", ()))
            best_expression = _reference_normalized_expression(model, best_individual)
            csv_row = dict(row)
            csv_row.update(
                {
                    "best_fitness": json.dumps(best_fitness),
                    "best_expression": best_expression,
                    "best_theta": json.dumps(best_theta),
                }
            )
            _append_csv_row(generation_log_path, csv_row)
            _save_json_atomic(
                best_so_far_path,
                {
                    "generation": row["gen"],
                    "statistics": row,
                    "best_expression": best_expression,
                    "best_theta": best_theta,
                    "best_fitness": best_fitness,
                },
            )

        return record_generation

    def _save_outputs(self, result: SGEPResult) -> dict[str, str]:
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        history_path = output_dir / "history.csv"
        summary_path = output_dir / "summary.json"
        output_paths = {"history_csv": str(history_path), "summary_json": str(summary_path)}
        output_paths.update(self._generation_output_paths)
        output_paths.update(_export_expression_tree(output_dir, result.model.best_individual))
        _save_history_csv(history_path, result.history)
        _save_json(
            summary_path,
            {
                "config": asdict(self.config),
                "dataset": self.dataset.name if self.dataset is not None else None,
                "best_expression": result.best_expression,
                "theta": result.theta.tolist(),
                "metrics": result.metrics,
                "timing": result.timing,
                "history": result.history,
                "output_paths": output_paths,
            },
        )
        return output_paths


def train_sgep(X, y, config: SGEPConfig | None = None) -> SGEP:
    return SGEP(config).fit(X, y)


def stress_feature_builder(
    dataset: StressDataset,
    variable_names: Sequence[str],
    derivative_step: float = 1e-6,
    value_limit: float = 1e8,
    duplicate_correlation: float = 0.999999,
):
    dvar_dF = variable_derivatives_wrt_F(dataset, variable_names)

    def build(model: SGEP, individual, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        variables = {
            name: X[:, index]
            for index, name in enumerate(variable_names)
        }
        base_outputs = model.gene_outputs(individual, X)
        features = np.zeros((dataset.target_vector.size, len(individual)), dtype=float)
        valid = np.zeros(len(individual), dtype=bool)
        normalized_columns = []

        for gene_index, output in enumerate(base_outputs):
            if not _valid(_as_vector(output, dataset.num_points), value_limit):
                continue

            stress = np.zeros_like(dataset.P)
            gene_valid = True
            for name in variable_names:
                base = variables[name]
                step = derivative_step * np.maximum(1.0, np.abs(base))
                plus = dict(variables)
                minus = dict(variables)
                plus[name] = base + step
                minus[name] = base - step
                derivative = (
                    _as_vector(
                        model.gene_outputs(individual, _variable_matrix(plus, variable_names))[gene_index],
                        dataset.num_points,
                    )
                    - _as_vector(
                        model.gene_outputs(individual, _variable_matrix(minus, variable_names))[gene_index],
                        dataset.num_points,
                    )
                ) / (2.0 * step)
                if not _valid(derivative, value_limit):
                    gene_valid = False
                    break
                stress += derivative.reshape(-1, 1) * dvar_dF[name]

            column = stress.reshape(-1)
            norm = np.linalg.norm(column)
            if not gene_valid or not _valid(column, value_limit) or norm < 1e-12:
                continue
            normalized = column / norm
            duplicate = any(abs(float(np.dot(normalized, existing))) >= duplicate_correlation for existing in normalized_columns)
            if duplicate:
                continue
            normalized_columns.append(normalized)
            features[:, gene_index] = column
            valid[gene_index] = True

        if model.config.fit_intercept:
            features = np.column_stack([features, np.zeros(dataset.target_vector.size, dtype=float)])
            valid = np.concatenate([valid, np.array([False], dtype=bool)])
        return features, valid

    return build


def geppy_feature_set_for_fem_data(
    model: SGEP,
    individual,
    data,
    variable_names: Sequence[str],
    gene_indices: Sequence[int],
    derivative_step: float = 1e-6,
    value_limit: float = 1e8,
) -> FeatureSet:
    dataset = build_stress_dataset_from_fem_data(data)
    variables = invariant_variables(dataset, variable_names)
    weak_config = SimpleNamespace(dim=2, numNodesPerElement=3)
    cache = _WeakFormDataCache(
        data=data,
        dataset=dataset,
        variables=variables,
        X=_variable_matrix(variables, variable_names),
        dvar_dI=variable_derivatives_wrt_invariants(dataset, variable_names),
        free_dofs=np.logical_not(zip_dofs(data.dirichlet_nodes)),
        reactions=tuple((zip_dofs(reaction.dofs), float(reaction.force)) for reaction in data.reactions),
        B_matrices=np.stack([assemble_B_matrix(data, element, weak_config) for element in range(data.numElements)]),
    )
    return geppy_feature_set_for_cached_fem_data(
        model,
        individual,
        cache,
        variable_names,
        gene_indices,
        derivative_step=derivative_step,
        value_limit=value_limit,
    )


def geppy_feature_set_for_cached_fem_data(
    model: SGEP,
    individual,
    cache: _WeakFormDataCache,
    variable_names: Sequence[str],
    gene_indices: Sequence[int],
    derivative_step: float = 1e-6,
    value_limit: float = 1e8,
) -> FeatureSet:
    dataset = cache.dataset
    variables = cache.variables
    X = cache.X
    outputs = model.gene_outputs(individual, X)
    perturbed_outputs = {}
    for name in variable_names:
        base = variables[name]
        step = derivative_step * np.maximum(1.0, np.abs(base))
        plus = dict(variables)
        minus = dict(variables)
        plus[name] = base + step
        minus[name] = base - step
        perturbed_outputs[name] = (
            step,
            model.gene_outputs(individual, _variable_matrix(plus, variable_names)),
            model.gene_outputs(individual, _variable_matrix(minus, variable_names)),
        )

    features = []
    dQdI1_columns = []
    dQdI2_columns = []
    dQdI3_columns = []
    for gene_index in gene_indices:
        gene_index = int(gene_index)
        energy = _as_vector(outputs[gene_index], dataset.num_points)
        if not _valid(energy, value_limit):
            raise ValueError("Invalid weak-form gene energy.")

        dQdI1 = np.zeros_like(dataset.I1)
        dQdI2 = np.zeros_like(dataset.I1)
        dQdI3 = np.zeros_like(dataset.I1)
        for name in variable_names:
            step, plus_outputs, minus_outputs = perturbed_outputs[name]
            derivative = (
                _as_vector(plus_outputs[gene_index], dataset.num_points)
                - _as_vector(minus_outputs[gene_index], dataset.num_points)
            ) / (2.0 * step)
            if not _valid(derivative, value_limit):
                raise ValueError("Invalid weak-form gene derivative.")
            dVdI1, dVdI2, dVdI3 = cache.dvar_dI[name]
            dQdI1 += derivative * dVdI1
            dQdI2 += derivative * dVdI2
            dQdI3 += derivative * dVdI3

        if not (_valid(dQdI1, value_limit) and _valid(dQdI2, value_limit) and _valid(dQdI3, value_limit)):
            raise ValueError("Invalid weak-form invariant derivative.")
        features.append(energy)
        dQdI1_columns.append(dQdI1)
        dQdI2_columns.append(dQdI2)
        dQdI3_columns.append(dQdI3)

    return FeatureSet(
        features=np.column_stack(features),
        d_features_dI1=np.column_stack(dQdI1_columns),
        d_features_dI2=np.column_stack(dQdI2_columns),
        d_features_dI3=np.column_stack(dQdI3_columns),
    )


def _compute_cached_weak_lhs(cache: _WeakFormDataCache, feature_set: FeatureSet) -> np.ndarray:
    data = cache.data
    num_features = int(feature_set.features.shape[1])
    lhs = np.zeros((2 * data.numNodes, num_features), dtype=float)
    for element in range(data.numElements):
        dQdF = (
            np.outer(feature_set.d_features_dI1[element, :], data.dI1dF[element, :])
            + np.outer(feature_set.d_features_dI2[element, :], data.dI2dF[element, :])
            + np.outer(feature_set.d_features_dI3[element, :], data.dI3dF[element, :])
        )
        element_lhs = cache.B_matrices[element].T.dot(dQdF.T) * data.qpWeights[element]
        for local_node in range(len(data.connectivity)):
            node = data.connectivity[local_node][element]
            lhs[2 * node, :] += element_lhs[2 * local_node, :]
            lhs[2 * node + 1, :] += element_lhs[2 * local_node + 1, :]
    return lhs


def _cached_reaction_balance(cache: _WeakFormDataCache, weak_lhs: np.ndarray, config) -> tuple[np.ndarray, np.ndarray]:
    balance = float(getattr(config, "balance", 100.0))
    lhs_bulk = weak_lhs[cache.free_dofs, :]
    lhs = 2.0 * lhs_bulk.T.dot(lhs_bulk)
    reaction_lhs = np.zeros_like(lhs)
    reaction_rhs = np.zeros(lhs.shape[0], dtype=float)

    for dofs, force in cache.reactions:
        one = np.ones(weak_lhs[dofs, :].shape[0], dtype=float)
        reaction_sensitivity = weak_lhs[dofs, :].T.dot(one)
        reaction_lhs += 2.0 * np.outer(reaction_sensitivity, reaction_sensitivity)
        reaction_rhs += 2.0 * reaction_sensitivity * force

    return lhs + balance * reaction_lhs, balance * reaction_rhs


def _cached_weak_residual_vector(
    caches: Sequence[_WeakFormDataCache],
    weak_lhs_by_step: Sequence[np.ndarray],
    theta: np.ndarray,
    config,
) -> np.ndarray:
    balance_sqrt = np.sqrt(float(getattr(config, "balance", 100.0)))
    residuals = []
    for cache, weak_lhs in zip(caches, weak_lhs_by_step):
        internal_force = weak_lhs.dot(theta)
        residuals.append(internal_force[cache.free_dofs])
        for dofs, force in cache.reactions:
            residuals.append(np.array([balance_sqrt * (np.sum(internal_force[dofs]) - force)]))
    if not residuals:
        return np.zeros(0, dtype=float)
    return np.concatenate(residuals)


def _residual_vector_from_operators(
    residual_operators: Sequence[tuple[np.ndarray, np.ndarray]],
    theta: np.ndarray,
) -> np.ndarray:
    residuals = [matrix.dot(theta) - target for matrix, target in residual_operators]
    if not residuals:
        return np.zeros(0, dtype=float)
    return np.concatenate(residuals)


def _device_to_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
    return np.asarray(value, dtype=float)


def _empty_jax_timing() -> dict[str, float]:
    return empty_backend_timing("jax")


def _reference_normalized_expression(model: SGEP, individual=None) -> str:
    expression = model.expression(individual)
    offset = _reference_energy_offset(model, individual)
    if not np.isfinite(offset) or abs(offset) < 1e-12:
        return expression
    if expression == "0":
        return "(%0.12g)" % float(-offset)
    return "%s + (%0.12g)" % (expression, float(-offset))


def _reference_energy_offset(model: SGEP, individual=None) -> float:
    individual = individual or model.best_individual
    theta = getattr(individual, "theta", None)
    if individual is None or theta is None:
        return 0.0
    theta = np.asarray(theta, dtype=float)
    active = getattr(
        getattr(individual, "sparse_fit", None),
        "active_mask",
        np.abs(theta) >= model.config.regression_threshold,
    )
    try:
        variables = reference_variables(model.config.variable_names)
        X_ref = _variable_matrix(variables, model.config.variable_names)
        outputs = model.gene_outputs(individual, X_ref)
    except (FloatingPointError, KeyError, ValueError, ZeroDivisionError):
        return 0.0
    n_gene_terms = min(len(individual), theta.size)
    offset = 0.0
    for index in range(n_gene_terms):
        if active[index]:
            value = _as_vector(outputs[index], 1)[0]
            if not np.isfinite(value):
                return 0.0
            offset += float(theta[index]) * float(value)
    intercept_index = len(individual)
    if model.config.fit_intercept and len(theta) > intercept_index and active[intercept_index]:
        offset += float(theta[intercept_index])
    return float(offset)


def _variable_matrix(variables: dict[str, np.ndarray], variable_names: Sequence[str]) -> np.ndarray:
    return np.column_stack([variables[name] for name in variable_names])


def _as_vector(values, n_rows: int) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.ndim == 0:
        return np.full(n_rows, float(values), dtype=float)
    values = values.reshape(-1)
    if values.size != n_rows:
        raise ValueError("Gene output has the wrong number of samples.")
    return values


def _valid(values, limit: float) -> bool:
    values = np.asarray(values, dtype=float)
    return bool(values.size and np.all(np.isfinite(values)) and np.max(np.abs(values)) <= limit)


def _save_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")


def _save_json_atomic(path: Path, payload: dict) -> None:
    temporary_path = path.with_name(path.name + ".tmp")
    temporary_path.write_text(
        json.dumps(_json_safe(payload), indent=2, default=_json_default),
        encoding="utf-8",
    )
    temporary_path.replace(path)


def _append_csv_row(path: Path, row: dict) -> None:
    write_header = path.stat().st_size == 0
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _save_history_csv(path: Path, history: Sequence[dict]) -> None:
    if not history:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


def _export_expression_tree(output_dir: Path, individual) -> dict[str, str]:
    tree_path = output_dir / "expression_tree.png"
    label_map = {
        "linked_add": "+",
        "add": "+",
        "sub": "-",
        "mul": "*",
        "protected_div": "protected_div",
        "neg": "neg",
        "square": "square",
        "cube": "cube",
        "protected_sqrt": "protected_sqrt",
        "protected_log": "protected_log",
        "protected_exp": "protected_exp",
        "sin": "sin",
        "cos": "cos",
    }
    try:
        gep.export_expression_tree(individual, label_map, str(tree_path))
    except Exception as exc:  # pragma: no cover - depends on local graphviz executables
        error_path = output_dir / "expression_tree_error.txt"
        error_path.write_text(
            "Failed to export SGEPPY expression tree with geppy.export_expression_tree(): %s\n" % exc,
            encoding="utf-8",
        )
        return {"expression_tree_error": str(error_path)}
    return {"expression_tree_png": str(tree_path)}


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError("Object of type %s is not JSON serializable" % type(value).__name__)


def _finite_values_or_none(values) -> list[float | None]:
    return [float(value) if np.isfinite(value) else None for value in np.asarray(values, dtype=float).reshape(-1)]


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value
