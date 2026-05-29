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

WEAK_FORM_JAX_TIMING_KEYS = (
    "weak_form_jax_gene_derivative_seconds",
    "weak_form_jax_weak_lhs_seconds",
    "weak_form_jax_transfer_seconds",
    "weak_form_jax_lp_seconds",
    "weak_form_jax_evaluation_seconds",
    "weak_form_jax_evaluations",
    "weak_form_jax_cache_hits",
    "weak_form_jax_cache_misses",
    "weak_form_jax_compile_cache_hits",
    "weak_form_jax_compile_cache_misses",
    "weak_form_jax_cache_entries",
    "weak_form_jax_gene_compile_seconds",
    "weak_form_jax_gene_execute_seconds",
)


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
    fitting_mode: str = "direct_stress"
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
    jax_precision: str = "float64"
    jax_cache_enabled: bool = True
    jax_cache_size: int = 256
    jax_cache_device_outputs: bool = True

    def __post_init__(self) -> None:
        if self.fitting_mode not in {"direct_stress", "weak_form", "weak_form_jax"}:
            raise ValueError("fitting_mode must be one of: direct_stress, weak_form, weak_form_jax.")
        if self.jax_precision not in {"float64", "float32"}:
            raise ValueError("jax_precision must be one of: float64, float32.")
        self.jax_cache_size = int(self.jax_cache_size)
        if self.jax_cache_size < 0:
            raise ValueError("jax_cache_size must be non-negative.")


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
        self._weak_form_jax_timing = _empty_weak_form_jax_timing()
        self._weak_form_jax_eval_cache = None

    def train(self) -> SGEPResult:
        wall_start = time.perf_counter()
        cpu_start = time.process_time()
        if self.config.fitting_mode in {"weak_form", "weak_form_jax"} and self.config.data_dir is None:
            raise ValueError("fitting_mode='%s' requires data_dir." % self.config.fitting_mode)
        if self.config.fitting_mode == "weak_form_jax":
            from .jax_backend import JaxWeakFormEvaluationCache, configure_jax_precision

            configure_jax_precision(self.config.jax_precision)
            self._weak_form_jax_eval_cache = JaxWeakFormEvaluationCache(
                enabled=self.config.jax_cache_enabled,
                max_size=self.config.jax_cache_size,
                device_outputs=self.config.jax_cache_device_outputs,
                timing=self._weak_form_jax_timing,
            )
        self.dataset = self._load_dataset()
        self.fem_datasets = self._load_fem_datasets() if self.config.fitting_mode in {"weak_form", "weak_form_jax"} else None
        self.weak_form_cache = self._build_weak_form_cache() if self.fem_datasets is not None else None
        variables = invariant_variables(self.dataset, self.config.model.variable_names)
        if self.config.fitting_mode == "weak_form_jax":
            from .jax_backend import stress_feature_builder as jax_stress_feature_builder

            builder = jax_stress_feature_builder(
                self.dataset,
                self.config.model.variable_names,
                value_limit=self.config.invalid_value_limit,
                duplicate_correlation=self.config.duplicate_correlation,
                precision=self.config.jax_precision,
                timing=self._weak_form_jax_timing,
                cache=self._weak_form_jax_eval_cache,
            )
        else:
            builder = stress_feature_builder(
                self.dataset,
                self.config.model.variable_names,
                derivative_step=self.config.derivative_step,
                value_limit=self.config.invalid_value_limit,
                duplicate_correlation=self.config.duplicate_correlation,
            )
        if self.config.fitting_mode == "weak_form_jax":
            evaluator = self._weak_form_jax_evaluator(builder)
        else:
            evaluator = self._weak_form_evaluator(builder) if self.config.fitting_mode == "weak_form" else None
        self.model = SGEP(self.config.model).fit(
            variables,
            self.dataset.target_vector,
            feature_builder=builder,
            evaluator=evaluator,
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
        if self.config.fitting_mode == "weak_form_jax":
            timing.update(self._weak_form_jax_timing)
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
            self._log("Loading direct-stress data from %s." % self.config.data_dir)
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

    def _weak_form_evaluator(self, stress_builder):
        if self.weak_form_cache is None and self.fem_datasets is not None:
            self.weak_form_cache = self._build_weak_form_cache()
        weak_caches = self.weak_form_cache or []
        fem_datasets = [cache.data for cache in weak_caches]
        variable_names = self.config.model.variable_names

        def evaluate(model: SGEP, individual, X: np.ndarray, y: np.ndarray):
            stress_features, valid = stress_builder(model, individual, X)
            gene_valid = np.asarray(valid[: len(individual)], dtype=bool)
            valid_indices = np.flatnonzero(gene_valid)
            if valid_indices.size == 0:
                raise ValueError("No valid weak-form genes.")

            weak_lhs_by_step = []
            weak_config = self._weak_form_config()
            lhs = np.zeros((valid_indices.size, valid_indices.size), dtype=float)
            rhs = np.zeros(valid_indices.size, dtype=float)
            for cache in weak_caches:
                feature_set = geppy_feature_set_for_cached_fem_data(
                    model,
                    individual,
                    cache,
                    variable_names,
                    valid_indices,
                    derivative_step=self.config.derivative_step,
                    value_limit=self.config.invalid_value_limit,
                )
                cache.data.featureSet = feature_set
                weak_lhs = _compute_cached_weak_lhs(cache, feature_set)
                step_lhs, step_rhs = _cached_reaction_balance(cache, weak_lhs, weak_config)
                lhs += step_lhs
                rhs += step_rhs
                weak_lhs_by_step.append(weak_lhs)

            theta_valid = apply_penalty_lp_iteration(
                fem_datasets,
                lhs,
                rhs,
                weak_config,
                verbose=False,
            )

            theta = np.zeros(stress_features.shape[1], dtype=float)
            theta[valid_indices] = theta_valid
            active = np.abs(theta) >= self.config.weak_form.threshold
            residual = _cached_weak_residual_vector(weak_caches, weak_lhs_by_step, theta_valid, weak_config)
            metrics = regression_metrics(
                np.zeros_like(residual),
                residual,
                num_parameters=int(np.count_nonzero(active)),
            )
            prediction = stress_features @ theta if stress_features.shape[1] == theta.size else np.zeros_like(y)
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

    def _weak_form_jax_evaluator(self, stress_builder):
        if self.weak_form_cache is None and self.fem_datasets is not None:
            self.weak_form_cache = self._build_weak_form_cache()
        weak_caches = self.weak_form_cache or []
        fem_datasets = [cache.data for cache in weak_caches]
        variable_names = self.config.model.variable_names

        from .jax_backend import (
            JaxWeakFormEvaluationCache,
            block_until_ready,
            compute_reaction_balance_device,
            compute_residual_operator_device,
            compute_weak_lhs_device,
            feature_values_and_dqdf_device,
            prepare_jax_case,
        )

        if self._weak_form_jax_eval_cache is None:
            self._weak_form_jax_eval_cache = JaxWeakFormEvaluationCache(
                enabled=self.config.jax_cache_enabled,
                max_size=self.config.jax_cache_size,
                device_outputs=self.config.jax_cache_device_outputs,
                timing=self._weak_form_jax_timing,
            )
        eval_cache = self._weak_form_jax_eval_cache
        jax_cases = [prepare_jax_case(cache, precision=self.config.jax_precision) for cache in weak_caches]

        def evaluate(model: SGEP, individual, X: np.ndarray, y: np.ndarray):
            evaluation_start = time.perf_counter()
            stress_features, valid = stress_builder(model, individual, X)
            gene_valid = np.asarray(valid[: len(individual)], dtype=bool)
            valid_indices = np.flatnonzero(gene_valid)
            if valid_indices.size == 0:
                raise ValueError("No valid JAX weak-form genes.")

            residual_operators = []
            weak_config = self._weak_form_config()
            lhs = np.zeros((valid_indices.size, valid_indices.size), dtype=float)
            rhs = np.zeros(valid_indices.size, dtype=float)
            balance = float(getattr(weak_config, "balance", 100.0))
            for case_index, jax_case in enumerate(jax_cases):
                case_key = ("loadstep", case_index, id(jax_case.data), tuple(getattr(jax_case.F, "shape", ())))
                artifact_key = eval_cache.weak_artifact_key(
                    case_key,
                    model,
                    individual,
                    valid_indices,
                    variable_names,
                    self.config.jax_precision,
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
                _, dqdf = feature_values_and_dqdf_device(
                    model,
                    individual,
                    jax_case.F,
                    variable_names,
                    gene_indices=valid_indices,
                    value_limit=self.config.invalid_value_limit,
                    precision=self.config.jax_precision,
                    cache=eval_cache,
                    data_key=("weak", case_index, id(jax_case.data), tuple(getattr(jax_case.F, "shape", ()))),
                )
                block_until_ready(dqdf)
                self._weak_form_jax_timing["weak_form_jax_gene_derivative_seconds"] += time.perf_counter() - start

                start = time.perf_counter()
                weak_lhs = compute_weak_lhs_device(jax_case, dqdf)
                step_lhs_device, step_rhs_device = compute_reaction_balance_device(jax_case, weak_lhs, balance)
                residual_matrix_device, residual_target_device = compute_residual_operator_device(jax_case, weak_lhs, balance)
                block_until_ready((step_lhs_device, step_rhs_device, residual_matrix_device, residual_target_device))
                self._weak_form_jax_timing["weak_form_jax_weak_lhs_seconds"] += time.perf_counter() - start

                start = time.perf_counter()
                step_lhs = np.asarray(step_lhs_device, dtype=float)
                step_rhs = np.asarray(step_rhs_device, dtype=float)
                residual_operators.append(
                    (
                        np.asarray(residual_matrix_device, dtype=float),
                        np.asarray(residual_target_device, dtype=float),
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
                self._weak_form_jax_timing["weak_form_jax_transfer_seconds"] += time.perf_counter() - start
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
            self._weak_form_jax_timing["weak_form_jax_lp_seconds"] += time.perf_counter() - start

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
            self._weak_form_jax_timing["weak_form_jax_evaluation_seconds"] += time.perf_counter() - evaluation_start
            self._weak_form_jax_timing["weak_form_jax_evaluations"] += 1.0
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

    def _save_outputs(self, result: SGEPResult) -> dict[str, str]:
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        history_path = output_dir / "history.csv"
        summary_path = output_dir / "summary.json"
        output_paths = {"history_csv": str(history_path), "summary_json": str(summary_path)}
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


def _empty_weak_form_jax_timing() -> dict[str, float]:
    return {key: 0.0 for key in WEAK_FORM_JAX_TIMING_KEYS}


def _reference_normalized_expression(model: SGEP) -> str:
    expression = model.expression()
    offset = _reference_energy_offset(model)
    if not np.isfinite(offset) or abs(offset) < 1e-12:
        return expression
    if expression == "0":
        return "(%0.12g)" % float(-offset)
    return "%s + (%0.12g)" % (expression, float(-offset))


def _reference_energy_offset(model: SGEP) -> float:
    individual = model.best_individual
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
        "div": "/",
        "neg": "neg",
        "square": "square",
        "sqrt": "sqrt",
        "log": "log",
        "exp": "exp",
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
