from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence
from types import SimpleNamespace
from time import perf_counter

import numpy as np

from sym_modeling.common.workflow import BaseTrainer
from sym_modeling.domains.fem.methods.common.lp_solver import apply_penalty_lp_iteration
from sym_modeling.domains.fem.methods.common.weak_form import (
    compute_strain_energy,
    extend_lhs_rhs,
    weak_residual_vector,
)
from sym_modeling.domains.fem.methods.euclid import EuclidConfig, EuclidWorkflow
from sym_modeling.domains.fem.methods.euclid.constraints import getPathX
from sym_modeling.domains.fem.methods.euclid.weak_form import computeFirstPiolaTheta
from sym_modeling.domains.fem.methods.sgep.aic import RegressionMetrics, regression_metrics
from sym_modeling.domains.fem.methods.sgep.gep_gene import BINARY_OPS, UNARY_OPS, Gene
from sym_modeling.domains.fem.methods.sgep.gep_population import (
    CandidateModel,
    CandidateRecord,
    PopulationEngine,
    selection_rank,
)
from sym_modeling.domains.fem.methods.sgep.hyperelastic_eval import (
    StressDataset,
    assemble_stress_feature_matrix,
    build_gene_feature_set_for_fem_data,
    build_stress_dataset_from_F,
    gene_feature_values_and_derivatives,
    load_stress_dataset_from_euclid_csv,
    resolve_loadsteps,
    synthetic_neo_hookean_dataset,
)
from sym_modeling.domains.fem.methods.sgep.postprocessing import (
    plot_history,
    plot_predicted_vs_true,
    plot_sgep_piola_field_comparisons,
    save_history_csv,
    save_json,
)
from sym_modeling.domains.fem.methods.sgep.sparse_fit import (
    SparseFitResult,
    fit_sparse_regression,
)
from sym_modeling.domains.fem.io.csv_loader import loadFemData


@dataclass
class SGEPConfig:
    genes_per_model: int = 4
    num_models: int = 12
    generations: int = 6
    max_depth: int = 3
    variable_names: tuple[str, ...] = ("K1", "K2", "Jm1")
    unary_operators: tuple[str, ...] = UNARY_OPS
    binary_operators: tuple[str, ...] = BINARY_OPS
    random_seed: int = 0

    mutation_rate: float = 0.55
    crossover_rate: float = 0.30
    random_fraction: float = 0.15
    elite_model_count: int = 1
    elite_gene_count: int = 4

    regression_method: str = "stlsq"
    regression_alpha: float = 1e-10
    sparsity_threshold: float = 1e-8
    refit_active_terms: bool = True
    derivative_step: float = 1e-6
    invalid_value_limit: float = 1e8
    duplicate_correlation: float = 0.999999
    fitting_mode: str = "auto"
    selection_objective: str = "epsilon_rmse"
    active_terms_epsilon: int = 2

    weak_form_balance: float = 100.0
    weak_form_penalty_lp: float = 1e-4
    weak_form_p: float = 1.0 / 4.0
    weak_form_num_increments: int = 5
    weak_form_factor_increments: float = 5.0
    weak_form_num_guesses: int = 1
    weak_form_num_iterations: int = 200
    weak_form_threshold_iter: float = 1e-6
    weak_form_threshold: float = 1e-2
    weak_form_energy_checks: bool = True

    data_dir: str | None = None
    loadsteps: list[int] | None = None
    noise_level: float = 0.0
    max_elements_per_loadstep: int | None = 600
    synthetic_samples: int = 200
    synthetic_mu: float = 1.0
    synthetic_bulk: float = 10.0

    output_dir: str = "output/sgep_results"
    save_plots: bool = True
    progress_log: bool = True
    compare_euclid: bool = False
    euclid_model: str = "NH2"

    def __post_init__(self) -> None:
        self.variable_names = tuple(self.variable_names)
        self.unary_operators = tuple(self.unary_operators)
        self.binary_operators = tuple(self.binary_operators)
        unsupported_unary = sorted(set(self.unary_operators) - set(UNARY_OPS))
        unsupported_binary = sorted(set(self.binary_operators) - set(BINARY_OPS))
        if unsupported_unary:
            raise ValueError("Unsupported unary operators: %s" % ", ".join(unsupported_unary))
        if unsupported_binary:
            raise ValueError("Unsupported binary operators: %s" % ", ".join(unsupported_binary))
        if not self.unary_operators and not self.binary_operators:
            raise ValueError("At least one unary or binary GEP operator must be enabled.")
        self.fitting_mode = str(self.fitting_mode).lower()
        if self.fitting_mode not in {"auto", "weak_form", "direct_stress"}:
            raise ValueError("fitting_mode must be one of: auto, weak_form, direct_stress.")
        self.selection_objective = str(self.selection_objective).lower()
        if self.selection_objective not in {"aicc", "epsilon_rmse"}:
            raise ValueError("selection_objective must be one of: aicc, epsilon_rmse.")
        self.active_terms_epsilon = int(self.active_terms_epsilon)
        if self.active_terms_epsilon < 1:
            raise ValueError("active_terms_epsilon must be at least 1.")
        if self.max_elements_per_loadstep is not None:
            max_elements = int(self.max_elements_per_loadstep)
            self.max_elements_per_loadstep = max_elements if max_elements > 0 else None


@dataclass
class SGEPResult:
    best_expression: str
    theta: np.ndarray
    genes: tuple[Gene, ...]
    metrics: dict
    history: list[dict]
    selected_gene_log: list[dict]
    output_paths: dict[str, str]
    piola_field_plots: dict[str, list[str]] | None = None
    piola_field_metrics: list[dict] | None = None
    piola_field_warnings: list[str] | None = None
    euclid_metrics: dict | None = None


@dataclass
class _Evaluation:
    candidate: CandidateModel
    fit: SparseFitResult
    theta_full: np.ndarray
    valid_mask: np.ndarray
    valid_reasons: list[str]
    gene_fitness: np.ndarray
    expression: str


class SGEPWorkflow(BaseTrainer):
    """Hybrid GEP + sparse-regression prototype for hyperelastic stress discovery."""

    def __init__(self, config: SGEPConfig | None = None):
        self.config = config or SGEPConfig()
        self.dataset: StressDataset | None = None
        self.fem_datasets: list | None = None
        self.result: SGEPResult | None = None

    def _load_dataset(self) -> StressDataset:
        fitting_mode = self._resolved_fitting_mode()
        if self.config.data_dir is not None:
            if fitting_mode == "direct_stress":
                self.fem_datasets = None
                return load_stress_dataset_from_euclid_csv(
                    self.config.data_dir,
                    loadsteps=self.config.loadsteps,
                    max_elements_per_loadstep=self.config.max_elements_per_loadstep,
                    noise_level=self.config.noise_level,
                )
            data_path = Path(self.config.data_dir)
            steps = list(self.config.loadsteps) if self.config.loadsteps is not None else resolve_loadsteps(data_path)
            self.fem_datasets = []
            F_parts = []
            P_parts = []
            for step in steps:
                data = loadFemData(
                    str(data_path / str(step)),
                    AD=True,
                    noiseLevel=self.config.noise_level,
                    noiseType="displacement",
                )
                data.convertToNumpy()
                self.fem_datasets.append(data)
                F_parts.append(data.F)
                if data.P is None:
                    P_parts.append(np.zeros((data.F.shape[0], 4), dtype=float))
                else:
                    P_parts.append(data.P)
            return build_stress_dataset_from_F(
                np.vstack(F_parts),
                np.vstack(P_parts),
                name=str(data_path),
            )
        self.fem_datasets = None
        return synthetic_neo_hookean_dataset(
            num_samples=self.config.synthetic_samples,
            seed=self.config.random_seed,
            mu=self.config.synthetic_mu,
            bulk=self.config.synthetic_bulk,
        )

    def _fit_candidate(self, candidate: CandidateModel, dataset: StressDataset) -> _Evaluation:
        if self._resolved_fitting_mode() == "weak_form":
            return self._fit_candidate_weak(candidate, dataset)
        return self._fit_candidate_direct(candidate, dataset)

    def _fit_candidate_direct(self, candidate: CandidateModel, dataset: StressDataset) -> _Evaluation:
        X_valid, valid_mask, reasons = assemble_stress_feature_matrix(
            candidate.genes,
            dataset,
            self.config.variable_names,
            derivative_step=self.config.derivative_step,
            value_limit=self.config.invalid_value_limit,
            duplicate_correlation=self.config.duplicate_correlation,
        )
        fit = fit_sparse_regression(
            X_valid,
            dataset.target_vector,
            method=self.config.regression_method,
            alpha=self.config.regression_alpha,
            threshold=self.config.sparsity_threshold,
            refit=self.config.refit_active_terms,
        )
        theta_full = np.zeros(len(candidate.genes), dtype=float)
        theta_full[valid_mask] = fit.theta
        gene_fitness = self._ablation_fitness(X_valid, fit, dataset.target_vector, valid_mask)
        expression = candidate.expression(theta_full)
        return _Evaluation(
            candidate=candidate,
            fit=fit,
            theta_full=theta_full,
            valid_mask=valid_mask,
            valid_reasons=reasons,
            gene_fitness=gene_fitness,
            expression=expression,
        )

    def _fit_candidate_weak(self, candidate: CandidateModel, dataset: StressDataset) -> _Evaluation:
        X_valid, valid_mask, reasons = assemble_stress_feature_matrix(
            candidate.genes,
            dataset,
            self.config.variable_names,
            derivative_step=self.config.derivative_step,
            value_limit=self.config.invalid_value_limit,
            duplicate_correlation=self.config.duplicate_correlation,
        )
        valid_genes = tuple(gene for gene, valid in zip(candidate.genes, valid_mask) if valid)
        weak_config = self._weak_form_config()
        try:
            self._attach_sgep_feature_sets(valid_genes)
            lhs, rhs = self._assemble_weak_system(len(valid_genes), weak_config)
            theta_valid = apply_penalty_lp_iteration(
                self.fem_datasets or [],
                lhs,
                rhs,
                weak_config,
                energy_check=self._sgep_energy_check(valid_genes),
                verbose=False,
            )
        except (ValueError, np.linalg.LinAlgError, FloatingPointError):
            return self._failed_candidate(candidate, dataset, valid_mask, reasons)

        residual = weak_residual_vector(self.fem_datasets or [], theta_valid, weak_config)
        active = np.abs(theta_valid) >= self.config.weak_form_threshold
        metrics = regression_metrics(
            np.zeros_like(residual),
            residual,
            num_parameters=int(np.count_nonzero(active)),
        )
        prediction = X_valid @ theta_valid if X_valid.shape[1] else np.zeros_like(dataset.target_vector)
        fit = SparseFitResult(
            theta=theta_valid,
            prediction=prediction,
            active_mask=active,
            metrics=metrics,
            column_scales=np.ones_like(theta_valid),
        )
        theta_full = np.zeros(len(candidate.genes), dtype=float)
        theta_full[valid_mask] = theta_valid
        gene_fitness = self._weak_ablation_fitness(theta_valid, metrics, valid_mask, weak_config)
        expression = candidate.expression(theta_full)
        return _Evaluation(
            candidate=candidate,
            fit=fit,
            theta_full=theta_full,
            valid_mask=valid_mask,
            valid_reasons=reasons,
            gene_fitness=gene_fitness,
            expression=expression,
        )

    def _failed_candidate(
        self,
        candidate: CandidateModel,
        dataset: StressDataset,
        valid_mask: np.ndarray,
        reasons: list[str],
    ) -> _Evaluation:
        metrics = RegressionMetrics(
            rss=float("inf"),
            rmse=float("inf"),
            aic=float("inf"),
            aicc=float("inf"),
            num_samples=dataset.target_vector.size,
            num_parameters=0,
        )
        theta_valid = np.zeros(int(np.count_nonzero(valid_mask)), dtype=float)
        fit = SparseFitResult(
            theta=theta_valid,
            prediction=np.zeros_like(dataset.target_vector),
            active_mask=np.zeros_like(theta_valid, dtype=bool),
            metrics=metrics,
            column_scales=np.ones_like(theta_valid),
        )
        return _Evaluation(
            candidate=candidate,
            fit=fit,
            theta_full=np.zeros(len(candidate.genes), dtype=float),
            valid_mask=valid_mask,
            valid_reasons=reasons,
            gene_fitness=np.zeros(len(candidate.genes), dtype=float),
            expression="0",
        )

    def _weak_form_config(self):
        return SimpleNamespace(
            dim=2,
            numNodesPerElement=3,
            balance=float(self.config.weak_form_balance),
            penaltyLp=float(self.config.weak_form_penalty_lp),
            penaltyLp_init=float(self.config.weak_form_penalty_lp),
            p=float(self.config.weak_form_p),
            numIncrements=int(self.config.weak_form_num_increments),
            factorIncrements=float(self.config.weak_form_factor_increments),
            numGuesses=int(self.config.weak_form_num_guesses),
            numIterations=int(self.config.weak_form_num_iterations),
            lowestCost=-1.0,
            lowestCostGuessID=-1,
            threshold_iter=float(self.config.weak_form_threshold_iter),
            threshold=float(self.config.weak_form_threshold),
        )

    def _attach_sgep_feature_sets(self, genes: Sequence[Gene]) -> None:
        for data in self.fem_datasets or []:
            data.featureSet = build_gene_feature_set_for_fem_data(
                data,
                genes,
                self.config.variable_names,
                derivative_step=self.config.derivative_step,
                value_limit=self.config.invalid_value_limit,
            )

    def _assemble_weak_system(self, num_features: int, weak_config) -> tuple[np.ndarray, np.ndarray]:
        lhs = np.zeros((num_features, num_features), dtype=float)
        rhs = np.zeros(num_features, dtype=float)
        for data in self.fem_datasets or []:
            lhs, rhs = extend_lhs_rhs(data, weak_config, lhs, rhs, verbose=False)
        return lhs, rhs

    def _sgep_energy_check(self, genes: Sequence[Gene]):
        def check(datasets, theta) -> bool:
            if not self.config.weak_form_energy_checks:
                return True
            for data in datasets:
                energy, _ = compute_strain_energy(data, theta)
                if energy.size and float(np.min(energy)) < -1e-10:
                    return False

            for path in ("tension", "bitension", "compression", "bicompression", "simpleShear", "pureShear"):
                previous = 0.0
                for x_value in getPathX():
                    F = self._deformation_path_F(path, x_value)
                    synthetic = build_stress_dataset_from_F(F, np.zeros((1, 4), dtype=float), name=path)
                    features, _, _, _ = gene_feature_values_and_derivatives(
                        genes,
                        synthetic,
                        self.config.variable_names,
                        derivative_step=self.config.derivative_step,
                        value_limit=self.config.invalid_value_limit,
                    )
                    energy = float(features[0, :].dot(theta)) if features.shape[1] else 0.0
                    if not np.isfinite(energy):
                        break
                    if energy < previous - 1e-10:
                        return False
                    previous = energy
            return True

        return check

    @staticmethod
    def _deformation_path_F(path: str, x_value: float) -> np.ndarray:
        F = np.zeros((1, 4), dtype=float)
        if path == "tension":
            F[:, 0] = x_value
            F[:, 3] = 1.0
        elif path == "bitension":
            F[:, 0] = x_value
            F[:, 3] = x_value
        elif path == "compression":
            F[:, 0] = 1.0 / x_value
            F[:, 3] = 1.0
        elif path == "bicompression":
            F[:, 0] = 1.0 / x_value
            F[:, 3] = 1.0 / x_value
        elif path == "simpleShear":
            F[:, 0] = 1.0
            F[:, 1] = x_value - 1.0
            F[:, 3] = 1.0
        elif path == "pureShear":
            F[:, 0] = x_value
            F[:, 3] = 1.0 / x_value
        else:
            F[:, 0] = 1.0
            F[:, 3] = 1.0
        return F

    def _ablation_fitness(
        self,
        X_valid: np.ndarray,
        fit: SparseFitResult,
        y: np.ndarray,
        valid_mask: np.ndarray,
    ) -> np.ndarray:
        fitness = np.zeros(valid_mask.shape[0], dtype=float)
        if X_valid.shape[1] == 0:
            return fitness

        valid_indices = np.flatnonzero(valid_mask)
        for local_idx, gene_idx in enumerate(valid_indices):
            if not fit.active_mask[local_idx]:
                continue
            keep = np.ones(X_valid.shape[1], dtype=bool)
            keep[local_idx] = False
            ablated = fit_sparse_regression(
                X_valid[:, keep],
                y,
                method=self.config.regression_method,
                alpha=self.config.regression_alpha,
                threshold=self.config.sparsity_threshold,
                refit=self.config.refit_active_terms,
            )
            delta = ablated.metrics.aicc - fit.metrics.aicc
            fitness[gene_idx] = max(0.0, float(delta))
        return fitness

    def _weak_ablation_fitness(
        self,
        theta_valid: np.ndarray,
        base_metrics: RegressionMetrics,
        valid_mask: np.ndarray,
        weak_config,
    ) -> np.ndarray:
        fitness = np.zeros(valid_mask.shape[0], dtype=float)
        valid_indices = np.flatnonzero(valid_mask)
        if theta_valid.size == 0 or not np.isfinite(base_metrics.aicc):
            return fitness
        for local_idx, gene_idx in enumerate(valid_indices):
            if abs(float(theta_valid[local_idx])) < self.config.weak_form_threshold:
                continue
            ablated_theta = np.copy(theta_valid)
            ablated_theta[local_idx] = 0.0
            residual = weak_residual_vector(self.fem_datasets or [], ablated_theta, weak_config)
            metrics = regression_metrics(
                np.zeros_like(residual),
                residual,
                num_parameters=max(0, int(np.count_nonzero(theta_valid)) - 1),
            )
            if np.isfinite(metrics.aicc):
                fitness[gene_idx] = max(0.0, float(metrics.aicc - base_metrics.aicc))
        return fitness

    def train(self) -> SGEPResult:
        dataset = self._load_dataset()
        self.dataset = dataset
        engine = PopulationEngine(
            variable_names=self.config.variable_names,
            genes_per_model=self.config.genes_per_model,
            num_models=self.config.num_models,
            max_depth=self.config.max_depth,
            random_seed=self.config.random_seed,
            mutation_rate=self.config.mutation_rate,
            crossover_rate=self.config.crossover_rate,
            random_fraction=self.config.random_fraction,
            elite_model_count=self.config.elite_model_count,
            elite_gene_count=self.config.elite_gene_count,
            unary_operators=self.config.unary_operators,
            binary_operators=self.config.binary_operators,
            selection_objective=self.config.selection_objective,
            active_terms_epsilon=self.config.active_terms_epsilon,
        )

        population = engine.initial_population()
        best: _Evaluation | None = None
        history: list[dict] = []
        selected_gene_log: list[dict] = []
        train_start = perf_counter()

        for generation in range(self.config.generations):
            generation_start = perf_counter()
            candidates = engine.group_candidates(population)
            self._log_progress(
                "Generation %d/%d: evaluating %d candidate models with %d genes each (%s)."
                % (
                    generation + 1,
                    self.config.generations,
                    len(candidates),
                    self.config.genes_per_model,
                    self._fitting_mode(),
                )
            )
            evaluations = [self._fit_candidate(candidate, dataset) for candidate in candidates]
            evaluations.sort(key=self._evaluation_rank)
            generation_best = evaluations[0]
            if best is None or self._evaluation_rank(generation_best) < self._evaluation_rank(best):
                best = generation_best
            generation_elapsed = perf_counter() - generation_start
            total_elapsed = perf_counter() - train_start
            generation_best_active_terms = self._active_terms(generation_best)

            history.append(
                {
                    "generation": generation,
                    "fitting_mode": self._fitting_mode(),
                    "selection_objective": self.config.selection_objective,
                    "active_terms_epsilon": self.config.active_terms_epsilon,
                    "best_feasible": self._is_epsilon_feasible(generation_best),
                    "elapsed_seconds": generation_elapsed,
                    "best_rmse": generation_best.fit.metrics.rmse,
                    "best_rss": generation_best.fit.metrics.rss,
                    "best_aicc": generation_best.fit.metrics.aicc,
                    "best_active_terms": generation_best_active_terms,
                    "best_expression": generation_best.expression,
                }
            )
            self._log_progress(
                "Generation %d/%d done in %.1fs (total %.1fs): RMSE %.6e, AICc %.6e, active terms %d."
                % (
                    generation + 1,
                    self.config.generations,
                    generation_elapsed,
                    total_elapsed,
                    generation_best.fit.metrics.rmse,
                    generation_best.fit.metrics.aicc,
                    generation_best_active_terms,
                )
            )
            self._log_progress("  Best: %s" % generation_best.expression)
            selected_gene_log.extend(self._generation_gene_log(generation, evaluations))

            records = [
                CandidateRecord(
                    candidate=evaluation.candidate,
                    aicc=evaluation.fit.metrics.aicc,
                    rmse=evaluation.fit.metrics.rmse,
                    active_terms=self._active_terms(evaluation),
                    complexity=self._active_complexity(evaluation),
                    gene_fitness=evaluation.gene_fitness,
                )
                for evaluation in evaluations
            ]
            if generation < self.config.generations - 1:
                population = engine.next_population(records)

        if best is None:
            raise RuntimeError("SGEP training produced no candidate evaluations.")

        output_paths = self._save_outputs(best, dataset, history, selected_gene_log)
        piola_field_plots = output_paths.pop("_piola_field_plots", None)
        piola_field_metrics = output_paths.pop("_piola_field_metrics", None)
        piola_field_warnings = output_paths.pop("_piola_field_warnings", None)
        euclid_metrics = self._compare_euclid() if self.config.compare_euclid else None
        metrics = self._metrics_dict(best.fit)
        metrics["fitting_mode"] = self._fitting_mode()
        self.result = SGEPResult(
            best_expression=best.expression,
            theta=best.theta_full,
            genes=best.candidate.genes,
            metrics=metrics,
            history=history,
            selected_gene_log=selected_gene_log,
            output_paths=output_paths,
            piola_field_plots=piola_field_plots,
            piola_field_metrics=piola_field_metrics,
            piola_field_warnings=piola_field_warnings,
            euclid_metrics=euclid_metrics,
        )
        self._save_summary(self.result)
        return self.result

    def _fitting_mode(self) -> str:
        return self._resolved_fitting_mode()

    def _resolved_fitting_mode(self) -> str:
        if self.config.fitting_mode == "auto":
            return "weak_form" if self.config.data_dir is not None else "direct_stress"
        if self.config.fitting_mode == "weak_form" and self.config.data_dir is None:
            raise ValueError("fitting_mode='weak_form' requires data_dir.")
        return self.config.fitting_mode

    def _log_progress(self, message: str) -> None:
        if self.config.progress_log:
            print("[SGEP] %s" % message, flush=True)

    def _evaluation_rank(self, evaluation: _Evaluation) -> tuple:
        return selection_rank(
            self.config.selection_objective,
            evaluation.fit.metrics.aicc,
            evaluation.fit.metrics.rmse,
            self._active_terms(evaluation),
            self._active_complexity(evaluation),
            self.config.active_terms_epsilon,
        )

    @staticmethod
    def _active_terms(evaluation: _Evaluation) -> int:
        return int(evaluation.fit.metrics.num_parameters)

    @staticmethod
    def _active_complexity(evaluation: _Evaluation) -> int:
        valid_indices = np.flatnonzero(evaluation.valid_mask)
        if evaluation.fit.active_mask.size == 0:
            return 0
        return int(
            sum(
                evaluation.candidate.genes[gene_idx].complexity
                for local_idx, gene_idx in enumerate(valid_indices)
                if local_idx < evaluation.fit.active_mask.size and bool(evaluation.fit.active_mask[local_idx])
            )
        )

    def _is_epsilon_feasible(self, evaluation: _Evaluation) -> bool:
        return self._active_terms(evaluation) <= self.config.active_terms_epsilon

    def _generation_gene_log(
        self,
        generation: int,
        evaluations: Sequence[_Evaluation],
    ) -> list[dict]:
        rows = []
        for rank, evaluation in enumerate(evaluations[: min(3, len(evaluations))]):
            for idx, gene in enumerate(evaluation.candidate.genes):
                rows.append(
                    {
                        "generation": generation,
                        "candidate_rank": rank,
                        "gene_index": idx,
                        "expression": gene.expression(),
                        "coefficient": float(evaluation.theta_full[idx]),
                        "fitness": float(evaluation.gene_fitness[idx]),
                        "valid": bool(evaluation.valid_mask[idx]),
                        "reason": evaluation.valid_reasons[idx],
                        "complexity": gene.complexity,
                    }
                )
        return rows

    def _save_outputs(
        self,
        best: _Evaluation,
        dataset: StressDataset,
        history: list[dict],
        selected_gene_log: list[dict],
    ) -> dict[str, str]:
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_paths: dict[str, str] = {}
        history_path = output_dir / "history.csv"
        genes_path = output_dir / "selected_genes.json"
        save_history_csv(history_path, history)
        save_json(genes_path, {"selected_genes": selected_gene_log})
        output_paths["history_csv"] = str(history_path)
        output_paths["selected_genes_json"] = str(genes_path)
        if self.config.save_plots:
            y = dataset.target_vector
            output_paths["predicted_vs_true_plot"] = plot_predicted_vs_true(
                y,
                best.fit.prediction,
                output_dir / "predicted_vs_true.png",
            )
            output_paths["history_plot"] = plot_history(history, output_dir / "history.png")
            if self.config.data_dir is not None:
                plot_paths, metrics, warnings = plot_sgep_piola_field_comparisons(
                    data_dir=self.config.data_dir,
                    loadsteps=self.config.loadsteps,
                    genes=best.candidate.genes,
                    theta=best.theta_full,
                    config=self.config,
                    output_dir=output_dir / "piola_fields",
                )
                output_paths["_piola_field_plots"] = plot_paths
                output_paths["_piola_field_metrics"] = metrics
                output_paths["_piola_field_warnings"] = warnings
        return output_paths

    def _save_summary(self, result: SGEPResult) -> None:
        summary_path = Path(self.config.output_dir) / "summary.json"
        payload = {
            "config": asdict(self.config),
            "dataset": self.dataset.name if self.dataset is not None else None,
            "fitting_mode": self._fitting_mode(),
            "best_expression": result.best_expression,
            "theta": result.theta.tolist(),
            "active_genes": [
                {
                    "coefficient": float(coefficient),
                    "expression": gene.expression(),
                    "complexity": gene.complexity,
                }
                for coefficient, gene in zip(result.theta, result.genes)
                if abs(float(coefficient)) >= self.config.sparsity_threshold
            ],
            "metrics": result.metrics,
            "history": result.history,
            "output_paths": result.output_paths,
            "piola_field_plots": result.piola_field_plots or {},
            "piola_field_metrics": result.piola_field_metrics or [],
            "piola_field_warnings": result.piola_field_warnings or [],
            "euclid_metrics": result.euclid_metrics,
        }
        save_json(summary_path, payload)
        result.output_paths["summary_json"] = str(summary_path)

    @staticmethod
    def _metrics_dict(fit: SparseFitResult) -> dict:
        return {
            "rss": fit.metrics.rss,
            "rmse": fit.metrics.rmse,
            "aic": fit.metrics.aic,
            "aicc": fit.metrics.aicc,
            "num_samples": fit.metrics.num_samples,
            "num_parameters": fit.metrics.num_parameters,
        }

    def _compare_euclid(self) -> dict | None:
        if self.config.data_dir is None:
            return None
        loadsteps = self.config.loadsteps or resolve_loadsteps(self.config.data_dir)
        euclid_config = EuclidConfig(
            str_model=self.config.euclid_model,
            femDataPathOverride=self.config.data_dir,
            loadstepsOverride=list(loadsteps),
            resultsDir=str(Path(self.config.output_dir) / "euclid_comparison"),
            appendResults=False,
            noiseLevel=self.config.noise_level,
        )
        workflow = EuclidWorkflow(config=euclid_config)
        euclid_result = workflow.train()
        errors = []
        for loadstep in loadsteps:
            data = loadFemData(
                str(Path(self.config.data_dir) / str(loadstep)),
                AD=True,
                noiseLevel=0.0,
                noiseType="displacement",
            )
            data.convertToNumpy()
            if data.P is None:
                continue
            predicted = computeFirstPiolaTheta(data, euclid_result.theta)
            errors.append((predicted - data.P).reshape(-1))
        if not errors:
            return None
        error = np.concatenate(errors)
        return {
            "rmse": float(np.sqrt(np.mean(np.square(error)))),
            "mae": float(np.mean(np.abs(error))),
            "max_abs_error": float(np.max(np.abs(error))),
            "theta": euclid_result.theta.tolist(),
        }

    def predict(self):
        if self.result is None or self.dataset is None:
            raise RuntimeError("SGEPWorkflow.predict() requires a completed train() call.")
        X, valid_mask, _ = assemble_stress_feature_matrix(
            self.result.genes,
            self.dataset,
            self.config.variable_names,
            derivative_step=self.config.derivative_step,
            value_limit=self.config.invalid_value_limit,
            duplicate_correlation=1.1,
        )
        return (X @ self.result.theta[valid_mask]).reshape(-1, 4)

    def evaluate(self) -> SGEPResult:
        if self.result is None:
            raise RuntimeError("SGEPWorkflow.evaluate() requires a completed train() call.")
        return self.result
