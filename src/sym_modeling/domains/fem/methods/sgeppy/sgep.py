from __future__ import annotations

import random
import operator
import time
from dataclasses import dataclass
from itertools import count
from typing import Mapping, Sequence

import geppy as gep
import numpy as np
import sympy as sp
from deap import base, creator, tools
from geppy.algorithms.basic import _apply_crossover, _apply_modification, _validate_basic_toolbox

from . import operator as ops
from sym_modeling.domains.fem.methods.common.regression import fit_sparse_regression


BINARY_OPS = {"add": ops.add, "sub": ops.sub, "mul": ops.mul, "div": ops.div}
UNARY_OPS = {
    "neg": ops.neg,
    "square": ops.square,
    "sqrt": ops.sqrt,
    "log": ops.log,
    "exp": ops.exp,
    "sin": ops.sin,
    "cos": ops.cos,
}
FITNESS_METRICS = {"mse", "rmse", "rss", "aic", "aicc"}
_CLASS_IDS = count()


def linked_add(*values):
    total = values[0]
    for value in values[1:]:
        total = operator.add(total, value)
    return total


class _UnlinkedChromosome(list):
    linker = None


SYMBOLIC_FUNCTIONS = {
    "add": operator.add,
    "sub": operator.sub,
    "mul": operator.mul,
    "div": operator.truediv,
    "neg": operator.neg,
    "square": lambda x: x**2,
    "sqrt": sp.sqrt,
    "log": sp.log,
    "exp": sp.exp,
    "sin": sp.sin,
    "cos": sp.cos,
    "linked_add": linked_add,
}


def _timed_gep_simple(population, toolbox, n_generations=100, n_elites=1, stats=None, hall_of_fame=None, verbose=True):
    _validate_basic_toolbox(toolbox)
    logbook = tools.Logbook()
    logbook.header = ["gen", "nevals", "wall_seconds", "cpu_seconds"] + (stats.fields if stats else [])

    for gen in range(n_generations + 1):
        wall_start = time.perf_counter()
        cpu_start = time.process_time()

        invalid_individuals = [ind for ind in population if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_individuals)
        for ind, fit in zip(invalid_individuals, fitnesses):
            ind.fitness.values = fit

        if hall_of_fame is not None:
            hall_of_fame.update(population)
        record = stats.compile(population) if stats else {}

        next_population = None
        if gen < n_generations:
            elites = tools.selBest(population, k=n_elites)
            offspring = toolbox.select(population, len(population) - n_elites)
            offspring = [toolbox.clone(ind) for ind in offspring]

            for op in toolbox.pbs:
                if op.startswith("mut"):
                    offspring = _apply_modification(offspring, getattr(toolbox, op), toolbox.pbs[op])
            for op in toolbox.pbs:
                if op.startswith("cx"):
                    offspring = _apply_crossover(offspring, getattr(toolbox, op), toolbox.pbs[op])
            next_population = elites + offspring

        logbook.record(
            gen=gen,
            nevals=len(invalid_individuals),
            wall_seconds=time.perf_counter() - wall_start,
            cpu_seconds=time.process_time() - cpu_start,
            **record,
        )
        if verbose:
            print(logbook.stream)
        if next_population is None:
            break
        population = next_population

    return population, logbook


@dataclass
class SGEPConfig:
    variable_names: tuple[str, ...] = ("K1", "K2", "Jm1")
    unary_operators: tuple[str, ...] = ()
    binary_operators: tuple[str, ...] = ("add", "sub", "mul", "div")
    random_seed: int = 0

    head_length: int = 7
    n_genes: int = 2
    population_size: int = 100
    n_generations: int = 200
    n_elites: int = 1
    hall_of_fame_size: int = 3
    tournament_size: int = 3
    mut_uniform_ind_pb: float = 0.05
    mut_uniform_pb: float = 1.0
    mut_invert_pb: float = 0.1
    mut_is_transpose_pb: float = 0.1
    mut_ris_transpose_pb: float = 0.1
    mut_gene_transpose_pb: float = 0.1
    cx_one_point_pb: float = 0.4
    cx_two_point_pb: float = 0.2
    cx_gene_pb: float = 0.1
    fitness_metrics: tuple[str, ...] = ("aicc",)
    epsilons: tuple[float | None, ...] | None = None
    fit_intercept: bool = True
    verbose: bool = True

    regression_method: str = "stlsq"
    regression_alpha: float = 1e-8
    regression_threshold: float = 1e-8
    regression_refit: bool = True
    regression_max_iter: int = 10

    def __post_init__(self) -> None:
        self.variable_names = tuple(self.variable_names)
        self.unary_operators = tuple(self.unary_operators)
        self.binary_operators = tuple(self.binary_operators)
        self.fitness_metrics = tuple(str(metric).lower() for metric in self.fitness_metrics)
        if self.epsilons is not None:
            self.epsilons = tuple(self.epsilons)

        unknown_unary = sorted(set(self.unary_operators) - set(UNARY_OPS))
        unknown_binary = sorted(set(self.binary_operators) - set(BINARY_OPS))
        if unknown_unary:
            raise ValueError("Unsupported unary operators: %s" % ", ".join(unknown_unary))
        if unknown_binary:
            raise ValueError("Unsupported binary operators: %s" % ", ".join(unknown_binary))
        if not self.unary_operators and not self.binary_operators:
            raise ValueError("At least one unary or binary operator is required.")
        if not self.fitness_metrics:
            raise ValueError("At least one fitness metric is required.")
        unknown_metrics = sorted(set(self.fitness_metrics) - FITNESS_METRICS)
        if unknown_metrics:
            raise ValueError("Unsupported fitness metrics: %s" % ", ".join(unknown_metrics))
        if self.epsilons is not None:
            if len(self.epsilons) != len(self.fitness_metrics):
                raise ValueError("epsilons must match fitness_metrics length.")
            if all(epsilon is not None for epsilon in self.epsilons):
                raise ValueError("At least one epsilon must be None.")
        if self.n_elites >= self.population_size:
            raise ValueError("n_elites must be smaller than population_size.")


class SGEP:
    """Small geppy SGEP loop that scores genes with the existing sparse fitter."""

    def __init__(self, config: SGEPConfig | None = None):
        self.config = config or SGEPConfig()
        self.pset = None
        self.toolbox = None
        self.population = None
        self.logbook = None
        self.hall_of_fame = None
        self.best_individual = None
        self.best_fit = None
        self._X = None
        self._y = None
        self._feature_builder = None
        self._evaluator = None

    def build(self) -> "SGEP":
        random.seed(self.config.random_seed)
        np.random.seed(self.config.random_seed)

        self.pset = gep.PrimitiveSet("Main", self.config.variable_names)
        for name in self.config.unary_operators:
            self.pset.add_function(UNARY_OPS[name], 1, name=name)
        for name in self.config.binary_operators:
            self.pset.add_function(BINARY_OPS[name], 2, name=name)

        suffix = next(_CLASS_IDS)
        weights = (-1.0, -1.0) if self.config.epsilons is not None else (-1.0,) * len(self.config.fitness_metrics)
        creator.create("SGEPFitness%d" % suffix, base.Fitness, weights=weights)
        creator.create(
            "SGEPIndividual%d" % suffix,
            gep.Chromosome,
            fitness=getattr(creator, "SGEPFitness%d" % suffix),
            theta=object,
            sparse_fit=object,
        )

        self.toolbox = gep.Toolbox()
        self.toolbox.register("gene_gen", gep.Gene, pset=self.pset, head_length=self.config.head_length)
        self.toolbox.register(
            "individual",
            getattr(creator, "SGEPIndividual%d" % suffix),
            gene_gen=self.toolbox.gene_gen,
            n_genes=self.config.n_genes,
            linker=linked_add,
        )
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("compile", gep.compile_, pset=self.pset)
        self.toolbox.register("evaluate", self.evaluate)
        self.toolbox.register("select", tools.selTournament, tournsize=self.config.tournament_size)

        self.toolbox.register("mut_uniform", gep.mutate_uniform, pset=self.pset, ind_pb=self.config.mut_uniform_ind_pb, pb=self.config.mut_uniform_pb)
        self.toolbox.register("mut_invert", gep.invert, pb=self.config.mut_invert_pb)
        self.toolbox.register("mut_is_transpose", gep.is_transpose, pb=self.config.mut_is_transpose_pb)
        self.toolbox.register("mut_ris_transpose", gep.ris_transpose, pb=self.config.mut_ris_transpose_pb)
        self.toolbox.register("mut_gene_transpose", gep.gene_transpose, pb=self.config.mut_gene_transpose_pb)
        self.toolbox.register("cx_1p", gep.crossover_one_point, pb=self.config.cx_one_point_pb)
        self.toolbox.register("cx_2p", gep.crossover_two_point, pb=self.config.cx_two_point_pb)
        self.toolbox.register("cx_gene", gep.crossover_gene, pb=self.config.cx_gene_pb)
        return self

    def fit(self, X: np.ndarray | Mapping[str, Sequence[float]], y: Sequence[float], feature_builder=None, evaluator=None) -> "SGEP":
        if self.toolbox is None:
            self.build()
        self._X = self._as_matrix(X)
        self._y = np.asarray(y, dtype=float).reshape(-1)
        self._feature_builder = feature_builder
        self._evaluator = evaluator
        if feature_builder is None and evaluator is None and self._X.shape[0] != self._y.size:
            raise ValueError("X and y must contain the same number of samples.")

        stats = tools.Statistics(key=lambda ind: ind.fitness.values[0])
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)

        self.population = self.toolbox.population(n=self.config.population_size)
        self.hall_of_fame = tools.HallOfFame(self.config.hall_of_fame_size)
        self.population, self.logbook = _timed_gep_simple(
            self.population,
            self.toolbox,
            n_generations=self.config.n_generations,
            n_elites=self.config.n_elites,
            stats=stats,
            hall_of_fame=self.hall_of_fame,
            verbose=self.config.verbose,
        )
        self.best_individual = self.hall_of_fame[0]
        self.best_fit = self.best_individual.sparse_fit
        return self

    def evaluate(self, individual) -> tuple[float, ...]:
        try:
            if self._evaluator is not None:
                fit, valid = self._evaluator(self, individual, self._X, self._y)
            else:
                features, valid = self._features_for(individual, self._X)
                fit = fit_sparse_regression(
                    features,
                    self._y,
                    method=self.config.regression_method,
                    alpha=self.config.regression_alpha,
                    threshold=self.config.regression_threshold,
                    refit=self.config.regression_refit,
                    max_iter=self.config.regression_max_iter,
                )
            individual.theta = fit.theta
            individual.valid_mask = valid
            individual.sparse_fit = fit
            return self._fitness_values(fit)
        except (ArithmeticError, FloatingPointError, ValueError, np.linalg.LinAlgError):
            n_features = self.config.n_genes + int(self.config.fit_intercept)
            individual.theta = np.zeros(n_features, dtype=float)
            individual.valid_mask = np.zeros(n_features, dtype=bool)
            individual.sparse_fit = None
            if self.config.epsilons is not None:
                return (float("inf"), float("inf"))
            return tuple(float("inf") for _ in self.config.fitness_metrics)

    def _fitness_values(self, fit) -> tuple[float, ...]:
        values = tuple(self._metric_value(fit, metric) for metric in self.config.fitness_metrics)
        if self.config.epsilons is None:
            return values

        violation = sum(
            max(0.0, value - float(epsilon))
            for value, epsilon in zip(values, self.config.epsilons)
            if epsilon is not None
        )
        objective = values[self.config.epsilons.index(None)]
        return (float(violation), float(objective))

    @staticmethod
    def _metric_value(fit, metric: str) -> float:
        if metric == "mse":
            return float(fit.metrics.rss / fit.metrics.num_samples)
        return float(getattr(fit.metrics, metric))

    def feature_matrix(self, X, individual=None) -> np.ndarray:
        return self._features_for(individual or self.best_individual, X)[0]

    def _features_for(self, individual, X) -> tuple[np.ndarray, np.ndarray]:
        individual = individual or self.best_individual
        if individual is None:
            raise RuntimeError("No individual available. Call fit first or pass one.")
        X = self._as_matrix(X)
        if self._feature_builder is not None:
            return self._feature_builder(self, individual, X)
        outputs = self.gene_outputs(individual, X)
        features = np.column_stack([self._as_column(values, X.shape[0]) for values in outputs])
        if not np.all(np.isfinite(features)):
            raise FloatingPointError("Non-finite gene feature values.")
        if self.config.fit_intercept:
            features = np.column_stack([features, np.ones(X.shape[0], dtype=float)])
        return features, np.ones(features.shape[1], dtype=bool)

    def gene_outputs(self, individual, X) -> tuple:
        X = self._as_matrix(X)
        func = self.toolbox.compile(_UnlinkedChromosome(individual))
        outputs = func(*(np.array(X[:, i], copy=True) for i in range(X.shape[1])))
        return outputs if isinstance(outputs, tuple) else (outputs,)

    def predict(self, X, individual=None) -> np.ndarray:
        individual = individual or self.best_individual
        theta = getattr(individual, "theta", None)
        if theta is None:
            raise RuntimeError("No sparse-regression coefficients available.")
        features, valid = self._features_for(individual, X)
        return features @ np.asarray(theta, dtype=float)

    def expression(self, individual=None, simplify: bool = True) -> str:
        individual = individual or self.best_individual
        theta = getattr(individual, "theta", None)
        if individual is None or theta is None:
            raise RuntimeError("No fitted expression available.")
        theta = np.asarray(theta, dtype=float)
        if theta.size == 0:
            return "0"
        active = getattr(getattr(individual, "sparse_fit", None), "active_mask", np.abs(theta) >= self.config.regression_threshold)
        parts = []
        n_gene_terms = min(len(individual), theta.size)
        for index, gene in enumerate(individual[:n_gene_terms]):
            if active[index]:
                expr = str(gep.simplify(gene, SYMBOLIC_FUNCTIONS)) if simplify else str(gene)
                parts.append("(%0.12g) * (%s)" % (float(theta[index]), expr))
        intercept_index = len(individual)
        if self.config.fit_intercept and len(theta) > intercept_index and active[intercept_index]:
            parts.append("(%0.12g)" % float(theta[intercept_index]))
        if not parts:
            return "0"
        return " + ".join(parts)

    def _as_matrix(self, X) -> np.ndarray:
        if isinstance(X, Mapping):
            X = np.column_stack([np.asarray(X[name], dtype=float).reshape(-1) for name in self.config.variable_names])
        else:
            X = np.asarray(X, dtype=float)
            if X.ndim == 1:
                X = X.reshape(-1, 1)
        if X.ndim != 2 or X.shape[1] != len(self.config.variable_names):
            raise ValueError("X must have one column per configured variable.")
        return X

    @staticmethod
    def _as_column(values, n_rows: int) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        if values.ndim == 0:
            return np.full(n_rows, float(values))
        values = values.reshape(-1)
        if values.size != n_rows:
            raise ValueError("Gene output has the wrong number of samples.")
        return values
