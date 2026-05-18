from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from sym_modeling.domains.fem.methods.sgep.gep_gene import (
    Gene,
    crossover_gene,
    mutate_gene,
    random_gene,
    seed_genes,
)


@dataclass(frozen=True)
class CandidateModel:
    genes: tuple[Gene, ...]

    def expression(self, theta: np.ndarray | None = None) -> str:
        if theta is None:
            return " + ".join(gene.expression() for gene in self.genes)
        terms = []
        for coefficient, gene in zip(theta, self.genes):
            if abs(float(coefficient)) > 0.0:
                terms.append("(%.8g)*%s" % (float(coefficient), gene.expression()))
        return " + ".join(terms) if terms else "0"


@dataclass(frozen=True)
class CandidateRecord:
    candidate: CandidateModel
    aicc: float
    rmse: float
    active_terms: int
    complexity: int
    gene_fitness: np.ndarray


def selection_rank(
    selection_objective: str,
    aicc: float,
    rmse: float,
    active_terms: int,
    complexity: int,
    active_terms_epsilon: int,
) -> tuple:
    """Return the candidate ordering key used by SGEP selection."""
    if selection_objective == "aicc":
        return (float(aicc), float(rmse))
    if selection_objective != "epsilon_rmse":
        raise ValueError("Unsupported selection objective: %s" % selection_objective)

    rmse_value = float(rmse)
    active_count = int(active_terms)
    complexity_value = int(complexity)
    epsilon = int(active_terms_epsilon)
    if not np.isfinite(rmse_value):
        return (2, float("inf"), active_count, complexity_value)
    if active_count <= epsilon:
        return (0, rmse_value, active_count, complexity_value)
    return (1, active_count - epsilon, rmse_value, complexity_value)


class PopulationEngine:
    """Small custom GEP-style population engine with a replaceable boundary."""

    def __init__(
        self,
        variable_names: Sequence[str],
        genes_per_model: int,
        num_models: int,
        max_depth: int,
        random_seed: int,
        mutation_rate: float = 0.55,
        crossover_rate: float = 0.30,
        random_fraction: float = 0.15,
        elite_model_count: int = 1,
        elite_gene_count: int = 4,
        unary_operators: Sequence[str] = ("neg", "square", "sqrt", "log", "exp", "sin", "cos"),
        binary_operators: Sequence[str] = ("add", "sub", "mul", "div"),
        selection_objective: str = "epsilon_rmse",
        active_terms_epsilon: int = 2,
    ) -> None:
        self.variable_names = tuple(variable_names)
        self.genes_per_model = int(genes_per_model)
        self.num_models = int(num_models)
        self.max_depth = int(max_depth)
        self.rng = np.random.default_rng(random_seed)
        self.mutation_rate = float(mutation_rate)
        self.crossover_rate = float(crossover_rate)
        self.random_fraction = float(random_fraction)
        self.elite_model_count = int(elite_model_count)
        self.elite_gene_count = int(elite_gene_count)
        self.unary_operators = tuple(unary_operators)
        self.binary_operators = tuple(binary_operators)
        self.selection_objective = str(selection_objective)
        self.active_terms_epsilon = int(active_terms_epsilon)

    @property
    def population_size(self) -> int:
        return self.genes_per_model * self.num_models

    def initial_population(self) -> list[Gene]:
        genes = seed_genes(
            self.variable_names,
            unary_operators=self.unary_operators,
            binary_operators=self.binary_operators,
        )
        while len(genes) < self.population_size:
            genes.append(
                random_gene(
                    self.rng,
                    self.variable_names,
                    self.max_depth,
                    unary_operators=self.unary_operators,
                    binary_operators=self.binary_operators,
                )
            )
        self.rng.shuffle(genes)
        return genes[: self.population_size]

    def group_candidates(self, genes: Sequence[Gene]) -> list[CandidateModel]:
        needed = self.population_size
        population = list(genes[:needed])
        while len(population) < needed:
            population.append(
                random_gene(
                    self.rng,
                    self.variable_names,
                    self.max_depth,
                    unary_operators=self.unary_operators,
                    binary_operators=self.binary_operators,
                )
            )
        return [
            CandidateModel(tuple(population[start : start + self.genes_per_model]))
            for start in range(0, needed, self.genes_per_model)
        ]

    def next_population(self, records: Sequence[CandidateRecord]) -> list[Gene]:
        ordered = sorted(
            records,
            key=lambda record: selection_rank(
                self.selection_objective,
                record.aicc,
                record.rmse,
                record.active_terms,
                record.complexity,
                self.active_terms_epsilon,
            ),
        )
        next_genes: list[Gene] = []

        for record in ordered[: self.elite_model_count]:
            next_genes.extend(record.candidate.genes)

        gene_scores: list[tuple[float, Gene]] = []
        for record in records:
            for score, gene in zip(record.gene_fitness, record.candidate.genes):
                gene_scores.append((float(score), gene))
        gene_scores.sort(key=lambda item: item[0], reverse=True)

        for score, gene in gene_scores[: self.elite_gene_count]:
            if score > 0.0:
                next_genes.append(gene)

        parents = [gene for score, gene in gene_scores if score > 0.0]
        if not parents:
            parents = [gene for record in ordered[: max(1, self.elite_model_count)] for gene in record.candidate.genes]

        random_count = int(round(self.population_size * self.random_fraction))
        while len(next_genes) < self.population_size:
            remaining = self.population_size - len(next_genes)
            if remaining <= random_count:
                next_genes.append(
                    random_gene(
                        self.rng,
                        self.variable_names,
                        self.max_depth,
                        unary_operators=self.unary_operators,
                        binary_operators=self.binary_operators,
                    )
                )
                continue

            parent = parents[int(self.rng.integers(0, len(parents)))]
            child = parent
            if self.rng.random() < self.crossover_rate and len(parents) > 1:
                other = parents[int(self.rng.integers(0, len(parents)))]
                child = crossover_gene(child, other, self.rng)
            if self.rng.random() < self.mutation_rate:
                child = mutate_gene(
                    child,
                    self.rng,
                    self.variable_names,
                    self.max_depth,
                    unary_operators=self.unary_operators,
                    binary_operators=self.binary_operators,
                )
            next_genes.append(child)

        self.rng.shuffle(next_genes)
        return next_genes[: self.population_size]
