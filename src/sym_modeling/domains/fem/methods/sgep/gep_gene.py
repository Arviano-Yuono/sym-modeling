from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


EPS = 1e-12
UNARY_OPS = ("neg", "square", "sqrt", "log", "exp", "sin", "cos")
BINARY_OPS = ("add", "sub", "mul", "div")


def _first_shape(variables: Mapping[str, np.ndarray]) -> tuple[int, ...]:
    for value in variables.values():
        return np.asarray(value, dtype=float).shape
    return (1,)


def _safe(values: np.ndarray, limit: float = 1e12) -> np.ndarray:
    return np.nan_to_num(values, nan=0.0, posinf=limit, neginf=-limit)


def protected_unary(op: str, x: np.ndarray) -> np.ndarray:
    with np.errstate(all="ignore"):
        if op == "neg":
            out = -x
        elif op == "square":
            out = np.square(x)
        elif op == "sqrt":
            out = np.sqrt(np.abs(x) + EPS)
        elif op == "log":
            out = np.log(np.abs(x) + EPS)
        elif op == "exp":
            out = np.exp(np.clip(x, -20.0, 20.0))
        elif op == "sin":
            out = np.sin(x)
        elif op == "cos":
            out = np.cos(x)
        else:
            raise ValueError("Unsupported unary operator: %s" % op)
    return _safe(out)


def protected_binary(op: str, left: np.ndarray, right: np.ndarray) -> np.ndarray:
    with np.errstate(all="ignore"):
        if op == "add":
            out = left + right
        elif op == "sub":
            out = left - right
        elif op == "mul":
            out = left * right
        elif op == "div":
            out = left / np.where(np.abs(right) < EPS, np.sign(right) * EPS + EPS, right)
        else:
            raise ValueError("Unsupported binary operator: %s" % op)
    return _safe(out)


@dataclass(frozen=True)
class Gene:
    """Small expression-tree gene used by the custom SGEP prototype."""

    op: str
    children: tuple["Gene", ...] = ()
    value: float | str | None = None

    @staticmethod
    def variable(name: str) -> "Gene":
        return Gene("var", value=name)

    @staticmethod
    def constant(value: float) -> "Gene":
        return Gene("const", value=float(value))

    @staticmethod
    def unary(op: str, child: "Gene") -> "Gene":
        if op not in UNARY_OPS:
            raise ValueError("Unsupported unary operator: %s" % op)
        return Gene(op, children=(child,))

    @staticmethod
    def binary(op: str, left: "Gene", right: "Gene") -> "Gene":
        if op not in BINARY_OPS:
            raise ValueError("Unsupported binary operator: %s" % op)
        return Gene(op, children=(left, right))

    @property
    def complexity(self) -> int:
        return 1 + sum(child.complexity for child in self.children)

    def evaluate(self, variables: Mapping[str, np.ndarray]) -> np.ndarray:
        if self.op == "var":
            name = str(self.value)
            if name not in variables:
                raise KeyError("Missing variable for gene evaluation: %s" % name)
            return np.asarray(variables[name], dtype=float)
        if self.op == "const":
            return np.full(_first_shape(variables), float(self.value), dtype=float)
        if self.op in UNARY_OPS:
            return protected_unary(self.op, self.children[0].evaluate(variables))
        if self.op in BINARY_OPS:
            return protected_binary(
                self.op,
                self.children[0].evaluate(variables),
                self.children[1].evaluate(variables),
            )
        raise ValueError("Unsupported gene operator: %s" % self.op)

    def expression(self) -> str:
        if self.op == "var":
            return str(self.value)
        if self.op == "const":
            return "%.6g" % float(self.value)
        if self.op == "neg":
            return "-(%s)" % self.children[0].expression()
        if self.op in ("square", "sqrt", "log", "exp", "sin", "cos"):
            return "%s(%s)" % (self.op, self.children[0].expression())
        left = self.children[0].expression()
        right = self.children[1].expression()
        symbol = {"add": "+", "sub": "-", "mul": "*", "div": "/"}[self.op]
        return "(%s %s %s)" % (left, symbol, right)

    def paths(self, prefix: tuple[int, ...] = ()) -> list[tuple[int, ...]]:
        found = [prefix]
        for idx, child in enumerate(self.children):
            found.extend(child.paths(prefix + (idx,)))
        return found

    def subtree(self, path: Sequence[int]) -> "Gene":
        node = self
        for idx in path:
            node = node.children[idx]
        return node

    def replace_subtree(self, path: Sequence[int], replacement: "Gene") -> "Gene":
        if not path:
            return replacement
        idx = path[0]
        children = list(self.children)
        children[idx] = children[idx].replace_subtree(path[1:], replacement)
        return Gene(self.op, tuple(children), self.value)


def random_gene(
    rng: np.random.Generator,
    variable_names: Sequence[str],
    max_depth: int,
    unary_operators: Sequence[str] = UNARY_OPS,
    binary_operators: Sequence[str] = BINARY_OPS,
    constant_probability: float = 0.15,
) -> Gene:
    unary_operators = tuple(unary_operators)
    binary_operators = tuple(binary_operators)
    if max_depth <= 0 or rng.random() < 0.35:
        if rng.random() < constant_probability:
            return Gene.constant(float(rng.uniform(-2.0, 2.0)))
        return Gene.variable(str(rng.choice(variable_names)))

    if not unary_operators and not binary_operators:
        return Gene.variable(str(rng.choice(variable_names)))

    if unary_operators and (not binary_operators or rng.random() < 0.45):
        op = str(rng.choice(unary_operators))
        return Gene.unary(
            op,
            random_gene(
                rng,
                variable_names,
                max_depth - 1,
                unary_operators=unary_operators,
                binary_operators=binary_operators,
            ),
        )

    op = str(rng.choice(binary_operators))
    return Gene.binary(
        op,
        random_gene(
            rng,
            variable_names,
            max_depth - 1,
            unary_operators=unary_operators,
            binary_operators=binary_operators,
        ),
        random_gene(
            rng,
            variable_names,
            max_depth - 1,
            unary_operators=unary_operators,
            binary_operators=binary_operators,
        ),
    )


def seed_genes(
    variable_names: Sequence[str],
    unary_operators: Sequence[str] = UNARY_OPS,
    binary_operators: Sequence[str] = BINARY_OPS,
) -> list[Gene]:
    genes: list[Gene] = []
    variables = [Gene.variable(name) for name in variable_names]
    genes.extend(variables)
    if "square" in unary_operators:
        genes.extend(Gene.unary("square", gene) for gene in variables)
    if "mul" in binary_operators and len(variables) >= 2:
        genes.append(Gene.binary("mul", variables[0], variables[1]))
    return genes


def mutate_gene(
    gene: Gene,
    rng: np.random.Generator,
    variable_names: Sequence[str],
    max_depth: int,
    unary_operators: Sequence[str] = UNARY_OPS,
    binary_operators: Sequence[str] = BINARY_OPS,
) -> Gene:
    path = gene.paths()[int(rng.integers(0, len(gene.paths())))]
    replacement = random_gene(
        rng,
        variable_names,
        max_depth=max(1, max_depth // 2),
        unary_operators=unary_operators,
        binary_operators=binary_operators,
    )
    return gene.replace_subtree(path, replacement)


def crossover_gene(left: Gene, right: Gene, rng: np.random.Generator) -> Gene:
    left_paths = left.paths()
    right_paths = right.paths()
    left_path = left_paths[int(rng.integers(0, len(left_paths)))]
    right_path = right_paths[int(rng.integers(0, len(right_paths)))]
    return left.replace_subtree(left_path, right.subtree(right_path))


def values_are_valid(values: np.ndarray, limit: float = 1e8) -> bool:
    array = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(array)):
        return False
    if array.size == 0:
        return False
    return bool(np.max(np.abs(array)) <= limit)
