from __future__ import annotations

import importlib
import importlib.util
import time
from collections import OrderedDict
from dataclasses import dataclass
from functools import lru_cache, partial
from typing import Sequence

import geppy as gep
import numpy as np
from geppy.tools.parser import _compile_gene


JAX_FEM_EXTRA = "jax_fem"
JAX_PRECISIONS = {"float64", "float32"}
JAX_CACHE_TIMING_KEYS = (
    "weak_form_jax_cache_hits",
    "weak_form_jax_cache_misses",
    "weak_form_jax_compile_cache_hits",
    "weak_form_jax_compile_cache_misses",
    "weak_form_jax_cache_entries",
    "weak_form_jax_gene_compile_seconds",
    "weak_form_jax_gene_execute_seconds",
)


def require_jax_fem_backend(enable_x64: bool | None = True):
    """Import optional JAX/JAX-FEM dependencies with an actionable error."""
    try:
        jax = importlib.import_module("jax")
        if enable_x64 is not None:
            jax.config.update("jax_enable_x64", bool(enable_x64))
        jnp = importlib.import_module("jax.numpy")
        if importlib.util.find_spec("jax_fem") is None:
            raise ModuleNotFoundError("No module named 'jax_fem'")
    except ModuleNotFoundError as exc:
        raise ImportError(
            "fitting_mode='weak_form_jax' requires optional dependencies. "
            'Install them with: pip install -e ".[jax_fem]"'
        ) from exc
    return jax, jnp


def configure_jax_precision(precision: str):
    if precision not in JAX_PRECISIONS:
        raise ValueError("jax_precision must be one of: float32, float64.")
    return require_jax_fem_backend(enable_x64=precision == "float64")


def _jax_dtype(precision: str):
    _, jnp = configure_jax_precision(precision)
    return jnp.float64 if precision == "float64" else jnp.float32


def block_until_ready(value):
    if isinstance(value, dict):
        for item in value.values():
            block_until_ready(item)
        return value
    if isinstance(value, (list, tuple)):
        for item in value:
            block_until_ready(item)
        return value
    if hasattr(value, "block_until_ready"):
        value.block_until_ready()
    return value


def is_jax_fem_backend_available() -> bool:
    try:
        require_jax_fem_backend()
    except ImportError:
        return False
    return True


@dataclass(frozen=True)
class JaxWeakFormCase:
    data: object
    F: object
    B_matrices: object
    qp_weights: object
    dof_indices: object
    free_dof_indices: object
    reaction_dofs: tuple
    reaction_forces: object
    num_nodes: int


class JaxWeakFormEvaluationCache:
    """Bounded LRU caches for repeated weak_form_jax individuals."""

    def __init__(
        self,
        enabled: bool = True,
        max_size: int = 256,
        device_outputs: bool = True,
        timing: dict[str, float] | None = None,
    ):
        self.enabled = bool(enabled)
        self.max_size = max(0, int(max_size))
        self.device_outputs = bool(device_outputs)
        self.timing = timing
        self._compiled = OrderedDict()
        self._artifacts = OrderedDict()
        self._sync_entries()

    def compiled_key(
        self,
        model,
        individual,
        selected: Sequence[int],
        variable_names: Sequence[str],
        precision: str | None,
    ) -> tuple:
        return (
            "compiled",
            _individual_signature(individual),
            tuple(int(index) for index in selected),
            tuple(variable_names),
            tuple(model.config.unary_operators),
            tuple(model.config.binary_operators),
            precision or "array",
        )

    def derivative_key(
        self,
        data_key,
        model,
        individual,
        selected: Sequence[int],
        variable_names: Sequence[str],
        precision: str | None,
        F,
    ) -> tuple:
        return (
            "dqdf",
            data_key,
            _individual_signature(individual),
            tuple(int(index) for index in selected),
            tuple(variable_names),
            precision or "array",
            tuple(getattr(F, "shape", ())),
            str(getattr(F, "dtype", "")),
        )

    def weak_artifact_key(
        self,
        case_key,
        model,
        individual,
        selected: Sequence[int],
        variable_names: Sequence[str],
        precision: str,
        balance: float,
    ) -> tuple:
        return (
            "weak_artifact",
            case_key,
            _individual_signature(individual),
            tuple(int(index) for index in selected),
            tuple(variable_names),
            precision,
            float(balance),
        )

    def get_compiled(self, key):
        if not self.enabled:
            return None
        if key in self._compiled:
            self._compiled.move_to_end(key)
            self._increment("weak_form_jax_compile_cache_hits")
            return self._compiled[key]
        self._increment("weak_form_jax_compile_cache_misses")
        return None

    def put_compiled(self, key, value) -> None:
        if not self.enabled or self.max_size <= 0:
            return
        self._compiled[key] = value
        self._compiled.move_to_end(key)
        self._evict(self._compiled)
        self._sync_entries()

    def get_artifact(self, key):
        if not self.enabled:
            return None
        if key in self._artifacts:
            self._artifacts.move_to_end(key)
            self._increment("weak_form_jax_cache_hits")
            return self._artifacts[key]
        self._increment("weak_form_jax_cache_misses")
        return None

    def put_artifact(self, key, value) -> None:
        if not self.enabled or self.max_size <= 0:
            return
        self._artifacts[key] = value
        self._artifacts.move_to_end(key)
        self._evict(self._artifacts)
        self._sync_entries()

    def _evict(self, store: OrderedDict) -> None:
        while len(store) > self.max_size:
            store.popitem(last=False)

    def _increment(self, key: str, amount: float = 1.0) -> None:
        if self.timing is not None:
            self.timing[key] = self.timing.get(key, 0.0) + amount

    def _sync_entries(self) -> None:
        if self.timing is not None:
            self.timing["weak_form_jax_cache_entries"] = float(len(self._compiled) + len(self._artifacts))


def _individual_signature(individual) -> tuple[str, ...]:
    return tuple(str(gene) for gene in individual)


def _jax_ops():
    _, jnp = require_jax_fem_backend()
    eps = 1e-12

    def div(a, b):
        return a / jnp.where(jnp.abs(b) < 1e-6, 1.0, b)

    return {
        "add": lambda a, b: a + b,
        "sub": lambda a, b: a - b,
        "mul": lambda a, b: a * b,
        "div": div,
        "neg": lambda a: -a,
        "square": lambda a: jnp.square(a),
        "sqrt": lambda a: jnp.sqrt(jnp.abs(a) + eps),
        "log": lambda a: jnp.log(jnp.abs(a) + eps),
        "exp": lambda a: jnp.exp(jnp.clip(a, -20.0, 20.0)),
        "sin": lambda a: jnp.sin(a),
        "cos": lambda a: jnp.cos(a),
    }


def _jax_pset(model):
    from .sgep import BINARY_OPS, UNARY_OPS

    ops = _jax_ops()
    pset = gep.PrimitiveSet("JaxMain", model.config.variable_names)
    for name in model.config.unary_operators:
        if name not in UNARY_OPS:
            raise ValueError("Unsupported unary operator for JAX backend: %s" % name)
        pset.add_function(ops[name], 1, name=name)
    for name in model.config.binary_operators:
        if name not in BINARY_OPS:
            raise ValueError("Unsupported binary operator for JAX backend: %s" % name)
        pset.add_function(ops[name], 2, name=name)
    return pset


def _compile_gene_functions(model, individual) -> tuple:
    pset = _jax_pset(model)
    return tuple(_compile_gene(gene, pset) for gene in individual)


def _variables_from_F(F, variable_names: Sequence[str]):
    _, jnp = require_jax_fem_backend(enable_x64=None)
    F11, F12, F21, F22 = F[0], F[1], F[2], F[3]
    C11 = F11**2 + F21**2
    C12 = F11 * F12 + F21 * F22
    C21 = C12
    C22 = F12**2 + F22**2
    I1 = C11 + C22 + 1.0
    I2 = C11 + C22 - C12 * C21 + C11 * C22
    I3 = C11 * C22 - C12 * C21
    J = F11 * F22 - F12 * F21
    K1 = I1 * jnp.power(I3, -1.0 / 3.0) - 3.0
    K2 = (I1 + I3 - 1.0) * jnp.power(I3, -2.0 / 3.0) - 3.0
    values = {
        "I1": I1,
        "I2": I2,
        "I3": I3,
        "J": J,
        "Jm1": J - 1.0,
        "K1": K1,
        "K2": K2,
        "logI13": jnp.log(K1 / 3.0 + 1.0),
        "logI23": jnp.log(K2 / 3.0 + 1.0),
    }
    return [values[name] for name in variable_names]


def prepare_jax_case(cache, precision: str = "float64") -> JaxWeakFormCase:
    _, jnp = configure_jax_precision(precision)
    dtype = _jax_dtype(precision)
    data = cache.data
    nodes = np.stack([np.asarray(part, dtype=int) for part in data.connectivity], axis=1)
    dof_indices = np.stack((2 * nodes, 2 * nodes + 1), axis=-1).reshape(nodes.shape[0], -1)
    reaction_dofs = tuple(jnp.asarray(np.flatnonzero(dofs), dtype=int) for dofs, _ in cache.reactions)
    return JaxWeakFormCase(
        data=data,
        F=jnp.asarray(data.F, dtype=dtype),
        B_matrices=jnp.asarray(cache.B_matrices, dtype=dtype),
        qp_weights=jnp.asarray(data.qpWeights, dtype=dtype),
        dof_indices=jnp.asarray(dof_indices, dtype=int),
        free_dof_indices=jnp.asarray(np.flatnonzero(cache.free_dofs), dtype=int),
        reaction_dofs=reaction_dofs,
        reaction_forces=jnp.asarray([force for _, force in cache.reactions], dtype=dtype),
        num_nodes=int(data.numNodes),
    )


def feature_values_and_dqdf_device(
    model,
    individual,
    F,
    variable_names: Sequence[str],
    gene_indices: Sequence[int] | None = None,
    value_limit: float = 1e8,
    precision: str | None = None,
    cache: JaxWeakFormEvaluationCache | None = None,
    data_key=None,
) -> tuple:
    _, jnp = require_jax_fem_backend(enable_x64=None if precision is not None else True)
    dtype = _jax_dtype(precision) if precision is not None else (F.dtype if hasattr(F, "dtype") else jnp.float64)
    F = jnp.asarray(F, dtype=dtype)
    selected = list(range(len(individual))) if gene_indices is None else [int(index) for index in gene_indices]

    if not selected:
        return jnp.zeros((F.shape[0], 0), dtype=dtype), jnp.zeros((F.shape[0], 0, 4), dtype=dtype)

    derivative_key = None
    if cache is not None and cache.device_outputs:
        derivative_key = cache.derivative_key(
            data_key,
            model,
            individual,
            selected,
            variable_names,
            precision,
            F,
        )
        cached = cache.get_artifact(derivative_key)
        if cached is not None:
            return cached

    compile_start = time.perf_counter()
    evaluate = _compiled_gene_evaluator(model, individual, selected, variable_names, precision, cache)
    if cache is not None and cache.timing is not None:
        cache.timing["weak_form_jax_gene_compile_seconds"] += time.perf_counter() - compile_start

    execute_start = time.perf_counter()
    features, dqdf = evaluate(F)
    block_until_ready((features, dqdf))
    if cache is not None and cache.timing is not None:
        cache.timing["weak_form_jax_gene_execute_seconds"] += time.perf_counter() - execute_start
    finite = jnp.all(jnp.isfinite(features)) & jnp.all(jnp.isfinite(dqdf))
    bounded = (jnp.max(jnp.abs(features)) <= value_limit) & (jnp.max(jnp.abs(dqdf)) <= value_limit)
    if not bool(np.asarray(finite & bounded)):
        raise ValueError("Invalid JAX weak-form gene value or derivative.")
    if cache is not None and cache.device_outputs and derivative_key is not None:
        cache.put_artifact(derivative_key, (features, dqdf))
    return features, dqdf


def _compiled_gene_evaluator(
    model,
    individual,
    selected: Sequence[int],
    variable_names: Sequence[str],
    precision: str | None,
    cache: JaxWeakFormEvaluationCache | None = None,
):
    key = cache.compiled_key(model, individual, selected, variable_names, precision) if cache is not None else None
    if cache is not None:
        cached = cache.get_compiled(key)
        if cached is not None:
            return cached

    jax, jnp = require_jax_fem_backend(enable_x64=None if precision is not None else True)
    gene_functions = _compile_gene_functions(model, individual)
    selected_gene_functions = tuple(gene_functions[int(index)] for index in selected)

    def energies(F_row):
        args = _variables_from_F(F_row, variable_names)
        return jnp.stack([jnp.reshape(gene_fn(*args), ()) for gene_fn in selected_gene_functions])

    jacobian_fn = jax.jacrev(energies)

    @jax.jit
    def evaluate(F_batch):
        return jax.vmap(lambda F_row: (energies(F_row), jacobian_fn(F_row)))(F_batch)

    if cache is not None:
        cache.put_compiled(key, evaluate)
    return evaluate


def feature_values_and_dqdf(
    model,
    individual,
    F,
    variable_names: Sequence[str],
    gene_indices: Sequence[int] | None = None,
    value_limit: float = 1e8,
    precision: str | None = None,
    cache: JaxWeakFormEvaluationCache | None = None,
    data_key=None,
) -> tuple[np.ndarray, np.ndarray]:
    features_device, dqdf_device = feature_values_and_dqdf_device(
        model,
        individual,
        F,
        variable_names,
        gene_indices=gene_indices,
        value_limit=value_limit,
        precision=precision,
        cache=cache,
        data_key=data_key,
    )
    features = np.asarray(features_device, dtype=float)
    dqdf = np.asarray(dqdf_device, dtype=float)
    if features.shape[1] == 0:
        return features, dqdf
    if not _valid(features, value_limit) or not _valid(dqdf, value_limit):
        raise ValueError("Invalid JAX weak-form gene value or derivative.")
    return features, dqdf


def stress_feature_builder(
    dataset,
    variable_names: Sequence[str],
    value_limit: float = 1e8,
    duplicate_correlation: float = 0.999999,
    precision: str = "float64",
    timing: dict[str, float] | None = None,
    cache: JaxWeakFormEvaluationCache | None = None,
):
    def build(model, individual, X):
        del X
        start = time.perf_counter()
        _, dqdf_device = feature_values_and_dqdf_device(
            model,
            individual,
            dataset.F,
            variable_names,
            value_limit=value_limit,
            precision=precision,
            cache=cache,
            data_key=("stress", id(dataset), dataset.num_points),
        )
        block_until_ready(dqdf_device)
        if timing is not None:
            timing["weak_form_jax_gene_derivative_seconds"] += time.perf_counter() - start
        start = time.perf_counter()
        dqdf = np.asarray(dqdf_device, dtype=float)
        if timing is not None:
            timing["weak_form_jax_transfer_seconds"] += time.perf_counter() - start
        features = np.zeros((dataset.target_vector.size, len(individual)), dtype=float)
        valid = np.zeros(len(individual), dtype=bool)
        normalized_columns = []
        for gene_index in range(len(individual)):
            column = dqdf[:, gene_index, :].reshape(-1)
            norm = np.linalg.norm(column)
            if not _valid(column, value_limit) or norm < 1e-12:
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


@lru_cache(maxsize=1)
def _weak_lhs_kernel():
    jax, jnp = require_jax_fem_backend(enable_x64=None)

    @partial(jax.jit, static_argnames=("num_nodes",))
    def kernel(B_matrices, qp_weights, dof_indices, dqdf, num_nodes: int):
        element_lhs = jnp.einsum(
            "edf,egf,e->edg",
            jnp.swapaxes(B_matrices, 1, 2),
            dqdf,
            qp_weights,
        )
        num_features = dqdf.shape[1]
        lhs = jnp.zeros((2 * num_nodes, num_features), dtype=element_lhs.dtype)
        lhs = lhs.at[dof_indices.reshape(-1), :].add(element_lhs.reshape(-1, num_features))
        return lhs

    return kernel


def compute_weak_lhs_device(case: JaxWeakFormCase, dqdf):
    return _weak_lhs_kernel()(case.B_matrices, case.qp_weights, case.dof_indices, dqdf, case.num_nodes)


def compute_weak_lhs(case: JaxWeakFormCase, dqdf: np.ndarray) -> np.ndarray:
    _, jnp = require_jax_fem_backend()
    dqdf_jax = jnp.asarray(dqdf, dtype=case.F.dtype)
    lhs = compute_weak_lhs_device(case, dqdf_jax)
    return np.asarray(lhs, dtype=float)


@lru_cache(maxsize=1)
def _reaction_balance_kernel():
    jax, jnp = require_jax_fem_backend(enable_x64=None)

    @jax.jit
    def kernel(weak_lhs, free_dof_indices, reaction_dofs, reaction_forces, balance):
        lhs_bulk = weak_lhs[free_dof_indices, :]
        lhs = 2.0 * lhs_bulk.T.dot(lhs_bulk)
        reaction_lhs = jnp.zeros_like(lhs)
        reaction_rhs = jnp.zeros(lhs.shape[0], dtype=weak_lhs.dtype)
        for dofs, force in zip(reaction_dofs, reaction_forces):
            reaction_sensitivity = jnp.sum(weak_lhs[dofs, :], axis=0)
            reaction_lhs = reaction_lhs + 2.0 * jnp.outer(reaction_sensitivity, reaction_sensitivity)
            reaction_rhs = reaction_rhs + 2.0 * reaction_sensitivity * force
        return lhs + balance * reaction_lhs, balance * reaction_rhs

    return kernel


def compute_reaction_balance_device(case: JaxWeakFormCase, weak_lhs, balance: float):
    _, jnp = require_jax_fem_backend(enable_x64=None)
    return _reaction_balance_kernel()(
        weak_lhs,
        case.free_dof_indices,
        case.reaction_dofs,
        case.reaction_forces,
        jnp.asarray(balance, dtype=weak_lhs.dtype),
    )


@lru_cache(maxsize=1)
def _residual_operator_kernel():
    jax, jnp = require_jax_fem_backend(enable_x64=None)

    @jax.jit
    def kernel(weak_lhs, free_dof_indices, reaction_dofs, reaction_forces, balance):
        balance_sqrt = jnp.sqrt(balance)
        rows = [weak_lhs[free_dof_indices, :]]
        targets = [jnp.zeros(free_dof_indices.shape[0], dtype=weak_lhs.dtype)]
        for dofs, force in zip(reaction_dofs, reaction_forces):
            rows.append(balance_sqrt * jnp.sum(weak_lhs[dofs, :], axis=0, keepdims=True))
            targets.append(jnp.asarray([balance_sqrt * force], dtype=weak_lhs.dtype))
        return jnp.concatenate(rows, axis=0), jnp.concatenate(targets, axis=0)

    return kernel


def compute_residual_operator_device(case: JaxWeakFormCase, weak_lhs, balance: float):
    _, jnp = require_jax_fem_backend(enable_x64=None)
    return _residual_operator_kernel()(
        weak_lhs,
        case.free_dof_indices,
        case.reaction_dofs,
        case.reaction_forces,
        jnp.asarray(balance, dtype=weak_lhs.dtype),
    )


def _valid(values, limit: float) -> bool:
    values = np.asarray(values, dtype=float)
    return bool(values.size and np.all(np.isfinite(values)) and np.max(np.abs(values)) <= limit)
