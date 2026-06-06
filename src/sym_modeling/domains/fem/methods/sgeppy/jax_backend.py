from __future__ import annotations

import importlib
import importlib.util
import time
from dataclasses import dataclass
from functools import lru_cache, partial
from typing import Sequence

import geppy as gep
import numpy as np

from .backend import WeakFormEvaluationCache, backend_timing_keys


JAX_FEM_EXTRA = "jax_fem"
JAX_PRECISIONS = {"float64", "float32"}
JAX_TIMING_KEYS = backend_timing_keys("jax")


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
            "backend='jax' requires optional dependencies. "
            'Install them with: pip install -e ".[jax_fem]"'
        ) from exc
    return jax, jnp


def configure_jax_precision(precision: str):
    if precision not in JAX_PRECISIONS:
        raise ValueError("precision must be one of: float32, float64.")
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


def jax_gene_operators():
    """JAX implementations of the supported arithmetic/math operators."""
    _, jnp = require_jax_fem_backend(enable_x64=None)
    eps = 1e-12

    def _protected_div(a, b):
        return a / jnp.where(jnp.abs(b) < 1e-6, 1.0, b)

    return {
        "add": lambda a, b: a + b,
        "sub": lambda a, b: a - b,
        "mul": lambda a, b: a * b,
        "protected_div": _protected_div,
        "neg": lambda a: -a,
        "square": lambda a: jnp.square(a),
        "cube": lambda a: jnp.power(a, 3),
        "protected_sqrt": lambda a: jnp.sqrt(jnp.abs(a) + eps),
        "protected_log": lambda a: jnp.log(jnp.abs(a) + eps),
        "protected_exp": lambda a: jnp.exp(jnp.clip(a, -20.0, 20.0)),
        "sin": lambda a: jnp.sin(a),
        "cos": lambda a: jnp.cos(a),
    }


def _variables_from_F(F_row, variable_names):
    """Compute named strain-invariant variables from one deformation gradient."""
    _, jnp = require_jax_fem_backend(enable_x64=None)
    F11, F12, F21, F22 = F_row[0], F_row[1], F_row[2], F_row[3]
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
    lookup = {
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
    return [lookup[name] for name in variable_names]


class JaxGeneEvaluationBackend:
    """JAX strategy for compiling one gene into value and gradient evaluators."""

    name = "jax"
    default_precision = "float64"
    compile_timing_key = "jax_gene_compile_seconds"

    def build_primitive_set(self, model):
        from .sgep import BINARY_OPS, UNARY_OPS

        operators = jax_gene_operators()
        pset = gep.PrimitiveSet("JaxMain", model.config.variable_names)
        for name in model.config.unary_operators:
            if name not in UNARY_OPS:
                raise ValueError("Unsupported unary operator for JAX backend: %s" % name)
            pset.add_function(operators[name], 1, name=name)
        for name in model.config.binary_operators:
            if name not in BINARY_OPS:
                raise ValueError("Unsupported binary operator for JAX backend: %s" % name)
            pset.add_function(operators[name], 2, name=name)
        return pset

    def make_gene_evaluator(self, gene_fn, variable_names: Sequence[str]):
        jax, _ = require_jax_fem_backend(enable_x64=None)

        def _scalar_eval(F_row):
            args = _variables_from_F(F_row, variable_names)
            return gene_fn(*args)

        _vmap_eval = jax.vmap(_scalar_eval)
        _vmap_grad = jax.vmap(jax.grad(_scalar_eval))

        @jax.jit
        def evaluate(F_batch):
            return _vmap_eval(F_batch), _vmap_grad(F_batch)

        return evaluate

    def empty_result(self, F_batch):
        _, jnp = require_jax_fem_backend(enable_x64=None)
        n = F_batch.shape[0]
        dtype = F_batch.dtype
        return jnp.zeros((n, 0), dtype=dtype), jnp.zeros((n, 0, 4), dtype=dtype)

    def stack_gene_results(self, values: Sequence, derivatives: Sequence):
        _, jnp = require_jax_fem_backend(enable_x64=None)
        return jnp.stack(values, axis=1), jnp.stack(derivatives, axis=1)

    def cache_precision(self, precision: str | None) -> str:
        return precision or self.default_precision


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


class JaxWeakFormEvaluationCache(WeakFormEvaluationCache):
    """Bounded LRU caches for repeated JAX weak-form individuals."""

    def __init__(
        self,
        device_name: str = "default",
        enabled: bool = True,
        max_size: int = 256,
        device_outputs: bool = True,
        timing: dict[str, float] | None = None,
    ):
        super().__init__(
            "jax",
            device_name=device_name,
            enabled=enabled,
            max_size=max_size,
            device_outputs=device_outputs,
            timing=timing,
        )


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
    gene_cache=None,
    data_key=None,
    gene_backend: JaxGeneEvaluationBackend | None = None,
) -> tuple:
    _, jnp = require_jax_fem_backend(enable_x64=None if precision is not None else True)
    dtype = _jax_dtype(precision) if precision is not None else (F.dtype if hasattr(F, "dtype") else jnp.float64)
    F = jnp.asarray(F, dtype=dtype)
    selected = list(range(len(individual))) if gene_indices is None else [int(index) for index in gene_indices]

    if not selected:
        return jnp.zeros((F.shape[0], 0), dtype=dtype), jnp.zeros((F.shape[0], 0, 4), dtype=dtype)

    # Check the per-individual artifact cache first (keyed by data + individual).
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

    # Evaluate each gene using its per-gene backend evaluator.
    from .gene_evaluator import evaluate_genes_on_F

    execute_start = time.perf_counter()
    features, dqdf = evaluate_genes_on_F(
        gene_cache,
        model,
        individual,
        selected,
        F,
        variable_names,
        precision=precision or "float64",
        backend=gene_backend or JaxGeneEvaluationBackend(),
    )
    block_until_ready((features, dqdf))
    if cache is not None and cache.timing is not None:
        timing_key = cache.timing_key("gene_execute_seconds")
        cache.timing[timing_key] = (
            cache.timing.get(timing_key, 0.0)
            + time.perf_counter()
            - execute_start
        )

    finite = jnp.all(jnp.isfinite(features)) & jnp.all(jnp.isfinite(dqdf))
    bounded = (jnp.max(jnp.abs(features)) <= value_limit) & (jnp.max(jnp.abs(dqdf)) <= value_limit)
    if not bool(np.asarray(finite & bounded)):
        raise ValueError("Invalid JAX weak-form gene value or derivative.")
    if cache is not None and cache.device_outputs and derivative_key is not None:
        cache.put_artifact(derivative_key, (features, dqdf))
    return features, dqdf


def feature_values_and_dqdf(
    model,
    individual,
    F,
    variable_names: Sequence[str],
    gene_indices: Sequence[int] | None = None,
    value_limit: float = 1e8,
    precision: str | None = None,
    cache: JaxWeakFormEvaluationCache | None = None,
    gene_cache=None,
    data_key=None,
    gene_backend: JaxGeneEvaluationBackend | None = None,
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
        gene_cache=gene_cache,
        data_key=data_key,
        gene_backend=gene_backend,
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
    gene_cache=None,
    gene_backend: JaxGeneEvaluationBackend | None = None,
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
            gene_cache=gene_cache,
            data_key=("stress", id(dataset), dataset.num_points),
            gene_backend=gene_backend,
        )
        block_until_ready(dqdf_device)
        if timing is not None:
            key = cache.timing_key("gene_derivative_seconds") if cache is not None else "jax_gene_derivative_seconds"
            timing[key] = timing.get(key, 0.0) + time.perf_counter() - start
        start = time.perf_counter()
        dqdf = np.asarray(dqdf_device, dtype=float)
        if timing is not None:
            key = cache.timing_key("transfer_seconds") if cache is not None else "jax_transfer_seconds"
            timing[key] = timing.get(key, 0.0) + time.perf_counter() - start
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


class JaxWeakFormBackend:
    """Strategy object that owns JAX weak-form configuration and caches."""

    name = "jax"

    def __init__(
        self,
        precision: str = "float64",
        cache_enabled: bool = True,
        cache_size: int = 256,
        cache_device_outputs: bool = True,
        gene_cache_size: int = 1024,
        timing: dict[str, float] | None = None,
    ):
        self.precision = precision
        self.cache_enabled = bool(cache_enabled)
        self.cache_size = int(cache_size)
        self.cache_device_outputs = bool(cache_device_outputs)
        self.gene_cache_size = int(gene_cache_size)
        self.timing = timing
        self.gene_backend = JaxGeneEvaluationBackend()
        self.eval_cache = None
        self.gene_cache = None

    def configure(self) -> None:
        from .gene_evaluator import PerGeneEvaluatorCache

        jax, _ = configure_jax_precision(self.precision)
        try:
            device_name = str(jax.default_backend())
        except Exception:
            device_name = "default"
        self.eval_cache = JaxWeakFormEvaluationCache(
            device_name=device_name,
            enabled=self.cache_enabled,
            max_size=self.cache_size,
            device_outputs=self.cache_device_outputs,
            timing=self.timing,
        )
        self.gene_cache = PerGeneEvaluatorCache(
            enabled=self.cache_enabled,
            max_size=self.gene_cache_size,
            timing=self.timing,
        )

    def make_stress_feature_builder(
        self,
        dataset,
        variable_names: Sequence[str],
        value_limit: float = 1e8,
        duplicate_correlation: float = 0.999999,
    ):
        self._ensure_configured()
        return stress_feature_builder(
            dataset,
            variable_names,
            value_limit=value_limit,
            duplicate_correlation=duplicate_correlation,
            precision=self.precision,
            timing=self.timing,
            cache=self.eval_cache,
            gene_cache=self.gene_cache,
            gene_backend=self.gene_backend,
        )

    def prepare_case(self, cache) -> JaxWeakFormCase:
        return prepare_jax_case(cache, precision=self.precision)

    def feature_values_and_dqdf_device(
        self,
        model,
        individual,
        F,
        variable_names: Sequence[str],
        gene_indices: Sequence[int] | None = None,
        value_limit: float = 1e8,
        data_key=None,
    ) -> tuple:
        self._ensure_configured()
        return feature_values_and_dqdf_device(
            model,
            individual,
            F,
            variable_names,
            gene_indices=gene_indices,
            value_limit=value_limit,
            precision=self.precision,
            cache=self.eval_cache,
            gene_cache=self.gene_cache,
            data_key=data_key,
            gene_backend=self.gene_backend,
        )

    def compute_weak_lhs_device(self, case: JaxWeakFormCase, dqdf):
        return compute_weak_lhs_device(case, dqdf)

    def compute_reaction_balance_device(self, case: JaxWeakFormCase, weak_lhs, balance: float):
        return compute_reaction_balance_device(case, weak_lhs, balance)

    def compute_residual_operator_device(self, case: JaxWeakFormCase, weak_lhs, balance: float):
        return compute_residual_operator_device(case, weak_lhs, balance)

    def block_until_ready(self, value):
        return block_until_ready(value)

    def timing_key(self, suffix: str) -> str:
        return "%s_%s" % (self.name, suffix)

    def _ensure_configured(self) -> None:
        if self.eval_cache is None or self.gene_cache is None:
            self.configure()
