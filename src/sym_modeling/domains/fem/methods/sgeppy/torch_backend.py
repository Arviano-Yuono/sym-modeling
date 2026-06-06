from __future__ import annotations

import importlib
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Sequence

import geppy as gep
import numpy as np

from .backend import WeakFormEvaluationCache, backend_timing_keys


TORCH_FEM_EXTRA = "torch_fem"
TORCH_PRECISIONS = {"float64", "float32"}
TORCH_TIMING_KEYS = backend_timing_keys("torch")


def require_torch_backend():
    """Import Torch and require a CUDA-capable runtime."""
    try:
        torch = importlib.import_module("torch")
    except (ImportError, OSError) as exc:
        raise ImportError(
            "backend='torch' requires optional dependencies and CUDA. "
            'Install them with: pip install -e ".[torch_fem]" and ensure an '
            "NVIDIA driver plus a CUDA-capable Torch build are available."
        ) from exc
    if not torch.cuda.is_available():
        raise ImportError(
            "backend='torch' requires CUDA, but torch.cuda.is_available() is false. "
            'Install with: pip install -e ".[torch_fem]" and ensure an NVIDIA '
            "driver plus a CUDA-capable Torch build are available."
        )
    return torch


def configure_torch_precision(precision: str):
    if precision not in TORCH_PRECISIONS:
        raise ValueError("precision must be one of: float32, float64.")
    return require_torch_backend()


def _torch_dtype(torch, precision: str):
    configure_torch_precision(precision)
    return torch.float64 if precision == "float64" else torch.float32


def _as_cuda_tensor(torch, value, dtype, device):
    if torch.is_tensor(value):
        return value.to(dtype=dtype, device=device)
    return torch.as_tensor(np.asarray(value).copy(), dtype=dtype, device=device)


def is_torch_cuda_backend_available() -> bool:
    try:
        require_torch_backend()
    except ImportError:
        return False
    return True


def block_until_ready(value):
    torch = require_torch_backend()
    if isinstance(value, dict):
        for item in value.values():
            block_until_ready(item)
        return value
    if isinstance(value, (list, tuple)):
        for item in value:
            block_until_ready(item)
        return value
    if hasattr(value, "is_cuda") and value.is_cuda:
        torch.cuda.synchronize(value.device)
    return value


def torch_gene_operators():
    """Torch implementations of the supported arithmetic/math operators."""
    torch = require_torch_backend()
    eps = 1e-12

    def _protected_div(a, b):
        return a / torch.where(torch.abs(b) < 1e-6, torch.ones_like(b), b)

    return {
        "add": lambda a, b: a + b,
        "sub": lambda a, b: a - b,
        "mul": lambda a, b: a * b,
        "protected_div": _protected_div,
        "neg": lambda a: -a,
        "square": lambda a: torch.square(a),
        "cube": lambda a: torch.pow(a, 3),
        "protected_sqrt": lambda a: torch.sqrt(torch.abs(a) + eps),
        "protected_log": lambda a: torch.log(torch.abs(a) + eps),
        "protected_exp": lambda a: torch.exp(torch.clamp(a, -20.0, 20.0)),
        "sin": lambda a: torch.sin(a),
        "cos": lambda a: torch.cos(a),
    }


def _variables_from_F(F_row, variable_names):
    torch = require_torch_backend()
    F11, F12, F21, F22 = F_row[0], F_row[1], F_row[2], F_row[3]
    C11 = F11**2 + F21**2
    C12 = F11 * F12 + F21 * F22
    C21 = C12
    C22 = F12**2 + F22**2
    I1 = C11 + C22 + 1.0
    I2 = C11 + C22 - C12 * C21 + C11 * C22
    I3 = C11 * C22 - C12 * C21
    J = F11 * F22 - F12 * F21
    K1 = I1 * torch.pow(I3, -1.0 / 3.0) - 3.0
    K2 = (I1 + I3 - 1.0) * torch.pow(I3, -2.0 / 3.0) - 3.0
    lookup = {
        "I1": I1,
        "I2": I2,
        "I3": I3,
        "J": J,
        "Jm1": J - 1.0,
        "K1": K1,
        "K2": K2,
        "logI13": torch.log(K1 / 3.0 + 1.0),
        "logI23": torch.log(K2 / 3.0 + 1.0),
    }
    return [lookup[name] for name in variable_names]


def torch_invariant_values_and_derivatives(F, variable_names: Sequence[str], precision: str = "float64"):
    """Compute selected invariant variables and their analytic d(variable)/dF.

    Parameters
    ----------
    F : torch.Tensor or array-like
        Flattened deformation gradients, shape ``[num_points, 4]`` with columns
        ``[F11, F12, F21, F22]``.
    variable_names : sequence of str
        Variable names requested by the symbolic genes.
    precision : str
        ``"float64"`` or ``"float32"``.

    Returns
    -------
    variables : torch.Tensor
        Shape ``[num_points, num_variables]``.
    dvariables_dF : torch.Tensor
        Shape ``[num_points, num_variables, 4]``.
    """
    torch = require_torch_backend()
    dtype = _torch_dtype(torch, precision)
    device = torch.device("cuda")
    F = _as_cuda_tensor(torch, F, dtype, device)

    F11 = F[:, 0]
    F12 = F[:, 1]
    F21 = F[:, 2]
    F22 = F[:, 3]
    zeros = torch.zeros_like(F11)

    C11 = F11**2 + F21**2
    C12 = F11 * F12 + F21 * F22
    C22 = F12**2 + F22**2
    I1 = C11 + C22 + 1.0
    I2 = C11 + C22 - C12 * C12 + C11 * C22
    I3 = C11 * C22 - C12 * C12
    J = F11 * F22 - F12 * F21
    K1 = I1 * torch.pow(I3, -1.0 / 3.0) - 3.0
    K2 = I2 * torch.pow(I3, -2.0 / 3.0) - 3.0

    dI1 = torch.stack((2.0 * F11, 2.0 * F12, 2.0 * F21, 2.0 * F22), dim=1)
    dJ = torch.stack((F22, -F21, -F12, F11), dim=1)
    dI3 = 2.0 * J[:, None] * dJ

    dC11 = torch.stack((2.0 * F11, zeros, 2.0 * F21, zeros), dim=1)
    dC12 = torch.stack((F12, F11, F22, F21), dim=1)
    dC22 = torch.stack((zeros, 2.0 * F12, zeros, 2.0 * F22), dim=1)
    dI2 = (1.0 + C22)[:, None] * dC11 + (1.0 + C11)[:, None] * dC22 - 2.0 * C12[:, None] * dC12

    dK1 = torch.pow(I3, -1.0 / 3.0)[:, None] * dI1 + (
        I1 * (-1.0 / 3.0) * torch.pow(I3, -4.0 / 3.0)
    )[:, None] * dI3
    dK2 = torch.pow(I3, -2.0 / 3.0)[:, None] * dI2 + (
        I2 * (-2.0 / 3.0) * torch.pow(I3, -5.0 / 3.0)
    )[:, None] * dI3
    dlogI13 = dK1 / (K1 + 3.0)[:, None]
    dlogI23 = dK2 / (K2 + 3.0)[:, None]

    value_lookup = {
        "I1": I1,
        "I2": I2,
        "I3": I3,
        "J": J,
        "Jm1": J - 1.0,
        "K1": K1,
        "K2": K2,
        "logI13": torch.log(K1 / 3.0 + 1.0),
        "logI23": torch.log(K2 / 3.0 + 1.0),
    }
    derivative_lookup = {
        "I1": dI1,
        "I2": dI2,
        "I3": dI3,
        "J": dJ,
        "Jm1": dJ,
        "K1": dK1,
        "K2": dK2,
        "logI13": dlogI13,
        "logI23": dlogI23,
    }
    variables = torch.stack(tuple(value_lookup[name] for name in variable_names), dim=1)
    dvariables_dF = torch.stack(tuple(derivative_lookup[name] for name in variable_names), dim=1)
    return variables, dvariables_dF


class TorchBatchGeneEvaluatorCache:
    """Bounded LRU cache for joint selected-gene Torch evaluators."""

    def __init__(self, enabled=True, max_size=1024, timing=None):
        self.enabled = bool(enabled)
        self.max_size = max(0, int(max_size))
        self.timing = timing
        self._store = OrderedDict()

    @staticmethod
    def _key(gene_strings, variable_names, precision):
        return (
            "torch_joint_chain_rule",
            tuple(str(gene) for gene in gene_strings),
            tuple(variable_names),
            precision or "float64",
        )

    def get(self, gene_strings, variable_names, precision):
        if not self.enabled:
            return None
        key = self._key(gene_strings, variable_names, precision)
        if key in self._store:
            self._store.move_to_end(key)
            return self._store[key]
        return None

    def put(self, gene_strings, variable_names, precision, evaluator):
        if not self.enabled or self.max_size <= 0:
            return
        key = self._key(gene_strings, variable_names, precision)
        self._store[key] = evaluator
        self._store.move_to_end(key)
        while len(self._store) > self.max_size:
            self._store.popitem(last=False)

    @property
    def size(self):
        return len(self._store)

    def clear(self):
        self._store.clear()


def _make_joint_gene_evaluator(model, individual, selected: Sequence[int], variable_names: Sequence[str], gene_backend):
    torch = require_torch_backend()
    torch_func = importlib.import_module("torch.func")
    from .gene_evaluator import compile_gene_function

    gene_fns = [
        compile_gene_function(individual[int(gene_index)], model, gene_backend)
        for gene_index in selected
    ]

    def _vector_eval(variable_row):
        args = [variable_row[index] for index in range(len(variable_names))]
        values = []
        for gene_fn in gene_fns:
            value = gene_fn(*args)
            if not torch.is_tensor(value):
                value = variable_row.sum() * 0.0 + float(value)
            values.append(value)
        return torch.stack(tuple(values), dim=0)

    vmap_eval = torch_func.vmap(_vector_eval)
    vmap_jacobian = torch_func.vmap(torch_func.jacrev(_vector_eval))

    def evaluate(variables, dvariables_dF):
        features = vmap_eval(variables)
        dQ_dvariables = vmap_jacobian(variables)
        dqdf = torch.einsum("ngv,nvf->ngf", dQ_dvariables, dvariables_dF)
        return features, dqdf

    return evaluate


def evaluate_genes_with_shared_invariants(
    batch_cache,
    model,
    individual,
    gene_indices,
    F_batch,
    variable_names,
    precision="float64",
    gene_backend=None,
):
    """Evaluate selected genes by sharing invariant work across all genes."""
    torch = require_torch_backend()
    selected = [int(index) for index in gene_indices]
    if not selected:
        dtype = _torch_dtype(torch, precision)
        device = torch.device("cuda")
        F_batch = _as_cuda_tensor(torch, F_batch, dtype, device)
        return (
            torch.zeros((F_batch.shape[0], 0), dtype=dtype, device=device),
            torch.zeros((F_batch.shape[0], 0, 4), dtype=dtype, device=device),
        )

    gene_backend = gene_backend or TorchGeneEvaluationBackend()
    precision_key = gene_backend.cache_precision(precision)
    gene_strings = tuple(str(individual[index]) for index in selected)
    evaluator = None
    if batch_cache is not None:
        evaluator = batch_cache.get(gene_strings, variable_names, precision_key)

    if evaluator is None:
        compile_start = time.perf_counter()
        evaluator = _make_joint_gene_evaluator(model, individual, selected, variable_names, gene_backend)
        if batch_cache is not None:
            batch_cache.put(gene_strings, variable_names, precision_key, evaluator)
            if batch_cache.timing is not None:
                batch_cache.timing[gene_backend.compile_timing_key] = (
                    batch_cache.timing.get(gene_backend.compile_timing_key, 0.0)
                    + time.perf_counter()
                    - compile_start
                )

    variables, dvariables_dF = torch_invariant_values_and_derivatives(
        F_batch,
        variable_names,
        precision=precision_key,
    )
    return evaluator(variables, dvariables_dF)


class TorchGeneEvaluationBackend:
    """Torch-CUDA strategy for compiling one gene into value and gradient evaluators."""

    name = "torch"
    default_precision = "float64"
    compile_timing_key = "torch_gene_compile_seconds"

    def build_primitive_set(self, model):
        from .sgep import BINARY_OPS, UNARY_OPS

        operators = torch_gene_operators()
        pset = gep.PrimitiveSet("TorchMain", model.config.variable_names)
        for name in model.config.unary_operators:
            if name not in UNARY_OPS:
                raise ValueError("Unsupported unary operator for Torch backend: %s" % name)
            pset.add_function(operators[name], 1, name=name)
        for name in model.config.binary_operators:
            if name not in BINARY_OPS:
                raise ValueError("Unsupported binary operator for Torch backend: %s" % name)
            pset.add_function(operators[name], 2, name=name)
        return pset

    def make_gene_evaluator(self, gene_fn, variable_names: Sequence[str]):
        torch = require_torch_backend()
        torch_func = importlib.import_module("torch.func")

        def _scalar_eval(F_row):
            args = _variables_from_F(F_row, variable_names)
            value = gene_fn(*args)
            if not torch.is_tensor(value):
                value = torch.as_tensor(value, dtype=F_row.dtype, device=F_row.device)
            return value

        _vmap_eval = torch_func.vmap(_scalar_eval)
        _vmap_grad = torch_func.vmap(torch_func.grad(_scalar_eval))

        def evaluate(F_batch):
            return _vmap_eval(F_batch), _vmap_grad(F_batch)

        return evaluate

    def empty_result(self, F_batch):
        n = F_batch.shape[0]
        dtype = F_batch.dtype
        device = F_batch.device
        torch = require_torch_backend()
        return (
            torch.zeros((n, 0), dtype=dtype, device=device),
            torch.zeros((n, 0, 4), dtype=dtype, device=device),
        )

    def stack_gene_results(self, values: Sequence, derivatives: Sequence):
        torch = require_torch_backend()
        return torch.stack(tuple(values), dim=1), torch.stack(tuple(derivatives), dim=1)

    def cache_precision(self, precision: str | None) -> str:
        return precision or self.default_precision


@dataclass(frozen=True)
class TorchWeakFormCase:
    data: object
    F: object
    B_matrices: object
    qp_weights: object
    dof_indices: object
    free_dof_indices: object
    reaction_dofs: tuple
    reaction_forces: object
    num_nodes: int


class TorchWeakFormEvaluationCache(WeakFormEvaluationCache):
    """Bounded LRU caches for repeated Torch weak-form individuals."""

    def __init__(
        self,
        device_name: str = "cuda",
        enabled: bool = True,
        max_size: int = 256,
        device_outputs: bool = True,
        timing: dict[str, float] | None = None,
    ):
        super().__init__(
            "torch",
            device_name=device_name,
            enabled=enabled,
            max_size=max_size,
            device_outputs=device_outputs,
            timing=timing,
        )


def prepare_torch_case(cache, precision: str = "float64") -> TorchWeakFormCase:
    torch = configure_torch_precision(precision)
    dtype = _torch_dtype(torch, precision)
    device = torch.device("cuda")
    data = cache.data
    nodes = np.stack([np.asarray(part, dtype=int) for part in data.connectivity], axis=1)
    dof_indices = np.stack((2 * nodes, 2 * nodes + 1), axis=-1).reshape(nodes.shape[0], -1)
    reaction_dofs = tuple(
        torch.as_tensor(np.flatnonzero(dofs), dtype=torch.long, device=device)
        for dofs, _ in cache.reactions
    )
    return TorchWeakFormCase(
        data=data,
        F=_as_cuda_tensor(torch, data.F, dtype, device),
        B_matrices=_as_cuda_tensor(torch, cache.B_matrices, dtype, device),
        qp_weights=_as_cuda_tensor(torch, data.qpWeights, dtype, device),
        dof_indices=torch.as_tensor(dof_indices, dtype=torch.long, device=device),
        free_dof_indices=torch.as_tensor(np.flatnonzero(cache.free_dofs), dtype=torch.long, device=device),
        reaction_dofs=reaction_dofs,
        reaction_forces=torch.as_tensor([force for _, force in cache.reactions], dtype=dtype, device=device),
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
    cache: TorchWeakFormEvaluationCache | None = None,
    gene_cache=None,
    data_key=None,
    gene_backend: TorchGeneEvaluationBackend | None = None,
) -> tuple:
    torch = require_torch_backend()
    dtype = _torch_dtype(torch, precision or "float64")
    device = torch.device("cuda")
    F = _as_cuda_tensor(torch, F, dtype, device)
    selected = list(range(len(individual))) if gene_indices is None else [int(index) for index in gene_indices]

    if not selected:
        return (
            torch.zeros((F.shape[0], 0), dtype=dtype, device=device),
            torch.zeros((F.shape[0], 0, 4), dtype=dtype, device=device),
        )

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

    execute_start = time.perf_counter()
    features, dqdf = evaluate_genes_with_shared_invariants(
        gene_cache,
        model,
        individual,
        selected,
        F,
        variable_names,
        precision=precision or "float64",
        gene_backend=gene_backend or TorchGeneEvaluationBackend(),
    )
    block_until_ready((features, dqdf))
    if cache is not None and cache.timing is not None:
        timing_key = cache.timing_key("gene_execute_seconds")
        cache.timing[timing_key] = cache.timing.get(timing_key, 0.0) + time.perf_counter() - execute_start

    finite = torch.all(torch.isfinite(features)) & torch.all(torch.isfinite(dqdf))
    bounded = (torch.max(torch.abs(features)) <= value_limit) & (torch.max(torch.abs(dqdf)) <= value_limit)
    if not bool((finite & bounded).detach().cpu().item()):
        raise ValueError("Invalid Torch weak-form gene value or derivative.")
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
    cache: TorchWeakFormEvaluationCache | None = None,
    gene_cache=None,
    data_key=None,
    gene_backend: TorchGeneEvaluationBackend | None = None,
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
    features = features_device.detach().cpu().numpy().astype(float, copy=False)
    dqdf = dqdf_device.detach().cpu().numpy().astype(float, copy=False)
    if features.shape[1] == 0:
        return features, dqdf
    if not _valid(features, value_limit) or not _valid(dqdf, value_limit):
        raise ValueError("Invalid Torch weak-form gene value or derivative.")
    return features, dqdf


def stress_feature_builder(
    dataset,
    variable_names: Sequence[str],
    value_limit: float = 1e8,
    duplicate_correlation: float = 0.999999,
    precision: str = "float64",
    timing: dict[str, float] | None = None,
    cache: TorchWeakFormEvaluationCache | None = None,
    gene_cache=None,
    gene_backend: TorchGeneEvaluationBackend | None = None,
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
            key = cache.timing_key("gene_derivative_seconds") if cache is not None else "torch_gene_derivative_seconds"
            timing[key] = timing.get(key, 0.0) + time.perf_counter() - start
        start = time.perf_counter()
        dqdf = dqdf_device.detach().cpu().numpy().astype(float, copy=False)
        if timing is not None:
            key = cache.timing_key("transfer_seconds") if cache is not None else "torch_transfer_seconds"
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


def compute_weak_lhs_device(case: TorchWeakFormCase, dqdf):
    torch = require_torch_backend()
    element_lhs = torch.einsum(
        "edf,egf,e->edg",
        torch.swapaxes(case.B_matrices, 1, 2),
        dqdf,
        case.qp_weights,
    )
    num_features = dqdf.shape[1]
    lhs = torch.zeros((2 * case.num_nodes, num_features), dtype=element_lhs.dtype, device=element_lhs.device)
    lhs.index_add_(0, case.dof_indices.reshape(-1), element_lhs.reshape(-1, num_features))
    return lhs


def compute_weak_lhs(case: TorchWeakFormCase, dqdf: np.ndarray) -> np.ndarray:
    torch = require_torch_backend()
    dqdf_torch = torch.as_tensor(dqdf, dtype=case.F.dtype, device=case.F.device)
    lhs = compute_weak_lhs_device(case, dqdf_torch)
    return lhs.detach().cpu().numpy().astype(float, copy=False)


def compute_reaction_balance_device(case: TorchWeakFormCase, weak_lhs, balance: float):
    torch = require_torch_backend()
    lhs_bulk = weak_lhs[case.free_dof_indices, :]
    lhs = 2.0 * lhs_bulk.T.matmul(lhs_bulk)
    reaction_lhs = torch.zeros_like(lhs)
    reaction_rhs = torch.zeros(lhs.shape[0], dtype=weak_lhs.dtype, device=weak_lhs.device)
    for dofs, force in zip(case.reaction_dofs, case.reaction_forces):
        reaction_sensitivity = torch.sum(weak_lhs[dofs, :], dim=0)
        reaction_lhs = reaction_lhs + 2.0 * torch.outer(reaction_sensitivity, reaction_sensitivity)
        reaction_rhs = reaction_rhs + 2.0 * reaction_sensitivity * force
    return lhs + balance * reaction_lhs, balance * reaction_rhs


def compute_residual_operator_device(case: TorchWeakFormCase, weak_lhs, balance: float):
    torch = require_torch_backend()
    balance_sqrt = torch.sqrt(torch.as_tensor(balance, dtype=weak_lhs.dtype, device=weak_lhs.device))
    rows = [weak_lhs[case.free_dof_indices, :]]
    targets = [torch.zeros(case.free_dof_indices.shape[0], dtype=weak_lhs.dtype, device=weak_lhs.device)]
    for dofs, force in zip(case.reaction_dofs, case.reaction_forces):
        rows.append(balance_sqrt * torch.sum(weak_lhs[dofs, :], dim=0, keepdim=True))
        targets.append((balance_sqrt * force).reshape(1))
    return torch.cat(rows, dim=0), torch.cat(targets, dim=0)


def _valid(values, limit: float) -> bool:
    values = np.asarray(values, dtype=float)
    return bool(values.size and np.all(np.isfinite(values)) and np.max(np.abs(values)) <= limit)


class TorchWeakFormBackend:
    """Strategy object that owns Torch-CUDA weak-form configuration and caches."""

    name = "torch"

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
        self.gene_backend = TorchGeneEvaluationBackend()
        self.eval_cache = None
        self.gene_cache = None
        self.device = None

    def configure(self) -> None:
        torch = configure_torch_precision(self.precision)
        self.device = torch.device("cuda")
        self.eval_cache = TorchWeakFormEvaluationCache(
            device_name=str(self.device),
            enabled=self.cache_enabled,
            max_size=self.cache_size,
            device_outputs=self.cache_device_outputs,
            timing=self.timing,
        )
        self.gene_cache = TorchBatchGeneEvaluatorCache(
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

    def prepare_case(self, cache) -> TorchWeakFormCase:
        return prepare_torch_case(cache, precision=self.precision)

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

    def compute_weak_lhs_device(self, case: TorchWeakFormCase, dqdf):
        return compute_weak_lhs_device(case, dqdf)

    def compute_reaction_balance_device(self, case: TorchWeakFormCase, weak_lhs, balance: float):
        return compute_reaction_balance_device(case, weak_lhs, balance)

    def compute_residual_operator_device(self, case: TorchWeakFormCase, weak_lhs, balance: float):
        return compute_residual_operator_device(case, weak_lhs, balance)

    def block_until_ready(self, value):
        return block_until_ready(value)

    def timing_key(self, suffix: str) -> str:
        return "%s_%s" % (self.name, suffix)

    def _ensure_configured(self) -> None:
        if self.eval_cache is None or self.gene_cache is None:
            self.configure()
