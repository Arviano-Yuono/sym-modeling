from __future__ import annotations

import importlib
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Sequence

import geppy as gep
import numpy as np

from .backend import WeakFormEvaluationCache, backend_timing_keys


TORCH_FEM_EXTRA = "torch_fem"
TORCH_PRECISIONS = {"float64", "float32"}
TORCH_POPULATION_GENE_CHUNK_SIZE = 128
TORCH_JACREV_CHUNK_SIZE = 16
TORCH_BATCH_TIMING_KEYS = (
    "torch_population_gene_seconds",
    "torch_gpu_filter_seconds",
    "torch_lazy_materialize_seconds",
    "torch_population_gene_oom_retries",
)
TORCH_TIMING_KEYS = backend_timing_keys("torch") + TORCH_BATCH_TIMING_KEYS


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


def _device_to_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
    return np.asarray(value, dtype=float)


def _is_torch_oom_error(torch, exc: BaseException) -> bool:
    oom_type = getattr(torch, "OutOfMemoryError", RuntimeError)
    return isinstance(exc, oom_type) or "out of memory" in str(exc).lower()


def _recover_after_torch_oom(torch) -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


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


def _make_joint_gene_evaluator_for_genes(model, genes: Sequence, variable_names: Sequence[str], gene_backend):
    torch = require_torch_backend()
    torch_func = importlib.import_module("torch.func")
    from .gene_evaluator import compile_gene_function

    gene_fns = [compile_gene_function(gene, model, gene_backend) for gene in genes]
    jacrev_chunk_size = max(1, min(TORCH_JACREV_CHUNK_SIZE, len(gene_fns)))

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
    vmap_jacobian = torch_func.vmap(torch_func.jacrev(_vector_eval, chunk_size=jacrev_chunk_size))

    def evaluate(variables, dvariables_dF):
        features = vmap_eval(variables)
        dQ_dvariables = vmap_jacobian(variables)
        dqdf = torch.einsum("ngv,nvf->ngf", dQ_dvariables, dvariables_dF)
        return features, dqdf

    return evaluate


def _make_joint_gene_evaluator(model, individual, selected: Sequence[int], variable_names: Sequence[str], gene_backend):
    genes = [individual[int(gene_index)] for gene_index in selected]
    return _make_joint_gene_evaluator_for_genes(model, genes, variable_names, gene_backend)


def evaluate_gene_objects_with_shared_invariants(
    batch_cache,
    model,
    genes: Sequence,
    F_batch,
    variable_names,
    precision="float64",
    gene_backend=None,
):
    """Evaluate an explicit sequence of genes with one shared invariant pass."""
    torch = require_torch_backend()
    if not genes:
        dtype = _torch_dtype(torch, precision)
        device = torch.device("cuda")
        F_batch = _as_cuda_tensor(torch, F_batch, dtype, device)
        return (
            torch.zeros((F_batch.shape[0], 0), dtype=dtype, device=device),
            torch.zeros((F_batch.shape[0], 0, 4), dtype=dtype, device=device),
        )

    gene_backend = gene_backend or TorchGeneEvaluationBackend()
    precision_key = gene_backend.cache_precision(precision)
    gene_strings = tuple(str(gene) for gene in genes)
    evaluator = None
    if batch_cache is not None:
        evaluator = batch_cache.get(gene_strings, variable_names, precision_key)

    if evaluator is None:
        compile_start = time.perf_counter()
        evaluator = _make_joint_gene_evaluator_for_genes(model, genes, variable_names, gene_backend)
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

    return evaluate_gene_objects_with_shared_invariants(
        batch_cache,
        model,
        [individual[index] for index in selected],
        F_batch,
        variable_names,
        precision=precision,
        gene_backend=gene_backend,
    )


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


@dataclass
class TorchStressFeatures:
    """Stress feature columns and valid mask kept on CUDA until needed."""

    features_device: object
    valid_mask: np.ndarray
    valid_indices: np.ndarray
    failed: bool = False
    _features_numpy: np.ndarray | None = field(default=None, init=False, repr=False)

    def features_numpy(self) -> np.ndarray:
        if self._features_numpy is None:
            self._features_numpy = _device_to_numpy(self.features_device)
        return self._features_numpy

    def prediction_numpy(self, theta: np.ndarray) -> np.ndarray:
        torch = require_torch_backend()
        theta_device = torch.as_tensor(theta, dtype=self.features_device.dtype, device=self.features_device.device)
        return _device_to_numpy(self.features_device.matmul(theta_device))

    @classmethod
    def failed_item(cls, num_rows: int, num_columns: int, dtype, device):
        torch = require_torch_backend()
        return cls(
            features_device=torch.zeros((num_rows, num_columns), dtype=dtype, device=device),
            valid_mask=np.zeros(num_columns, dtype=bool),
            valid_indices=np.zeros(0, dtype=int),
            failed=True,
        )


@dataclass
class TorchWeakFormArtifacts:
    """Weak-form matrices kept on CUDA with lazy NumPy materialization for LP."""

    lhs_device: object
    rhs_device: object
    residual_matrix_device: object
    residual_target_device: object
    _numpy_values: tuple[np.ndarray, np.ndarray, tuple[np.ndarray, np.ndarray]] | None = field(
        default=None,
        init=False,
        repr=False,
    )

    def block_until_ready(self):
        block_until_ready(
            (
                self.lhs_device,
                self.rhs_device,
                self.residual_matrix_device,
                self.residual_target_device,
            )
        )
        return self

    def to_numpy(self) -> tuple[np.ndarray, np.ndarray, tuple[np.ndarray, np.ndarray]]:
        if self._numpy_values is None:
            self._numpy_values = (
                _device_to_numpy(self.lhs_device),
                _device_to_numpy(self.rhs_device),
                (
                    _device_to_numpy(self.residual_matrix_device),
                    _device_to_numpy(self.residual_target_device),
                ),
            )
        return self._numpy_values


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


def population_feature_values_and_dqdf_device(
    model,
    individuals: Sequence,
    F,
    variable_names: Sequence[str],
    gene_indices_by_individual: Sequence[Sequence[int]] | None = None,
    value_limit: float = 1e8,
    precision: str = "float64",
    gene_cache=None,
    gene_backend: TorchGeneEvaluationBackend | None = None,
    timing: dict[str, float] | None = None,
    chunk_size: int = TORCH_POPULATION_GENE_CHUNK_SIZE,
) -> list[tuple]:
    """Evaluate unique genes from many individuals in CUDA-sized chunks."""
    torch = require_torch_backend()
    dtype = _torch_dtype(torch, precision)
    device = torch.device("cuda")
    F = _as_cuda_tensor(torch, F, dtype, device)
    gene_backend = gene_backend or TorchGeneEvaluationBackend()
    selected_by_individual = _selected_gene_indices(individuals, gene_indices_by_individual)
    unique_genes, individual_columns = _collect_unique_gene_columns(individuals, selected_by_individual)

    if not unique_genes:
        return [
            (
                torch.zeros((F.shape[0], 0), dtype=dtype, device=device),
                torch.zeros((F.shape[0], 0, 4), dtype=dtype, device=device),
            )
            for _ in individuals
        ]

    start = time.perf_counter()
    feature_chunks = []
    dqdf_chunks = []
    offset = 0
    active_chunk_size = max(1, int(chunk_size))
    while offset < len(unique_genes):
        chunk = unique_genes[offset: offset + active_chunk_size]
        try:
            features, dqdf = evaluate_gene_objects_with_shared_invariants(
                gene_cache,
                model,
                chunk,
                F,
                variable_names,
                precision=precision,
                gene_backend=gene_backend,
            )
            block_until_ready((features, dqdf))
        except RuntimeError as exc:
            if not _is_torch_oom_error(torch, exc) or active_chunk_size <= 1:
                raise
            _recover_after_torch_oom(torch)
            active_chunk_size = max(1, active_chunk_size // 2)
            if timing is not None:
                timing["torch_population_gene_oom_retries"] = timing.get("torch_population_gene_oom_retries", 0.0) + 1.0
            continue
        feature_chunks.append(features)
        dqdf_chunks.append(dqdf)
        offset += len(chunk)
    unique_features = torch.cat(tuple(feature_chunks), dim=1)
    unique_dqdf = torch.cat(tuple(dqdf_chunks), dim=1)
    block_until_ready((unique_features, unique_dqdf))
    if timing is not None:
        elapsed = time.perf_counter() - start
        timing["torch_population_gene_seconds"] = timing.get("torch_population_gene_seconds", 0.0) + elapsed
        timing["torch_gene_execute_seconds"] = timing.get("torch_gene_execute_seconds", 0.0) + elapsed

    results = []
    for columns in individual_columns:
        if not columns:
            results.append(
                (
                    torch.zeros((F.shape[0], 0), dtype=dtype, device=device),
                    torch.zeros((F.shape[0], 0, 4), dtype=dtype, device=device),
                )
            )
            continue
        column_index = torch.as_tensor(columns, dtype=torch.long, device=device)
        features = unique_features.index_select(1, column_index)
        dqdf = unique_dqdf.index_select(1, column_index)
        results.append((features, dqdf))
    return results


def build_population_stress_features(
    model,
    individuals: Sequence,
    dataset,
    variable_names: Sequence[str],
    value_limit: float = 1e8,
    duplicate_correlation: float = 0.999999,
    precision: str = "float64",
    timing: dict[str, float] | None = None,
    gene_cache=None,
    gene_backend: TorchGeneEvaluationBackend | None = None,
) -> list[TorchStressFeatures]:
    """Build stress feature matrices for many individuals with CUDA filtering."""
    gene_batches = population_feature_values_and_dqdf_device(
        model,
        individuals,
        dataset.F,
        variable_names,
        value_limit=value_limit,
        precision=precision,
        gene_cache=gene_cache,
        gene_backend=gene_backend,
        timing=timing,
    )
    stress_items = []
    for individual, (features, dqdf) in zip(individuals, gene_batches):
        num_columns = len(individual) + int(model.config.fit_intercept)
        if not _feature_batch_is_valid(features, dqdf, value_limit):
            dtype = dqdf.dtype
            device = dqdf.device
            stress_items.append(TorchStressFeatures.failed_item(dataset.target_vector.size, num_columns, dtype, device))
            continue
        filter_start = time.perf_counter()
        columns = stress_columns_from_dqdf(dqdf)
        filtered_columns, gene_valid_device = filter_stress_columns_device(
            columns,
            value_limit=value_limit,
            duplicate_correlation=duplicate_correlation,
        )
        features_device, valid_device = append_intercept_column_device(
            filtered_columns,
            gene_valid_device,
            fit_intercept=model.config.fit_intercept,
        )
        block_until_ready((features_device, valid_device))
        if timing is not None:
            timing["torch_gpu_filter_seconds"] = timing.get("torch_gpu_filter_seconds", 0.0) + time.perf_counter() - filter_start
        valid_mask = valid_device.detach().cpu().numpy().astype(bool, copy=False)
        stress_items.append(
            TorchStressFeatures(
                features_device=features_device,
                valid_mask=valid_mask,
                valid_indices=np.flatnonzero(valid_mask[: len(individual)]),
            )
        )
    return stress_items


def stress_columns_from_dqdf(dqdf):
    """Return columns shaped like the CPU stress builder: ``[point*4, gene]``."""
    return dqdf.permute(0, 2, 1).reshape(dqdf.shape[0] * dqdf.shape[2], dqdf.shape[1])


def filter_stress_columns_device(columns, value_limit: float = 1e8, duplicate_correlation: float = 0.999999):
    """Filter stress columns on CUDA while preserving first-column-wins order."""
    torch = require_torch_backend()
    if columns.shape[1] == 0:
        return columns, torch.zeros((0,), dtype=torch.bool, device=columns.device)

    filtered = torch.zeros_like(columns)
    valid = torch.zeros((columns.shape[1],), dtype=torch.bool, device=columns.device)
    normalized_columns = []
    for gene_index in range(columns.shape[1]):
        column = columns[:, gene_index]
        finite = torch.all(torch.isfinite(column))
        bounded = torch.max(torch.abs(column)) <= value_limit
        norm = torch.linalg.vector_norm(column)
        if not bool((finite & bounded & (norm >= 1e-12)).detach().cpu().item()):
            continue
        normalized = column / norm
        if normalized_columns:
            correlations = torch.stack(tuple(torch.abs(torch.dot(normalized, existing)) for existing in normalized_columns))
            if bool(torch.any(correlations >= duplicate_correlation).detach().cpu().item()):
                continue
        normalized_columns.append(normalized)
        filtered[:, gene_index] = column
        valid[gene_index] = True
    return filtered, valid


def append_intercept_column_device(features, valid, fit_intercept: bool):
    if not fit_intercept:
        return features, valid
    torch = require_torch_backend()
    intercept_column = torch.zeros((features.shape[0], 1), dtype=features.dtype, device=features.device)
    intercept_valid = torch.zeros((1,), dtype=torch.bool, device=features.device)
    return torch.cat((features, intercept_column), dim=1), torch.cat((valid, intercept_valid), dim=0)


def _feature_batch_is_valid(features, dqdf, value_limit: float) -> bool:
    torch = require_torch_backend()
    if features.shape[1] == 0:
        return True
    finite = torch.all(torch.isfinite(features)) & torch.all(torch.isfinite(dqdf))
    bounded = (torch.max(torch.abs(features)) <= value_limit) & (torch.max(torch.abs(dqdf)) <= value_limit)
    return bool((finite & bounded).detach().cpu().item())


def _selected_gene_indices(individuals: Sequence, selected_by_individual: Sequence[Sequence[int]] | None) -> list[list[int]]:
    if selected_by_individual is None:
        return [list(range(len(individual))) for individual in individuals]
    if len(selected_by_individual) != len(individuals):
        raise ValueError("selected_by_individual must match the number of individuals.")
    return [[int(index) for index in selected] for selected in selected_by_individual]


def _collect_unique_gene_columns(individuals: Sequence, selected_by_individual: Sequence[Sequence[int]]):
    unique_by_string = OrderedDict()
    unique_indices = {}
    columns_by_individual = []
    for individual, selected in zip(individuals, selected_by_individual):
        columns = []
        for gene_index in selected:
            gene = individual[int(gene_index)]
            key = str(gene)
            if key not in unique_by_string:
                unique_indices[key] = len(unique_by_string)
                unique_by_string[key] = gene
            columns.append(unique_indices[key])
        columns_by_individual.append(columns)
    return list(unique_by_string.values()), columns_by_individual


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
        columns = stress_columns_from_dqdf(dqdf_device)
        filtered_device, valid_device = filter_stress_columns_device(
            columns,
            value_limit=value_limit,
            duplicate_correlation=duplicate_correlation,
        )
        features_device, valid_device = append_intercept_column_device(
            filtered_device,
            valid_device,
            fit_intercept=model.config.fit_intercept,
        )
        block_until_ready((features_device, valid_device))
        if timing is not None:
            timing["torch_gpu_filter_seconds"] = timing.get("torch_gpu_filter_seconds", 0.0) + time.perf_counter() - start
        start = time.perf_counter()
        features = features_device.detach().cpu().numpy().astype(float, copy=False)
        valid = valid_device.detach().cpu().numpy().astype(bool, copy=False)
        if timing is not None:
            key = cache.timing_key("transfer_seconds") if cache is not None else "torch_transfer_seconds"
            timing[key] = timing.get(key, 0.0) + time.perf_counter() - start
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


def build_weak_form_artifacts_device(case: TorchWeakFormCase, dqdf, balance: float) -> TorchWeakFormArtifacts:
    weak_lhs = compute_weak_lhs_device(case, dqdf)
    lhs, rhs = compute_reaction_balance_device(case, weak_lhs, balance)
    residual_matrix, residual_target = compute_residual_operator_device(case, weak_lhs, balance)
    return TorchWeakFormArtifacts(
        lhs_device=lhs,
        rhs_device=rhs,
        residual_matrix_device=residual_matrix,
        residual_target_device=residual_target,
    ).block_until_ready()


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
        if self.timing is not None:
            for key in TORCH_BATCH_TIMING_KEYS:
                self.timing.setdefault(key, 0.0)
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

    def population_stress_features(
        self,
        model,
        individuals: Sequence,
        dataset,
        variable_names: Sequence[str],
        value_limit: float = 1e8,
        duplicate_correlation: float = 0.999999,
    ) -> list[TorchStressFeatures]:
        self._ensure_configured()
        return build_population_stress_features(
            model,
            individuals,
            dataset,
            variable_names,
            value_limit=value_limit,
            duplicate_correlation=duplicate_correlation,
            precision=self.precision,
            timing=self.timing,
            gene_cache=self.gene_cache,
            gene_backend=self.gene_backend,
        )

    def population_feature_values_and_dqdf_device(
        self,
        model,
        individuals: Sequence,
        F,
        variable_names: Sequence[str],
        gene_indices_by_individual: Sequence[Sequence[int]],
        value_limit: float = 1e8,
    ) -> list[tuple]:
        self._ensure_configured()
        return population_feature_values_and_dqdf_device(
            model,
            individuals,
            F,
            variable_names,
            gene_indices_by_individual=gene_indices_by_individual,
            value_limit=value_limit,
            precision=self.precision,
            gene_cache=self.gene_cache,
            gene_backend=self.gene_backend,
            timing=self.timing,
        )

    def feature_batch_is_valid(self, features, dqdf, value_limit: float = 1e8) -> bool:
        return _feature_batch_is_valid(features, dqdf, value_limit)

    def compute_weak_lhs_device(self, case: TorchWeakFormCase, dqdf):
        return compute_weak_lhs_device(case, dqdf)

    def compute_reaction_balance_device(self, case: TorchWeakFormCase, weak_lhs, balance: float):
        return compute_reaction_balance_device(case, weak_lhs, balance)

    def compute_residual_operator_device(self, case: TorchWeakFormCase, weak_lhs, balance: float):
        return compute_residual_operator_device(case, weak_lhs, balance)

    def build_weak_form_artifacts_device(self, case: TorchWeakFormCase, dqdf, balance: float) -> TorchWeakFormArtifacts:
        return build_weak_form_artifacts_device(case, dqdf, balance)

    def block_until_ready(self, value):
        return block_until_ready(value)

    def timing_key(self, suffix: str) -> str:
        return "%s_%s" % (self.name, suffix)

    def _ensure_configured(self) -> None:
        if self.eval_cache is None or self.gene_cache is None:
            self.configure()
