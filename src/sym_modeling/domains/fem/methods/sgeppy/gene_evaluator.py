"""Backend-agnostic per-gene evaluators for SGEPPY weak-form fitting.

The key insight: GEP individuals share genes across the population (elitism
preserves them, crossover recombines them).  Instead of compiling/evaluating a
fresh program for every individual, we compile each unique gene once and compose
individual-level results by stacking.

For a typical run (population=100, generations=200, n_genes=2), there are at
most a few hundred unique gene strings.  With per-gene caching, the expensive
backend compilation step drops from ~20 000 (one per individual) to ~a few
hundred.
"""

from __future__ import annotations

import time
from collections import OrderedDict
from typing import Protocol, Sequence

from geppy.tools.parser import _compile_gene


# ---------------------------------------------------------------------------
# Backend interface
# ---------------------------------------------------------------------------


class GeneEvaluationBackend(Protocol):
    """Strategy interface for compiling and evaluating one symbolic gene."""

    name: str
    default_precision: str
    compile_timing_key: str

    def build_primitive_set(self, model):
        """Return a geppy PrimitiveSet using this backend's operators."""

    def make_gene_evaluator(self, gene_fn, variable_names: Sequence[str]):
        """Return ``evaluate(F_batch) -> (values, dqdf)`` for one gene."""

    def empty_result(self, F_batch):
        """Return empty feature and derivative arrays for no selected genes."""

    def stack_gene_results(self, values: Sequence, derivatives: Sequence):
        """Stack per-gene results into individual-level feature arrays."""

    def cache_precision(self, precision: str | None) -> str:
        """Normalize the precision token used in cache keys."""


def default_gene_backend() -> GeneEvaluationBackend:
    """Return the default gene evaluation backend.

    JAX remains the default so existing weak-form callers keep their
    behavior.  The import is intentionally lazy because JAX/JAX-FEM are
    optional dependencies.
    """
    from .jax_backend import JaxGeneEvaluationBackend

    return JaxGeneEvaluationBackend()


# ---------------------------------------------------------------------------
# Gene-to-function compilation
# ---------------------------------------------------------------------------


def _jax_operators():
    """Compatibility wrapper for tests/imports that inspect JAX operators."""
    from .jax_backend import jax_gene_operators

    return jax_gene_operators()


def _build_jax_pset(model):
    """Compatibility wrapper for the JAX primitive set builder."""
    from .jax_backend import JaxGeneEvaluationBackend

    return JaxGeneEvaluationBackend().build_primitive_set(model)


def compile_gene_function(gene, model, backend: GeneEvaluationBackend | None = None):
    """Convert a geppy gene to a backend-specific callable Python function.

    The returned function accepts one scalar argument per configured variable
    (in the order of ``model.config.variable_names``) and returns a scalar.

    Parameters
    ----------
    gene : geppy.Gene
        The gene whose expression tree should be compiled.
    model : SGEP
        The SGEP model (provides operator and variable configuration).
    backend : GeneEvaluationBackend, optional
        Backend strategy used to resolve operator implementations.

    Returns
    -------
    callable
        A function ``f(v0, v1, ...) -> scalar`` using backend operations.
    """
    backend = backend or default_gene_backend()
    pset = backend.build_primitive_set(model)
    return _compile_gene(gene, pset)


# ---------------------------------------------------------------------------
# Per-gene evaluator cache
# ---------------------------------------------------------------------------

class PerGeneEvaluatorCache:
    """Bounded LRU cache of backend-compiled per-gene evaluator functions.

    Each unique gene (identified by its string representation) is compiled
    once into a backend evaluator that returns the gene value and its gradient
    w.r.t. ``F`` over a batch of deformation gradients.

    Parameters
    ----------
    enabled : bool
        If ``False``, every call compiles fresh (useful for debugging).
    max_size : int
        Maximum number of cached gene evaluators.  Oldest entries are
        evicted (LRU) when the limit is reached.
    timing : dict, optional
        If provided, the backend compile timing key is incremented with time
        spent compiling/building the evaluator.
    """

    def __init__(self, enabled=True, max_size=1024, timing=None):
        self.enabled = bool(enabled)
        self.max_size = max(0, int(max_size))
        self.timing = timing
        self._store = OrderedDict()

    @staticmethod
    def _key(gene_str, variable_names, precision, backend_name=None):
        return (
            backend_name or "default",
            str(gene_str),
            tuple(variable_names),
            precision or "float64",
        )

    def get(self, gene_str, variable_names, precision, backend_name=None):
        """Return the cached evaluator for this gene, or ``None`` on miss."""
        if not self.enabled:
            return None
        key = self._key(gene_str, variable_names, precision, backend_name)
        if key in self._store:
            self._store.move_to_end(key)
            return self._store[key]
        return None

    def put(self, gene_str, variable_names, precision, evaluator, backend_name=None):
        """Store an evaluator, evicting the oldest entry if at capacity."""
        if not self.enabled or self.max_size <= 0:
            return
        key = self._key(gene_str, variable_names, precision, backend_name)
        self._store[key] = evaluator
        self._store.move_to_end(key)
        while len(self._store) > self.max_size:
            self._store.popitem(last=False)

    @property
    def size(self):
        """Current number of cached gene evaluators."""
        return len(self._store)

    def clear(self):
        """Drop all cached entries."""
        self._store.clear()


# ---------------------------------------------------------------------------
# Batch evaluation of all selected genes in an individual
# ---------------------------------------------------------------------------

def evaluate_genes_on_F(
    gene_cache,
    model,
    individual,
    gene_indices,
    F_batch,
    variable_names,
    precision="float64",
    backend: GeneEvaluationBackend | None = None,
):
    """Evaluate selected genes on a batch of deformation gradients.

    Each unique gene is compiled once (via *gene_cache*) and reused across all
    individuals that contain it.  The per-gene results are
    stacked into individual-level feature and derivative matrices.

    Parameters
    ----------
    gene_cache : PerGeneEvaluatorCache or None
        Cache for compiled per-gene evaluators.  ``None`` disables caching.
    model : SGEP
        The SGEP model (provides operator and variable configuration).
    individual : geppy.Chromosome
        The individual whose genes are being evaluated.
    gene_indices : sequence of int
        Indices into *individual* selecting which genes to evaluate.
    F_batch : jax.numpy.ndarray
        Deformation gradients, shape ``[n_elements, 4]``.
    variable_names : sequence of str
        Strain-invariant variable names.
    precision : str
        Backend precision token, usually ``"float64"`` or ``"float32"``.
    backend : GeneEvaluationBackend, optional
        Strategy that provides operator, autograd, and array operations.

    Returns
    -------
    features : jax.numpy.ndarray, shape ``[n, n_genes]``
        Gene output values at each integration point.
    dqdf : jax.numpy.ndarray, shape ``[n, n_genes, 4]``
        Derivative of each gene output w.r.t. ``[F11, F12, F21, F22]``.
    """
    backend = backend or default_gene_backend()
    precision_key = backend.cache_precision(precision)

    if not gene_indices:
        return backend.empty_result(F_batch)

    all_values = []
    all_dqdf = []

    for gene_index in gene_indices:
        gene = individual[int(gene_index)]
        gene_str = str(gene)

        # Look up a previously compiled evaluator for this gene.
        evaluator = None
        if gene_cache is not None:
            evaluator = gene_cache.get(
                gene_str,
                variable_names,
                precision_key,
                backend_name=backend.name,
            )

        if evaluator is None:
            # First time seeing this gene: compile and optionally cache it.
            compile_start = time.perf_counter()
            gene_fn = compile_gene_function(gene, model, backend)
            evaluator = backend.make_gene_evaluator(gene_fn, variable_names)
            if gene_cache is not None:
                gene_cache.put(
                    gene_str,
                    variable_names,
                    precision_key,
                    evaluator,
                    backend_name=backend.name,
                )
                if gene_cache.timing is not None:
                    gene_cache.timing[backend.compile_timing_key] = (
                        gene_cache.timing.get(backend.compile_timing_key, 0.0)
                        + time.perf_counter()
                        - compile_start
                    )

        values, dqdf = evaluator(F_batch)
        all_values.append(values)
        all_dqdf.append(dqdf)

    return backend.stack_gene_results(all_values, all_dqdf)
