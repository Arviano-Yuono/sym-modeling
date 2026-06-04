"""Per-gene JIT-compiled evaluators for the SGEPPY JAX weak-form backend.

The key insight: GEP individuals share genes across the population (elitism
preserves them, crossover recombines them).  Instead of JIT-compiling a fresh
XLA program for every individual, we compile each unique gene once and compose
individual-level results by stacking.

For a typical run (population=100, generations=200, n_genes=2), there are at
most a few hundred unique gene strings.  With per-gene caching, the number of
JIT compilations drops from ~20 000 (one per individual) to ~a few hundred.
"""

from __future__ import annotations

import time
from collections import OrderedDict
from typing import Sequence

import geppy as gep
from geppy.tools.parser import _compile_gene


# ---------------------------------------------------------------------------
# Gene-to-function compilation
# ---------------------------------------------------------------------------

def _jax_operators():
    """JAX implementations of the supported arithmetic/math operators.

    Mirrors the NumPy safe-operator definitions in ``operator.py``, but uses
    ``jax.numpy`` so the resulting callables are traceable by ``jax.jit``.
    """
    _, jnp = _require_jax()
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


def _build_jax_pset(model):
    """Build a geppy PrimitiveSet that resolves operator names to JAX callables.

    The pset is used by ``_compile_gene`` to convert a gene's expression tree
    into a Python function whose arithmetic uses JAX operations.
    """
    from .sgep import BINARY_OPS, UNARY_OPS

    ops = _jax_operators()
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


def compile_gene_function(gene, model):
    """Convert a geppy gene to a callable Python function using JAX operators.

    The returned function accepts one scalar argument per configured variable
    (in the order of ``model.config.variable_names``) and returns a scalar.

    Parameters
    ----------
    gene : geppy.Gene
        The gene whose expression tree should be compiled.
    model : SGEP
        The SGEP model (provides operator and variable configuration).

    Returns
    -------
    callable
        A function ``f(v0, v1, ...) -> scalar`` using JAX operations.
    """
    pset = _build_jax_pset(model)
    return _compile_gene(gene, pset)


# ---------------------------------------------------------------------------
# Invariant variable computation (JAX-traceable)
# ---------------------------------------------------------------------------

def _variables_from_F(F_row, variable_names):
    """Compute named strain-invariant variables from a single deformation gradient.

    ``F_row`` is a length-4 vector ``[F11, F12, F21, F22]``.  All arithmetic
    uses ``jax.numpy`` so this function is traceable by ``jax.jit`` and
    ``jax.grad``.

    Returns a list of scalar values in the order of *variable_names*.
    """
    _, jnp = _require_jax()
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


def _require_jax():
    """Lazy import of JAX (avoids module-level JAX dependency)."""
    from .jax_backend import require_jax_fem_backend
    return require_jax_fem_backend(enable_x64=None)


# ---------------------------------------------------------------------------
# Per-gene JIT cache
# ---------------------------------------------------------------------------

class PerGeneJitCache:
    """Bounded LRU cache of JIT-compiled per-gene evaluator functions.

    Each unique gene (identified by its string representation) is compiled
    once into a ``jax.jit``-ed function that evaluates the gene and its
    gradient w.r.t. ``F`` over a batch of deformation gradients.

    Parameters
    ----------
    enabled : bool
        If ``False``, every call compiles fresh (useful for debugging).
    max_size : int
        Maximum number of cached gene evaluators.  Oldest entries are
        evicted (LRU) when the limit is reached.
    timing : dict, optional
        If provided, ``"gene_jit_compile_seconds"`` is incremented with
        time spent inside JIT compilation.
    """

    def __init__(self, enabled=True, max_size=1024, timing=None):
        self.enabled = bool(enabled)
        self.max_size = max(0, int(max_size))
        self.timing = timing
        self._store = OrderedDict()

    @staticmethod
    def _key(gene_str, variable_names, precision):
        return (str(gene_str), tuple(variable_names), precision or "float64")

    def get(self, gene_str, variable_names, precision):
        """Return the cached evaluator for this gene, or ``None`` on miss."""
        if not self.enabled:
            return None
        key = self._key(gene_str, variable_names, precision)
        if key in self._store:
            self._store.move_to_end(key)
            return self._store[key]
        return None

    def put(self, gene_str, variable_names, precision, evaluator):
        """Store an evaluator, evicting the oldest entry if at capacity."""
        if not self.enabled or self.max_size <= 0:
            return
        key = self._key(gene_str, variable_names, precision)
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
# Single-gene JIT compilation
# ---------------------------------------------------------------------------

def _make_gene_evaluator(gene_fn, variable_names):
    """JIT-compile a single gene's value-and-gradient evaluator.

    Parameters
    ----------
    gene_fn : callable
        Python function from :func:`compile_gene_function`.
    variable_names : sequence of str
        Names of the strain-invariant variables.

    Returns
    -------
    callable
        ``evaluate(F_batch) -> (values, dqdf)`` where *F_batch* has shape
        ``[n, 4]``, *values* has shape ``[n]``, and *dqdf* has shape
        ``[n, 4]``.
    """
    import jax

    def _scalar_eval(F_row):
        args = _variables_from_F(F_row, variable_names)
        return gene_fn(*args)

    _vmap_eval = jax.vmap(_scalar_eval)
    _vmap_grad = jax.vmap(jax.grad(_scalar_eval))

    @jax.jit
    def evaluate(F_batch):
        return _vmap_eval(F_batch), _vmap_grad(F_batch)

    return evaluate


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
):
    """Evaluate selected genes on a batch of deformation gradients.

    Each unique gene is JIT-compiled once (via *gene_cache*) and reused
    across all individuals that contain it.  The per-gene results are
    stacked into individual-level feature and derivative matrices.

    Parameters
    ----------
    gene_cache : PerGeneJitCache or None
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
        ``"float64"`` or ``"float32"``.

    Returns
    -------
    features : jax.numpy.ndarray, shape ``[n, n_genes]``
        Gene output values at each integration point.
    dqdf : jax.numpy.ndarray, shape ``[n, n_genes, 4]``
        Derivative of each gene output w.r.t. ``[F11, F12, F21, F22]``.
    """
    import jax.numpy as jnp

    if not gene_indices:
        n = F_batch.shape[0]
        dtype = F_batch.dtype
        return jnp.zeros((n, 0), dtype=dtype), jnp.zeros((n, 0, 4), dtype=dtype)

    all_values = []
    all_dqdf = []

    for gene_index in gene_indices:
        gene = individual[int(gene_index)]
        gene_str = str(gene)

        # Look up a previously compiled evaluator for this gene.
        evaluator = None
        if gene_cache is not None:
            evaluator = gene_cache.get(gene_str, variable_names, precision)

        if evaluator is None:
            # First time seeing this gene — compile and (optionally) cache it.
            compile_start = time.perf_counter()
            gene_fn = compile_gene_function(gene, model)
            evaluator = _make_gene_evaluator(gene_fn, variable_names)
            if gene_cache is not None:
                gene_cache.put(gene_str, variable_names, precision, evaluator)
                if gene_cache.timing is not None:
                    gene_cache.timing["gene_jit_compile_seconds"] += (
                        time.perf_counter() - compile_start
                    )

        values, dqdf = evaluator(F_batch)
        all_values.append(values)
        all_dqdf.append(dqdf)

    features = jnp.stack(all_values, axis=1)
    dqdf = jnp.stack(all_dqdf, axis=1)
    return features, dqdf
