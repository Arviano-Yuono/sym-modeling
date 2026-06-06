from __future__ import annotations

from collections import OrderedDict
from typing import Sequence


BACKENDS = {"jax", "torch"}
PRECISIONS = {"float64", "float32"}

BACKEND_TIMING_SUFFIXES = (
    "gene_derivative_seconds",
    "weak_lhs_seconds",
    "transfer_seconds",
    "lp_seconds",
    "evaluation_seconds",
    "evaluations",
    "cache_hits",
    "cache_misses",
    "cache_entries",
    "gene_execute_seconds",
    "gene_compile_seconds",
)


def backend_timing_keys(backend_name: str) -> tuple[str, ...]:
    return tuple("%s_%s" % (backend_name, suffix) for suffix in BACKEND_TIMING_SUFFIXES)


def empty_backend_timing(backend_name: str) -> dict[str, float]:
    return {key: 0.0 for key in backend_timing_keys(backend_name)}


def create_weak_form_backend(config, timing: dict[str, float] | None = None):
    backend_name = str(config.backend)
    kwargs = {
        "precision": config.precision,
        "cache_enabled": config.cache_enabled,
        "cache_size": config.cache_size,
        "cache_device_outputs": config.cache_device_outputs,
        "gene_cache_size": config.gene_cache_size,
        "timing": timing,
    }
    if backend_name == "jax":
        from .jax_backend import JaxWeakFormBackend

        return JaxWeakFormBackend(**kwargs)
    if backend_name == "torch":
        from .torch_backend import TorchWeakFormBackend

        return TorchWeakFormBackend(**kwargs)
    raise ValueError("backend must be one of: jax, torch.")


class WeakFormEvaluationCache:
    """Bounded LRU cache for backend weak-form artifacts."""

    def __init__(
        self,
        backend_name: str,
        device_name: str = "default",
        enabled: bool = True,
        max_size: int = 256,
        device_outputs: bool = True,
        timing: dict[str, float] | None = None,
    ):
        self.backend_name = str(backend_name)
        self.device_name = str(device_name)
        self.enabled = bool(enabled)
        self.max_size = max(0, int(max_size))
        self.device_outputs = bool(device_outputs)
        self.timing = timing
        self._artifacts = OrderedDict()
        self._sync_entries()

    def timing_key(self, suffix: str) -> str:
        return "%s_%s" % (self.backend_name, suffix)

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
        del model
        return (
            self.backend_name,
            "dqdf",
            self.device_name,
            data_key,
            _individual_signature(individual),
            tuple(int(index) for index in selected),
            tuple(variable_names),
            precision or "array",
            tuple(getattr(F, "shape", ())),
            str(getattr(F, "dtype", "")),
            _device_name(F),
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
        del model
        return (
            self.backend_name,
            "weak_artifact",
            self.device_name,
            case_key,
            _individual_signature(individual),
            tuple(int(index) for index in selected),
            tuple(variable_names),
            precision,
            float(balance),
        )

    def get_artifact(self, key):
        if not self.enabled:
            return None
        if key in self._artifacts:
            self._artifacts.move_to_end(key)
            self._increment(self.timing_key("cache_hits"))
            return self._artifacts[key]
        self._increment(self.timing_key("cache_misses"))
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
            self.timing[self.timing_key("cache_entries")] = float(len(self._artifacts))


def _individual_signature(individual) -> tuple[str, ...]:
    return tuple(str(gene) for gene in individual)


def _device_name(value) -> str:
    device = getattr(value, "device", None)
    if device is not None:
        return str(device)
    devices = getattr(value, "devices", None)
    if callable(devices):
        try:
            return ",".join(sorted(str(device) for device in devices()))
        except TypeError:
            pass
    return "default"
