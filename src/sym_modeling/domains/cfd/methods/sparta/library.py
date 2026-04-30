"""
------------
Builds a feature/library matrix Theta consisting of terms:

    (I1^a * I2^b) * Tn

for each tensor basis Tn in {T1, T2, T3} and for all exponent pairs (a,b)
such that:

    a + b <= max_degree

By default it *includes* the (a,b) = (0,0) term, i.e. the plain tensor basis Tn.
This matches your previous behavior (you had deg=0..max for I1^deg * Tn).

Shapes
------
Tn: (N, 3, 3)
I1,I2: (N, 1, 1)

Each feature column is flattened as (N*9,).
Theta returned is: (N*9, n_features)
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

from sym_modeling.domains.cfd.data import FlowData


@dataclass(frozen=True)
class FeatureDescription:
    """Metadata for a single column in the library."""

    name: str  # ex: "I1^2 * I2 * T2"
    invariant_power: Tuple[int, ...]  # ex: (2, 1) meaning I1^2 * I2^1
    tensor_basis: int  # ex: 2 for T2


@dataclass(frozen=True)
class ScalarFeatureDescription:
    """Metadata for a single scalar (R-library) feature."""

    name: str
    invariant_power: Tuple[int, ...]
    tensor_basis: int
    contraction: str


def _iter_total_degree_powers(
    n_invariants: int, max_degree: int, *, include_zero: bool
) -> Iterable[Tuple[int, ...]]:
    """
    Generate exponent tuples (a1, ..., ak) with total degree <= max_degree.
    If include_zero=True, includes the all-zero tuple (0,0,...,0).
    """
    if n_invariants <= 0:
        raise ValueError("n_invariants must be >= 1")
    if max_degree < 0:
        raise ValueError("max_degree must be >= 0")

    for exps in product(range(max_degree + 1), repeat=n_invariants):
        s = sum(exps)
        if s > max_degree:
            continue
        if not include_zero and s == 0:
            continue
        yield exps


def _monomial(invariants: Sequence[np.ndarray], powers: Sequence[int]) -> np.ndarray:
    """
    Compute monomial: prod_i invariants[i] ** powers[i]

    invariants are expected to be broadcastable together (and later with T).
    Typical invariant shape: (N,1,1).
    """
    if len(invariants) != len(powers):
        raise ValueError("len(invariants) must match len(powers)")

    out = np.ones_like(invariants[0])
    for I, p in zip(invariants, powers):
        if p:
            out = out * (I**p)
    return out


def _format_invariant_name(powers: Sequence[int]) -> str:
    """
    Convert (a,b,...) -> "I1^a * I2^b * ..."
    Skips zero powers. If all are zero, returns "1".
    """
    parts: List[str] = []
    for i, p in enumerate(powers, start=1):
        if p == 0:
            continue
        if p == 1:
            parts.append(f"I{i}")
        else:
            parts.append(f"I{i}^{p}")
    return " * ".join(parts) if parts else "1"


class BaseCandidateLibrary:
    """Shared utilities for candidate library builders."""

    def __init__(self, max_degree: int = 2):
        self.max_degree = int(max_degree)
        self.include_invariant_constant: bool = True

    def _prepare_tensor_basis(self, flow_data: FlowData) -> Tuple[list[np.ndarray], int]:
        assert flow_data.T1.shape[0] != 0, "Tensor basis is empty"

        T_basis = [
            np.asarray(flow_data.T1, dtype=float),
            np.asarray(flow_data.T2, dtype=float),
            np.asarray(flow_data.T3, dtype=float),
        ]
        N = T_basis[0].shape[0]

        for k, T in enumerate(T_basis, start=1):
            if T.shape != (N, 3, 3):
                raise ValueError(f"T{k} must have shape (N,3,3). Got {T.shape}.")

        return T_basis, N

    def _prepare_invariants(self, flow_data: FlowData, N: int) -> list[np.ndarray]:
        assert flow_data.I1.shape[0] != 0, "Invariant is empty"

        invariants_raw = [flow_data.I1, flow_data.I2]
        invariants: list[np.ndarray] = []

        for i, I_raw in enumerate(invariants_raw, start=1):
            I = np.asarray(I_raw, dtype=float)
            if I.shape[0] != N:
                raise ValueError(
                    f"I{i} must have same N as T. Got {I.shape[0]} vs {N}."
                )

            if I.ndim == 1:
                I = I.reshape(N, 1, 1)
            elif I.ndim == 2 and I.shape[1] == 1:
                I = I.reshape(N, 1, 1)
            elif I.ndim == 3 and I.shape[1:] == (1, 1):
                pass
            else:
                raise ValueError(
                    f"I{i} must be shape (N,1,1) or broadcastable to it; got {I.shape}."
                )

            invariants.append(I)

        return invariants

    def _precompute_monomials(
        self, invariants: Sequence[np.ndarray]
    ) -> Tuple[list[Tuple[int, ...]], list[np.ndarray]]:
        power_tuples = list(
            _iter_total_degree_powers(
                n_invariants=len(invariants),
                max_degree=self.max_degree,
                include_zero=self.include_invariant_constant,
            )
        )
        scalar_terms = [_monomial(invariants, p) for p in power_tuples]
        return power_tuples, scalar_terms


class BDeltaCandidateLibrary(BaseCandidateLibrary):
    def __init__(self, max_degree: int = 2):
        super().__init__(max_degree=max_degree)
        self.feature_names: list[FeatureDescription] = []
        self._numerical_matrix: Optional[np.ndarray] = None

    def fit_transform(self, flow_data: FlowData, threshold: float = 1e5) -> np.ndarray:
        """
        Builds the tensor library matrix Theta for b_ij correction.

        Returns
        -------
        Theta: np.ndarray
            Shape: (N*9, n_features)
        """
        T_basis, N = self._prepare_tensor_basis(flow_data=flow_data)
        invariants = self._prepare_invariants(flow_data=flow_data, N=N)
        power_tuples, scalar_terms = self._precompute_monomials(invariants=invariants)

        features_list: list[np.ndarray] = []
        descriptions: list[FeatureDescription] = []

        for tensor_idx, T in enumerate(T_basis, start=1):
            for powers, I_term in zip(power_tuples, scalar_terms):
                term = T * I_term  # (N,3,3)

                # SpaRTA magnitude filter
                if np.max(np.abs(term)) > threshold:
                    continue

                features_list.append(term.reshape(-1))

                inv_name = _format_invariant_name(powers)
                name = (
                    f"T{tensor_idx}"
                    if inv_name == "1"
                    else f"{inv_name} * T{tensor_idx}"
                )
                descriptions.append(
                    FeatureDescription(
                        name=name,
                        invariant_power=tuple(int(p) for p in powers),
                        tensor_basis=tensor_idx,
                    )
                )

        if not features_list:
            raise ValueError(
                "All candidates were filtered out! Check your data scaling."
            )

        self._numerical_matrix = np.stack(features_list, axis=1)
        self.feature_names = descriptions
        return self._numerical_matrix

    def get_feature_name(self, idx: int) -> str:
        return self.feature_names[idx].name

    def get_features(self) -> List[str]:
        return [self.get_feature_name(idx) for idx in range(len(self))]

    def __len__(self):
        return len(self.feature_names)


class RCandidateLibrary(BaseCandidateLibrary):
    def __init__(self, max_degree: int = 2):
        super().__init__(max_degree=max_degree)
        self.feature_names: list[ScalarFeatureDescription] = []
        self._numerical_matrix: Optional[np.ndarray] = None

    def fit_transform(self, flow_data: FlowData, threshold: float = 1e5) -> np.ndarray:
        """
        Builds scalar library Theta_R for k-equation residual correction.

        Returns
        -------
        Theta_R: np.ndarray
            Shape: (N, n_features)
        """
        T_basis, N = self._prepare_tensor_basis(flow_data=flow_data)
        invariants = self._prepare_invariants(flow_data=flow_data, N=N)

        if not hasattr(flow_data, "gradU"):
            raise AttributeError(
                "FlowData must provide gradU with shape (N,3,3) for R library."
            )
        gradU = np.asarray(flow_data.gradU, dtype=float)
        if gradU.shape != (N, 3, 3):
            raise ValueError(
                f"gradU must have shape (N,3,3). Got {gradU.shape} with N={N}."
            )

        power_tuples, scalar_terms = self._precompute_monomials(invariants=invariants)
        features_list: list[np.ndarray] = []
        descriptions: list[ScalarFeatureDescription] = []

        gradU_all_zero = np.allclose(gradU, 0.0)
        invariants_all_one = all(np.allclose(I, 1.0) for I in invariants)

        for tensor_idx, T in enumerate(T_basis, start=1):
            base_contraction = np.einsum("nij,nij->n", T, gradU)

            for powers, I_term in zip(power_tuples, scalar_terms):
                tensor_candidate = T * I_term
                scalar_candidate = np.einsum("nij,nij->n", tensor_candidate, gradU)

                # Lightweight sanity checks:
                if gradU_all_zero:
                    assert np.allclose(scalar_candidate, 0.0)
                if invariants_all_one and all(int(p) == 0 for p in powers):
                    assert np.allclose(scalar_candidate, base_contraction)

                # SpaRTA magnitude filter
                if np.max(np.abs(scalar_candidate)) > threshold:
                    continue

                features_list.append(scalar_candidate)

                inv_name = _format_invariant_name(powers)
                if inv_name == "1":
                    name = f"T{tensor_idx} : (T_ij dU_i/dx_j)"
                else:
                    name = f"{inv_name} * T{tensor_idx} : (T_ij dU_i/dx_j)"

                descriptions.append(
                    ScalarFeatureDescription(
                        name=name,
                        invariant_power=tuple(int(p) for p in powers),
                        tensor_basis=tensor_idx,
                        contraction="T_ij dU_i/dx_j",
                    )
                )

        if not features_list:
            raise ValueError(
                "All R candidates were filtered out! Check your data scaling."
            )

        self._numerical_matrix = np.stack(features_list, axis=1)
        self.feature_names = descriptions
        return self._numerical_matrix

    def get_feature_name(self, idx: int) -> str:
        return self.feature_names[idx].name

    def get_features(self) -> List[str]:
        return [self.get_feature_name(idx) for idx in range(len(self))]

    def __len__(self):
        return len(self.feature_names)


class CandidateLibrary(BDeltaCandidateLibrary):
    """
    Backward-compatible wrapper for the b_ij candidate library.

    Existing projects that called `CandidateLibrary.fit_transform_R(...)` keep
    working via delegation to `RCandidateLibrary`.
    """

    def __init__(self, max_degree: int = 2):
        super().__init__(max_degree=max_degree)
        self.feature_names_R: list[ScalarFeatureDescription] = []
        self._numerical_matrix_R: Optional[np.ndarray] = None

    def fit_transform_R(self, flow_data: FlowData, threshold: float = 1e5) -> np.ndarray:
        r_library = RCandidateLibrary(max_degree=self.max_degree)
        r_library.include_invariant_constant = self.include_invariant_constant
        theta_r = r_library.fit_transform(flow_data=flow_data, threshold=threshold)
        self.feature_names_R = r_library.feature_names.copy()
        self._numerical_matrix_R = theta_r
        return theta_r

    def get_feature_name_R(self, idx: int) -> str:
        return self.feature_names_R[idx].name

    def get_features_R(self) -> List[str]:
        return [self.get_feature_name_R(idx) for idx in range(self.n_features_R)]

    @property
    def n_features_R(self) -> int:
        return len(self.feature_names_R)
