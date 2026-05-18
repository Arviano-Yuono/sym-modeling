from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from sym_modeling.domains.fem.data import FeatureSet
from sym_modeling.domains.fem.io.csv_loader import loadFemData
from sym_modeling.domains.fem.operators.kinematics import (
    computeCauchyGreenStrain,
    computeJacobian,
    computeStrainInvariantDerivatives,
    computeStrainInvariants,
)
from sym_modeling.domains.fem.methods.sgep.gep_gene import Gene, values_are_valid


@dataclass(frozen=True)
class StressDataset:
    name: str
    F: np.ndarray
    P: np.ndarray
    I1: np.ndarray
    I2: np.ndarray
    I3: np.ndarray
    J: np.ndarray
    dI1dF: np.ndarray
    dI2dF: np.ndarray
    dI3dF: np.ndarray

    @property
    def target_vector(self) -> np.ndarray:
        return self.P.reshape(-1)

    @property
    def num_points(self) -> int:
        return int(self.F.shape[0])


def build_stress_dataset_from_F(F: np.ndarray, P: np.ndarray, name: str = "stress") -> StressDataset:
    F = np.asarray(F, dtype=float)
    P = np.asarray(P, dtype=float)
    C = computeCauchyGreenStrain(F)
    I1, I2, I3 = computeStrainInvariants(C)
    return StressDataset(
        name=name,
        F=F,
        P=P,
        I1=I1.reshape(-1),
        I2=I2.reshape(-1),
        I3=I3.reshape(-1),
        J=computeJacobian(F).reshape(-1),
        dI1dF=computeStrainInvariantDerivatives(F, 1),
        dI2dF=computeStrainInvariantDerivatives(F, 2),
        dI3dF=computeStrainInvariantDerivatives(F, 3),
    )


def build_stress_dataset_from_fem_data(data, name: str | None = None) -> StressDataset:
    piola = data.P
    if piola is None:
        piola = np.zeros((data.F.shape[0], 4), dtype=float)
    return build_stress_dataset_from_F(
        data.F,
        piola,
        name=name if name is not None else getattr(data, "path", "fem_data"),
    )


def resolve_loadsteps(data_dir: str | Path) -> list[int]:
    data_path = Path(data_dir)
    manifest_path = data_path / "generation_manifest.json"
    if manifest_path.exists():
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        steps = [int(entry["load_step"]) for entry in payload.get("load_steps", [])]
        if steps:
            return steps

    steps = []
    for child in data_path.iterdir():
        if child.is_dir():
            try:
                steps.append(int(child.name))
            except ValueError:
                continue
    steps.sort()
    if not steps:
        raise FileNotFoundError("No numeric load-step directories found in %s." % data_path)
    return steps


def _select_rows(array: np.ndarray, max_rows: int | None) -> np.ndarray:
    if max_rows is None or max_rows <= 0 or array.shape[0] <= max_rows:
        return array
    indices = np.linspace(0, array.shape[0] - 1, max_rows).astype(int)
    return array[indices]


def load_stress_dataset_from_euclid_csv(
    data_dir: str | Path,
    loadsteps: Sequence[int] | None = None,
    max_elements_per_loadstep: int | None = None,
    noise_level: float = 0.0,
) -> StressDataset:
    data_path = Path(data_dir)
    steps = list(loadsteps) if loadsteps is not None else resolve_loadsteps(data_path)
    F_parts = []
    P_parts = []
    for step in steps:
        data = loadFemData(
            str(data_path / str(step)),
            AD=True,
            noiseLevel=noise_level,
            noiseType="displacement",
        )
        data.convertToNumpy()
        if data.P is None:
            raise ValueError("Reference Piola stresses are missing for load step %s." % step)
        F_parts.append(_select_rows(data.F, max_elements_per_loadstep))
        P_parts.append(_select_rows(data.P, max_elements_per_loadstep))
    return build_stress_dataset_from_F(
        np.vstack(F_parts),
        np.vstack(P_parts),
        name=str(data_path),
    )


def invariant_variables(
    dataset: StressDataset,
    variable_names: Sequence[str],
) -> dict[str, np.ndarray]:
    I1 = dataset.I1
    I2 = dataset.I2
    I3 = dataset.I3
    J = dataset.J
    values = {
        "I1": I1,
        "I2": I2,
        "I3": I3,
        "J": J,
        "Jm1": J - 1.0,
        "K1": I1 * np.power(I3, -1.0 / 3.0) - 3.0,
        "K2": (I1 + I3 - 1.0) * np.power(I3, -2.0 / 3.0) - 3.0,
    }
    return {name: values[name] for name in variable_names}


def reference_variables(variable_names: Sequence[str]) -> dict[str, np.ndarray]:
    values = {
        "I1": np.array([3.0]),
        "I2": np.array([3.0]),
        "I3": np.array([1.0]),
        "J": np.array([1.0]),
        "Jm1": np.array([0.0]),
        "K1": np.array([0.0]),
        "K2": np.array([0.0]),
    }
    return {name: values[name] for name in variable_names}


def variable_derivatives_wrt_F(
    dataset: StressDataset,
    variable_names: Sequence[str],
) -> dict[str, np.ndarray]:
    I1 = dataset.I1.reshape(-1, 1)
    I3 = dataset.I3.reshape(-1, 1)
    dI1 = dataset.dI1dF
    dI2 = dataset.dI2dF
    dI3 = dataset.dI3dF
    dJ = 0.5 * np.power(I3, -0.5) * dI3
    values = {
        "I1": dI1,
        "I2": dI2,
        "I3": dI3,
        "J": dJ,
        "Jm1": dJ,
        "K1": np.power(I3, -1.0 / 3.0) * dI1
        + I1 * (-1.0 / 3.0) * np.power(I3, -4.0 / 3.0) * dI3,
        "K2": np.power(I3, -2.0 / 3.0) * dI1
        + (
            np.power(I3, -2.0 / 3.0)
            - (2.0 / 3.0) * (I1 + I3 - 1.0) * np.power(I3, -5.0 / 3.0)
        )
        * dI3,
    }
    return {name: values[name] for name in variable_names}


def variable_derivatives_wrt_invariants(
    dataset: StressDataset,
    variable_names: Sequence[str],
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    I1 = dataset.I1
    I3 = dataset.I3
    zeros = np.zeros_like(I1)
    ones = np.ones_like(I1)
    dK1dI1 = np.power(I3, -1.0 / 3.0)
    dK1dI3 = (-1.0 / 3.0) * I1 * np.power(I3, -4.0 / 3.0)
    dK2dI1 = np.power(I3, -2.0 / 3.0)
    dK2dI3 = np.power(I3, -2.0 / 3.0) - (
        2.0 / 3.0
    ) * (I1 + I3 - 1.0) * np.power(I3, -5.0 / 3.0)
    dJdI3 = 0.5 * np.power(I3, -0.5)
    values = {
        "I1": (ones, zeros, zeros),
        "I2": (zeros, ones, zeros),
        "I3": (zeros, zeros, ones),
        "J": (zeros, zeros, dJdI3),
        "Jm1": (zeros, zeros, dJdI3),
        "K1": (dK1dI1, zeros, dK1dI3),
        "K2": (dK2dI1, zeros, dK2dI3),
    }
    return {name: values[name] for name in variable_names}


def normalized_gene_values(
    gene: Gene,
    variables: Mapping[str, np.ndarray],
    reference: Mapping[str, np.ndarray],
) -> np.ndarray:
    return gene.evaluate(variables) - float(gene.evaluate(reference)[0])


def gene_variable_derivatives(
    gene: Gene,
    variables: Mapping[str, np.ndarray],
    variable_names: Sequence[str],
    derivative_step: float,
) -> dict[str, np.ndarray]:
    derivatives = {}
    for name in variable_names:
        base = np.asarray(variables[name], dtype=float)
        step = derivative_step * np.maximum(1.0, np.abs(base))
        plus = dict(variables)
        minus = dict(variables)
        plus[name] = base + step
        minus[name] = base - step
        derivatives[name] = (gene.evaluate(plus) - gene.evaluate(minus)) / (2.0 * step)
    return derivatives


def gene_feature_values_and_derivatives(
    genes: Sequence[Gene],
    dataset: StressDataset,
    variable_names: Sequence[str],
    derivative_step: float = 1e-6,
    value_limit: float = 1e8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    variables = invariant_variables(dataset, variable_names)
    reference = reference_variables(variable_names)
    dVardI = variable_derivatives_wrt_invariants(dataset, variable_names)

    features = []
    dQdI1_columns = []
    dQdI2_columns = []
    dQdI3_columns = []
    for gene in genes:
        energy = normalized_gene_values(gene, variables, reference)
        if not values_are_valid(energy, limit=value_limit):
            raise ValueError("Invalid SGEP energy feature: %s" % gene.expression())

        dGdvar = gene_variable_derivatives(gene, variables, variable_names, derivative_step)
        dQdI1 = np.zeros_like(dataset.I1)
        dQdI2 = np.zeros_like(dataset.I1)
        dQdI3 = np.zeros_like(dataset.I1)
        for name in variable_names:
            if not values_are_valid(dGdvar[name], limit=value_limit):
                raise ValueError("Invalid SGEP feature derivative: %s" % gene.expression())
            dVdI1, dVdI2, dVdI3 = dVardI[name]
            dQdI1 += dGdvar[name] * dVdI1
            dQdI2 += dGdvar[name] * dVdI2
            dQdI3 += dGdvar[name] * dVdI3

        for derivative_values in (dQdI1, dQdI2, dQdI3):
            if not values_are_valid(derivative_values, limit=value_limit):
                raise ValueError("Invalid SGEP feature derivative: %s" % gene.expression())
        features.append(energy)
        dQdI1_columns.append(dQdI1)
        dQdI2_columns.append(dQdI2)
        dQdI3_columns.append(dQdI3)

    if not features:
        empty = np.zeros((dataset.num_points, 0), dtype=float)
        return empty, empty, empty, empty
    return (
        np.column_stack(features),
        np.column_stack(dQdI1_columns),
        np.column_stack(dQdI2_columns),
        np.column_stack(dQdI3_columns),
    )


def build_gene_feature_set_for_fem_data(
    data,
    genes: Sequence[Gene],
    variable_names: Sequence[str],
    derivative_step: float = 1e-6,
    value_limit: float = 1e8,
) -> FeatureSet:
    dataset = build_stress_dataset_from_fem_data(data)
    features, dQdI1, dQdI2, dQdI3 = gene_feature_values_and_derivatives(
        genes,
        dataset,
        variable_names,
        derivative_step=derivative_step,
        value_limit=value_limit,
    )
    return FeatureSet(
        features=features,
        d_features_dI1=dQdI1,
        d_features_dI2=dQdI2,
        d_features_dI3=dQdI3,
    )


def stress_feature_for_gene(
    gene: Gene,
    dataset: StressDataset,
    variable_names: Sequence[str],
    derivative_step: float = 1e-6,
    value_limit: float = 1e8,
) -> np.ndarray | None:
    variables = invariant_variables(dataset, variable_names)
    reference = reference_variables(variable_names)
    energy_values = normalized_gene_values(gene, variables, reference)
    if not values_are_valid(energy_values, limit=value_limit):
        return None

    dGdvar = gene_variable_derivatives(gene, variables, variable_names, derivative_step)
    dVardF = variable_derivatives_wrt_F(dataset, variable_names)
    stress = np.zeros_like(dataset.P)
    for name in variable_names:
        if not values_are_valid(dGdvar[name], limit=value_limit):
            return None
        stress += dGdvar[name].reshape(-1, 1) * dVardF[name]
    if not values_are_valid(stress, limit=value_limit):
        return None
    return stress


def assemble_stress_feature_matrix(
    genes: Sequence[Gene],
    dataset: StressDataset,
    variable_names: Sequence[str],
    derivative_step: float = 1e-6,
    value_limit: float = 1e8,
    duplicate_correlation: float = 0.999999,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    columns = []
    valid = []
    reasons: list[str] = []
    normalized_columns = []
    for gene in genes:
        feature = stress_feature_for_gene(
            gene,
            dataset,
            variable_names,
            derivative_step=derivative_step,
            value_limit=value_limit,
        )
        if feature is None:
            valid.append(False)
            reasons.append("invalid")
            continue
        column = feature.reshape(-1)
        norm = np.linalg.norm(column)
        if norm < 1e-12:
            valid.append(False)
            reasons.append("constant_or_zero")
            continue
        normalized = column / norm
        duplicate = any(abs(float(np.dot(normalized, existing))) >= duplicate_correlation for existing in normalized_columns)
        if duplicate:
            valid.append(False)
            reasons.append("duplicate")
            continue
        normalized_columns.append(normalized)
        columns.append(column)
        valid.append(True)
        reasons.append("ok")

    if columns:
        X = np.column_stack(columns)
    else:
        X = np.zeros((dataset.target_vector.shape[0], 0), dtype=float)
    return X, np.asarray(valid, dtype=bool), reasons


def synthetic_neo_hookean_dataset(
    num_samples: int = 200,
    seed: int = 0,
    mu: float = 1.0,
    bulk: float = 10.0,
) -> StressDataset:
    rng = np.random.default_rng(seed)
    F_values = []
    while len(F_values) < num_samples:
        sx = rng.uniform(0.82, 1.25)
        sy = rng.uniform(0.82, 1.25)
        fxy = rng.uniform(-0.18, 0.18)
        fyx = rng.uniform(-0.18, 0.18)
        F = np.array([sx, fxy, fyx, sy], dtype=float)
        if F[0] * F[3] - F[1] * F[2] > 0.35:
            F_values.append(F)
    F_array = np.vstack(F_values)
    C = computeCauchyGreenStrain(F_array)
    I1, _, I3 = computeStrainInvariants(C)
    J = computeJacobian(F_array)
    dI1 = computeStrainInvariantDerivatives(F_array, 1)
    dI3 = computeStrainInvariantDerivatives(F_array, 3)
    dK1 = np.power(I3, -1.0 / 3.0) * dI1 + I1 * (
        -1.0 / 3.0
    ) * np.power(I3, -4.0 / 3.0) * dI3
    dJ = 0.5 * np.power(I3, -0.5) * dI3
    P = (mu / 2.0) * dK1 + bulk * (J - 1.0) * dJ
    return build_stress_dataset_from_F(F_array, P, name="synthetic_neo_hookean")
