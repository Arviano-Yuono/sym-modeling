from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from sym_modeling.domains.fem.io.csv_loader import loadFemData
from sym_modeling.domains.fem.operators.kinematics import (
    computeCauchyGreenStrain,
    computeJacobian,
    computeStrainInvariantDerivatives,
    computeStrainInvariants,
)


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
    K1 = I1 * np.power(I3, -1.0 / 3.0) - 3.0
    K2 = (I1 + I3 - 1.0) * np.power(I3, -2.0 / 3.0) - 3.0
    values = {
        "I1": I1,
        "I2": I2,
        "I3": I3,
        "J": J,
        "Jm1": J - 1.0,
        "K1": K1,
        "K2": K2,
        "logI13": np.log(K1 / 3.0 + 1.0),
        "logI23": np.log(K2 / 3.0 + 1.0),
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
        "logI13": np.array([0.0]),
        "logI23": np.array([0.0]),
    }
    return {name: values[name] for name in variable_names}


def variable_derivatives_wrt_F(
    dataset: StressDataset,
    variable_names: Sequence[str],
) -> dict[str, np.ndarray]:
    I1 = dataset.I1.reshape(-1, 1)
    I2 = dataset.I2.reshape(-1, 1)
    I3 = dataset.I3.reshape(-1, 1)
    dI1 = dataset.dI1dF
    dI2 = dataset.dI2dF
    dI3 = dataset.dI3dF
    dJ = 0.5 * np.power(I3, -0.5) * dI3
    dK1 = np.power(I3, -1.0 / 3.0) * dI1 + I1 * (-1.0 / 3.0) * np.power(I3, -4.0 / 3.0) * dI3
    dK2 = np.power(I3, -2.0 / 3.0) * dI1 + (
        np.power(I3, -2.0 / 3.0)
        - (2.0 / 3.0) * (I1 + I3 - 1.0) * np.power(I3, -5.0 / 3.0)
    ) * dI3
    dlogI13 = (
        dI1 * np.power(I3, -1.0 / 3.0)
        + I1 * (-1.0 / 3.0) * np.power(I3, -4.0 / 3.0) * dI3
    ) / (I1 * np.power(I3, -1.0 / 3.0))
    dlogI23 = (
        dI2 * np.power(I3, -2.0 / 3.0)
        + I2 * (-2.0 / 3.0) * np.power(I3, -5.0 / 3.0) * dI3
    ) / (I2 * np.power(I3, -2.0 / 3.0))
    values = {
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
    return {name: values[name] for name in variable_names}


def variable_derivatives_wrt_invariants(
    dataset: StressDataset,
    variable_names: Sequence[str],
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    I1 = dataset.I1
    I2 = dataset.I2
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
    dlogI13dI1 = 1 / I1
    dlogI13dI3 = -1 / (3 * I3)
    dlogI23dI2 = 1 / I2
    dlogI23dI3 = -2 / (3 * I3)
    values = {
        "I1": (ones, zeros, zeros),
        "I2": (zeros, ones, zeros),
        "I3": (zeros, zeros, ones),
        "J": (zeros, zeros, dJdI3),
        "Jm1": (zeros, zeros, dJdI3),
        "K1": (dK1dI1, zeros, dK1dI3),
        "K2": (dK2dI1, zeros, dK2dI3),
        "logI13": (dlogI13dI1, zeros, dlogI13dI3),
        "logI23": (zeros, dlogI23dI2, dlogI23dI3),
    }
    return {name: values[name] for name in variable_names}


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
