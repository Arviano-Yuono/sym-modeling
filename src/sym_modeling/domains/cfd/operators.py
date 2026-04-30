"""
This module consists of physics computation functions.

Input shape notes:
    N (int): Number of data points
    gradU (np.ndarray): Gradient of velocity field (N, 3, 3)
    Rij (np.ndarray): Reynolds stress tensor field (N, 3, 3)
    k (np.ndarray): Turbulence kinetic energy field (N,)
    Sij (np.ndarray): Strain rate tensor field (N, 3, 3)
    Wij (np.ndarray): Rotation rate tensor field (N, 3, 3)
"""

from typing import Optional, Tuple

import numpy as np

from sym_modeling.common import tensor_ops as ops


def compute_incompressibility(gradU: np.ndarray) -> None:
    """
    Verify incompressibility by computing dV/dy + dW/dz

    Args:
        gradU (np.ndarray): Gradient of velocity field
    """
    div_u = gradU[:, 0, 0] + gradU[:, 1, 1] + gradU[:, 2, 2]
    max_divergence = np.max(np.abs(div_u))
    mean_divergence = np.mean(np.abs(div_u))
    min_divergence = np.min(np.abs(div_u))

    print(f"""
    Computing Incompressibility:
    Max divergence = {max_divergence}
    Mean divergence = {mean_divergence}
    Min divergence = {min_divergence}
    """)


def compute_strain_rate(
    gradU: np.ndarray, omega: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Calculate the strain rate tensor field: S = 0.5 * (gradU + gradU^T)

    Args:
        gradU (np.ndarray): Gradient of velocity field
        omega (Optional[np.ndarray]): Optional omega field of shape (N,). If provided,
            S is normalized by omega as S / omega.
    """
    assert gradU.shape[0] != 0, "gradU must not be empty"
    Sij = 0.5 * (gradU + ops.transpose(gradU))

    if omega is not None:
        omega_arr = np.asarray(omega, dtype=float).reshape(-1)
        if omega_arr.shape[0] != gradU.shape[0]:
            raise ValueError(
                "omega must have same N as gradU. "
                f"Got omega={omega_arr.shape[0]} and gradU={gradU.shape[0]}."
            )
        Sij = Sij / omega_arr[:, None, None]

    return Sij


def compute_rotation_rate(
    gradU: np.ndarray, omega: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Calculate the rotation rate tensor field: W = 0.5 * (gradU - gradU^T)

    Args:
        gradU (np.ndarray): Gradient of velocity field
        omega (Optional[np.ndarray]): Optional omega field of shape (N,). If provided,
            W is normalized by omega as W / omega.
    """
    assert gradU.shape[0] != 0, "gradU must not be empty"
    Wij = 0.5 * (gradU - ops.transpose(gradU))

    if omega is not None:
        omega_arr = np.asarray(omega, dtype=float).reshape(-1)
        if omega_arr.shape[0] != gradU.shape[0]:
            raise ValueError(
                "omega must have same N as gradU. "
                f"Got omega={omega_arr.shape[0]} and gradU={gradU.shape[0]}."
            )
        Wij = Wij / omega_arr[:, None, None]

    return Wij


def compute_k_field(Rij: np.ndarray, epsilon: float = 1e-10) -> np.ndarray:
    """
    Calculate the turbulent kinetic energy (k) field from Reynolds stress tensor.
    k = 0.5 * tr(Rij)

    Args:
        Rij (np.ndarray): Reynolds stress tensor field
        epsilon (float): Small value to avoid division by zero
    """
    assert Rij.shape[0] != 0, "Reynolds stress data is not loaded."

    # We use manual np.trace here because we want a flat scalar array (N,)
    # whereas ops.trace returns (N, 1, 1) for tensor broadcasting.
    k = 0.5 * np.trace(Rij, axis1=1, axis2=2)
    k[k < epsilon] = epsilon  # safety to avoid 0 k near wall
    return k


def compute_anisotropy(Rij: np.ndarray, k: np.ndarray) -> np.ndarray:
    """
    Calculate target anisotropy field: bij = Rij / (2k) - I/3

    Args:
        Rij (np.ndarray): Reynolds stress tensor field
        k (np.ndarray): Turbulence kinetic energy field
    """
    assert Rij.shape[0] != 0, "Reynolds stress data is not loaded."
    assert k.shape[0] != 0, "Turbulence kinetic energy data is not loaded."

    # Broadcasting k (N,) -> (N, 1, 1) to divide the (N, 3, 3) tensor
    normalized_covariance = Rij / (2 * k[:, None, None])

    isotropic_term = (1 / 3) * np.eye(3)
    bij = normalized_covariance - isotropic_term
    return bij


def compute_basis_tensor(
    Sij: np.ndarray, Wij: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate basis tensors based on SpaRTA paper (T1, T2, T3) using tensor operators.

    T1 = S
    T2 = SW - WS
    T3 = S^2 - 1/3 tr(S^2)I  (Deviatoric part of S^2)

    Args:
        Sij (np.ndarray): Strain rate tensor field
        Wij (np.ndarray): Rotation rate tensor field

    """
    assert Sij.shape[0] != 0, "strain rate must be computed"
    assert Wij.shape[0] != 0, "rotation rate must be computed"

    # T1 = S
    T1 = Sij.copy()

    # T2 = S @ W - W @ S
    SW = ops.matmul(Sij, Wij)
    WS = ops.matmul(Wij, Sij)
    T2 = SW - WS

    # T3 = dev(S^2)
    S2 = ops.matmul(Sij, Sij)
    T3 = ops.deviatoric(S2)

    return (T1, T2, T3)


def compute_invariants(
    Sij: np.ndarray, Wij: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes I1 and I2 according to Pope/SpaRTA using double dot products.
    I1 = S : S = tr(S^2)
    I2 = W : W = tr(W^2)

    Args:
        Sij (np.ndarray): Strain rate tensor field
        Wij (np.ndarray): Rotation rate tensor field
    """
    # Uses tensor operator for double dot product (trace of product)
    I1 = ops.double_dot(Sij, Sij)
    I2 = ops.double_dot(Wij, Wij)

    return I1, I2
