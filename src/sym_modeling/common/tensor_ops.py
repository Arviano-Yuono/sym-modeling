"""
This module provides tensor operators for batched tensor fields.

"""

import numpy as np


def matmul(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Computes the batch matrix multiplication of two tensor fields.
    Result[i, j, k] = sum_m (A[i, j, m] * B[i, m, k])

    Args:
        A: Tensor of shape (N, 3, 3)
        B: Tensor of shape (N, 3, 3)
    """
    return np.matmul(A, B)


def transpose(A: np.ndarray) -> np.ndarray:
    """
    Computes the batch transpose of a tensor field.
    """
    return np.transpose(A, axes=(0, 2, 1))


def trace(A: np.ndarray) -> np.ndarray:
    """
    Computes the batch trace of a tensor field.
    Returns shape (N, 1, 1) to allow for broadcasting back against tensors.
    """
    # Sum along the last two dimensions' diagonal
    tr = np.trace(A, axis1=1, axis2=2)
    return tr[:, None, None]  # Reshape to (N, 1, 1) for broadcasting


def deviatoric(A: np.ndarray) -> np.ndarray:
    """
    Computes the deviatoric part of a tensor: A_dev = A - (1/3)*tr(A)*I
    """
    N = A.shape[0]
    identity = np.eye(3).reshape(1, 3, 3)
    # Using the broadcastable trace defined above
    return A - (1 / 3) * trace(A) * identity


def double_dot(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Computes the double dot product A : B = A_ij * B_ji = tr(A @ B).
    Note: For symmetric tensors (like S), A : A = tr(S^2).
    For antisymmetric (like Omega), A : A = tr(Omega^2) (which is negative).

    Returns shape (N, 1, 1) for broadcasting.
    """
    return trace(matmul(A, B))
