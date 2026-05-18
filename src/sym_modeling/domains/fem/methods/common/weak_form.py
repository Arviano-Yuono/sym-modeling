from __future__ import annotations

from typing import Sequence

import numpy as np


def zip_dofs(matrix: np.ndarray) -> np.ndarray:
    """Turn nodal x/y flags into the global degree-of-freedom order."""
    return np.asarray(matrix).reshape(-1)


def assemble_feature_derivative_wrt_F(data, element: int) -> np.ndarray:
    """Apply the chain rule: dQ/dF = dQ/dI1 dI1/dF + dQ/dI2 dI2/dF + dQ/dI3 dI3/dF."""
    feature_set = data.featureSet
    dQdI1 = np.asarray(feature_set.d_features_dI1[element, :], dtype=float)
    dQdI2 = np.asarray(feature_set.d_features_dI2[element, :], dtype=float)
    dQdI3 = np.asarray(feature_set.d_features_dI3[element, :], dtype=float)
    return (
        np.outer(dQdI1, data.dI1dF[element, :])
        + np.outer(dQdI2, data.dI2dF[element, :])
        + np.outer(dQdI3, data.dI3dF[element, :])
    )


def compute_first_piola(data, theta: np.ndarray, element: int | None = None) -> np.ndarray:
    """Compute P = sum_i theta_i dQ_i/dF at one element or every element."""
    theta = np.asarray(theta, dtype=float)
    if element is not None:
        dQdF = assemble_feature_derivative_wrt_F(data, element)
        return dQdF.T.dot(theta)

    piola = np.zeros((data.numElements, 4), dtype=float)
    for ele in range(data.numElements):
        piola[ele, :] = compute_first_piola(data, theta, element=ele)
    return piola


def compute_strain_energy(data, theta: np.ndarray) -> tuple[np.ndarray, float]:
    """Compute element strain energy and the quadrature-weighted total."""
    theta = np.asarray(theta, dtype=float)
    energy = np.asarray(data.featureSet.features, dtype=float).dot(theta)
    total = float(np.dot(energy, data.qpWeights))
    return energy, total


def assemble_B_matrix(data, element: int, config) -> np.ndarray:
    """Map nodal test-function values to the gradient in Voigt order."""
    dim = int(getattr(config, "dim", 2))
    nodes_per_element = int(getattr(config, "numNodesPerElement", 3))
    B = np.zeros((dim * dim, dim * nodes_per_element), dtype=float)
    for local_node in range(nodes_per_element):
        dN1 = data.gradNa[local_node][element, 0]
        dN2 = data.gradNa[local_node][element, 1]
        B[:, 2 * local_node : 2 * local_node + 2] = np.array(
            [(dN1, 0.0), (dN2, 0.0), (0.0, dN1), (0.0, dN2)],
            dtype=float,
        )
    return B


def assemble_global_vector(
    global_vector: np.ndarray,
    element_vector: np.ndarray,
    connectivity: Sequence[np.ndarray],
    element: int,
    config,
) -> np.ndarray:
    nodes_per_element = int(getattr(config, "numNodesPerElement", 3))
    for local_node in range(nodes_per_element):
        node = connectivity[local_node][element]
        global_vector[2 * node] += element_vector[2 * local_node]
        global_vector[2 * node + 1] += element_vector[2 * local_node + 1]
    return global_vector


def assemble_global_matrix(
    global_matrix: np.ndarray,
    element_matrix: np.ndarray,
    connectivity: Sequence[np.ndarray],
    element: int,
    config,
) -> np.ndarray:
    nodes_per_element = int(getattr(config, "numNodesPerElement", 3))
    for local_node in range(nodes_per_element):
        node = connectivity[local_node][element]
        global_matrix[2 * node, :] += element_matrix[2 * local_node, :]
        global_matrix[2 * node + 1, :] += element_matrix[2 * local_node + 1, :]
    return global_matrix


def compute_internal_force(data, theta: np.ndarray, config) -> np.ndarray:
    """Integrate P against shape-function gradients and assemble nodal internal forces."""
    dim = int(getattr(config, "dim", 2))
    nodal_force = np.zeros(dim * data.numNodes, dtype=float)
    for ele in range(data.numElements):
        piola = compute_first_piola(data, theta, element=ele)
        B = assemble_B_matrix(data, ele, config)
        element_force = B.T.dot(piola) * data.qpWeights[ele]
        nodal_force = assemble_global_vector(
            nodal_force,
            element_force,
            data.connectivity,
            ele,
            config,
        )
    return nodal_force


def compute_weak_lhs(data, config) -> np.ndarray:
    """Assemble the linear map theta -> internal nodal force."""
    dim = int(getattr(config, "dim", 2))
    num_features = int(data.featureSet.features.shape[1])
    lhs = np.zeros((dim * data.numNodes, num_features), dtype=float)
    for ele in range(data.numElements):
        dQdF = assemble_feature_derivative_wrt_F(data, ele)
        B = assemble_B_matrix(data, ele, config)
        element_lhs = B.T.dot(dQdF.T) * data.qpWeights[ele]
        lhs = assemble_global_matrix(lhs, element_lhs, data.connectivity, ele, config)
    return lhs


def add_global_reaction_balance(data, weak_lhs: np.ndarray, config) -> tuple[np.ndarray, np.ndarray]:
    """Convert equilibrium and global reaction measurements into normal equations."""
    balance = float(getattr(config, "balance", 100.0))
    free_dofs = np.logical_not(zip_dofs(data.dirichlet_nodes))
    lhs_bulk = weak_lhs[free_dofs, :]
    lhs = 2.0 * lhs_bulk.T.dot(lhs_bulk)
    reaction_lhs = np.zeros_like(lhs)
    reaction_rhs = np.zeros(lhs.shape[0], dtype=float)

    for reaction in data.reactions:
        dofs = zip_dofs(reaction.dofs)
        one = np.ones(weak_lhs[dofs, :].shape[0], dtype=float)
        reaction_sensitivity = weak_lhs[dofs, :].T.dot(one)
        reaction_lhs += 2.0 * np.outer(reaction_sensitivity, reaction_sensitivity)
        reaction_rhs += 2.0 * reaction_sensitivity * reaction.force

    return lhs + balance * reaction_lhs, balance * reaction_rhs


def extend_lhs_rhs(data, config, lhs: np.ndarray, rhs: np.ndarray, verbose: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Add one load step to the accumulated weak-form normal equations."""
    if verbose:
        print("\n-----------------------------------------------------")
        print("Consider new load step in LHS and RHS.")
        print("Balance factor: ", getattr(config, "balance", 100.0))
    weak_lhs = compute_weak_lhs(data, config)
    step_lhs, step_rhs = add_global_reaction_balance(data, weak_lhs, config)
    lhs += step_lhs
    rhs += step_rhs
    if verbose:
        print("-----------------------------------------------------\n")
    return lhs, rhs


def compute_weak_overdetermined(data, config) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return the explicit free-DOF and reaction equations before normal-equation assembly."""
    weak_lhs = compute_weak_lhs(data, config)
    free_dofs = np.logical_not(zip_dofs(data.dirichlet_nodes))
    lhs_free = weak_lhs[free_dofs, :]
    rhs_free = np.zeros(lhs_free.shape[0], dtype=float)
    lhs_fix = np.zeros((len(data.reactions), lhs_free.shape[1]), dtype=float)
    rhs_fix = np.zeros(len(data.reactions), dtype=float)
    for idx, reaction in enumerate(data.reactions):
        dofs = zip_dofs(reaction.dofs)
        one = np.ones(weak_lhs[dofs, :].shape[0], dtype=float)
        lhs_fix[idx] = weak_lhs[dofs, :].T.dot(one)
        rhs_fix[idx] = reaction.force
    return lhs_free, rhs_free, lhs_fix, rhs_fix


def weak_residual_vector(datasets: Sequence, theta: np.ndarray, config) -> np.ndarray:
    """Collect weak equilibrium and reaction residuals for scoring."""
    balance_sqrt = np.sqrt(float(getattr(config, "balance", 100.0)))
    residuals = []
    for data in datasets:
        internal_force = compute_internal_force(data, theta, config)
        free_dofs = np.logical_not(zip_dofs(data.dirichlet_nodes))
        residuals.append(internal_force[free_dofs])
        for reaction in data.reactions:
            dofs = zip_dofs(reaction.dofs)
            residuals.append(
                np.array([balance_sqrt * (np.sum(internal_force[dofs]) - reaction.force)])
            )
    if not residuals:
        return np.zeros(0, dtype=float)
    return np.concatenate(residuals)


def compute_lp_cost(datasets: Sequence, theta: np.ndarray, config) -> tuple[float, float, float]:
    """Compute weak residual cost plus the Lp coefficient penalty."""
    residual = weak_residual_vector(datasets, theta, config)
    weak_cost = float(np.sum(np.square(residual)))
    penalty = float(getattr(config, "penaltyLp", 0.0)) * float(
        np.sum(np.power(np.abs(theta), float(getattr(config, "p", 1.0))))
    )
    return weak_cost, penalty, weak_cost + penalty
