from __future__ import annotations

from sym_modeling.domains.fem.methods.common.weak_form import (
    add_global_reaction_balance,
    assemble_B_matrix,
    assemble_feature_derivative_wrt_F,
    assemble_global_matrix,
    assemble_global_vector,
    compute_first_piola,
    compute_internal_force,
    compute_strain_energy,
    compute_weak_lhs,
    compute_weak_overdetermined,
    extend_lhs_rhs,
    zip_dofs,
)


def extendLHSRHS(data, c, LHS, RHS):
    return extend_lhs_rhs(data, c, LHS, RHS, verbose=True)


def computeInternalForceTheta(data, theta, c):
    return compute_internal_force(data, theta, c)


def computeFirstPiolaTheta(data, theta, ele=None):
    return compute_first_piola(data, theta, element=ele)


def computeStrainEnergyTheta(data, theta):
    return compute_strain_energy(data, theta)


def computeWeakOverdetermined(data, c):
    print("\n-----------------------------------------------------")
    print("Assemble overdetermined system of equations.")
    result = compute_weak_overdetermined(data, c)
    print("-----------------------------------------------------\n")
    return result


def computeWeakLHS(data, c):
    return compute_weak_lhs(data, c)


def considerReactionGlobal(data, LHS, c):
    return add_global_reaction_balance(data, LHS, c)


def assembleFeatureDerivative(data, ele):
    return assemble_feature_derivative_wrt_F(data, ele)


def assembleB(data, ele, c):
    return assemble_B_matrix(data, ele, c)


def assembleGlobalVector(vector_global, vector_element, connectivity, ele, c):
    return assemble_global_vector(vector_global, vector_element, connectivity, ele, c)


def assembleGlobalMatrix(matrix_global, matrix_element, connectivity, ele, c):
    return assemble_global_matrix(matrix_global, matrix_element, connectivity, ele, c)


def zipper(matrix):
    return zip_dofs(matrix)
