from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from sym_modeling.domains.fem.methods.common.weak_form import compute_lp_cost


EnergyCheck = Callable[[Sequence, np.ndarray], bool]


def _solve(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    if rhs.size == 0:
        return np.zeros(0, dtype=float)
    return np.linalg.solve(lhs, rhs)


def apply_threshold(lhs: np.ndarray, rhs: np.ndarray, theta: np.ndarray, config, verbose: bool = True) -> np.ndarray:
    """Apply EUCLID's threshold refit to a weak-form linear system."""
    threshold = float(getattr(config, "threshold", 1e-2))
    theta = np.asarray(theta, dtype=float).copy()
    if verbose:
        print("\n-----------------------------------------------------")
        print("Apply threshold: ", threshold)
        print()
    for iteration in range(len(theta)):
        below = np.abs(theta) < threshold
        if all(theta[below] == 0.0):
            if verbose:
                print("Converged after iteration:", iteration)
            break
        above = np.abs(theta) >= threshold
        theta[below] = 0.0
        if np.any(above):
            active_lhs = lhs[above, :][:, above]
            theta[above] = _solve(active_lhs, rhs[above])
    if verbose:
        print("Solution:")
        print(theta)
        print("-----------------------------------------------------\n")
    return theta


def apply_penalty_lp_threshold(
    datasets: Sequence,
    lhs: np.ndarray,
    rhs: np.ndarray,
    theta: np.ndarray,
    config,
) -> tuple[np.ndarray, bool]:
    """Fixed-point solve for one Lp-regularized weak-form system."""
    theta = np.asarray(theta, dtype=float).copy()
    converged = False
    for _ in range(int(getattr(config, "numIterations", 200))):
        previous = np.copy(theta)
        active = np.abs(theta) >= float(getattr(config, "threshold_iter", 1e-6))
        penalty_weighted = np.zeros_like(theta)
        for idx in range(len(penalty_weighted)):
            if active[idx]:
                penalty_weighted[idx] = np.power(np.abs(theta[idx]), float(getattr(config, "p", 0.25)) - 2.0)
        lhs_lp = lhs + float(getattr(config, "p", 0.25)) * float(getattr(config, "penaltyLp", 1e-4)) * np.diag(
            penalty_weighted
        )
        if np.any(active):
            theta[active] = _solve(lhs_lp[active, :][:, active], rhs[active])
        theta[np.logical_not(active)] = 0.0
        if all(np.absolute(previous - theta) < 1e-3):
            converged = True
            break
    return theta, converged


def apply_penalty_lp_random_start(
    datasets: Sequence,
    lhs: np.ndarray,
    rhs: np.ndarray,
    config,
    verbose: bool = True,
) -> tuple[np.ndarray, bool]:
    """Solve the Lp problem from EUCLID-style deterministic and random starts."""
    if verbose:
        print("\n-----------------------------------------------------")
        print("Apply Lp-norm penalty.")
        print("Lp-norm penalty factor: ", getattr(config, "penaltyLp", 1e-4))
        print("Lp-norm: p=", getattr(config, "p", 0.25))
        print("Number of initial guesses:", getattr(config, "numGuesses", 1))
        print()
    theta = _solve(lhs, rhs)
    at_least_one_converged = False
    best_cost = None
    for restart in range(int(getattr(config, "numGuesses", 1))):
        if restart == 0:
            theta_start = theta
        elif restart < 50:
            theta_start = 2.0 * np.random.rand(*rhs.shape) - 1.0
        else:
            theta_start = 10.0 * (2.0 * np.random.rand(*rhs.shape) - 1.0)
        theta_candidate, converged = apply_penalty_lp_threshold(datasets, lhs, rhs, theta_start, config)
        if converged:
            _, _, total_cost = compute_lp_cost(datasets, theta_candidate, config)
            if not at_least_one_converged or total_cost < best_cost:
                best_cost = total_cost
                theta = np.copy(theta_candidate)
                config.lowestCost = total_cost
                config.lowestCostGuessID = restart
                at_least_one_converged = True
        elif verbose:
            print("Fixed-point iteration not converged for guess number: ", restart + 1)
    if verbose:
        print("Solution with the lowest cost:")
        print(theta)
        print("-----------------------------------------------------\n")
    return theta, at_least_one_converged


def check_local_minimum_lp(datasets: Sequence, theta: np.ndarray, config, verbose: bool = True) -> None:
    """Probe the fitted coefficients with small perturbations, matching EUCLID's diagnostic."""
    num_checks = 10
    perturbation_magnitude = 1e-3
    if verbose:
        print("\n-----------------------------------------------------")
        print("Check if the solution is a local minimum.")
        print("Number of checks: ", num_checks)
    local_minimum = True
    _, _, base_cost = compute_lp_cost(datasets, theta, config)
    for _ in range(num_checks):
        theta_perturbed = np.copy(theta)
        for idx in range(len(theta_perturbed)):
            if np.abs(theta[idx]) > float(getattr(config, "threshold", 1e-2)):
                theta_perturbed[idx] += perturbation_magnitude * (2.0 * np.random.rand() - 1.0)
        _, _, perturbed_cost = compute_lp_cost(datasets, theta_perturbed, config)
        if base_cost > perturbed_cost:
            local_minimum = False
            if verbose:
                print("Solution is not a local minimum.")
            break
    if local_minimum and verbose:
        print("Solution is likely to be a local minimum.")
    if verbose:
        print("-----------------------------------------------------\n")


def apply_penalty_lp_iteration(
    datasets: Sequence,
    lhs: np.ndarray,
    rhs: np.ndarray,
    config,
    energy_check: EnergyCheck | None = None,
    verbose: bool = True,
) -> np.ndarray:
    """Run EUCLID's penalty-increment loop with an optional library-specific energy check."""
    for increment in range(int(getattr(config, "numIncrements", 5))):
        if increment > 0:
            config.penaltyLp = float(getattr(config, "penaltyLp", 1e-4)) * float(getattr(config, "factorIncrements", 5.0))
        theta, converged = apply_penalty_lp_random_start(datasets, lhs, rhs, config, verbose=verbose)
        check_local_minimum_lp(datasets, theta, config, verbose=verbose)
        theta = apply_threshold(lhs, rhs, theta, config, verbose=verbose)
        energy_ok = True if energy_check is None else bool(energy_check(datasets, theta))
        if energy_ok and converged:
            break
    return theta
