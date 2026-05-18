from __future__ import annotations

import os
from pathlib import Path

from sym_modeling.domains.fem.methods.common.lp_solver import (
    apply_penalty_lp_iteration,
    apply_penalty_lp_random_start,
    apply_penalty_lp_threshold,
    check_local_minimum_lp,
)
from sym_modeling.domains.fem.methods.common.weak_form import compute_lp_cost
from sym_modeling.domains.fem.methods.euclid.constraints import checkEnergyRequirementsRigorous


def applyPenaltyLpThreshold(datasets, LHS, RHS, theta, c):
    return apply_penalty_lp_threshold(datasets, LHS, RHS, theta, c)


def applyPenaltyLpRandomStart(datasets, LHS, RHS, c):
    return apply_penalty_lp_random_start(datasets, LHS, RHS, c, verbose=True)


def applyPenaltyLpIteration(datasets, LHS, RHS, c):
    return apply_penalty_lp_iteration(
        datasets,
        LHS,
        RHS,
        c,
        energy_check=checkEnergyRequirementsRigorous,
        verbose=True,
    )


def checkLocalMinimumLp(datasets, theta, c):
    return check_local_minimum_lp(datasets, theta, c, verbose=True)


def computeCostLp(datasets, theta, c):
    return compute_lp_cost(datasets, theta, c)


def saveResultsLp(theta, c, counter_load=None):
    """
    Save EUCLID results and hyperparameters in the legacy text format.
    """
    default_results_dir = str(
        Path(__file__).resolve().parents[6] / "output" / "euclid_results"
    )
    results_dir = os.path.abspath(getattr(c, "resultsDir", default_results_dir))
    os.makedirs(results_dir, exist_ok=True)

    print("\n-----------------------------------------------------")
    print("Save results.")
    resultfile = os.path.join(results_dir, "results_" + c.saveResultsName + ".txt")
    write_mode = "a+" if getattr(c, "appendResults", True) else "w"
    with open(resultfile, write_mode, encoding="utf-8") as handle:
        handle.write("-------------------- Data --------------------\n")
        if hasattr(c, "caseName"):
            handle.write("Case Name: " + str(c.caseName) + "\n")
        if hasattr(c, "dataGroup"):
            handle.write("Data Group: " + str(c.dataGroup) + "\n")
        if hasattr(c, "str_model"):
            handle.write("Model: " + str(c.str_model) + "\n")
        if hasattr(c, "str_mesh"):
            handle.write("Mesh: " + str(c.str_mesh) + "\n")
        handle.write(c.femDataPath + "\n")
        if counter_load is None:
            handle.write("Load Steps: " + str(c.loadsteps) + "\n")
        else:
            handle.write("Load Steps: " + str(c.loadsteps[0 : counter_load + 1]) + "\n")
        handle.write("Additional Noise Level: " + str(c.noiseLevel) + "\n")
        handle.write("-------------------- Hyperparameters --------------------\n")
        handle.write("Balance Factor: " + str(c.balance) + "\n")
        handle.write("Lp-Norm Penalty Factor (initial): " + str(c.penaltyLp_init) + "\n")
        handle.write("Lp-Norm Penalty Factor: " + str(c.penaltyLp) + "\n")
        handle.write("p: " + str(c.p) + "\n")
        handle.write("Max. Number Penalty Increments: " + str(c.numIncrements) + "\n")
        handle.write("Factor Penalty Increments: " + str(c.factorIncrements) + "\n")
        handle.write("Number Initial Guesses: " + str(c.numGuesses) + "\n")
        handle.write("Max. Number Iterations: " + str(c.numIterations) + "\n")
        handle.write("Lowest Cost: " + str(c.lowestCost) + "\n")
        handle.write("Lowest Cost Guess Identifier: " + str(c.lowestCostGuessID) + "\n")
        handle.write("Threshold (Fixed-Point Iteration): " + str(c.threshold_iter) + "\n")
        handle.write("Threshold: " + str(c.threshold) + "\n")
        handle.write("-------------------- Results --------------------\n")
        handle.write("Theta (Lp-Norm Penalty + Threshold): \n" + str(theta) + "\n")
        handle.write("\n\n\n")
    print("-----------------------------------------------------\n")
