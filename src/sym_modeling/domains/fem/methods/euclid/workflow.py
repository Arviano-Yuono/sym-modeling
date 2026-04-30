from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from sym_modeling.common.workflow import BaseTrainer
from sym_modeling.domains.fem.io.csv_loader import loadFemData
from sym_modeling.domains.fem.methods.euclid.config import EuclidConfig
from sym_modeling.domains.fem.methods.euclid.feature_library import getNumberOfFeatures
from sym_modeling.domains.fem.methods.euclid.lp_solver import (
    applyPenaltyLpIteration,
    saveResultsLp,
)
from sym_modeling.domains.fem.methods.euclid.weak_form import extendLHSRHS
from sym_modeling.domains.fem.runtime import initCUDA


@dataclass
class EuclidResult:
    theta: np.ndarray
    lhs: np.ndarray
    rhs: np.ndarray
    num_datasets: int


class EuclidWorkflow(BaseTrainer):
    """Method-level EUCLID workflow for constitutive-law discovery on FEM cases."""

    def __init__(self, config: Optional[EuclidConfig] = None):
        self.config = config or EuclidConfig()
        initCUDA(self.config.cuda)
        self.datasets = []
        self.result: Optional[EuclidResult] = None

    def train(self) -> EuclidResult:
        self.config.penaltyLp = float(self.config.penaltyLp_init)
        lhs = None
        rhs = None
        for counter_load, loadstep in enumerate(self.config.loadsteps):
            data = loadFemData(
                self.config.femDataPath + "/" + str(loadstep),
                AD=True,
                noiseLevel=self.config.noiseLevel,
                noiseType="displacement",
            )
            self.datasets.append(data)
            data.convertToNumpy()

            if lhs is None or rhs is None:
                num_features = getNumberOfFeatures()
                lhs = np.zeros((num_features, num_features))
                rhs = np.zeros((num_features,))
            lhs, rhs = extendLHSRHS(data, self.config, lhs, rhs)

        if lhs is None or rhs is None:
            raise ValueError("No FEM datasets were loaded; cannot train EUCLID workflow.")

        theta = applyPenaltyLpIteration(self.datasets, lhs, rhs, self.config)
        saveResultsLp(theta, self.config, counter_load=len(self.config.loadsteps) - 1)
        self.config.penaltyLp = float(self.config.penaltyLp_init)
        self.result = EuclidResult(
            theta=theta,
            lhs=lhs,
            rhs=rhs,
            num_datasets=len(self.datasets),
        )
        return self.result

    def evaluate(self) -> EuclidResult:
        if self.result is None:
            raise RuntimeError("EuclidWorkflow.evaluate() requires a completed train() call.")
        return self.result
