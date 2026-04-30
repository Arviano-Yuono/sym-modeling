from abc import ABC, abstractmethod

import numpy as np
from typing import List

class BaseRegressionModel(ABC):
    """
    BaseRegressionModel is an abstract base class for making a regression model
    """

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> "BaseRegressionModel":
        """Fit the model to the data"""
        return self

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Run inference on the model"""
        pass

    @abstractmethod
    def get_expression(self, features: List[str]) -> str:
        """Return string of math expression from the model"""
        pass
