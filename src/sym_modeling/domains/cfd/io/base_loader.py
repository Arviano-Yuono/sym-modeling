import os
from abc import ABC, abstractmethod
from typing import Optional, Tuple

import numpy as np

from sym_modeling.domains.cfd.data import FlowData


class BaseLoader(ABC):
    """
    BaseLoader is an abstract class to load flow data obtained from a data source to a FlowData object.
    This is done to prepare the data for further processing or analysis using a standardized format.
    """

    @abstractmethod
    def __init__(self, data_path: str, flow_data: FlowData) -> None:
        clean_path = data_path.rstrip(os.sep)
        self.data_path = clean_path
        self.flow_data = flow_data
        self.simulation_config: Optional[str] = None

    @abstractmethod
    def load(self) -> FlowData:
        # Load and format data from the specified path to a FlowData object
        return self.flow_data

    @abstractmethod
    def _load_coords(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # Load the x,y,z unique vector and the flatten coordinates from the specified path
        pass

    @abstractmethod
    def _load_mean_velocities(self) -> np.ndarray:
        # Load velocities from the specified path
        pass

    def set_metadata(self) -> FlowData:
        # set metadata of the flow data
        assert self.simulation_config is not None, "Simulation configuration is not set"

        self.flow_data.simulation_config = self.simulation_config
        self.flow_data.is_loaded = True
        self.flow_data.n_points = self.flow_data.U.shape[0]
        self.flow_data.grid_shape = np.array(
            [
                self.flow_data.x_vec.shape[0],
                self.flow_data.y_vec.shape[0],
                self.flow_data.z_vec.shape[0],
            ]
        )
        return self.flow_data
