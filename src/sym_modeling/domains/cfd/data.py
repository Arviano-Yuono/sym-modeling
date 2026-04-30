"""
This module consists of a standardized data structure for flow data.

"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np


@dataclass
class CFDCaseData:
    """
    Canonical data class for flow data.

    Args:
        coords: Coordinates of the flow data points.
        U: Mean flow velocity at each point.
        p: Pressure at each point.
        T: Temperature at each point.
        rho: Density at each point.
        nu: Kinematic viscosity at each point.
        k: Turbulent kinetic energy at each point.
        epsilon: Turbulent dissipation rate at each point.
        omega: Turbulent frequency at each point.
        nut: Turbulent viscosity at each point.
        Rij: Reynolds stress tensor at each point.
        Sij: Strain rate tensor at each point.
        Wij: Rotation rate tensor at each point.

    Methods:
        validate(): Validate the data in the FlowData object.
                returns True if valid, False otherwise.
        __save__(): Save the FlowData object to .pkl at save_path.
    """

    # Metadata
    simulation_config: Optional[str] = None
    n_points: int = 0
    grid_shape: np.ndarray = field(
        default_factory=lambda: np.empty(
            3,
        )
    )  # (3,)

    # Coordinates and mean flow
    x_vec: np.ndarray = field(default_factory=lambda: np.empty((0,)))  # (nx,)
    y_vec: np.ndarray = field(default_factory=lambda: np.empty((0,)))  # (ny,)
    z_vec: np.ndarray = field(default_factory=lambda: np.empty((0,)))  # (nz,)
    coords: np.ndarray = field(default_factory=lambda: np.empty((0, 3)))  # (N,3)
    U: np.ndarray = field(default_factory=lambda: np.empty((0, 3)))  # (N,3)

    # Scalar fields
    p: np.ndarray = field(default_factory=lambda: np.empty((0,)))  # (N,)
    T: np.ndarray = field(default_factory=lambda: np.empty((0,)))  # (N,)
    rho: np.ndarray = field(default_factory=lambda: np.empty((0,)))  # (N,)
    nu: np.ndarray = field(default_factory=lambda: np.empty((0,)))  # (N,)

    # Turbulence scalars
    k: np.ndarray = field(default_factory=lambda: np.empty((0,)))  # (N,)
    epsilon: np.ndarray = field(default_factory=lambda: np.empty((0,)))  # (N,)
    omega: np.ndarray = field(default_factory=lambda: np.empty((0,)))  # (N,)
    nut: np.ndarray = field(default_factory=lambda: np.empty((0,)))  # (N,)
    residual: np.ndarray = field(default_factory=lambda: np.empty((0,)))  # (N,)

    # Tensors (tensor format: (N, i, j))
    Rij: np.ndarray = field(default_factory=lambda: np.empty((0, 3, 3)))  # (N,3,3)
    bij: np.ndarray = field(
        default_factory=lambda: np.empty((0, 6))
    )  # (N,6) because anisotropy is symmetric

    # Derived tensors (optional, computed later)
    gradU: np.ndarray = field(default_factory=lambda: np.empty((0, 3, 3)))  # (N,3,3)
    anisotropy: np.ndarray = field(
        default_factory=lambda: np.empty((0, 3, 3))
    )  # (N,3,3)
    Sij: np.ndarray = field(default_factory=lambda: np.empty((0, 3, 3)))  # (N,3,3)
    Wij: np.ndarray = field(default_factory=lambda: np.empty((0, 3, 3)))  # (N,3,3)

    # Tensor basis / Invariants
    T1: np.ndarray = field(default_factory=lambda: np.empty((0, 3, 3)))  # (N,3,3)
    T2: np.ndarray = field(default_factory=lambda: np.empty((0, 3, 3)))  # (N,3,3)
    T3: np.ndarray = field(default_factory=lambda: np.empty((0, 3, 3)))  # (N,3,3)

    I1: np.ndarray = field(default_factory=lambda: np.empty((0, 1, 1)))  # (N,1,1)
    I2: np.ndarray = field(default_factory=lambda: np.empty((0, 1, 1)))  # (N,1,1)

    # Bool
    is_loaded: bool = False
    is_preprocessed: bool = False

    def get_field_names(self):
        "return a list of class properties dynamically"
        return [field.name for field in self.__dataclass_fields__.values()]

    def validate(self):
        # coordinate arrays
        if self.coords.size != 0 and (self.coords.ndim != 2 or self.coords.shape[1] != 3):
            raise ValueError(f"coords must have shape (N,3), got {self.coords.shape}")

        for vec_name in ("x_vec", "y_vec", "z_vec"):
            vec = np.asarray(getattr(self, vec_name))
            if vec.size != 0 and vec.ndim != 1:
                raise ValueError(f"{vec_name} must be 1D, got {vec.shape}")

        if self.grid_shape.size != 0 and self.grid_shape.shape != (3,):
            raise ValueError(f"grid_shape must have shape (3,), got {self.grid_shape.shape}")

        n_coords = self.coords.shape[0]
        if self.n_points not in (0, n_coords):
            raise ValueError(
                f"n_points mismatch: n_points={self.n_points}, coords count={n_coords}"
            )

        point_fields: Dict[str, Tuple[int, ...]] = {
            # vectors/scalars
            "U": (3,),
            "p": (),
            "T": (),
            "rho": (),
            "nu": (),
            "k": (),
            "epsilon": (),
            "omega": (),
            "nut": (),
            "residual": (),
            # tensors
            "Rij": (3, 3),
            "bij": (6,),
            "gradU": (3, 3),
            "anisotropy": (3, 3),
            "Sij": (3, 3),
            "Wij": (3, 3),
            "T1": (3, 3),
            "T2": (3, 3),
            "T3": (3, 3),
            "I1": (1, 1),
            "I2": (1, 1),
        }

        for field_name, expected_tail in point_fields.items():
            arr = np.asarray(getattr(self, field_name))
            if arr.size == 0:
                continue

            expected_ndim = 1 + len(expected_tail)
            if arr.ndim != expected_ndim or arr.shape[1:] != expected_tail:
                raise ValueError(
                    f"{field_name} must have shape (N,{','.join(str(x) for x in expected_tail)}) "
                    f"or be empty; got {arr.shape}"
                )

            if n_coords == 0:
                raise ValueError(
                    f"{field_name} is populated but coords are empty; coords are required for point fields."
                )

            if arr.shape[0] != n_coords:
                raise ValueError(
                    f"{field_name} first axis must match coords count ({n_coords}), got {arr.shape}"
                )

        return True

    def save(self, save_path: str):
        "Save FlowData object to a .pkl file."
        import pickle

        with open(save_path, "wb") as f:
            pickle.dump(self, f)
        return save_path

    def load(self, load_path: str):
        "Load FlowData object from a .pkl file into this instance."
        import pickle

        with open(load_path, "rb") as f:
            loaded = pickle.load(f)

        if not isinstance(loaded, CFDCaseData):
            raise TypeError(
                f"Expected CFDCaseData object in '{load_path}', got {type(loaded)}"
            )

        self.__dict__.update(loaded.__dict__)
        self.validate()
        self.is_loaded = True
        return self

    @classmethod
    def from_file(cls, load_path: str) -> "CFDCaseData":
        "Load a CFDCaseData object from a .pkl file."
        instance = cls()
        return instance.load(load_path=load_path)

    def __repr__(self):
        repr_str = "CFDCaseData(\n"
        for field_name in self.get_field_names():
            field_value = getattr(self, field_name)
            if isinstance(field_value, np.ndarray):
                repr_str += f"    {field_name}: {field_value.shape}\n"
            else:
                repr_str += f"    {field_name}: {field_value}\n"
        repr_str += ")"
        return repr_str


FlowData = CFDCaseData
