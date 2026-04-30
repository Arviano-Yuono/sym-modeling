from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from sym_modeling.domains.cfd.data import FlowData
from sym_modeling.domains.cfd.preprocessing import Preprocessor

from .base_loader import BaseLoader
from .openfoam import read_flow_data_from_openfoam


class FOAMLoader(BaseLoader):
    """
    Load OpenFOAM data through foam_parser and populate FlowData.

    OpenFOAM I/O is delegated to `read_flow_data_from_openfoam` so this loader
    focuses only on metadata and derived physics fields.
    """

    _DEFAULT_FIELDS: Tuple[str, ...] = ("U", "k", "omega", "Rij")

    def __init__(
        self,
        data_path: str,
        flow_data: FlowData,
        time: Optional[str] = None,
        region: str = "region0",
        fields: Optional[Sequence[str]] = None,
        field_map: Optional[Dict[str, str]] = None,
        sample_location: str = "point",
        streamwise_average: bool = False,
        streamwise_fields: Optional[Sequence[str]] = None,
        streamwise_yz_atol: float = 1e-10,
    ) -> None:
        super().__init__(data_path, flow_data)

        self.time = time  # e.g. "0", "1000", or None -> latest
        self.region = region  # reserved for future multi-region support
        self.fields = self._validate_fields(fields)
        self.field_map = None if field_map is None else dict(field_map)
        self.sample_location = sample_location
        self.streamwise_average = streamwise_average
        self.streamwise_fields = (
            list(streamwise_fields) if streamwise_fields is not None else None
        )
        self.streamwise_yz_atol = streamwise_yz_atol
        self.simulation_config = Path(self.data_path).name

        self._raw_flow_data: Optional[FlowData] = None

    @classmethod
    def _validate_fields(cls, fields: Optional[Sequence[str]]) -> List[str]:
        requested = list(cls._DEFAULT_FIELDS if fields is None else fields)
        if not requested:
            raise ValueError("FOAMLoader: 'fields' must not be empty")

        known_fields = set(FlowData().get_field_names())
        unknown = [name for name in requested if name not in known_fields]
        if unknown:
            raise ValueError(
                "FOAMLoader: unknown field(s) in 'fields': "
                f"{unknown}. Available FlowData fields: {sorted(known_fields)}"
            )
        return requested

    def _read_raw(self) -> FlowData:
        if self._raw_flow_data is None:
            self._raw_flow_data = read_flow_data_from_openfoam(
                case_path=self.data_path,
                time=self.time,
                fields=self.fields,
                field_map=self.field_map,
                sample_location=self.sample_location,
            )
            if self.streamwise_average:
                self._raw_flow_data = Preprocessor.streamwise_average_flow_data(
                    flow_data=self._raw_flow_data,
                    field_names=self.streamwise_fields,
                    yz_atol=self.streamwise_yz_atol,
                )
        return self._raw_flow_data

    def load(self) -> FlowData:
        if "U" not in self.fields:
            raise ValueError(
                "FOAMLoader.load() requires 'U' in requested fields to build "
                "coordinates, gradU, and derived metadata."
            )

        # required fields
        (
            self.flow_data.x_vec,
            self.flow_data.y_vec,
            self.flow_data.z_vec,
            self.flow_data.coords,
        ) = self._load_coords()
        self.flow_data.U = self._load_mean_velocities()

        # metadata needed before gradU
        self.flow_data = self.set_metadata()

        # compute gradU
        self.flow_data = Preprocessor().compute_gradU(flow_data=self.flow_data)

        raw = self._read_raw()
        non_point_fields = {
            "simulation_config",
            "n_points",
            "grid_shape",
            "x_vec",
            "y_vec",
            "z_vec",
            "coords",
            "is_loaded",
            "is_preprocessed",
        }

        for field_name in self.fields:
            if field_name == "U" or field_name in non_point_fields:
                continue
            values = np.asarray(getattr(raw, field_name))
            if values.size == 0:
                continue
            setattr(self.flow_data, field_name, values.copy())

        if self.flow_data.omega.size == 0:
            self.flow_data.omega = np.ones((self.flow_data.n_points,), dtype=float)

        return self.flow_data

    def _load_coords(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        raw = self._read_raw()
        coords = np.asarray(raw.coords, dtype=float)
        if coords.ndim != 2 or coords.shape[1] != 3:
            raise RuntimeError(f"FOAMLoader: coordinates have unexpected shape: {coords.shape}")
        return raw.x_vec.copy(), raw.y_vec.copy(), raw.z_vec.copy(), coords.copy()

    def _load_mean_velocities(self) -> np.ndarray:
        raw = self._read_raw()
        U = np.asarray(raw.U, dtype=float)
        if U.ndim != 2 or U.shape[1] != 3:
            raise RuntimeError(f"FOAMLoader: U has unexpected shape: {U.shape}")
        return U.copy()


OpenFOAMLoader = FOAMLoader
