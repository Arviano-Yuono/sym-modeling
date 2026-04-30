import os.path as osp
from pathlib import Path
from typing import Tuple

import numpy as np

from sym_modeling.domains.cfd.data import FlowData
from sym_modeling.domains.cfd.operators import compute_anisotropy, compute_k_field
from sym_modeling.domains.cfd.preprocessing import Preprocessor

from .base_loader import BaseLoader


class KTHLoader(BaseLoader):
    """
    KTHLoader is a class inheriting from BaseLoader that loads DNS data obtained from KTH Duct flow
    DNS simulation result into a FlowData object.

    Note:
        In order to download this dataset, please visit the following link:

        https://www.mech.kth.se/~rvinuesa/DuctData/index.html

        Or run the download script provided in "symbolic_turb/utils/download_dns.py"

    Args:
        data_path (str): Path to the DNS data directory.
        flow_data (FlowData): FlowData object to be populated with DNS data.

    Method:
        format(): Load and format the KTH DNS data for the FlowData object.
                returns FlowData object populated with DNS data.

    """

    def __init__(self, data_path: str, flow_data: FlowData) -> None:
        super().__init__(data_path, flow_data)

        self.simulation_config = str(
            Path(self.data_path).name.split("AR_")[-1]
        )  # taking the sim config, ex: "1_80" from "AR_1_180"

    def load(self) -> FlowData:
        """Load the KTH DNS data for the FlowData object"""
        # actual data
        (
            self.flow_data.x_vec,
            self.flow_data.y_vec,
            self.flow_data.z_vec,
            self.flow_data.coords,
        ) = self._load_coords()
        self.flow_data.U = self._load_mean_velocities()
        self.flow_data.Rij = self._load_reynold_stress()
        # metadata needed before gradU
        self.flow_data = self._set_metadata()

        # before compute phyisics, we need to compute gradU
        self.flow_data = Preprocessor().compute_gradU(flow_data=self.flow_data)

        # compute the physics
        self.flow_data.k = compute_k_field(Rij=self.flow_data.Rij)
        # DNS loader uses unit omega by default; if a case-specific omega is needed,
        # provide it later and recompute normalized Sij/Wij accordingly.
        self.flow_data.omega = np.ones_like(self.flow_data.k)
        self.flow_data.anisotropy = compute_anisotropy(
            Rij=self.flow_data.Rij, k=self.flow_data.k
        )

        return self.flow_data

    def _set_metadata(self) -> FlowData:
        return super().set_metadata()

    def _load_coords(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # Load coordinates data based on the original dataset file structure
        _coords_np = np.loadtxt(
            osp.join(self.data_path, f"zcoord_{self.simulation_config}.prof.txt"),
            encoding="latin1",
            comments="%",
            skiprows=1,
        )

        z_vec = _coords_np[1]

        # this is based on the coods documentations on the original dataset text files
        assert self.simulation_config is not None  # for mypy
        if self.simulation_config.split("_")[0] == "1":
            y_vec = _coords_np[2]
        else:
            y_vec = np.loadtxt(
                osp.join(self.data_path, f"ycoord_{self.simulation_config}.prof.txt"),
                encoding="latin1",
                comments="%",
                skiprows=1,
            )

        x_vec = np.ones(shape=(1))
        y_vec = y_vec + 1
        z_vec = z_vec + 1

        YY, ZZ = np.meshgrid(y_vec, z_vec, indexing="ij")
        y_flat = YY.ravel()  # y: [y1, y1, y1, y2, y2, y2...]
        z_flat = ZZ.ravel()  # z: [z1, z2, z3, z1, z2, z3...]
        x_flat = np.ones_like(y_flat)

        _coords = np.stack([x_flat, y_flat, z_flat], axis=1)

        return (x_vec, y_vec, z_vec, _coords)

    def _load_mean_velocities(self) -> np.ndarray:
        # Load Mean Velocity Data
        U = np.loadtxt(
            osp.join(self.data_path, f"U_{self.simulation_config}.prof.txt"),
            encoding="latin1",
            comments="%",
            skiprows=1,
        ).reshape(-1, 1)  # reshape from (Nz, Ny) to (Nz*Ny, 1)
        V = np.loadtxt(
            osp.join(self.data_path, f"V_{self.simulation_config}.prof.txt"),
            encoding="latin1",
            comments="%",
            skiprows=1,
        ).reshape(-1, 1)
        W = np.loadtxt(
            osp.join(self.data_path, f"W_{self.simulation_config}.prof.txt"),
            encoding="latin1",
            comments="%",
            skiprows=1,
        ).reshape(-1, 1)

        return np.concatenate([U, V, W], axis=1)

    def _load_reynold_stress(self) -> np.ndarray:
        # Load Covariance Data
        uu = np.loadtxt(
            osp.join(self.data_path, f"uu_{self.simulation_config}.prof.txt"),
            encoding="latin1",
            comments="%",
            skiprows=1,
        ).reshape(
            -1,
        )  # reshape from (Nz, Ny) to (Nz*Ny,)
        vv = np.loadtxt(
            osp.join(self.data_path, f"vv_{self.simulation_config}.prof.txt"),
            encoding="latin1",
            comments="%",
            skiprows=1,
        ).reshape(
            -1,
        )
        ww = np.loadtxt(
            osp.join(self.data_path, f"ww_{self.simulation_config}.prof.txt"),
            encoding="latin1",
            comments="%",
            skiprows=1,
        ).reshape(
            -1,
        )
        uv = np.loadtxt(
            osp.join(self.data_path, f"uv_{self.simulation_config}.prof.txt"),
            encoding="latin1",
            comments="%",
            skiprows=1,
        ).reshape(
            -1,
        )
        uw = np.loadtxt(
            osp.join(self.data_path, f"uw_{self.simulation_config}.prof.txt"),
            encoding="latin1",
            comments="%",
            skiprows=1,
        ).reshape(
            -1,
        )
        vw = np.loadtxt(
            osp.join(self.data_path, f"vw_{self.simulation_config}.prof.txt"),
            encoding="latin1",
            comments="%",
            skiprows=1,
        ).reshape(
            -1,
        )

        num_points = uu.shape[0]
        _cov_tensor_field = np.zeros((num_points, 3, 3))

        _cov_tensor_field[:, 0, 0] = uu
        _cov_tensor_field[:, 1, 1] = vv
        _cov_tensor_field[:, 2, 2] = ww
        _cov_tensor_field[:, 0, 1] = _cov_tensor_field[:, 1, 0] = uv
        _cov_tensor_field[:, 0, 2] = _cov_tensor_field[:, 2, 0] = uw
        _cov_tensor_field[:, 1, 2] = _cov_tensor_field[:, 2, 1] = vw

        return _cov_tensor_field
