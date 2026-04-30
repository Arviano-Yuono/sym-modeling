from typing import List, Optional, Sequence, Tuple

import numpy as np
from scipy.interpolate import griddata

from sym_modeling.domains.cfd.data import FlowData
from .coord_mapping import build_index_map_by_coords


class Preprocessor:
    """
    Preprocesses the flow data by calculating the derivated fields of the FlowData object.
    Static Methods:
        compute_gradU: Calculates the velocity gradient tensor adaptively.
        interpolate_scalar_field: Interpolates a scalar field to a new grid.
    """

    @staticmethod
    def _is_uniform_spacing(
        values: np.ndarray,
        rtol: float = 1e-8,
        atol: float = 1e-12,
    ) -> bool:
        values = np.asarray(values, dtype=float).reshape(-1)
        if values.size <= 2:
            return True
        diffs = np.diff(values)
        return np.allclose(diffs, diffs[0], rtol=rtol, atol=atol)

    @staticmethod
    def _trapezoidal_point_weights(axis_values: np.ndarray) -> np.ndarray:
        axis_values = np.asarray(axis_values, dtype=float).reshape(-1)
        n = axis_values.size
        if n == 0:
            raise ValueError("Axis values are empty")
        if n == 1:
            return np.ones((1,), dtype=float)
        if n == 2:
            delta = axis_values[1] - axis_values[0]
            return np.array([0.5 * delta, 0.5 * delta], dtype=float)

        weights = np.zeros((n,), dtype=float)
        weights[0] = 0.5 * (axis_values[1] - axis_values[0])
        weights[-1] = 0.5 * (axis_values[-1] - axis_values[-2])
        weights[1:-1] = 0.5 * (axis_values[2:] - axis_values[:-2])
        return weights

    @staticmethod
    def compute_bulk_velocity(
        flow_data: FlowData,
        component: int = 0,
        yz_atol: float = 1e-10,
    ) -> float:
        """
        Compute bulk streamwise velocity U_b as cross-section mean of U_component.

        Rules:
            - Uniform YZ spacing -> simple arithmetic mean.
            - Non-uniform YZ spacing -> area-weighted mean using point-control areas.
            - If YZ does not form a full tensor-product grid, fallback to simple mean.
        """
        if component not in (0, 1, 2):
            raise ValueError(f"component must be 0, 1, or 2; got {component}")
        if yz_atol <= 0:
            raise ValueError(f"yz_atol must be > 0, got {yz_atol}")

        coords = np.asarray(flow_data.coords, dtype=float)
        U = np.asarray(flow_data.U, dtype=float)
        if coords.ndim != 2 or coords.shape[1] != 3:
            raise ValueError(f"Expected coords shape (N,3), got {coords.shape}")
        if U.ndim != 2 or U.shape[1] != 3:
            raise ValueError(f"Expected U shape (N,3), got {U.shape}")
        if coords.shape[0] != U.shape[0]:
            raise ValueError(
                f"coords and U must have same N. Got coords={coords.shape[0]}, U={U.shape[0]}"
            )
        if coords.shape[0] == 0:
            raise ValueError("Cannot compute bulk velocity from empty flow data")

        u_stream = U[:, component]

        # Collapse repeated (y,z) locations across x by averaging first.
        yz_quantized = np.rint(coords[:, 1:3] / yz_atol).astype(np.int64)
        yz_unique_q, yz_inverse = np.unique(yz_quantized, axis=0, return_inverse=True)
        n_groups = yz_unique_q.shape[0]

        yz_mean = Preprocessor._average_by_group(
            values=coords[:, 1:3],
            group_index=yz_inverse,
            group_count=n_groups,
        )
        u_yz = Preprocessor._average_by_group(
            values=u_stream,
            group_index=yz_inverse,
            group_count=n_groups,
        ).reshape(-1)

        y_unique_q = np.sort(np.unique(yz_unique_q[:, 0]))
        z_unique_q = np.sort(np.unique(yz_unique_q[:, 1]))
        ny = y_unique_q.shape[0]
        nz = z_unique_q.shape[0]

        # Not a complete tensor-product YZ grid: fallback to simple mean.
        if ny * nz != n_groups:
            return float(np.mean(u_yz))

        y_index = {val: idx for idx, val in enumerate(y_unique_q.tolist())}
        z_index = {val: idx for idx, val in enumerate(z_unique_q.tolist())}
        iy = np.array([y_index[val] for val in yz_unique_q[:, 0]], dtype=np.int64)
        iz = np.array([z_index[val] for val in yz_unique_q[:, 1]], dtype=np.int64)

        y_vec = np.zeros((ny,), dtype=float)
        z_vec = np.zeros((nz,), dtype=float)
        np.add.at(y_vec, iy, yz_mean[:, 0])
        np.add.at(z_vec, iz, yz_mean[:, 1])
        y_counts = np.bincount(iy, minlength=ny).astype(float)
        z_counts = np.bincount(iz, minlength=nz).astype(float)
        y_vec = y_vec / y_counts
        z_vec = z_vec / z_counts

        if Preprocessor._is_uniform_spacing(y_vec) and Preprocessor._is_uniform_spacing(z_vec):
            return float(np.mean(u_yz))

        wy = Preprocessor._trapezoidal_point_weights(y_vec)
        wz = Preprocessor._trapezoidal_point_weights(z_vec)
        weights = wy[iy] * wz[iz]
        weight_sum = float(np.sum(weights))
        if weight_sum == 0.0:
            raise ValueError("Area-weighted bulk velocity failed: zero total weight")
        return float(np.sum(u_yz * weights) / weight_sum)

    @staticmethod
    def compute_bulk_normalized_velocity_fields(
        flow_data: FlowData,
        ub: Optional[float] = None,
        ub_epsilon: float = 1e-12,
        yz_atol: float = 1e-10,
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Return bulk velocity and normalized fields (U/U_b, gradU/U_b).
        """
        if ub_epsilon <= 0:
            raise ValueError(f"ub_epsilon must be > 0, got {ub_epsilon}")

        if ub is None:
            ub = Preprocessor.compute_bulk_velocity(
                flow_data=flow_data,
                component=0,
                yz_atol=yz_atol,
            )

        ub_scalar = float(ub)
        if abs(ub_scalar) < ub_epsilon:
            raise ValueError(
                f"Bulk velocity is too close to zero for normalization: U_b={ub_scalar:.6e}"
            )

        U = np.asarray(flow_data.U, dtype=float)
        grad_u = np.asarray(flow_data.gradU, dtype=float)
        if U.ndim != 2 or U.shape[1] != 3:
            raise ValueError(f"Expected U shape (N,3), got {U.shape}")
        if grad_u.ndim != 3 or grad_u.shape[1:] != (3, 3):
            raise ValueError(f"Expected gradU shape (N,3,3), got {grad_u.shape}")
        if U.shape[0] != grad_u.shape[0]:
            raise ValueError(
                f"U and gradU must have same N. Got U={U.shape[0]}, gradU={grad_u.shape[0]}"
            )

        U_norm = U / ub_scalar
        grad_u_norm = grad_u / ub_scalar
        return ub_scalar, U_norm, grad_u_norm

    @staticmethod
    def compute_gradU(flow_data: FlowData) -> FlowData:
        nx, ny, nz = (
            flow_data.grid_shape[0],
            flow_data.grid_shape[1],
            flow_data.grid_shape[2],
        )
        N = nx * ny * nz
        edge = (
            2 if min([n for n in (nx, ny, nz) if n > 1]) >= 3 else 1
        )  # for safety in calculating gradient
        assert flow_data.U.shape == (N, 3), f"Expected U (N,3), got {flow_data.U.shape}"

        U = flow_data.U.reshape(nx, ny, nz, 3)
        grad = np.zeros((nx, ny, nz, 3, 3), dtype=flow_data.U.dtype)

        def grad_scalar(F):
            dFx = np.zeros_like(F)
            dFy = np.zeros_like(F)
            dFz = np.zeros_like(F)

            if nx > 1 and ny > 1 and nz > 1:
                dFx, dFy, dFz = np.gradient(
                    F,
                    flow_data.x_vec,
                    flow_data.y_vec,
                    flow_data.z_vec,
                    edge_order=edge,
                )
                return dFx, dFy, dFz

            if nz == 1:  # (x,y)
                F2 = F[:, :, 0]
                if nx > 1 and ny > 1:
                    gx, gy = np.gradient(
                        F2, flow_data.x_vec, flow_data.y_vec, edge_order=edge
                    )
                    dFx[:, :, 0], dFy[:, :, 0] = gx, gy
                elif nx > 1:
                    (gx,) = np.gradient(F2[:, 0], flow_data.x_vec, edge_order=edge)
                    dFx[:, 0, 0] = gx
                elif ny > 1:
                    (gy,) = np.gradient(F2[0, :], flow_data.y_vec, edge_order=edge)
                    dFy[0, :, 0] = gy
                return dFx, dFy, dFz

            if ny == 1:  # (x,z)
                F2 = F[:, 0, :]
                if nx > 1 and nz > 1:
                    gx, gz = np.gradient(
                        F2, flow_data.x_vec, flow_data.z_vec, edge_order=edge
                    )
                    dFx[:, 0, :], dFz[:, 0, :] = gx, gz
                elif nx > 1:
                    (gx,) = np.gradient(F2[:, 0], flow_data.x_vec, edge_order=edge)
                    dFx[:, 0, 0] = gx
                elif nz > 1:
                    (gz,) = np.gradient(F2[0, :], flow_data.z_vec, edge_order=edge)
                    dFz[0, 0, :] = gz
                return dFx, dFy, dFz

            if nx == 1:  # (y,z)
                F2 = F[0, :, :]
                if ny > 1 and nz > 1:
                    gy, gz = np.gradient(
                        F2, flow_data.y_vec, flow_data.z_vec, edge_order=edge
                    )
                    dFy[0, :, :], dFz[0, :, :] = gy, gz
                elif ny > 1:
                    (gy,) = np.gradient(F2[:, 0], flow_data.y_vec, edge_order=edge)
                    dFy[0, :, 0] = gy
                elif nz > 1:
                    (gz,) = np.gradient(F2[0, :], flow_data.z_vec, edge_order=edge)
                    dFz[0, 0, :] = gz
                return dFx, dFy, dFz

            return dFx, dFy, dFz

        for comp in range(3):
            dFx, dFy, dFz = grad_scalar(U[..., comp])
            grad[..., comp, 0] = dFx
            grad[..., comp, 1] = dFy
            grad[..., comp, 2] = dFz

        flow_data.gradU = grad.reshape(N, 3, 3)
        return flow_data

    @staticmethod
    def _build_index_map_by_coords(
        source_coords: np.ndarray,
        target_coords: np.ndarray,
        atol: float = 1e-10,
    ) -> np.ndarray:
        return build_index_map_by_coords(
            source_coords=source_coords,
            target_coords=target_coords,
            atol=atol,
        )

    @staticmethod
    def map_fields_by_coords(
        source_data: FlowData,
        target_data: FlowData,
        field_names: List[str],
        atol: float = 1e-10,
    ) -> FlowData:
        """
        Map fields from source_data to target_data by coordinates.

        Useful when both datasets represent the same points but not in the same order,
        e.g. OpenFOAM internal field order vs another FlowData point ordering.
        """
        assert source_data.coords.shape[0] != 0, "source_data has no coordinates"
        assert target_data.coords.shape[0] != 0, "target_data has no coordinates"

        source_to_target_idx = Preprocessor._build_index_map_by_coords(
            source_coords=source_data.coords,
            target_coords=target_data.coords,
            atol=atol,
        )

        n_source = source_data.coords.shape[0]
        for field_name in field_names:
            assert field_name in source_data.get_field_names(), (
                f"Field '{field_name}' is not in source_data. "
                f"Available fields: {source_data.get_field_names()}"
            )
            assert field_name in target_data.get_field_names(), (
                f"Field '{field_name}' is not in target_data. "
                f"Available fields: {target_data.get_field_names()}"
            )

            source_field = np.asarray(getattr(source_data, field_name))
            assert source_field.shape[0] == n_source, (
                f"Field '{field_name}' first axis must match source point count "
                f"({n_source}), got {source_field.shape}"
            )

            mapped_field = source_field[source_to_target_idx]
            setattr(target_data, field_name, mapped_field.copy())

        return target_data

    @staticmethod
    def interpolate_tensor_field(
        source_data: FlowData,
        target_data: FlowData,
        method: str = "cubic",
    ):
        """
        Interpolate tensor field from source_data grid to target_data with different resolutions.

        Args:
            source_data (FlowData): source data that will be interpolated.
            target_data (FlowData): target data.

        Returns:
            target_data (FlowData): interpolated data.
        """
        # assert field_name in source_data.get_field_names(), (
        #     f"Field {field_name} not found in source data field name list:\n{source_data.get_field_names()}."
        # )
        assert method in ["linear", "cubic"], f"Invalid interpolation method: {method}"
        assert source_data.coords.shape[0] != 0, "Source data has no points"
        assert target_data.coords.shape[0] != 0, "Target data has no points"

        dns_points = source_data.coords[:, 1:3]
        rans_points = target_data.coords[:, 1:3]

        for i in range(3):
            target_data.U[:, i] = griddata(
                dns_points, source_data.U[:, i], rans_points, method="cubic"
            )

        target_data.k = griddata(dns_points, source_data.k, rans_points, method="cubic")

        _Rij_holder = np.zeros((target_data.k.shape[0], 3, 3))
        for i in range(3):
            for j in range(i, 3):
                mapped_comp = griddata(
                    dns_points, source_data.Rij[:, i, j], rans_points, method="cubic"
                )
                _Rij_holder[:, i, j] = mapped_comp
                _Rij_holder[:, j, i] = mapped_comp  # symmetry

        target_data.Rij = _Rij_holder
        target_data.is_loaded = True

        return target_data

    @staticmethod
    def _average_by_group(
        values: np.ndarray,
        group_index: np.ndarray,
        group_count: int,
    ) -> np.ndarray:
        """
        Average `values` over groups defined by `group_index` on axis 0.
        """
        summed = np.zeros((group_count,) + values.shape[1:], dtype=float)
        np.add.at(summed, group_index, values)

        counts = np.bincount(group_index, minlength=group_count).astype(float)
        reshape = (group_count,) + (1,) * (values.ndim - 1)
        return summed / counts.reshape(reshape)

    @staticmethod
    def streamwise_average_flow_data(
        flow_data: FlowData,
        field_names: Optional[Sequence[str]] = None,
        yz_atol: float = 1e-10,
        x_plane_value: Optional[float] = None,
    ) -> FlowData:
        """
        Collapse a 3D field to a 2D YZ plane by averaging over streamwise (x).

        Grouping is performed by YZ coordinates (with quantization tolerance),
        so it is robust to arbitrary OpenFOAM point ordering.
        """
        coords = np.asarray(flow_data.coords, dtype=float)
        if coords.ndim != 2 or coords.shape[1] != 3:
            raise ValueError(
                f"streamwise_average_flow_data expects coords (N,3), got {coords.shape}"
            )
        if coords.shape[0] == 0:
            raise ValueError("streamwise_average_flow_data received empty coordinates")
        if yz_atol <= 0:
            raise ValueError(f"yz_atol must be > 0, got {yz_atol}")

        n_points = coords.shape[0]
        yz_quantized = np.rint(coords[:, 1:3] / yz_atol).astype(np.int64)

        _, inverse, _ = np.unique(
            yz_quantized,
            axis=0,
            return_inverse=True,
            return_counts=True,
        )
        n_groups = int(inverse.max()) + 1

        yz_mean = Preprocessor._average_by_group(
            values=coords[:, 1:3],
            group_index=inverse,
            group_count=n_groups,
        )

        x_value = float(np.mean(coords[:, 0])) if x_plane_value is None else float(x_plane_value)
        coords_out = np.column_stack((np.full((n_groups,), x_value), yz_mean))

        # Stable ordering for deterministic output
        order = np.lexsort((coords_out[:, 2], coords_out[:, 1]))
        order_inv = np.empty_like(order)
        order_inv[order] = np.arange(order.size)
        grouped_index_sorted = order_inv[inverse]
        coords_out = coords_out[order]

        if field_names is None:
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
            candidate_fields = []
            for field_name in flow_data.get_field_names():
                if field_name in non_point_fields:
                    continue
                values = np.asarray(getattr(flow_data, field_name))
                if values.size == 0 or values.ndim == 0:
                    continue
                if values.shape[0] != n_points:
                    continue
                candidate_fields.append(field_name)
        else:
            candidate_fields = list(field_names)

        for field_name in candidate_fields:
            if field_name not in flow_data.get_field_names():
                raise ValueError(f"Unknown FlowData field: '{field_name}'")

            values = np.asarray(getattr(flow_data, field_name))
            if values.size == 0:
                continue
            if values.ndim == 0:
                continue
            if values.shape[0] != n_points:
                raise ValueError(
                    f"Field '{field_name}' first axis ({values.shape[0]}) does not match "
                    f"coords size ({n_points})"
                )

            values_mean = Preprocessor._average_by_group(
                values=np.asarray(values, dtype=float),
                group_index=grouped_index_sorted,
                group_count=n_groups,
            )
            setattr(flow_data, field_name, values_mean)

        flow_data.coords = coords_out
        flow_data.x_vec = np.array([x_value], dtype=float)
        flow_data.y_vec = np.unique(coords_out[:, 1])
        flow_data.z_vec = np.unique(coords_out[:, 2])
        flow_data.n_points = coords_out.shape[0]
        flow_data.grid_shape = np.array(
            [flow_data.x_vec.shape[0], flow_data.y_vec.shape[0], flow_data.z_vec.shape[0]]
        )
        flow_data.is_preprocessed = True

        return flow_data
