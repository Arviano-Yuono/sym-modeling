"""
Generic field transfer utilities between FlowData objects.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.interpolate import griddata

from sym_modeling.domains.cfd.data import FlowData


def _new_flow_from_coords(coords: np.ndarray, simulation_config: Optional[str]) -> FlowData:
    coords = np.asarray(coords, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"coords must have shape (N,3), got {coords.shape}")

    flow = FlowData()
    flow.coords = coords.copy()
    flow.x_vec = np.unique(coords[:, 0])
    flow.y_vec = np.unique(coords[:, 1])
    flow.z_vec = np.unique(coords[:, 2])
    flow.n_points = coords.shape[0]
    flow.grid_shape = np.array(
        [flow.x_vec.shape[0], flow.y_vec.shape[0], flow.z_vec.shape[0]]
    )
    flow.simulation_config = simulation_config
    flow.is_loaded = True
    return flow


def build_reduced_target_from_reference(
    reference_flow: FlowData,
    keep_dims: Tuple[int, ...] = (1, 2),
    atol: float = 1e-10,
    fill_dim_values: Optional[Dict[int, float]] = None,
) -> Tuple[FlowData, np.ndarray]:
    """
    Build a reduced target mesh by keeping unique coordinates in selected dimensions.

    Example:
        keep_dims=(1,2) builds a unique (y,z) mesh and fills x with a constant.

    Returns:
        reduced_flow: reduced target FlowData
        inverse_map: indices so full_values = reduced_values[inverse_map]
    """
    if reference_flow.coords.shape[0] == 0:
        raise ValueError("reference_flow.coords is empty")

    if not keep_dims:
        raise ValueError("keep_dims must not be empty")

    for dim in keep_dims:
        if dim not in (0, 1, 2):
            raise ValueError(f"keep_dims contains invalid axis: {dim}")
    if len(set(keep_dims)) != len(keep_dims):
        raise ValueError(f"keep_dims contains duplicates: {keep_dims}")

    coords_3d = np.asarray(reference_flow.coords, dtype=float)
    kept = coords_3d[:, keep_dims]

    decimals = int(max(0, np.ceil(-np.log10(atol))))
    kept_rounded = np.round(kept, decimals=decimals)
    kept_unique, inverse_map = np.unique(kept_rounded, axis=0, return_inverse=True)

    reduced_coords = np.zeros((kept_unique.shape[0], 3), dtype=float)
    for idx, dim in enumerate(keep_dims):
        reduced_coords[:, dim] = kept_unique[:, idx]

    fill_dim_values = {} if fill_dim_values is None else dict(fill_dim_values)
    for dim in (0, 1, 2):
        if dim in keep_dims:
            continue
        value = fill_dim_values.get(dim, float(np.mean(coords_3d[:, dim])))
        reduced_coords[:, dim] = float(value)

    reduced_flow = _new_flow_from_coords(
        coords=reduced_coords,
        simulation_config=reference_flow.simulation_config,
    )
    return reduced_flow, inverse_map


def _interpolate_scalar(
    source_points: np.ndarray,
    source_values: np.ndarray,
    target_points: np.ndarray,
    method: str,
    fallback_method: Optional[str],
) -> np.ndarray:
    out = griddata(source_points, source_values, target_points, method=method)
    if fallback_method and np.any(np.isnan(out)):
        fallback = griddata(
            source_points,
            source_values,
            target_points,
            method=fallback_method,
        )
        out = np.where(np.isnan(out), fallback, out)
    return np.asarray(out, dtype=float).reshape(-1)


def _interpolate_array(
    source_points: np.ndarray,
    source_values: np.ndarray,
    target_points: np.ndarray,
    method: str,
    fallback_method: Optional[str],
) -> np.ndarray:
    source_values = np.asarray(source_values, dtype=float)
    if source_values.shape[0] != source_points.shape[0]:
        raise ValueError(
            "source_values first axis must match source_points count. "
            f"Got values={source_values.shape}, points={source_points.shape}"
        )

    if source_values.ndim == 1:
        return _interpolate_scalar(
            source_points=source_points,
            source_values=source_values,
            target_points=target_points,
            method=method,
            fallback_method=fallback_method,
        )

    tail_shape = source_values.shape[1:]
    source_flat = source_values.reshape(source_values.shape[0], -1)
    target_flat = np.zeros((target_points.shape[0], source_flat.shape[1]), dtype=float)

    for col in range(source_flat.shape[1]):
        target_flat[:, col] = _interpolate_scalar(
            source_points=source_points,
            source_values=source_flat[:, col],
            target_points=target_points,
            method=method,
            fallback_method=fallback_method,
        )

    return target_flat.reshape((target_points.shape[0],) + tail_shape)


def interpolate_fields_between_flows(
    source_flow: FlowData,
    target_flow: FlowData,
    field_names: List[str],
    source_dims: Tuple[int, ...] = (1, 2),
    target_dims: Tuple[int, ...] = (1, 2),
    method: str = "linear",
    fallback_method: Optional[str] = "nearest",
) -> FlowData:
    """
    Interpolate fields from source_flow to target_flow using selected coordinate dimensions.
    """
    if source_flow.coords.shape[0] == 0:
        raise ValueError("source_flow.coords is empty")
    if target_flow.coords.shape[0] == 0:
        raise ValueError("target_flow.coords is empty")
    if len(source_dims) != len(target_dims):
        raise ValueError(
            "source_dims and target_dims must have the same length. "
            f"Got {source_dims} and {target_dims}"
        )

    source_points = source_flow.coords[:, source_dims]
    target_points = target_flow.coords[:, target_dims]

    mapped = _new_flow_from_coords(
        coords=target_flow.coords,
        simulation_config=target_flow.simulation_config,
    )

    source_field_names = set(source_flow.get_field_names())
    target_field_names = set(mapped.get_field_names())

    for field_name in field_names:
        if field_name not in source_field_names:
            raise ValueError(f"Unknown source field: {field_name}")
        if field_name not in target_field_names:
            raise ValueError(f"Unknown target field: {field_name}")

        source_field = np.asarray(getattr(source_flow, field_name))
        if source_field.size == 0:
            raise ValueError(f"Source field '{field_name}' is empty")

        mapped_field = _interpolate_array(
            source_points=source_points,
            source_values=source_field,
            target_points=target_points,
            method=method,
            fallback_method=fallback_method,
        )
        setattr(mapped, field_name, mapped_field)

    return mapped


def expand_fields_by_inverse_map(
    reduced_flow: FlowData,
    full_reference_flow: FlowData,
    inverse_map: np.ndarray,
    field_names: List[str],
) -> FlowData:
    """
    Expand reduced fields back to a full mesh using an inverse map.
    """
    if reduced_flow.coords.shape[0] == 0:
        raise ValueError("reduced_flow.coords is empty")
    if full_reference_flow.coords.shape[0] == 0:
        raise ValueError("full_reference_flow.coords is empty")

    inverse_map = np.asarray(inverse_map, dtype=np.int64).reshape(-1)
    n_full = full_reference_flow.coords.shape[0]
    n_reduced = reduced_flow.coords.shape[0]

    if inverse_map.shape[0] != n_full:
        raise ValueError(
            "inverse_map length must match full_reference_flow point count. "
            f"Got inverse_map={inverse_map.shape[0]}, full={n_full}"
        )
    if np.any(inverse_map < 0) or np.any(inverse_map >= n_reduced):
        raise ValueError(
            "inverse_map has out-of-range indices. "
            f"Expected [0, {n_reduced - 1}]"
        )

    expanded = _new_flow_from_coords(
        coords=full_reference_flow.coords,
        simulation_config=full_reference_flow.simulation_config,
    )

    reduced_field_names = set(reduced_flow.get_field_names())
    expanded_field_names = set(expanded.get_field_names())

    for field_name in field_names:
        if field_name not in reduced_field_names:
            raise ValueError(f"Unknown reduced field: {field_name}")
        if field_name not in expanded_field_names:
            raise ValueError(f"Unknown expanded field: {field_name}")

        values = np.asarray(getattr(reduced_flow, field_name))
        if values.size == 0:
            raise ValueError(f"Reduced field '{field_name}' is empty")
        if values.shape[0] != n_reduced:
            raise ValueError(
                f"Reduced field '{field_name}' first axis must match reduced coords count "
                f"({n_reduced}), got {values.shape}"
            )

        setattr(expanded, field_name, values[inverse_map].copy())

    return expanded
