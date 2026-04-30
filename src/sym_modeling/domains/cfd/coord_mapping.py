"""
Coordinate-based point mapping utilities.
"""

import numpy as np
from scipy.spatial import cKDTree


def build_index_map_by_coords(
    source_coords: np.ndarray,
    target_coords: np.ndarray,
    atol: float = 1e-10,
) -> np.ndarray:
    """
    Build source->target index mapping using 3D coordinates.

    Returns an index array `idx` so that:
        mapped_values = source_values[idx]
    produces values ordered like `target_coords`.
    """
    source_coords = np.asarray(source_coords, dtype=float)
    target_coords = np.asarray(target_coords, dtype=float)

    if source_coords.ndim != 2 or source_coords.shape[1] != 3:
        raise ValueError(f"Expected source_coords shape (N,3), got {source_coords.shape}")
    if target_coords.ndim != 2 or target_coords.shape[1] != 3:
        raise ValueError(f"Expected target_coords shape (N,3), got {target_coords.shape}")
    if source_coords.shape[0] != target_coords.shape[0]:
        raise ValueError(
            "Source and target point counts must match. "
            f"Got {source_coords.shape[0]} and {target_coords.shape[0]}"
        )

    decimals = int(max(0, np.ceil(-np.log10(atol))))
    source_keys = np.round(source_coords, decimals=decimals)
    target_keys = np.round(target_coords, decimals=decimals)

    key_to_idx = {}
    for idx, row in enumerate(source_keys):
        key = tuple(row.tolist())
        if key in key_to_idx:
            raise ValueError(
                "Duplicate coordinates detected in source coordinates after rounding. "
                "Cannot build one-to-one mapping."
            )
        key_to_idx[key] = idx

    mapped = np.full(target_coords.shape[0], -1, dtype=np.int64)
    for idx, row in enumerate(target_keys):
        mapped[idx] = key_to_idx.get(tuple(row.tolist()), -1)

    if np.all(mapped >= 0):
        return mapped

    tree = cKDTree(source_coords)
    distances, nn_idx = tree.query(target_coords, k=1)

    if np.any(distances > atol):
        raise ValueError(
            "Unable to map all points within tolerance. "
            f"Max nearest distance={float(np.max(distances)):.3e}, atol={atol:.3e}"
        )

    if np.unique(nn_idx).size != target_coords.shape[0]:
        raise ValueError(
            "Point mapping is not one-to-one (duplicate nearest neighbours found). "
            "Check coordinate consistency or tighten tolerance."
        )

    return nn_idx.astype(np.int64)
