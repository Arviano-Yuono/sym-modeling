"""
Utilities to read/write OpenFOAM fields using foamlib and FlowData.

This module is intentionally independent from loaders so it can be used as a
standalone bridge:
    - OpenFOAM -> FlowData
    - FlowData -> OpenFOAM
"""

import os
import subprocess
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from sym_modeling.domains.cfd.data import FlowData
from sym_modeling.domains.cfd.coord_mapping import build_index_map_by_coords


DEFAULT_FLOW_TO_FOAM_FIELD_MAP = {
    "U": "U",
    "p": "p",
    "T": "T",
    "rho": "rho",
    "nu": "nu",
    "k": "k",
    "epsilon": "epsilon",
    "omega": "omega",
    "nut": "nut",
    "Rij": "R",
    "gradU": "gradU",
    "Sij": "Sij",
    "Wij": "Wij",
    "T1": "T1",
    "T2": "T2",
    "T3": "T3",
    "I1": "I1",
    "I2": "I2",
}

DEFAULT_RIJ_READ_CANDIDATES = ["R", "Rij", "ReynoldsStress", "tauRANS", "tau_ij"]
DEFAULT_TEMPLATE_FIELDS = ["U", "p", "k", "omega", "nut"]
DEFAULT_FIELD_CLASS = {
    "U": "volVectorField",
    "Rij": "volSymmTensorField",
}
SUPPORTED_SAMPLE_LOCATIONS = ("cell", "point")
FLOW_FIELD_TAIL_SHAPES = {
    "U": (3,),
    "p": (),
    "T": (),
    "rho": (),
    "nu": (),
    "k": (),
    "epsilon": (),
    "omega": (),
    "nut": (),
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


def _get_case(case_path: str) -> Any:
    try:
        import foamlib  # type: ignore
    except Exception as e:
        raise ImportError(
            "foamlib is not installed/importable in this environment. Install it first."
        ) from e

    return foamlib.FoamCase(case_path)


def _resolve_time_name(case: Any, case_path: str, time: Optional[str]) -> str:
    if time and str(time).lower() not in ("latest", "end", "final"):
        return str(time)

    time_names = [str(td.name) for td in case]
    if not time_names:
        dirs = [
            d
            for d in os.listdir(case_path)
            if os.path.isdir(os.path.join(case_path, d))
        ]
        time_names = dirs

    def _is_float(value: str) -> bool:
        try:
            float(value)
            return True
        except Exception:
            return False

    numeric_names = [t for t in time_names if _is_float(t)]
    if numeric_names:
        return sorted(numeric_names, key=lambda x: float(x))[-1]

    if time_names:
        return sorted(time_names)[-1]

    raise RuntimeError("No OpenFOAM time directories found")


def _read_cell_centers(case: Any, time_name: str) -> np.ndarray:
    time_dir = case[time_name]
    C_file = time_dir.cell_centers()
    C = np.asarray(C_file["internalField"], dtype=float)
    if C.ndim != 2 or C.shape[1] != 3:
        raise RuntimeError(f"cell centres have unexpected shape: {C.shape}")
    return C


def _read_mesh_points(case: Any) -> np.ndarray:
    points_file = case.file("constant/polyMesh/points")
    points = np.asarray(points_file[None], dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise RuntimeError(f"mesh points have unexpected shape: {points.shape}")
    return points

def _build_cell_to_point_connectivity(case: Any) -> Tuple[List[np.ndarray], int]:
    faces_raw = case.file("constant/polyMesh/faces")[None]
    owner = np.asarray(case.file("constant/polyMesh/owner")[None], dtype=np.int64).reshape(-1)
    neighbour = np.asarray(case.file("constant/polyMesh/neighbour")[None], dtype=np.int64).reshape(
        -1
    )

    if owner.size == 0:
        raise RuntimeError("OpenFOAM mesh has empty owner list")

    max_owner = int(owner.max())
    max_neighbour = int(neighbour.max()) if neighbour.size != 0 else -1
    n_cells = max(max_owner, max_neighbour) + 1

    cell_points: List[set] = [set() for _ in range(n_cells)]
    for face_idx, face in enumerate(faces_raw):
        face_points = np.asarray(face, dtype=np.int64).reshape(-1)
        if face_points.size == 0:
            continue

        own = int(owner[face_idx])
        cell_points[own].update(face_points.tolist())

        if face_idx < neighbour.size:
            nei = int(neighbour[face_idx])
            cell_points[nei].update(face_points.tolist())

    cell_point_index = [
        np.fromiter(sorted(point_ids), dtype=np.int64) if point_ids else np.empty((0,), dtype=np.int64)
        for point_ids in cell_points
    ]
    return cell_point_index, n_cells


def _interpolate_cell_values_to_points(
    cell_values: np.ndarray,
    cell_to_point: Sequence[np.ndarray],
    n_points: int,
) -> np.ndarray:
    cell_values = np.asarray(cell_values, dtype=float)
    n_cells = len(cell_to_point)

    if cell_values.shape[0] != n_cells:
        raise ValueError(
            "Cannot interpolate cell values to points: first axis of values must "
            f"match number of mesh cells. Got values={cell_values.shape}, n_cells={n_cells}"
        )

    flat = cell_values.reshape(n_cells, -1)
    summed = np.zeros((n_points, flat.shape[1]), dtype=float)
    counts = np.zeros((n_points,), dtype=float)

    for cell_idx, point_idx in enumerate(cell_to_point):
        if point_idx.size == 0:
            continue
        summed[point_idx] += flat[cell_idx]
        counts[point_idx] += 1.0

    if np.any(counts == 0):
        orphan_points = int(np.sum(counts == 0))
        raise RuntimeError(
            "Found mesh points that are not connected to any cell while interpolating "
            f"cell values to points. orphan_points={orphan_points}"
        )

    point_flat = summed / counts[:, None]
    return point_flat.reshape((n_points,) + cell_values.shape[1:])


def _extract_internal_field_array(field_file: Any, n_points: Optional[int]) -> np.ndarray:
    internal = field_file["internalField"]

    if isinstance(internal, np.ndarray):
        return np.asarray(internal, dtype=float)

    # OpenFOAM uniform fields may come as ("uniform", value_or_$name)
    if (
        isinstance(internal, tuple)
        and len(internal) == 2
        and str(internal[0]).lower() == "uniform"
    ):
        if n_points is None:
            raise ValueError(
                "Cannot expand uniform internalField without known point count"
            )

        value = internal[1]
        if isinstance(value, str) and value.startswith("$"):
            ref_name = value[1:]
            value = field_file[ref_name]

        value_np = np.asarray(value, dtype=float)
        if value_np.ndim == 0:
            return np.full((n_points,), float(value_np), dtype=float)
        return np.tile(value_np.reshape(1, -1), (n_points, 1))

    return np.asarray(internal, dtype=float)


def _read_openfoam_field(
    case: Any,
    time_name: str,
    foam_field_name: str,
    n_points: Optional[int],
) -> np.ndarray:
    field_file = case.file(f"{time_name}/{foam_field_name}")
    return _extract_internal_field_array(field_file=field_file, n_points=n_points)


def _read_openfoam_field_at_location(
    case: Any,
    time_name: str,
    foam_field_name: str,
    sample_location: str,
    point_count: int,
    get_cell_connectivity: Optional[Callable[[], Tuple[List[np.ndarray], int]]] = None,
) -> np.ndarray:
    if sample_location == "cell":
        return _read_openfoam_field(
            case=case,
            time_name=time_name,
            foam_field_name=foam_field_name,
            n_points=point_count,
        )

    if sample_location != "point":
        raise ValueError(
            f"Unsupported sample_location='{sample_location}'. "
            f"Expected one of {SUPPORTED_SAMPLE_LOCATIONS}"
        )

    # First try direct point field (if available)
    point_values = _read_openfoam_field(
        case=case,
        time_name=time_name,
        foam_field_name=foam_field_name,
        n_points=point_count,
    )
    if point_values.shape[0] == point_count:
        return point_values

    if get_cell_connectivity is None:
        raise RuntimeError(
            "Point sampling requires cell connectivity helper for volField -> point interpolation"
        )

    cell_to_point, n_cells = get_cell_connectivity()
    cell_values = _read_openfoam_field(
        case=case,
        time_name=time_name,
        foam_field_name=foam_field_name,
        n_points=n_cells,
    )
    if cell_values.shape[0] != n_cells:
        raise RuntimeError(
            f"OpenFOAM field '{foam_field_name}' has unsupported size {cell_values.shape[0]} "
            f"for point sampling (expected point_count={point_count} or cell_count={n_cells})"
        )
    return _interpolate_cell_values_to_points(
        cell_values=cell_values,
        cell_to_point=cell_to_point,
        n_points=point_count,
    )


def _coerce_values_to_flow_field_shape(
    flow_field_name: str,
    raw_values: np.ndarray,
    n_points: int,
) -> np.ndarray:
    values = np.asarray(raw_values, dtype=float)
    expected_tail = FLOW_FIELD_TAIL_SHAPES.get(flow_field_name, ())

    if expected_tail == ():
        if values.ndim == 2 and values.shape[1] == 1:
            values = values[:, 0]
        return values.reshape(-1)

    if values.ndim == 1 + len(expected_tail) and values.shape[1:] == expected_tail:
        return values

    if values.ndim == 2 and expected_tail == (3, 3) and values.shape[1] == 9:
        return values.reshape(n_points, 3, 3)

    if values.ndim == 2 and expected_tail == (3, 3) and values.shape[1] == 6:
        return _symm6_openfoam_to_tensor(values)

    expected_size = int(n_points * np.prod(expected_tail))
    if values.size == expected_size:
        return values.reshape((n_points,) + expected_tail)

    raise ValueError(
        f"OpenFOAM field for '{flow_field_name}' has incompatible shape {values.shape}. "
        f"Expected (N,{','.join(str(x) for x in expected_tail)}) with N={n_points}."
    )


def _to_rij_tensor(field_values: np.ndarray) -> np.ndarray:
    if field_values.ndim == 2 and field_values.shape[1] == 6:
        xx, yy, zz, xy, yz, xz = [field_values[:, j] for j in range(6)]
        out = np.zeros((field_values.shape[0], 3, 3), dtype=float)
        out[:, 0, 0] = xx
        out[:, 1, 1] = yy
        out[:, 2, 2] = zz
        out[:, 0, 1] = out[:, 1, 0] = xy
        out[:, 1, 2] = out[:, 2, 1] = yz
        out[:, 0, 2] = out[:, 2, 0] = xz
        return out

    if field_values.ndim == 2 and field_values.shape[1] == 9:
        out = field_values.reshape(-1, 3, 3)
        out = 0.5 * (out + np.transpose(out, (0, 2, 1)))
        return out

    if field_values.ndim == 3 and field_values.shape[1:] == (3, 3):
        out = 0.5 * (field_values + np.transpose(field_values, (0, 2, 1)))
        return out

    raise ValueError(
        "Unsupported Rij field shape for conversion. "
        f"Expected (N,6), (N,9), or (N,3,3), got {field_values.shape}"
    )


def _symm6_openfoam_to_tensor(field_values: np.ndarray) -> np.ndarray:
    """
    Convert OpenFOAM symmTensor compact values (N,6) to full tensors (N,3,3).

    OpenFOAM compact order:
        (xx, xy, xz, yy, yz, zz)
    """
    if field_values.ndim != 2 or field_values.shape[1] != 6:
        raise ValueError(
            f"Expected OpenFOAM symmTensor compact shape (N,6), got {field_values.shape}"
        )

    xx = field_values[:, 0]
    xy = field_values[:, 1]
    xz = field_values[:, 2]
    yy = field_values[:, 3]
    yz = field_values[:, 4]
    zz = field_values[:, 5]

    out = np.zeros((field_values.shape[0], 3, 3), dtype=float)
    out[:, 0, 0] = xx
    out[:, 1, 1] = yy
    out[:, 2, 2] = zz
    out[:, 0, 1] = out[:, 1, 0] = xy
    out[:, 0, 2] = out[:, 2, 0] = xz
    out[:, 1, 2] = out[:, 2, 1] = yz
    return out


def _to_openfoam_rij_symm6(rij: np.ndarray) -> np.ndarray:
    if rij.ndim == 2 and rij.shape[1] == 6:
        return np.asarray(rij, dtype=float)

    if rij.ndim != 3 or rij.shape[1:] != (3, 3):
        raise ValueError(
            f"Rij must be shape (N,3,3) or (N,6), got {rij.shape}"
        )

    # OpenFOAM compact symmTensor order:
    # (xx, xy, xz, yy, yz, zz)
    out = np.zeros((rij.shape[0], 6), dtype=float)
    out[:, 0] = rij[:, 0, 0]  # xx
    out[:, 1] = 0.5 * (rij[:, 0, 1] + rij[:, 1, 0])  # xy
    out[:, 2] = 0.5 * (rij[:, 0, 2] + rij[:, 2, 0])  # xz
    out[:, 3] = rij[:, 1, 1]  # yy
    out[:, 4] = 0.5 * (rij[:, 1, 2] + rij[:, 2, 1])  # yz
    out[:, 5] = rij[:, 2, 2]  # zz
    return out


def _build_index_map_by_coords(
    source_coords: np.ndarray,
    target_coords: np.ndarray,
    atol: float,
) -> np.ndarray:
    return build_index_map_by_coords(
        source_coords=source_coords,
        target_coords=target_coords,
        atol=atol,
    )


def _prepare_internal_field_for_write(flow_field_name: str, values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)

    if flow_field_name == "U":
        if values.ndim != 2 or values.shape[1] != 3:
            raise ValueError(f"U must have shape (N,3), got {values.shape}")
        return values

    if flow_field_name == "Rij":
        return _to_openfoam_rij_symm6(values)

    if values.ndim == 1:
        return values

    raise ValueError(
        f"Field '{flow_field_name}' must be 1D scalar, (N,3) vector, or Rij tensor; got {values.shape}"
    )


def _find_template_file(
    case: Any,
    time_name: str,
    template_time: Optional[str],
    foam_field_name: str,
) -> Optional[Any]:
    candidate_paths = []

    if template_time:
        candidate_paths.append(f"{template_time}/{foam_field_name}")
        candidate_paths.extend(
            [f"{template_time}/{name}" for name in DEFAULT_TEMPLATE_FIELDS]
        )

    candidate_paths.append(f"{time_name}/{foam_field_name}")
    candidate_paths.extend([f"{time_name}/{name}" for name in DEFAULT_TEMPLATE_FIELDS])

    case_root = str(case.path)
    for rel_path in candidate_paths:
        abs_path = os.path.join(case_root, rel_path)
        if os.path.exists(abs_path):
            return case.file(rel_path)

    return None


def _initialize_missing_field_file(
    field_file: Any,
    template_file: Any,
    n_points: int,
    flow_field_name: str,
) -> None:
    for key in template_file.keys():
        if key != "internalField":
            field_file[key] = template_file[key]

    field_file["internalField"] = np.zeros((n_points,), dtype=float)

    expected_class = DEFAULT_FIELD_CLASS.get(flow_field_name)
    if expected_class is not None:
        field_file["FoamFile"] = {"class": expected_class}


def read_flow_data_from_openfoam(
    case_path: str,
    flow_data: Optional[FlowData] = None,
    time: Optional[str] = None,
    fields: Optional[List[str]] = None,
    field_map: Optional[Dict[str, str]] = None,
    sample_location: str = "cell",
) -> FlowData:
    """
    Read OpenFOAM fields into FlowData.

    Args:
        case_path: OpenFOAM case directory path.
        flow_data: Optional FlowData to populate. If None, a new instance is created.
        time: Time directory to read (e.g. "3000"). If None/"latest", uses latest.
        fields: FlowData field names to load. If None, uses supported defaults.
        field_map: Optional map {flow_field_name: openfoam_field_name}.
        sample_location: "cell" for cell-centre values or "point" for mesh-point values.
            In "point" mode, volFields are interpolated from cells to points if needed.

    Returns:
        FlowData populated with coordinates and requested fields.
    """
    flow_data = FlowData() if flow_data is None else flow_data
    field_map = {} if field_map is None else dict(field_map)

    if sample_location not in SUPPORTED_SAMPLE_LOCATIONS:
        raise ValueError(
            f"Unsupported sample_location='{sample_location}'. "
            f"Expected one of {SUPPORTED_SAMPLE_LOCATIONS}"
        )

    case = _get_case(case_path=case_path)
    time_name = _resolve_time_name(case=case, case_path=case_path, time=time)

    if sample_location == "cell":
        coords = _read_cell_centers(case=case, time_name=time_name)
    else:
        coords = _read_mesh_points(case=case)
    n_points = coords.shape[0]

    cell_connectivity_cache: Optional[Tuple[List[np.ndarray], int]] = None

    def _get_cell_connectivity() -> Tuple[List[np.ndarray], int]:
        nonlocal cell_connectivity_cache
        if cell_connectivity_cache is None:
            cell_connectivity_cache = _build_cell_to_point_connectivity(case=case)
        return cell_connectivity_cache

    flow_data.coords = coords
    flow_data.x_vec = np.unique(coords[:, 0])
    flow_data.y_vec = np.unique(coords[:, 1])
    flow_data.z_vec = np.unique(coords[:, 2])

    requested_fields = (
        fields
        if fields is not None
        else [
            "U",
            "p",
            "T",
            "rho",
            "nu",
            "k",
            "epsilon",
            "omega",
            "nut",
            "Rij",
        ]
    )

    for flow_field_name in requested_fields:
        if flow_field_name not in flow_data.get_field_names():
            raise ValueError(f"Unknown FlowData field: '{flow_field_name}'")

        try:
            if flow_field_name == "Rij" and "Rij" not in field_map:
                rij_values = None
                for candidate in DEFAULT_RIJ_READ_CANDIDATES:
                    try:
                        arr = _read_openfoam_field_at_location(
                            case=case,
                            time_name=time_name,
                            foam_field_name=candidate,
                            sample_location=sample_location,
                            point_count=n_points,
                            get_cell_connectivity=_get_cell_connectivity
                            if sample_location == "point"
                            else None,
                        )
                        rij_values = _to_rij_tensor(arr)
                        break
                    except Exception:
                        continue
                if rij_values is None:
                    continue
                setattr(flow_data, flow_field_name, rij_values)
                continue

            foam_field_name = field_map.get(
                flow_field_name, DEFAULT_FLOW_TO_FOAM_FIELD_MAP.get(flow_field_name)
            )
            if foam_field_name is None:
                continue

            raw_values = _read_openfoam_field_at_location(
                case=case,
                time_name=time_name,
                foam_field_name=foam_field_name,
                sample_location=sample_location,
                point_count=n_points,
                get_cell_connectivity=_get_cell_connectivity
                if sample_location == "point"
                else None,
            )

            if flow_field_name == "U":
                if raw_values.ndim != 2 or raw_values.shape[1] != 3:
                    raise ValueError(
                        f"OpenFOAM field '{foam_field_name}' expected (N,3), got {raw_values.shape}"
                    )
                setattr(flow_data, flow_field_name, raw_values)
            elif flow_field_name == "Rij":
                setattr(flow_data, flow_field_name, _to_rij_tensor(raw_values))
            else:
                setattr(
                    flow_data,
                    flow_field_name,
                    _coerce_values_to_flow_field_shape(
                        flow_field_name=flow_field_name,
                        raw_values=raw_values,
                        n_points=n_points,
                    ),
                )
        except FileNotFoundError:
            continue

    flow_data.simulation_config = Path(case_path.rstrip(os.sep)).name
    flow_data.n_points = n_points
    flow_data.grid_shape = np.array(
        [
            flow_data.x_vec.shape[0],
            flow_data.y_vec.shape[0],
            flow_data.z_vec.shape[0],
        ]
    )
    flow_data.is_loaded = True

    return flow_data


def write_flow_data_to_openfoam(
    case_path: str,
    flow_data: FlowData,
    time: Optional[str] = None,
    fields: Optional[List[str]] = None,
    field_map: Optional[Dict[str, str]] = None,
    atol: float = 1e-10,
    create_if_missing: bool = False,
    template_time: Optional[str] = "0",
) -> str:
    """
    Write FlowData fields into OpenFOAM field files.

    Mapping is done by coordinates, not by index order, so this is robust when
    FlowData point order differs from OpenFOAM internal cell order.

    Args:
        case_path: OpenFOAM case directory path.
        flow_data: Source FlowData.
        time: Target time directory to write (e.g. "3000"). If None/"latest", uses latest.
        fields: FlowData fields to write. If None, writes known supported fields when available.
        field_map: Optional map {flow_field_name: openfoam_field_name}.
        atol: Coordinate matching tolerance.
        create_if_missing: If True, create missing target field files from templates.
        template_time: Preferred time directory for template headers/boundary when creating fields.

    Returns:
        The resolved time directory name written to.
    """
    if flow_data.coords.shape[0] == 0:
        raise ValueError("flow_data.coords is empty")

    field_map = {} if field_map is None else dict(field_map)
    requested_fields = (
        fields
        if fields is not None
        else [
            "U",
            "p",
            "T",
            "rho",
            "nu",
            "k",
            "epsilon",
            "omega",
            "nut",
            "Rij",
        ]
    )

    case = _get_case(case_path=case_path)
    time_name = _resolve_time_name(case=case, case_path=case_path, time=time)
    foam_coords = _read_cell_centers(case=case, time_name=time_name)

    source_to_target_idx = _build_index_map_by_coords(
        source_coords=flow_data.coords,
        target_coords=foam_coords,
        atol=atol,
    )

    n_points = foam_coords.shape[0]
    for flow_field_name in requested_fields:
        if flow_field_name not in flow_data.get_field_names():
            raise ValueError(f"Unknown FlowData field: '{flow_field_name}'")

        values = np.asarray(getattr(flow_data, flow_field_name))
        if values.size == 0:
            continue
        if values.shape[0] != flow_data.coords.shape[0]:
            raise ValueError(
                f"Field '{flow_field_name}' first axis must match flow_data.coords count "
                f"({flow_data.coords.shape[0]}), got {values.shape}"
            )

        foam_field_name = field_map.get(
            flow_field_name, DEFAULT_FLOW_TO_FOAM_FIELD_MAP.get(flow_field_name)
        )
        if foam_field_name is None:
            continue

        field_values = _prepare_internal_field_for_write(
            flow_field_name=flow_field_name,
            values=values,
        )
        field_values = field_values[source_to_target_idx]

        rel_path = f"{time_name}/{foam_field_name}"
        abs_path = os.path.join(str(case.path), rel_path)
        target_file = case.file(rel_path)

        if not os.path.exists(abs_path):
            if not create_if_missing:
                raise FileNotFoundError(
                    f"Target field file does not exist: {abs_path}. "
                    "Set create_if_missing=True to create it."
                )

            template = _find_template_file(
                case=case,
                time_name=time_name,
                template_time=template_time,
                foam_field_name=foam_field_name,
            )
            if template is None:
                raise FileNotFoundError(
                    "Cannot create missing field file because no template field was found. "
                    f"Target={rel_path}, template_time={template_time}"
                )
            _initialize_missing_field_file(
                field_file=target_file,
                template_file=template,
                n_points=n_points,
                flow_field_name=flow_field_name,
            )

        target_file["internalField"] = field_values

    return time_name


def _build_openfoam_time_args(time: Optional[str]) -> List[str]:
    """
    Build OpenFOAM CLI time-selection args.

    Rules:
        - None / latest / end / final -> ["-latestTime"]
        - otherwise                    -> ["-time", "<time>"]
    """
    if time is None:
        return ["-latestTime"]

    value = str(time).strip()
    if value.lower() in {"latest", "end", "final"}:
        return ["-latestTime"]

    return ["-time", value]


def run_sparta_feature_postprocess(
    case_path: str,
    time: Optional[str] = "latest",
    omega_floor: float = 1e-30,
    run_surface_sampling: bool = False,
    surface_function_name: str = "surfaceSampling",
    utility_executable: str = "spartaFeaturePostProcess",
    postprocess_executable: str = "postProcess",
    check: bool = True,
) -> Tuple[int, Optional[int]]:
    """
    Run OpenFOAM SpaRTA feature post-processing for the 2D-plane workflow.

    This helper:
        1) Runs the custom utility `spartaFeaturePostProcess` to write:
           gradU, tau, Sij, Wij, I1, I2, T1, T2, T3 in the selected time dir.
        2) Optionally runs your existing surface sampling function object
           (usually `surfaceSampling`) so these fields are available on the
           2D sampled plane.

    Notes:
        - This function does not modify dictionaries.
        - It assumes `surface_function_name` already exists in the case setup.
        - It assumes OpenFOAM environment is sourced.
        - `run_surface_sampling` is disabled by default because the
          `surfaceSampling` function-object dictionary syntax is
          OpenFOAM-version dependent.

    Returns:
        (feature_returncode, surface_sampling_returncode_or_None)
    """
    if omega_floor <= 0:
        raise ValueError(f"omega_floor must be > 0, got {omega_floor}")

    case_dir = Path(case_path)
    if not case_dir.is_dir():
        raise FileNotFoundError(f"OpenFOAM case path does not exist: {case_path}")

    time_args = _build_openfoam_time_args(time=time)

    feature_cmd = [
        utility_executable,
        "-case",
        str(case_dir),
        *time_args,
        "-omegaFloor",
        f"{float(omega_floor):.16g}",
    ]
    feature_result = subprocess.run(
        feature_cmd,
        check=check,
    )

    sampling_code: Optional[int] = None
    if run_surface_sampling:
        sampling_cmd = [
            postprocess_executable,
            "-case",
            str(case_dir),
            *time_args,
            "-func",
            surface_function_name,
        ]
        sampling_result = subprocess.run(
            sampling_cmd,
            check=check,
        )
        sampling_code = int(sampling_result.returncode)

    return int(feature_result.returncode), sampling_code
