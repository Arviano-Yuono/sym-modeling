from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from sym_modeling.domains.fem.io.hyperelastic import _compute_triangle_gradients  # noqa: E402


DEFAULT_INPUT_DIR = Path("dataset/fem_data/thermocorr")
DEFAULT_OUTPUT_DIR = DEFAULT_INPUT_DIR / "formatted"
THERMOCORR_HDF5_NAME = "TPS_2.hdf5"
FORCE_FILE_NAME = "force_vfm.csv"
DOC_URLS = (
    "https://thermocorr.ippt.pan.pl/output-file-structure.html",
    "https://zenodo.org/records/18617429",
)


@dataclass(frozen=True)
class ThermoCorrMesh:
    x_nodes: np.ndarray
    source_node_indices: np.ndarray
    connectivity: np.ndarray
    grad_na: np.ndarray
    qp_weights: np.ndarray
    scale: float
    x_min: float
    y_min: float
    y_max: float
    raw_grid_dx: float
    raw_grid_dy: float
    num_valid_points: int
    num_dropped_points: int
    num_quads: int


@dataclass(frozen=True)
class ThermoCorrFormatResult:
    input_dir: Path
    output_dir: Path
    manifest_path: Path
    load_steps: tuple[int, ...]
    num_nodes: int
    num_elements: int
    num_quads: int
    num_uniaxial_curves: int


@dataclass(frozen=True)
class ReactionScaling:
    mode: str
    factor: float
    physical_height: float | None = None
    thickness: float | None = None


@dataclass(frozen=True)
class ThermoCorrSamplingInfo:
    enabled: bool
    method: str
    requested_node_count: int | None
    actual_node_count: int
    sample_seed: int
    sample_max_edge_factor: float
    boundary_nodes_preserved: int
    interior_nodes_selected: int
    selected_source_node_indices: tuple[int, ...]
    boundary_source_node_indices: tuple[int, ...]


def _require_scipy_spatial():
    try:
        from scipy.spatial import Delaunay, cKDTree
    except Exception as exc:  # pragma: no cover - exercised only when scipy is unavailable
        raise ImportError(
            "ThermoCorr node sampling requires scipy. Install the project dependencies "
            "or run the formatter without --sample-nodes."
        ) from exc
    return Delaunay, cKDTree


def read_force_values(path: Path) -> np.ndarray:
    values: list[float] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if header is None:
            raise ValueError("Force file is empty: %s" % path)
        for row in reader:
            if not row or not row[0].strip():
                continue
            values.append(float(row[0]))
    if not values:
        raise ValueError("Force file does not contain any force values: %s" % path)
    return np.asarray(values, dtype=float)


def _grid_spacing(values: np.ndarray, axis_name: str) -> float:
    unique_values = np.unique(values)
    diffs = np.diff(unique_values)
    diffs = diffs[diffs > 0.0]
    if diffs.size == 0:
        raise ValueError("Cannot infer %s grid spacing from fewer than two coordinates." % axis_name)
    return float(np.min(diffs))


def _normalize_points(raw_points: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    x_min = float(np.min(raw_points[:, 0]))
    y_min = float(np.min(raw_points[:, 1]))
    y_max = float(np.max(raw_points[:, 1]))
    scale = y_max - y_min
    if scale <= 0.0:
        raise ValueError("Cannot normalize ThermoCorr coordinates with non-positive y span.")
    normalized = np.column_stack(
        (
            (raw_points[:, 0] - x_min) / scale,
            (y_max - raw_points[:, 1]) / scale,
        )
    )
    return normalized, {"x_min": x_min, "y_min": y_min, "y_max": y_max, "scale": scale}


def _orient_triangle_ccw(triangle: tuple[int, int, int], points: np.ndarray) -> tuple[int, int, int]:
    p0, p1, p2 = points[list(triangle)]
    det = (p1[0] - p0[0]) * (p2[1] - p0[1]) - (p2[0] - p0[0]) * (p1[1] - p0[1])
    if np.isclose(det, 0.0):
        raise ValueError("Encountered a degenerate ThermoCorr triangle.")
    if det < 0.0:
        return (triangle[0], triangle[2], triangle[1])
    return triangle


def infer_thermocorr_mesh(raw_nodes: np.ndarray) -> ThermoCorrMesh:
    if raw_nodes.ndim != 2 or raw_nodes.shape[1] < 3:
        raise ValueError("ThermoCorr /region/nodes must be an n x 3 array.")

    valid_mask = raw_nodes[:, 2].astype(bool)
    valid_source_indices = np.flatnonzero(valid_mask)
    raw_points = np.asarray(raw_nodes[valid_source_indices, :2], dtype=float)
    if raw_points.shape[0] < 4:
        raise ValueError("At least four valid DIC points are needed to infer a mesh.")

    normalized_points, normalization = _normalize_points(raw_points)
    raw_grid_dx = _grid_spacing(raw_points[:, 0], "x")
    raw_grid_dy = _grid_spacing(raw_points[:, 1], "y")

    point_to_local = {
        (float(x), float(y)): local_id for local_id, (x, y) in enumerate(raw_points)
    }
    xs = np.unique(raw_points[:, 0])
    ys = np.unique(raw_points[:, 1])

    triangles: list[tuple[int, int, int]] = []
    num_quads = 0
    for y_top, y_bottom in zip(ys[:-1], ys[1:]):
        if not np.isclose(y_bottom - y_top, raw_grid_dy):
            continue
        for x_left, x_right in zip(xs[:-1], xs[1:]):
            if not np.isclose(x_right - x_left, raw_grid_dx):
                continue
            top_left = (float(x_left), float(y_top))
            top_right = (float(x_right), float(y_top))
            bottom_right = (float(x_right), float(y_bottom))
            bottom_left = (float(x_left), float(y_bottom))
            corners = (top_left, top_right, bottom_right, bottom_left)
            if not all(corner in point_to_local for corner in corners):
                continue

            tl = point_to_local[top_left]
            tr = point_to_local[top_right]
            br = point_to_local[bottom_right]
            bl = point_to_local[bottom_left]
            triangles.append(_orient_triangle_ccw((bl, br, tr), normalized_points))
            triangles.append(_orient_triangle_ccw((bl, tr, tl), normalized_points))
            num_quads += 1

    if not triangles:
        raise ValueError("No complete quadrilateral cells could be inferred from ThermoCorr nodes.")

    used_local_nodes = set(node for triangle in triangles for node in triangle)
    connected_local_nodes = np.asarray(
        [local_id for local_id in range(raw_points.shape[0]) if local_id in used_local_nodes],
        dtype=int,
    )
    remap = {int(local_id): new_id for new_id, local_id in enumerate(connected_local_nodes)}
    connectivity = np.asarray(
        [[remap[int(node)] for node in triangle] for triangle in triangles],
        dtype=int,
    )
    x_nodes = normalized_points[connected_local_nodes]
    source_node_indices = valid_source_indices[connected_local_nodes]

    grad_na = np.zeros((connectivity.shape[0], 3, 2), dtype=float)
    qp_weights = np.zeros(connectivity.shape[0], dtype=float)
    for element, node_ids in enumerate(connectivity):
        grad_na[element], qp_weights[element] = _compute_triangle_gradients(x_nodes[node_ids])

    return ThermoCorrMesh(
        x_nodes=x_nodes,
        source_node_indices=source_node_indices,
        connectivity=connectivity,
        grad_na=grad_na,
        qp_weights=qp_weights,
        scale=float(normalization["scale"]),
        x_min=float(normalization["x_min"]),
        y_min=float(normalization["y_min"]),
        y_max=float(normalization["y_max"]),
        raw_grid_dx=raw_grid_dx,
        raw_grid_dy=raw_grid_dy,
        num_valid_points=int(raw_points.shape[0]),
        num_dropped_points=int(raw_points.shape[0] - connected_local_nodes.shape[0]),
        num_quads=int(num_quads),
    )


def _mesh_boundary_node_indices(mesh: ThermoCorrMesh) -> np.ndarray:
    edge_counts: dict[tuple[int, int], int] = {}
    for node1, node2, node3 in mesh.connectivity:
        for raw_edge in ((node1, node2), (node2, node3), (node3, node1)):
            edge = tuple(sorted((int(raw_edge[0]), int(raw_edge[1]))))
            edge_counts[edge] = edge_counts.get(edge, 0) + 1

    boundary_nodes = sorted(
        {
            node
            for edge, count in edge_counts.items()
            if count == 1
            for node in edge
        }
    )
    return np.asarray(boundary_nodes, dtype=int)


def _reaction_boundary_node_indices(mesh: ThermoCorrMesh) -> np.ndarray:
    return np.flatnonzero(np.isclose(mesh.x_nodes[:, 1], np.max(mesh.x_nodes[:, 1])))


def _preserved_boundary_node_indices(mesh: ThermoCorrMesh) -> np.ndarray:
    return np.union1d(_mesh_boundary_node_indices(mesh), _reaction_boundary_node_indices(mesh))


def _make_sampling_info(
    final_mesh: ThermoCorrMesh,
    full_mesh: ThermoCorrMesh,
    method: str,
    requested_node_count: int | None,
    sample_seed: int,
    sample_max_edge_factor: float,
    boundary_indices: np.ndarray,
    selected_indices: np.ndarray,
    enabled: bool,
) -> ThermoCorrSamplingInfo:
    selected_boundary = np.intersect1d(selected_indices, boundary_indices, assume_unique=True)
    return ThermoCorrSamplingInfo(
        enabled=enabled,
        method=method,
        requested_node_count=None if requested_node_count is None else int(requested_node_count),
        actual_node_count=int(final_mesh.x_nodes.shape[0]),
        sample_seed=int(sample_seed),
        sample_max_edge_factor=float(sample_max_edge_factor),
        boundary_nodes_preserved=int(selected_boundary.shape[0]),
        interior_nodes_selected=int(selected_indices.shape[0] - selected_boundary.shape[0]),
        selected_source_node_indices=tuple(int(index) for index in final_mesh.source_node_indices),
        boundary_source_node_indices=tuple(int(index) for index in full_mesh.source_node_indices[boundary_indices]),
    )


def _random_sample_indices(mesh: ThermoCorrMesh, sample_nodes: int, sample_seed: int) -> np.ndarray:
    if sample_nodes < 3:
        raise ValueError("At least three sampled nodes are needed to build a triangular mesh.")

    rng = np.random.default_rng(int(sample_seed))
    selected = rng.choice(mesh.x_nodes.shape[0], size=sample_nodes, replace=False)

    top_candidates = _reaction_boundary_node_indices(mesh)
    if top_candidates.size and not np.intersect1d(selected, top_candidates).size:
        selected[0] = int(rng.choice(top_candidates))

    selected = np.asarray(sorted(set(int(index) for index in selected)), dtype=int)
    if selected.shape[0] != sample_nodes:
        available = np.setdiff1d(np.arange(mesh.x_nodes.shape[0]), selected, assume_unique=True)
        fill = rng.choice(available, size=sample_nodes - selected.shape[0], replace=False)
        selected = np.sort(np.concatenate((selected, fill.astype(int))))

    return selected


def _select_interior_nodes_deterministically(
    mesh: ThermoCorrMesh,
    interior_candidates: np.ndarray,
    anchor_indices: np.ndarray,
    num_to_select: int,
) -> np.ndarray:
    if num_to_select <= 0:
        return np.empty(0, dtype=int)
    if num_to_select >= interior_candidates.shape[0]:
        return np.sort(interior_candidates.astype(int))

    _, cKDTree = _require_scipy_spatial()
    candidate_points = mesh.x_nodes[interior_candidates]
    if anchor_indices.size:
        distances, _ = cKDTree(mesh.x_nodes[anchor_indices]).query(candidate_points, k=1)
        min_distance_sq = np.asarray(distances, dtype=float) ** 2
    else:
        center = np.mean(mesh.x_nodes, axis=0)
        deltas = candidate_points - center
        min_distance_sq = np.einsum("ij,ij->i", deltas, deltas)

    available = np.ones(interior_candidates.shape[0], dtype=bool)
    selected_positions: list[int] = []
    source_order = mesh.source_node_indices[interior_candidates]
    for _ in range(num_to_select):
        scores = min_distance_sq.copy()
        scores[~available] = -np.inf
        best_score = float(np.max(scores))
        best_positions = np.flatnonzero(np.isclose(scores, best_score))
        best_position = int(best_positions[np.argmin(source_order[best_positions])])
        selected_positions.append(best_position)
        available[best_position] = False

        deltas = candidate_points - candidate_points[best_position]
        min_distance_sq = np.minimum(min_distance_sq, np.einsum("ij,ij->i", deltas, deltas))

    return np.sort(interior_candidates[np.asarray(selected_positions, dtype=int)])


def _boundary_coarsen_indices(mesh: ThermoCorrMesh, sample_nodes: int, boundary_indices: np.ndarray) -> np.ndarray:
    all_indices = np.arange(mesh.x_nodes.shape[0], dtype=int)
    interior_candidates = np.setdiff1d(all_indices, boundary_indices, assume_unique=True)
    num_interior_to_select = max(0, int(sample_nodes) - int(boundary_indices.shape[0]))
    selected_interiors = _select_interior_nodes_deterministically(
        mesh,
        interior_candidates=interior_candidates,
        anchor_indices=boundary_indices,
        num_to_select=num_interior_to_select,
    )
    return np.sort(np.concatenate((boundary_indices, selected_interiors)))


def _point_inside_any_triangle(point: np.ndarray, triangles: np.ndarray, tolerance: float = 1e-12) -> bool:
    if triangles.size == 0:
        return False

    a = triangles[:, 0, :]
    b = triangles[:, 1, :]
    c = triangles[:, 2, :]
    denominator = (b[:, 1] - c[:, 1]) * (a[:, 0] - c[:, 0]) + (c[:, 0] - b[:, 0]) * (a[:, 1] - c[:, 1])
    valid = np.abs(denominator) > tolerance
    if not np.any(valid):
        return False

    a = a[valid]
    b = b[valid]
    c = c[valid]
    denominator = denominator[valid]
    alpha = ((b[:, 1] - c[:, 1]) * (point[0] - c[:, 0]) + (c[:, 0] - b[:, 0]) * (point[1] - c[:, 1])) / denominator
    beta = ((c[:, 1] - a[:, 1]) * (point[0] - c[:, 0]) + (a[:, 0] - c[:, 0]) * (point[1] - c[:, 1])) / denominator
    gamma = 1.0 - alpha - beta
    inside = (alpha >= -tolerance) & (beta >= -tolerance) & (gamma >= -tolerance)
    inside &= (alpha <= 1.0 + tolerance) & (beta <= 1.0 + tolerance) & (gamma <= 1.0 + tolerance)
    return bool(np.any(inside))


def _points_inside_mesh_domain(
    points: np.ndarray,
    domain_triangles: np.ndarray,
    centroid_tree,
    search_radius: float,
) -> np.ndarray:
    if points.size == 0:
        return np.zeros(0, dtype=bool)

    candidate_triangle_ids = centroid_tree.query_ball_point(points, r=search_radius)
    inside = np.zeros(points.shape[0], dtype=bool)
    for point_id, triangle_ids in enumerate(candidate_triangle_ids):
        if not triangle_ids:
            continue
        inside[point_id] = _point_inside_any_triangle(points[point_id], domain_triangles[np.asarray(triangle_ids, dtype=int)])
    return inside


def _filter_simplices_to_domain(
    simplices: np.ndarray,
    sampled_points: np.ndarray,
    domain_mesh: ThermoCorrMesh,
    cKDTree,
) -> np.ndarray:
    if simplices.size == 0:
        return simplices

    domain_triangles = domain_mesh.x_nodes[domain_mesh.connectivity]
    domain_centroids = np.mean(domain_triangles, axis=1)
    domain_radii = np.max(np.linalg.norm(domain_triangles - domain_centroids[:, None, :], axis=2), axis=1)
    search_radius = float(np.max(domain_radii)) + 1e-12
    centroid_tree = cKDTree(domain_centroids)

    keep = []
    for simplex in simplices:
        vertices = sampled_points[simplex]
        query_points = np.asarray(
            (
                np.mean(vertices, axis=0),
                0.5 * (vertices[0] + vertices[1]),
                0.5 * (vertices[1] + vertices[2]),
                0.5 * (vertices[2] + vertices[0]),
            ),
            dtype=float,
        )
        keep.append(bool(np.all(_points_inside_mesh_domain(query_points, domain_triangles, centroid_tree, search_radius))))

    return simplices[np.asarray(keep, dtype=bool)]


def _triangulate_selected_mesh(
    mesh: ThermoCorrMesh,
    selected: np.ndarray,
    sample_max_edge_factor: float,
    domain_mesh: ThermoCorrMesh | None = None,
) -> ThermoCorrMesh:
    selected = np.asarray(sorted(set(int(index) for index in selected)), dtype=int)
    if selected.shape[0] < 3:
        raise ValueError("At least three sampled nodes are needed to build a triangular mesh.")

    Delaunay, cKDTree = _require_scipy_spatial()
    sampled_points = mesh.x_nodes[selected]
    try:
        triangulation = Delaunay(sampled_points)
    except Exception as exc:
        raise ValueError("Could not triangulate sampled ThermoCorr points.") from exc

    simplices = np.asarray(triangulation.simplices, dtype=int)
    if sample_max_edge_factor > 0.0 and sampled_points.shape[0] > 3:
        tree = cKDTree(sampled_points)
        distances, _ = tree.query(sampled_points, k=2)
        median_spacing = float(np.median(distances[:, 1]))
        if median_spacing > 0.0:
            max_edge = float(sample_max_edge_factor) * median_spacing
            keep = []
            for simplex in simplices:
                vertices = sampled_points[simplex]
                edges = (
                    np.linalg.norm(vertices[0] - vertices[1]),
                    np.linalg.norm(vertices[1] - vertices[2]),
                    np.linalg.norm(vertices[2] - vertices[0]),
                )
                keep.append(max(edges) <= max_edge)
            simplices = simplices[np.asarray(keep, dtype=bool)]

    if domain_mesh is not None:
        simplices = _filter_simplices_to_domain(simplices, sampled_points, domain_mesh, cKDTree)

    if simplices.size == 0:
        raise ValueError("Sampled ThermoCorr points did not produce any usable triangles.")

    triangles = [_orient_triangle_ccw(tuple(int(node) for node in simplex), sampled_points) for simplex in simplices]
    connectivity = np.asarray(triangles, dtype=int)
    grad_na = np.zeros((connectivity.shape[0], 3, 2), dtype=float)
    qp_weights = np.zeros(connectivity.shape[0], dtype=float)
    for element, node_ids in enumerate(connectivity):
        grad_na[element], qp_weights[element] = _compute_triangle_gradients(sampled_points[node_ids])

    return ThermoCorrMesh(
        x_nodes=sampled_points,
        source_node_indices=mesh.source_node_indices[selected],
        connectivity=connectivity,
        grad_na=grad_na,
        qp_weights=qp_weights,
        scale=mesh.scale,
        x_min=mesh.x_min,
        y_min=mesh.y_min,
        y_max=mesh.y_max,
        raw_grid_dx=mesh.raw_grid_dx,
        raw_grid_dy=mesh.raw_grid_dy,
        num_valid_points=mesh.num_valid_points,
        num_dropped_points=int(mesh.num_valid_points - sampled_points.shape[0]),
        num_quads=0,
    )


def _sample_thermocorr_mesh(
    mesh: ThermoCorrMesh,
    sample_nodes: int | None,
    sample_seed: int,
    sample_max_edge_factor: float,
    sample_method: str,
) -> tuple[ThermoCorrMesh, ThermoCorrSamplingInfo]:
    if sample_method not in {"boundary_coarsen", "random"}:
        raise ValueError("sample_method must be 'boundary_coarsen' or 'random'.")

    boundary_indices = _preserved_boundary_node_indices(mesh)
    all_indices = np.arange(mesh.x_nodes.shape[0], dtype=int)
    if sample_nodes is None:
        sampling_info = _make_sampling_info(
            mesh,
            full_mesh=mesh,
            method="none",
            requested_node_count=None,
            sample_seed=sample_seed,
            sample_max_edge_factor=sample_max_edge_factor,
            boundary_indices=boundary_indices,
            selected_indices=all_indices,
            enabled=False,
        )
        return mesh, sampling_info

    sample_nodes = int(sample_nodes)
    if sample_nodes <= 0:
        raise ValueError("--sample-nodes must be a positive integer.")
    if sample_nodes >= mesh.x_nodes.shape[0]:
        sampling_info = _make_sampling_info(
            mesh,
            full_mesh=mesh,
            method=sample_method,
            requested_node_count=sample_nodes,
            sample_seed=sample_seed,
            sample_max_edge_factor=sample_max_edge_factor,
            boundary_indices=boundary_indices,
            selected_indices=all_indices,
            enabled=False,
        )
        return mesh, sampling_info

    if sample_method == "random":
        selected = _random_sample_indices(mesh, sample_nodes=sample_nodes, sample_seed=sample_seed)
        sampled_mesh = _triangulate_selected_mesh(
            mesh,
            selected=selected,
            sample_max_edge_factor=sample_max_edge_factor,
            domain_mesh=None,
        )
    else:
        selected = _boundary_coarsen_indices(mesh, sample_nodes=sample_nodes, boundary_indices=boundary_indices)
        sampled_mesh = _triangulate_selected_mesh(
            mesh,
            selected=selected,
            sample_max_edge_factor=sample_max_edge_factor,
            domain_mesh=mesh,
        )

    sampling_info = _make_sampling_info(
        sampled_mesh,
        full_mesh=mesh,
        method=sample_method,
        requested_node_count=sample_nodes,
        sample_seed=sample_seed,
        sample_max_edge_factor=sample_max_edge_factor,
        boundary_indices=boundary_indices,
        selected_indices=selected,
        enabled=True,
    )
    return sampled_mesh, sampling_info


def _normalized_displacements(raw_displacements: np.ndarray, mesh: ThermoCorrMesh) -> np.ndarray:
    if raw_displacements.ndim != 2 or raw_displacements.shape[1] < 2:
        raise ValueError("ThermoCorr displacement fields must be n x 3 arrays.")
    selected = np.asarray(raw_displacements[mesh.source_node_indices, :2], dtype=float)
    return np.column_stack((selected[:, 0] / mesh.scale, -selected[:, 1] / mesh.scale))


def _write_csv(path: Path, header: tuple[str, ...], rows: np.ndarray) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for row in rows:
            writer.writerow(row.tolist())


def _write_step_csvs(
    output_dir: Path,
    mesh: ThermoCorrMesh,
    u_nodes: np.ndarray,
    force_value: float,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    bcx = np.zeros(mesh.x_nodes.shape[0], dtype=int)
    bcy = np.zeros(mesh.x_nodes.shape[0], dtype=int)
    top_nodes = np.isclose(mesh.x_nodes[:, 1], np.max(mesh.x_nodes[:, 1]))
    bcy[top_nodes] = 1

    _write_csv(
        output_dir / "output_nodes.csv",
        ("x", "y", "ux", "uy", "bcx", "bcy"),
        np.column_stack((mesh.x_nodes, u_nodes, bcx, bcy)),
    )
    _write_csv(
        output_dir / "output_elements.csv",
        ("node1", "node2", "node3"),
        mesh.connectivity,
    )
    _write_csv(
        output_dir / "output_integrator.csv",
        (
            "gradNa_node1_x",
            "gradNa_node1_y",
            "gradNa_node2_x",
            "gradNa_node2_y",
            "gradNa_node3_x",
            "gradNa_node3_y",
            "qpWeight",
        ),
        np.column_stack(
            (
                mesh.grad_na[:, 0, 0],
                mesh.grad_na[:, 0, 1],
                mesh.grad_na[:, 1, 0],
                mesh.grad_na[:, 1, 1],
                mesh.grad_na[:, 2, 0],
                mesh.grad_na[:, 2, 1],
                mesh.qp_weights,
            )
        ),
    )
    _write_csv(
        output_dir / "output_reactions.csv",
        ("forces",),
        np.asarray([[float(force_value)]], dtype=float),
    )


def export_uniaxial_curves(input_dir: Path, output_dir: Path) -> list[dict[str, object]]:
    uniaxial_output_dir = output_dir / "uniaxial"
    uniaxial_output_dir.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, object]] = []

    for npz_path in sorted(input_dir.glob("e*.npz")):
        with np.load(npz_path) as payload:
            for curve_name in sorted(payload.files):
                curve = np.asarray(payload[curve_name], dtype=float)
                if curve.ndim != 2 or curve.shape[0] != 2:
                    raise ValueError(
                        "Expected curve %s in %s to have shape (2, n), got %s."
                        % (curve_name, npz_path, curve.shape)
                    )
                output_path = uniaxial_output_dir / ("%s_%s.csv" % (npz_path.stem, curve_name))
                _write_csv(
                    output_path,
                    ("strain", "stress"),
                    np.column_stack((curve[0], curve[1])),
                )
                entries.append(
                    {
                        "source": str(npz_path),
                        "curve": curve_name,
                        "path": str(output_path),
                        "num_points": int(curve.shape[1]),
                    }
                )
    return entries


def _prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(
                "Output directory already exists: %s. Pass --overwrite to replace it." % output_dir
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def _manifest_entry_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _resolve_reaction_scaling(
    mesh: ThermoCorrMesh,
    reaction_scale: float | None,
    physical_height: float | None,
    thickness: float | None,
) -> ReactionScaling:
    if reaction_scale is not None and (physical_height is not None or thickness is not None):
        raise ValueError("Pass either reaction_scale or physical_height/thickness, not both.")

    if physical_height is not None or thickness is not None:
        if physical_height is None or thickness is None:
            raise ValueError("physical_height and thickness must be passed together.")
        physical_height = float(physical_height)
        thickness = float(thickness)
        if physical_height <= 0.0 or thickness <= 0.0:
            raise ValueError("physical_height and thickness must be positive.")
        return ReactionScaling(
            mode="physical_stress",
            factor=1.0 / (physical_height * thickness),
            physical_height=physical_height,
            thickness=thickness,
        )

    if reaction_scale is not None:
        reaction_scale = float(reaction_scale)
        if reaction_scale <= 0.0:
            raise ValueError("reaction_scale must be positive.")
        mode = "raw" if np.isclose(reaction_scale, 1.0) else "custom"
        return ReactionScaling(mode=mode, factor=reaction_scale)

    return ReactionScaling(mode="pixel_height", factor=1.0 / mesh.scale)


def format_thermocorr_dataset(
    input_dir: str | Path = DEFAULT_INPUT_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    overwrite: bool = False,
    sample_nodes: int | None = None,
    sample_seed: int = 0,
    sample_max_edge_factor: float = 6.0,
    sample_method: str = "boundary_coarsen",
    reaction_scale: float | None = None,
    physical_height: float | None = None,
    thickness: float | None = None,
) -> ThermoCorrFormatResult:
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    hdf5_path = input_dir / THERMOCORR_HDF5_NAME
    force_path = input_dir / FORCE_FILE_NAME
    if not hdf5_path.is_file():
        raise FileNotFoundError("ThermoCorr HDF5 file does not exist: %s" % hdf5_path)
    if not force_path.is_file():
        raise FileNotFoundError("ThermoCorr force file does not exist: %s" % force_path)

    force_values = read_force_values(force_path)
    _prepare_output_dir(output_dir, overwrite=overwrite)

    load_step_entries: list[dict[str, object]] = []
    with h5py.File(hdf5_path, "r") as hdf5_file:
        raw_nodes = np.asarray(hdf5_file["region/nodes"][:], dtype=float)
        mesh = infer_thermocorr_mesh(raw_nodes)
        mesh, sampling_info = _sample_thermocorr_mesh(
            mesh,
            sample_nodes=sample_nodes,
            sample_seed=sample_seed,
            sample_max_edge_factor=sample_max_edge_factor,
            sample_method=sample_method,
        )
        reaction_scaling = _resolve_reaction_scaling(
            mesh,
            reaction_scale=reaction_scale,
            physical_height=physical_height,
            thickness=thickness,
        )

        frame_keys = sorted(hdf5_file["output/u"].keys())
        if len(frame_keys) < 2 or frame_keys[0] != "00000":
            raise ValueError("Expected ThermoCorr displacement frames to start at 00000.")
        export_frame_keys = frame_keys[1:]
        if len(export_frame_keys) != force_values.shape[0]:
            raise ValueError(
                "Expected %d force values for frames 00001.., found %d."
                % (len(export_frame_keys), force_values.shape[0])
            )

        for step, (frame_key, force_value) in enumerate(zip(export_frame_keys, force_values), start=1):
            displacement_dataset = hdf5_file["output/u"][frame_key]
            u_nodes = _normalized_displacements(np.asarray(displacement_dataset[:], dtype=float), mesh)
            reaction_value = float(force_value) * reaction_scaling.factor
            step_output_dir = output_dir / str(step)
            _write_step_csvs(step_output_dir, mesh, u_nodes, reaction_value)
            load_step_entries.append(
                {
                    "load_step": int(step),
                    "frame": frame_key,
                    "path": _manifest_entry_path(step_output_dir),
                    "time": float(displacement_dataset.attrs.get("time", np.nan)),
                    "ref": int(displacement_dataset.attrs.get("ref", -1)),
                    "force_N": float(force_value),
                    "reaction_force": float(reaction_value),
                }
            )

    uniaxial_entries = export_uniaxial_curves(input_dir, output_dir)
    manifest = {
        "generator": "scripts/format_thermocorr_dataset.py",
        "source": {
            "input_dir": _manifest_entry_path(input_dir),
            "hdf5": _manifest_entry_path(hdf5_path),
            "force_csv": _manifest_entry_path(force_path),
            "documentation": list(DOC_URLS),
        },
        "coordinate_system": {
            "description": "Normalized Cartesian coordinates from ThermoCorr image pixels.",
            "x": "(x_raw - x_min) / scale",
            "y": "(y_max - y_raw) / scale",
            "ux": "ux_raw / scale",
            "uy": "-uy_raw / scale",
            "scale": float(mesh.scale),
            "x_min": float(mesh.x_min),
            "y_min": float(mesh.y_min),
            "y_max": float(mesh.y_max),
        },
        "mesh": {
            "source_region_type": "quad",
            "raw_grid_dx": float(mesh.raw_grid_dx),
            "raw_grid_dy": float(mesh.raw_grid_dy),
            "num_valid_points": int(mesh.num_valid_points),
            "num_exported_nodes": int(mesh.x_nodes.shape[0]),
            "num_dropped_points": int(mesh.num_dropped_points),
            "num_recovered_quads": int(mesh.num_quads),
            "num_triangles": int(mesh.connectivity.shape[0]),
        },
        "sampling": {
            "enabled": bool(sampling_info.enabled),
            "method": sampling_info.method,
            "requested_node_count": sampling_info.requested_node_count,
            "actual_node_count": sampling_info.actual_node_count,
            "sample_nodes": sampling_info.requested_node_count,
            "sample_seed": sampling_info.sample_seed,
            "sample_max_edge_factor": sampling_info.sample_max_edge_factor,
            "boundary_nodes_preserved": sampling_info.boundary_nodes_preserved,
            "interior_nodes_selected": sampling_info.interior_nodes_selected,
            "selected_source_node_indices": list(sampling_info.selected_source_node_indices),
            "boundary_source_node_indices": list(sampling_info.boundary_source_node_indices),
        },
        "reaction_boundary": {
            "description": "Scaled measured force attached to top vertical DOFs.",
            "bcx": "all zero",
            "bcy": "1 on max-y nodes, else 0",
        },
        "reaction_scaling": {
            "mode": reaction_scaling.mode,
            "factor": float(reaction_scaling.factor),
            "raw_force_units": "N",
            "output_reaction": "force_N * factor",
            "physical_height": reaction_scaling.physical_height,
            "thickness": reaction_scaling.thickness,
            "description": (
                "The default pixel_height mode divides force_vfm by the raw ThermoCorr "
                "y-span so the normalized-coordinate weak form has the same coefficient "
                "scale as raw pixel coordinates. Pass physical_height and thickness to "
                "obtain stress-like units."
            ),
        },
        "load_steps": load_step_entries,
        "uniaxial_curves": uniaxial_entries,
    }
    manifest_path = output_dir / "generation_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return ThermoCorrFormatResult(
        input_dir=input_dir,
        output_dir=output_dir,
        manifest_path=manifest_path,
        load_steps=tuple(entry["load_step"] for entry in load_step_entries),
        num_nodes=int(mesh.x_nodes.shape[0]),
        num_elements=int(mesh.connectivity.shape[0]),
        num_quads=int(mesh.num_quads),
        num_uniaxial_curves=len(uniaxial_entries),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Format ThermoCorr DIC data as FEM CSV load-step directories."
    )
    parser.add_argument(
        "--input-dir",
        default=str(DEFAULT_INPUT_DIR),
        help="Directory containing TPS_2.hdf5, force_vfm.csv, and e*.npz files.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where formatted FEM CSV files will be written.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the output directory if it already exists.",
    )
    parser.add_argument(
        "--sample-nodes",
        type=int,
        default=None,
        help="Sample this many DIC points once and reuse them for every load step.",
    )
    parser.add_argument(
        "--sample-method",
        choices=("boundary_coarsen", "random"),
        default="boundary_coarsen",
        help=(
            "Sampling method. boundary_coarsen preserves all full-mesh boundary and top "
            "reaction nodes; random keeps the previous exact-count random sampler."
        ),
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=0,
        help="Random seed used by --sample-method random and recorded for sampled outputs.",
    )
    parser.add_argument(
        "--sample-max-edge-factor",
        type=float,
        default=6.0,
        help=(
            "Reject sampled Delaunay triangles whose longest edge is larger than this "
            "factor times the sampled median nearest-neighbor spacing. Use <=0 to disable."
        ),
    )
    parser.add_argument(
        "--reaction-scale",
        type=float,
        default=None,
        help=(
            "Direct multiplier for force_vfm values. By default, reactions are divided "
            "by the raw ThermoCorr y-span to match the normalized-coordinate weak form."
        ),
    )
    parser.add_argument(
        "--raw-force",
        action="store_true",
        help="Write force_vfm values unchanged to output_reactions.csv.",
    )
    parser.add_argument(
        "--physical-height",
        type=float,
        default=None,
        help="Physical specimen height represented by normalized y=0..1, in the same units as thickness.",
    )
    parser.add_argument(
        "--thickness",
        type=float,
        default=None,
        help="Specimen thickness. With --physical-height, reactions are divided by height * thickness.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.raw_force and args.reaction_scale is not None:
        parser.error("--raw-force cannot be combined with --reaction-scale.")
    if args.raw_force and (args.physical_height is not None or args.thickness is not None):
        parser.error("--raw-force cannot be combined with --physical-height/--thickness.")
    reaction_scale = 1.0 if args.raw_force else args.reaction_scale
    result = format_thermocorr_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
        sample_nodes=args.sample_nodes,
        sample_seed=args.sample_seed,
        sample_max_edge_factor=args.sample_max_edge_factor,
        sample_method=args.sample_method,
        reaction_scale=reaction_scale,
        physical_height=args.physical_height,
        thickness=args.thickness,
    )
    print("[thermocorr] wrote %d load steps to %s" % (len(result.load_steps), result.output_dir))
    print(
        "[thermocorr] mesh: %d nodes, %d triangles from %d quads"
        % (result.num_nodes, result.num_elements, result.num_quads)
    )
    print("[thermocorr] uniaxial curves: %d" % result.num_uniaxial_curves)
    print("[thermocorr] manifest: %s" % result.manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
