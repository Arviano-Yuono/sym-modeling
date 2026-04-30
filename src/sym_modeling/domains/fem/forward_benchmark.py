from __future__ import annotations

import csv
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


SUPPORTED_FORWARD_BENCHMARK_MODELS = ("NH2", "NH4", "IH", "HW", "GT")

BENCHMARK_BOUNDARY_TAGS = {
    "LEFT": 1,
    "BOTTOM": 2,
    "RIGHT": 3,
    "TOP": 4,
    "HOLE": 5,
}
BENCHMARK_CELL_TAG = 11


@dataclass(frozen=True)
class ForwardFEMBenchmarkConfig:
    """Configuration for the paper-style forward FEM benchmark."""

    material_model: str = "NH2"
    output_root: str | Path = Path("dataset/fem_data/plate_hole_fenics")
    output_dir: str | Path | None = None
    load_steps: tuple[float, ...] | None = None
    input_msh_path: str | Path | None = None

    # Quarter plate with hole geometry
    outer_size: float = 1.0
    hole_radius: float = 0.1
    mesh_gdim: int = 2

    # Gmsh mesh sizing
    mesh_size_hole: float = 0.003
    mesh_size_outer: float = 0.02
    refine_dist_min: float = 0.08
    refine_dist_max: float = 0.35

    quadrature_degree: int = 4
    solver_atol: float = 1e-8
    solver_rtol: float = 1e-8
    solver_max_it: int = 50
    solver_linesearch_type: str = "bt"
    initial_load_subdivisions: int = 1
    max_load_subdivisions: int = 32

    save_debug_fields: bool = True
    use_comm_self: bool = True

    # Facet/cell tags used for BC and validation.
    left_tag: int = BENCHMARK_BOUNDARY_TAGS["LEFT"]
    bottom_tag: int = BENCHMARK_BOUNDARY_TAGS["BOTTOM"]
    right_tag: int = BENCHMARK_BOUNDARY_TAGS["RIGHT"]
    top_tag: int = BENCHMARK_BOUNDARY_TAGS["TOP"]
    hole_tag: int = BENCHMARK_BOUNDARY_TAGS["HOLE"]
    domain_tag: int = BENCHMARK_CELL_TAG

    def __post_init__(self) -> None:
        if self.material_model not in SUPPORTED_FORWARD_BENCHMARK_MODELS:
            raise ValueError(
                "Unsupported material_model: %s. Supported models: %s"
                % (self.material_model, SUPPORTED_FORWARD_BENCHMARK_MODELS)
            )
        if self.outer_size <= 0.0:
            raise ValueError("outer_size must be positive.")
        if self.hole_radius <= 0.0 or self.hole_radius >= self.outer_size:
            raise ValueError("hole_radius must be positive and strictly less than outer_size.")
        if self.mesh_gdim not in (2, 3):
            raise ValueError("mesh_gdim must be 2 or 3.")
        if self.mesh_size_hole <= 0.0 or self.mesh_size_outer <= 0.0:
            raise ValueError("mesh_size_hole and mesh_size_outer must be positive.")
        if self.refine_dist_min <= 0.0 or self.refine_dist_max <= 0.0:
            raise ValueError("refine_dist_min and refine_dist_max must be positive.")
        if self.refine_dist_min >= self.refine_dist_max:
            raise ValueError("refine_dist_min must be strictly less than refine_dist_max.")
        if self.quadrature_degree < 1:
            raise ValueError("quadrature_degree must be >= 1.")
        if self.solver_atol <= 0.0 or self.solver_rtol <= 0.0:
            raise ValueError("solver tolerances must be positive.")
        if self.solver_max_it < 1:
            raise ValueError("solver_max_it must be >= 1.")
        if not isinstance(self.solver_linesearch_type, str) or not self.solver_linesearch_type:
            raise ValueError("solver_linesearch_type must be a non-empty string.")
        if self.initial_load_subdivisions < 1:
            raise ValueError("initial_load_subdivisions must be >= 1.")
        if self.max_load_subdivisions < self.initial_load_subdivisions:
            raise ValueError(
                "max_load_subdivisions must be >= initial_load_subdivisions."
            )
        tag_values = [self.left_tag, self.bottom_tag, self.right_tag, self.top_tag, self.hole_tag]
        if any(tag <= 0 for tag in tag_values):
            raise ValueError("All boundary tags must be positive integers.")
        if len(set(tag_values)) != len(tag_values):
            raise ValueError("Boundary tags must be unique.")
        if self.domain_tag <= 0:
            raise ValueError("domain_tag must be a positive integer.")

        if self.load_steps is not None:
            if len(self.load_steps) == 0:
                raise ValueError("load_steps must contain at least one value.")
            if any(step <= 0.0 for step in self.load_steps):
                raise ValueError("load_steps must be strictly positive.")
            if tuple(sorted(self.load_steps)) != tuple(self.load_steps):
                raise ValueError("load_steps must be sorted in ascending order.")
            if len(set(self.load_steps)) != len(self.load_steps):
                raise ValueError("load_steps must not contain duplicates.")

    @property
    def resolved_load_steps(self) -> tuple[float, ...]:
        if self.load_steps is not None:
            return tuple(float(step) for step in self.load_steps)

        num_steps = 4 if self.material_model in {"NH2", "NH4"} else 8
        return tuple(0.1 * float(step) for step in range(1, num_steps + 1))

    @property
    def resolved_output_root(self) -> Path:
        if self.output_dir is not None:
            return Path(self.output_dir)
        return Path(self.output_root)

    @property
    def resolved_output_dir(self) -> Path:
        return self.resolved_output_root

    @property
    def boundary_tags(self) -> dict[str, int]:
        return {
            "LEFT": int(self.left_tag),
            "BOTTOM": int(self.bottom_tag),
            "RIGHT": int(self.right_tag),
            "TOP": int(self.top_tag),
            "HOLE": int(self.hole_tag),
        }


def _import_fenicsx():
    try:
        from mpi4py import MPI
        from petsc4py import PETSc
        import ufl
        from dolfinx import fem, io
        from dolfinx.fem import petsc as fem_petsc

        try:
            from dolfinx.nls import petsc as nls_petsc
        except ImportError:
            nls_petsc = None
    except Exception as exc:  # pragma: no cover - runtime dependent
        raise ImportError(
            "Forward benchmark requires DOLFINx/FEniCSx + gmsh Python bindings. "
            "Use the Conda environment from environment.yml."
        ) from exc

    return MPI, PETSc, ufl, fem, io, fem_petsc, nls_petsc


def _import_gmsh():
    try:
        import gmsh
    except Exception as exc:  # pragma: no cover - runtime dependent
        raise ImportError(
            "Forward benchmark mesh generation requires the gmsh Python module."
        ) from exc
    return gmsh


def _create_vector_function_space(fem, domain, degree: int):
    if hasattr(fem, "functionspace"):
        return fem.functionspace(domain, ("Lagrange", degree, (domain.geometry.dim,)))
    return fem.VectorFunctionSpace(domain, ("Lagrange", degree))


def _model_to_mesh(dolfinx_io, gmsh_model, comm, gdim: int, msh_path: Path):
    if hasattr(dolfinx_io, "gmsh") and hasattr(dolfinx_io.gmsh, "model_to_mesh"):
        mesh_data = dolfinx_io.gmsh.model_to_mesh(gmsh_model, comm, 0, gdim=gdim)
    elif hasattr(dolfinx_io, "gmshio") and hasattr(dolfinx_io.gmshio, "model_to_mesh"):
        mesh_data = dolfinx_io.gmshio.model_to_mesh(gmsh_model, comm, 0, gdim=gdim)
    elif hasattr(dolfinx_io, "gmsh") and hasattr(dolfinx_io.gmsh, "read_from_msh"):
        mesh_data = dolfinx_io.gmsh.read_from_msh(str(msh_path), comm, gdim=gdim)
    elif hasattr(dolfinx_io, "gmshio") and hasattr(dolfinx_io.gmshio, "read_from_msh"):
        mesh_data = dolfinx_io.gmshio.read_from_msh(str(msh_path), comm, gdim=gdim)
    else:  # pragma: no cover - depends on DOLFINx version
        raise AttributeError(
            "Could not find a compatible DOLFINx gmsh import API "
            "(model_to_mesh/read_from_msh)."
        )

    if hasattr(mesh_data, "mesh"):
        return mesh_data.mesh, mesh_data.cell_tags, mesh_data.facet_tags
    return mesh_data


def _read_from_msh(dolfinx_io, msh_path: Path, comm, gdim: int):
    if hasattr(dolfinx_io, "gmsh") and hasattr(dolfinx_io.gmsh, "read_from_msh"):
        mesh_data = dolfinx_io.gmsh.read_from_msh(str(msh_path), comm, gdim=gdim)
    elif hasattr(dolfinx_io, "gmshio") and hasattr(dolfinx_io.gmshio, "read_from_msh"):
        mesh_data = dolfinx_io.gmshio.read_from_msh(str(msh_path), comm, gdim=gdim)
    else:  # pragma: no cover - depends on DOLFINx version
        raise AttributeError(
            "Could not find a compatible DOLFINx gmsh import API "
            "for read_from_msh."
        )

    if hasattr(mesh_data, "mesh"):
        return mesh_data.mesh, mesh_data.cell_tags, mesh_data.facet_tags
    return mesh_data


def _build_quarter_plate_hole_mesh(dolfinx_io, comm, config: ForwardFEMBenchmarkConfig, msh_path: Path):
    gmsh = _import_gmsh()
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("Mesh.RecombineAll", 0)
        gmsh.option.setNumber("Mesh.ElementOrder", 1)

        gmsh.model.add("quarter_plate_with_hole")
        occ = gmsh.model.occ

        L = float(config.outer_size)
        R = float(config.hole_radius)

        p0 = occ.addPoint(0.0, 0.0, 0.0, config.mesh_size_hole)
        p1 = occ.addPoint(R, 0.0, 0.0, config.mesh_size_hole)
        p2 = occ.addPoint(L, 0.0, 0.0, config.mesh_size_outer)
        p3 = occ.addPoint(L, L, 0.0, config.mesh_size_outer)
        p4 = occ.addPoint(0.0, L, 0.0, config.mesh_size_outer)
        p5 = occ.addPoint(0.0, R, 0.0, config.mesh_size_hole)

        bottom = occ.addLine(p1, p2)
        right = occ.addLine(p2, p3)
        top = occ.addLine(p3, p4)
        left = occ.addLine(p4, p5)
        hole = occ.addCircleArc(p5, p0, p1)

        curve_loop = occ.addCurveLoop([bottom, right, top, left, hole])
        surface = occ.addPlaneSurface([curve_loop])
        occ.synchronize()

        gmsh.model.addPhysicalGroup(2, [surface], tag=config.domain_tag)
        gmsh.model.setPhysicalName(2, config.domain_tag, "DOMAIN")

        boundary_tags = config.boundary_tags
        gmsh.model.addPhysicalGroup(1, [left], tag=boundary_tags["LEFT"])
        gmsh.model.setPhysicalName(1, boundary_tags["LEFT"], "LEFT")
        gmsh.model.addPhysicalGroup(1, [bottom], tag=boundary_tags["BOTTOM"])
        gmsh.model.setPhysicalName(1, boundary_tags["BOTTOM"], "BOTTOM")
        gmsh.model.addPhysicalGroup(1, [right], tag=boundary_tags["RIGHT"])
        gmsh.model.setPhysicalName(1, boundary_tags["RIGHT"], "RIGHT")
        gmsh.model.addPhysicalGroup(1, [top], tag=boundary_tags["TOP"])
        gmsh.model.setPhysicalName(1, boundary_tags["TOP"], "TOP")
        gmsh.model.addPhysicalGroup(1, [hole], tag=boundary_tags["HOLE"])
        gmsh.model.setPhysicalName(1, boundary_tags["HOLE"], "HOLE")

        # Radial refinement near the hole with coarser mesh away from the hole.
        distance = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(distance, "CurvesList", [hole])
        gmsh.model.mesh.field.setNumber(distance, "Sampling", 200)
        threshold = gmsh.model.mesh.field.add("Threshold")
        gmsh.model.mesh.field.setNumber(threshold, "InField", distance)
        gmsh.model.mesh.field.setNumber(threshold, "SizeMin", config.mesh_size_hole)
        gmsh.model.mesh.field.setNumber(threshold, "SizeMax", config.mesh_size_outer)
        gmsh.model.mesh.field.setNumber(threshold, "DistMin", config.refine_dist_min)
        gmsh.model.mesh.field.setNumber(threshold, "DistMax", config.refine_dist_max)
        gmsh.model.mesh.field.setAsBackgroundMesh(threshold)

        gmsh.model.mesh.generate(2)
        msh_path.parent.mkdir(parents=True, exist_ok=True)
        gmsh.write(str(msh_path))
        return _model_to_mesh(dolfinx_io, gmsh.model, comm, gdim=config.mesh_gdim, msh_path=msh_path)
    finally:
        gmsh.finalize()


def _locate_component_dofs(fem, space, component: int, facet_dim: int, facets: np.ndarray) -> np.ndarray:
    subspace = space.sub(component)
    try:
        dofs = fem.locate_dofs_topological(subspace, facet_dim, facets)
    except TypeError:  # pragma: no cover - compatibility path
        collapsed, _ = subspace.collapse()
        dofs = fem.locate_dofs_topological((subspace, collapsed), facet_dim, facets)
    return np.asarray(dofs, dtype=np.int32)


def _component_dirichlet_bc(fem, space, component: int, dofs: np.ndarray, value: float, keepalive: list[Any]):
    subspace = space.sub(component)
    scalar_value = np.float64(value)
    try:
        return fem.dirichletbc(scalar_value, dofs, subspace)
    except TypeError:  # pragma: no cover - compatibility path
        collapsed, _ = subspace.collapse()
        value_function = fem.Function(collapsed)
        value_function.x.array[:] = scalar_value
        keepalive.append(value_function)
        try:
            return fem.dirichletbc(value_function, dofs, subspace)
        except TypeError:
            return fem.dirichletbc(value_function, dofs)


def _build_biaxial_dirichlet_bcs(fem, space, dof_sets: dict[str, np.ndarray], delta: float):
    keepalive: list[Any] = []
    bcs = [
        _component_dirichlet_bc(fem, space, 0, dof_sets["left_x"], 0.0, keepalive),
        _component_dirichlet_bc(fem, space, 1, dof_sets["bottom_y"], 0.0, keepalive),
        _component_dirichlet_bc(fem, space, 0, dof_sets["right_x"], 0.5 * delta, keepalive),
        _component_dirichlet_bc(fem, space, 1, dof_sets["top_y"], delta, keepalive),
    ]
    return bcs, keepalive


def _plane_strain_invariants(ufl, displacement):
    identity_2d = ufl.Identity(2)
    F2 = ufl.variable(identity_2d + ufl.grad(displacement))
    F3 = ufl.as_tensor(
        (
            (F2[0, 0], F2[0, 1], 0.0),
            (F2[1, 0], F2[1, 1], 0.0),
            (0.0, 0.0, 1.0),
        )
    )
    C = ufl.variable(F3.T * F3)
    I1 = ufl.variable(ufl.tr(C))
    I2 = ufl.variable(0.5 * ((ufl.tr(C) ** 2) - ufl.tr(C * C)))
    I3 = ufl.variable(ufl.det(C))
    J = ufl.variable(ufl.det(F3))
    I1_bar = ufl.variable((J ** (-2.0 / 3.0)) * I1)
    I2_bar = ufl.variable((J ** (-4.0 / 3.0)) * I2)
    return {
        "F2": F2,
        "F3": F3,
        "C": C,
        "I1": I1,
        "I2": I2,
        "I3": I3,
        "J": J,
        "I1_bar": I1_bar,
        "I2_bar": I2_bar,
    }


def _benchmark_energy_density(ufl, invariants: dict[str, Any], material_model: str):
    I1_bar = invariants["I1_bar"]
    I2_bar = invariants["I2_bar"]
    J = invariants["J"]

    if material_model == "NH2":
        return 0.5 * (I1_bar - 3.0) + 1.5 * ((J - 1.0) ** 2)
    if material_model == "NH4":
        return 0.5 * (I1_bar - 3.0) + 1.5 * ((J - 1.0) ** 4)
    if material_model == "IH":
        return (
            0.5 * (I1_bar - 3.0)
            + 1.0 * (I2_bar - 3.0)
            + 1.0 * ((I1_bar - 3.0) ** 2)
            + 1.5 * ((J - 1.0) ** 2)
        )
    if material_model == "HW":
        return (
            0.5 * (I1_bar - 3.0)
            + 1.0 * (I2_bar - 3.0)
            + 0.7 * (I1_bar - 3.0) * (I2_bar - 3.0)
            + 0.2 * ((I1_bar - 3.0) ** 3)
            + 1.5 * ((J - 1.0) ** 2)
        )
    if material_model == "GT":
        return 0.5 * (I1_bar - 3.0) + 1.5 * ((J - 1.0) ** 2) + 1.0 * ufl.ln(I2_bar / 3.0)

    raise ValueError("Unsupported material_model: %s" % material_model)


def _solve_nonlinear_step(
    domain,
    residual_form,
    jacobian_form,
    displacement,
    bcs,
    fem_petsc,
    nls_petsc,
    config: ForwardFEMBenchmarkConfig,
) -> int:
    petsc_options = {
        "snes_type": "newtonls",
        "snes_linesearch_type": config.solver_linesearch_type,
        "snes_atol": config.solver_atol,
        "snes_rtol": config.solver_rtol,
        "snes_stol": config.solver_rtol,
        "snes_max_it": config.solver_max_it,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
    }
    try:
        problem = fem_petsc.NonlinearProblem(
            residual_form,
            displacement,
            bcs=bcs,
            J=jacobian_form,
            petsc_options=petsc_options,
            petsc_options_prefix="forward_benchmark",
        )
    except TypeError:
        try:
            problem = fem_petsc.NonlinearProblem(
                residual_form,
                displacement,
                bcs=bcs,
                J=jacobian_form,
            )
        except TypeError:
            problem = fem_petsc.NonlinearProblem(
                residual_form,
                displacement,
                bcs=bcs,
            )

        if nls_petsc is None:
            raise RuntimeError(
                "This DOLFINx build needs dolfinx.nls.petsc.NewtonSolver for nonlinear solves."
            )

        solver = nls_petsc.NewtonSolver(domain.comm, problem)
        for name, value in (
            ("atol", config.solver_atol),
            ("rtol", config.solver_rtol),
            ("max_it", config.solver_max_it),
        ):
            if hasattr(solver, name):
                setattr(solver, name, value)
        if hasattr(solver, "convergence_criterion"):
            solver.convergence_criterion = "incremental"
        iterations, converged = solver.solve(displacement)
        if not converged:
            raise RuntimeError("Forward benchmark step failed to converge.")
        return int(iterations)

    problem.solve()
    solver = getattr(problem, "solver", None)
    if solver is None:
        return -1
    converged_reason = solver.getConvergedReason()
    if converged_reason <= 0:
        reason_labels = {
            -1: "DIVERGED_FUNCTION_DOMAIN",
            -2: "DIVERGED_FUNCTION_COUNT",
            -3: "DIVERGED_LINEAR_SOLVE",
            -4: "DIVERGED_FNORM_NAN",
            -5: "DIVERGED_MAX_IT",
            -6: "DIVERGED_LINE_SEARCH",
            -7: "DIVERGED_INNER",
            -8: "DIVERGED_LOCAL_MIN",
            -9: "DIVERGED_DTOL",
            -10: "DIVERGED_JACOBIAN_DOMAIN",
        }
        reason_label = reason_labels.get(int(converged_reason), "UNKNOWN_DIVERGENCE")
        raise RuntimeError(
            "Forward benchmark step failed to converge "
            "(reason=%s: %s)." % (converged_reason, reason_label)
        )
    return int(solver.getIterationNumber())


def _assemble_internal_force_vector(
    fem_petsc,
    PETSc,
    residual_form_compiled,
    displacement,
) -> np.ndarray:
    residual_vector = None
    try:
        residual_vector = fem_petsc.assemble_vector(residual_form_compiled)
    except TypeError:
        try:
            residual_vector = fem_petsc.create_vector(displacement.function_space)
        except Exception:
            if hasattr(displacement.x, "petsc_vec"):
                residual_vector = displacement.x.petsc_vec.duplicate()
            else:  # pragma: no cover - compatibility fallback
                residual_vector = fem_petsc.create_vector([displacement.function_space])

        if hasattr(residual_vector, "localForm"):
            with residual_vector.localForm() as local_form:
                local_form.set(0.0)
        else:  # pragma: no cover - compatibility fallback
            residual_vector.set(0.0)
        fem_petsc.assemble_vector(residual_vector, residual_form_compiled)

    residual_vector.ghostUpdate(
        addv=PETSc.InsertMode.ADD_VALUES,
        mode=PETSc.ScatterMode.REVERSE,
    )

    if hasattr(residual_vector, "array"):
        return np.asarray(residual_vector.array, dtype=float).copy()
    elif hasattr(residual_vector, "array_r"):
        return np.asarray(residual_vector.array_r, dtype=float).copy()
    return np.asarray(  # pragma: no cover - compatibility fallback
        residual_vector.getArray(readonly=True), dtype=float
    ).copy()


def _compute_reactions(
    dof_sets: dict[str, np.ndarray],
    internal_force_vector: np.ndarray,
):
    array = np.asarray(internal_force_vector, dtype=float)

    def constrained_sum(name: str) -> float:
        dofs = dof_sets[name]
        if dofs.size == 0:
            return 0.0
        return float(np.sum(array[dofs]))

    # Reaction on constrained dofs is minus the internal residual contribution.
    reactions = {
        "Rx_right": -constrained_sum("right_x"),
        "Ry_top": -constrained_sum("top_y"),
        "Rx_left": -constrained_sum("left_x"),
        "Ry_bottom": -constrained_sum("bottom_y"),
    }
    return reactions


def _compute_triangle_gradients(vertices: np.ndarray) -> tuple[np.ndarray, float]:
    x1, y1 = vertices[0]
    x2, y2 = vertices[1]
    x3, y3 = vertices[2]
    det_jacobian = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
    if np.isclose(det_jacobian, 0.0):
        raise ValueError("Encountered a degenerate triangle while exporting FEM data.")

    gradients = np.array(
        [
            [y2 - y3, x3 - x2],
            [y3 - y1, x1 - x3],
            [y1 - y2, x2 - x1],
        ],
        dtype=float,
    ) / det_jacobian
    area = 0.5 * abs(det_jacobian)
    return gradients, area


def _boundary_vertices_from_facets(domain, facet_dim: int, facets: np.ndarray) -> np.ndarray:
    domain.topology.create_connectivity(facet_dim, 0)
    connectivity = domain.topology.connectivity(facet_dim, 0)
    if connectivity is None:
        return np.empty((0,), dtype=np.int32)

    vertex_ids: list[np.ndarray] = []
    for facet in np.asarray(facets, dtype=np.int32):
        links = connectivity.links(int(facet))
        if links.size:
            vertex_ids.append(np.asarray(links, dtype=np.int32))
    if not vertex_ids:
        return np.empty((0,), dtype=np.int32)
    return np.unique(np.concatenate(vertex_ids)).astype(np.int32, copy=False)


def _write_step_nodes_csv(
    path: Path,
    coordinates_xy: np.ndarray,
    displacement_xy: np.ndarray,
    internal_force_xy: np.ndarray,
    bcx: np.ndarray,
    bcy: np.ndarray,
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("id", "x", "y", "ux", "uy", "fintx", "finty", "bcx", "bcy"))
        for node_id in range(coordinates_xy.shape[0]):
            writer.writerow(
                (
                    int(node_id),
                    float(coordinates_xy[node_id, 0]),
                    float(coordinates_xy[node_id, 1]),
                    float(displacement_xy[node_id, 0]),
                    float(displacement_xy[node_id, 1]),
                    float(internal_force_xy[node_id, 0]),
                    float(internal_force_xy[node_id, 1]),
                    int(bcx[node_id]),
                    int(bcy[node_id]),
                )
            )


def _write_step_elements_csv(
    path: Path,
    connectivity: np.ndarray,
    deformation_gradient: np.ndarray,
    piola: np.ndarray,
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("node1", "node2", "node3", "Fxx", "Fxy", "Fyx", "Fyy", "Pxx", "Pxy", "Pyx", "Pyy"))
        for cell in range(connectivity.shape[0]):
            writer.writerow(
                (
                    int(connectivity[cell, 0]),
                    int(connectivity[cell, 1]),
                    int(connectivity[cell, 2]),
                    float(deformation_gradient[cell, 0, 0]),
                    float(deformation_gradient[cell, 0, 1]),
                    float(deformation_gradient[cell, 1, 0]),
                    float(deformation_gradient[cell, 1, 1]),
                    float(piola[cell, 0, 0]),
                    float(piola[cell, 0, 1]),
                    float(piola[cell, 1, 0]),
                    float(piola[cell, 1, 1]),
                )
            )


def _write_step_integrator_csv(
    path: Path,
    coordinates_xy: np.ndarray,
    connectivity: np.ndarray,
    grad_na: np.ndarray,
    qp_weights: np.ndarray,
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "node1_x",
                "node1_y",
                "node2_x",
                "node2_y",
                "node3_x",
                "node3_y",
                "gradNa_node1_x",
                "gradNa_node1_y",
                "gradNa_node2_x",
                "gradNa_node2_y",
                "gradNa_node3_x",
                "gradNa_node3_y",
                "qpWeight",
            )
        )
        for cell in range(connectivity.shape[0]):
            n0 = int(connectivity[cell, 0])
            n1 = int(connectivity[cell, 1])
            n2 = int(connectivity[cell, 2])
            writer.writerow(
                (
                    float(coordinates_xy[n0, 0]),
                    float(coordinates_xy[n0, 1]),
                    float(coordinates_xy[n1, 0]),
                    float(coordinates_xy[n1, 1]),
                    float(coordinates_xy[n2, 0]),
                    float(coordinates_xy[n2, 1]),
                    float(grad_na[cell, 0, 0]),
                    float(grad_na[cell, 0, 1]),
                    float(grad_na[cell, 1, 0]),
                    float(grad_na[cell, 1, 1]),
                    float(grad_na[cell, 2, 0]),
                    float(grad_na[cell, 2, 1]),
                    float(qp_weights[cell]),
                )
            )


def _write_step_reactions_csv(path: Path, forces: np.ndarray) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("forces",))
        for value in np.asarray(forces, dtype=float).reshape(-1):
            writer.writerow((float(value),))


def _interpolation_points(element):
    points = element.interpolation_points
    return points() if callable(points) else points


def _interpolate_scalar_expression(fem, space, expression, name: str):
    field = fem.Function(space)
    field.name = name
    expr = fem.Expression(expression, _interpolation_points(space.element))
    field.interpolate(expr)
    return field


def _write_reaction_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "load_step",
        "delta",
        "load_substeps",
        "Rx_right",
        "Ry_top",
        "Rx_left",
        "Ry_bottom",
        "converged",
        "newton_iterations",
        "max_displacement",
        "J_min",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _solve_with_adaptive_substepping(
    fem,
    domain,
    residual_form,
    jacobian_form,
    displacement,
    dof_sets: dict[str, np.ndarray],
    previous_delta: float,
    target_delta: float,
    fem_petsc,
    nls_petsc,
    config: ForwardFEMBenchmarkConfig,
) -> tuple[int, int]:
    """
    Solve one load step with optional adaptive substepping for robustness.

    Returns `(total_newton_iterations, used_substeps)`.
    """

    start_state = displacement.x.array.copy()
    subdivisions = int(config.initial_load_subdivisions)
    max_subdivisions = int(config.max_load_subdivisions)
    last_error: Exception | None = None

    while subdivisions <= max_subdivisions:
        displacement.x.array[:] = start_state
        total_iterations = 0
        success = True
        last_error = None

        for sub in range(1, subdivisions + 1):
            alpha = float(sub) / float(subdivisions)
            sub_delta = previous_delta + (target_delta - previous_delta) * alpha
            bcs, _ = _build_biaxial_dirichlet_bcs(
                fem=fem,
                space=displacement.function_space,
                dof_sets=dof_sets,
                delta=float(sub_delta),
            )
            try:
                iterations = _solve_nonlinear_step(
                    domain=domain,
                    residual_form=residual_form,
                    jacobian_form=jacobian_form,
                    displacement=displacement,
                    bcs=bcs,
                    fem_petsc=fem_petsc,
                    nls_petsc=nls_petsc,
                    config=config,
                )
                total_iterations += max(int(iterations), 0)
            except RuntimeError as exc:
                success = False
                last_error = exc
                break

        if success:
            return total_iterations, subdivisions
        subdivisions *= 2

    message = (
        "Failed to converge load step delta=%s even with adaptive substepping "
        "(max_subdivisions=%s)."
    ) % (target_delta, max_subdivisions)
    if last_error is not None:
        raise RuntimeError(message + " Last error: %s" % last_error) from last_error
    raise RuntimeError(message)


def plot_forward_benchmark_loadsteps(
    results: dict[str, Any],
    load_step_indices: tuple[int, ...] | list[int] | None = None,
    warp_factor: float = 1.0,
    cmap: str = "viridis",
    max_cols: int = 4,
    show: bool = True,
):
    """
    Plot displacement magnitude for each available load step.

    This function expects the dictionary returned by `run_forward_hyperelastic_benchmark`.
    """

    if max_cols < 1:
        raise ValueError("max_cols must be >= 1.")

    load_steps = np.asarray(results.get("load_steps", []), dtype=float)
    displacements = results.get("nodal_displacements")
    coordinates = results.get("nodal_coordinates")
    triangles = results.get("triangles")

    if displacements is None or coordinates is None:
        raise ValueError(
            "results does not include nodal field arrays. "
            "Pass the direct return value from run_forward_hyperelastic_benchmark."
        )

    displacements = [np.asarray(step, dtype=float) for step in displacements]
    coordinates = np.asarray(coordinates, dtype=float)
    triangles_array = None if triangles is None else np.asarray(triangles, dtype=np.int32)

    if len(displacements) != len(load_steps):
        raise ValueError("Mismatch between number of load_steps and nodal_displacements.")

    if load_step_indices is None:
        indices = list(range(len(load_steps)))
    else:
        indices = [int(idx) for idx in load_step_indices]
    if len(indices) == 0:
        raise ValueError("No load step indices selected.")
    if min(indices) < 0 or max(indices) >= len(load_steps):
        raise IndexError("load_step_indices contains out-of-range index.")

    n_plots = len(indices)
    n_cols = min(max_cols, n_plots)
    n_rows = int(np.ceil(n_plots / n_cols))

    if show:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 4.0 * n_rows), squeeze=False)
    else:
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure

        fig = Figure(figsize=(4.5 * n_cols, 4.0 * n_rows))
        FigureCanvasAgg(fig)
        axes = np.empty((n_rows, n_cols), dtype=object)
        for row in range(n_rows):
            for col in range(n_cols):
                axes[row, col] = fig.add_subplot(n_rows, n_cols, row * n_cols + col + 1)

    for flat_index, load_index in enumerate(indices):
        row = flat_index // n_cols
        col = flat_index % n_cols
        ax = axes[row, col]
        displacement = np.asarray(displacements[load_index], dtype=float)
        displaced = coordinates.copy()
        displaced[:, : displacement.shape[1]] += float(warp_factor) * displacement
        magnitude = np.linalg.norm(displacement, axis=1)

        if triangles_array is not None and triangles_array.ndim == 2 and triangles_array.shape[1] >= 3:
            import matplotlib.tri as mtri

            tri = mtri.Triangulation(
                displaced[:, 0],
                displaced[:, 1],
                triangles=triangles_array[:, :3],
            )
            artist = ax.tripcolor(tri, magnitude, shading="gouraud", cmap=cmap)
        else:
            artist = ax.scatter(
                displaced[:, 0],
                displaced[:, 1],
                c=magnitude,
                cmap=cmap,
                s=8,
            )

        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(f"step={load_index + 1}, delta={load_steps[load_index]:.3f}")
        fig.colorbar(artist, ax=ax, label="|u|")

    for flat_index in range(n_plots, n_rows * n_cols):
        row = flat_index // n_cols
        col = flat_index % n_cols
        axes[row, col].axis("off")

    fig.tight_layout()

    if show:
        displayed = False
        try:
            from IPython import get_ipython
            from IPython.display import display

            if get_ipython() is not None:
                display(fig)
                displayed = True
        except Exception:
            displayed = False
        if not displayed:
            import matplotlib.pyplot as plt

            plt.show()

    return fig, axes


def run_forward_hyperelastic_benchmark(
    config: ForwardFEMBenchmarkConfig | None = None,
) -> dict[str, Any]:
    """
    Run the paper-style forward FEM benchmark with DOLFINx + Gmsh.

    Returns a dictionary with mesh stats, load-step data, reactions, checks, and file paths.
    """

    config = config or ForwardFEMBenchmarkConfig()
    output_root = config.resolved_output_dir
    output_root.mkdir(parents=True, exist_ok=True)

    mesh_msh_path = output_root / "quarter_plate_hole.msh"
    displacement_xdmf_path = output_root / "displacement.xdmf"
    debug_xdmf_path = output_root / "debug_fields.xdmf"
    reactions_csv_path = output_root / "reactions.csv"
    summary_json_path = output_root / "summary.json"

    MPI, PETSc, ufl, fem, io, fem_petsc, nls_petsc = _import_fenicsx()
    comm = MPI.COMM_SELF if config.use_comm_self else MPI.COMM_WORLD
    if comm.size != 1:
        raise NotImplementedError(
            "run_forward_hyperelastic_benchmark currently supports single-rank runs."
        )
    boundary_tags = config.boundary_tags

    if config.input_msh_path is None:
        mesh_source = "generated"
        domain, cell_tags, facet_tags = _build_quarter_plate_hole_mesh(
            dolfinx_io=io,
            comm=comm,
            config=config,
            msh_path=mesh_msh_path,
        )
    else:
        mesh_source = "provided"
        input_msh_path = Path(config.input_msh_path)
        if not input_msh_path.exists():
            raise FileNotFoundError(input_msh_path)
        domain, cell_tags, facet_tags = _read_from_msh(
            dolfinx_io=io,
            msh_path=input_msh_path,
            comm=comm,
            gdim=config.mesh_gdim,
        )
        if input_msh_path.resolve() != mesh_msh_path.resolve():
            mesh_msh_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(input_msh_path, mesh_msh_path)

    if facet_tags is None:
        raise ValueError("Facet tags are missing from imported mesh.")
    if cell_tags is None:
        raise ValueError("Cell tags are missing from imported mesh.")

    required_facet_tags = set(boundary_tags.values())
    available_facet_tags = set(np.unique(np.asarray(facet_tags.values, dtype=np.int64)).tolist())
    if not required_facet_tags.issubset(available_facet_tags):
        raise ValueError(
            "Mesh does not contain all required boundary facet tags. "
            "required=%s available=%s"
            % (sorted(required_facet_tags), sorted(available_facet_tags))
        )
    available_cell_tags = set(np.unique(np.asarray(cell_tags.values, dtype=np.int64)).tolist())
    if config.domain_tag not in available_cell_tags:
        raise ValueError(
            "Mesh does not contain the required domain cell tag %s. available=%s"
            % (config.domain_tag, sorted(available_cell_tags))
        )

    displacement_space = _create_vector_function_space(fem, domain, degree=1)
    displacement = fem.Function(displacement_space)
    displacement.name = "u"

    facet_dim = domain.topology.dim - 1
    left_facets = facet_tags.find(boundary_tags["LEFT"])
    bottom_facets = facet_tags.find(boundary_tags["BOTTOM"])
    right_facets = facet_tags.find(boundary_tags["RIGHT"])
    top_facets = facet_tags.find(boundary_tags["TOP"])

    dof_sets = {
        "left_x": _locate_component_dofs(fem, displacement_space, 0, facet_dim, left_facets),
        "bottom_y": _locate_component_dofs(fem, displacement_space, 1, facet_dim, bottom_facets),
        "right_x": _locate_component_dofs(fem, displacement_space, 0, facet_dim, right_facets),
        "top_y": _locate_component_dofs(fem, displacement_space, 1, facet_dim, top_facets),
    }
    for name, dofs in dof_sets.items():
        if dofs.size == 0:
            raise ValueError("No constrained dofs found for boundary component %s." % name)

    tdim = domain.topology.dim
    num_cells = int(domain.topology.index_map(tdim).size_local)
    num_nodes = int(domain.geometry.x.shape[0])
    nodal_coordinates = np.asarray(domain.geometry.x, dtype=float).copy()
    coordinates_xy = nodal_coordinates[:, :2].copy()

    triangles = np.asarray(domain.geometry.dofmap[:num_cells], dtype=np.int32)
    if triangles.ndim != 2 or triangles.shape[1] < 3:
        raise RuntimeError("Expected linear triangular geometry dofmap.")
    triangles = triangles[:, :3]
    connectivity = triangles.copy()

    grad_na = np.zeros((num_cells, 3, 2), dtype=float)
    qp_weights = np.zeros(num_cells, dtype=float)
    for cell in range(num_cells):
        grad_na[cell], qp_weights[cell] = _compute_triangle_gradients(coordinates_xy[connectivity[cell]])

    left_nodes = _boundary_vertices_from_facets(domain, facet_dim, left_facets)
    right_nodes = _boundary_vertices_from_facets(domain, facet_dim, right_facets)
    bottom_nodes = _boundary_vertices_from_facets(domain, facet_dim, bottom_facets)
    top_nodes = _boundary_vertices_from_facets(domain, facet_dim, top_facets)

    bcx = np.zeros(num_nodes, dtype=np.int32)
    bcy = np.zeros(num_nodes, dtype=np.int32)
    bcx[left_nodes] = 1
    bcx[right_nodes] = 2
    bcy[bottom_nodes] = 3
    bcy[top_nodes] = 4

    test_function = ufl.TestFunction(displacement_space)
    trial_increment = ufl.TrialFunction(displacement_space)
    invariants = _plane_strain_invariants(ufl, displacement)
    strain_energy_density = _benchmark_energy_density(
        ufl=ufl,
        invariants=invariants,
        material_model=config.material_model,
    )
    piola_expr = ufl.diff(strain_energy_density, invariants["F2"])
    metadata = {"quadrature_degree": config.quadrature_degree}
    dx = ufl.Measure("dx", domain=domain, metadata=metadata)
    potential = strain_energy_density * dx
    residual_form = ufl.derivative(potential, displacement, test_function)
    jacobian_form = ufl.derivative(residual_form, displacement, trial_increment)
    residual_form_compiled = fem.form(residual_form)

    if hasattr(fem, "functionspace"):
        scalar_space = fem.functionspace(domain, ("DG", 0))
    else:
        scalar_space = fem.FunctionSpace(domain, ("DG", 0))

    load_steps = config.resolved_load_steps
    rows: list[dict[str, Any]] = []
    nodal_displacements: list[np.ndarray] = []
    displacement_max_values: list[float] = []
    j_min_values: list[float] = []
    newton_iterations: list[int] = []
    load_substeps_used: list[int] = []
    converged_steps: list[bool] = []
    step_directories: list[str] = []
    bc_force_rows: list[list[float]] = []
    step_file_rows: list[dict[str, str]] = []
    used_step_dir_names: set[str] = set()

    displacement_writer = io.XDMFFile(domain.comm, str(displacement_xdmf_path), "w")
    displacement_writer.write_mesh(domain)
    debug_writer = io.XDMFFile(domain.comm, str(debug_xdmf_path), "w") if config.save_debug_fields else None
    if debug_writer is not None:
        debug_writer.write_mesh(domain)

    try:
        previous_delta = 0.0
        for step_index, delta in enumerate(load_steps, start=1):
            iterations, used_substeps = _solve_with_adaptive_substepping(
                fem=fem,
                domain=domain,
                residual_form=residual_form,
                jacobian_form=jacobian_form,
                displacement=displacement,
                dof_sets=dof_sets,
                previous_delta=previous_delta,
                target_delta=float(delta),
                fem_petsc=fem_petsc,
                nls_petsc=nls_petsc,
                config=config,
            )
            previous_delta = float(delta)

            displacement_array = displacement.x.array.reshape(num_nodes, domain.geometry.dim).copy()
            displacement_xy = displacement_array[:, :2].copy()
            displacement_norm = np.linalg.norm(displacement_xy, axis=1)
            max_displacement = float(np.max(displacement_norm)) if displacement_norm.size else 0.0

            internal_force_vector = _assemble_internal_force_vector(
                fem_petsc=fem_petsc,
                PETSc=PETSc,
                residual_form_compiled=residual_form_compiled,
                displacement=displacement,
            )
            expected_internal_size = num_nodes * domain.geometry.dim
            if internal_force_vector.size != expected_internal_size:
                raise RuntimeError(
                    "Unexpected internal force vector layout. "
                    "Expected %s entries but got %s."
                    % (expected_internal_size, internal_force_vector.size)
                )
            internal_force_xy = internal_force_vector.reshape(num_nodes, domain.geometry.dim)[:, :2].copy()
            reactions = _compute_reactions(
                dof_sets=dof_sets,
                internal_force_vector=internal_force_vector,
            )

            bc_forces = np.array(
                [
                    float(np.sum(internal_force_xy[bcx == 1, 0])),
                    float(np.sum(internal_force_xy[bcx == 2, 0])),
                    float(np.sum(internal_force_xy[bcy == 3, 1])),
                    float(np.sum(internal_force_xy[bcy == 4, 1])),
                ],
                dtype=float,
            )

            j_field = _interpolate_scalar_expression(fem, scalar_space, invariants["J"], "J")
            j_min = float(np.min(j_field.x.array))

            pxx = _interpolate_scalar_expression(fem, scalar_space, piola_expr[0, 0], "Pxx").x.array.copy()
            pxy = _interpolate_scalar_expression(fem, scalar_space, piola_expr[0, 1], "Pxy").x.array.copy()
            pyx = _interpolate_scalar_expression(fem, scalar_space, piola_expr[1, 0], "Pyx").x.array.copy()
            pyy = _interpolate_scalar_expression(fem, scalar_space, piola_expr[1, 1], "Pyy").x.array.copy()
            piola = np.zeros((num_cells, 2, 2), dtype=float)
            piola[:, 0, 0] = pxx
            piola[:, 0, 1] = pxy
            piola[:, 1, 0] = pyx
            piola[:, 1, 1] = pyy

            u_local = displacement_xy[connectivity, :]
            grad_u = np.einsum("cai,caj->cij", u_local, grad_na)
            deformation_gradient = grad_u.copy()
            deformation_gradient[:, 0, 0] += 1.0
            deformation_gradient[:, 1, 1] += 1.0

            step_dir_name = str(int(round(float(delta) * 100.0)))
            if step_dir_name in used_step_dir_names:
                raise ValueError(
                    "Two load steps map to the same folder name '%s'. "
                    "Adjust load_steps to avoid duplicates."
                    % step_dir_name
                )
            used_step_dir_names.add(step_dir_name)
            step_dir = output_root / step_dir_name
            step_dir.mkdir(parents=True, exist_ok=True)

            nodes_csv_path = step_dir / "output_nodes.csv"
            elements_csv_path = step_dir / "output_elements.csv"
            integrator_csv_path = step_dir / "output_integrator.csv"
            reactions_step_csv_path = step_dir / "output_reactions.csv"

            _write_step_nodes_csv(
                nodes_csv_path,
                coordinates_xy=coordinates_xy,
                displacement_xy=displacement_xy,
                internal_force_xy=internal_force_xy,
                bcx=bcx,
                bcy=bcy,
            )
            _write_step_elements_csv(
                elements_csv_path,
                connectivity=connectivity,
                deformation_gradient=deformation_gradient,
                piola=piola,
            )
            _write_step_integrator_csv(
                integrator_csv_path,
                coordinates_xy=coordinates_xy,
                connectivity=connectivity,
                grad_na=grad_na,
                qp_weights=qp_weights,
            )
            _write_step_reactions_csv(reactions_step_csv_path, bc_forces)

            displacement_writer.write_function(displacement, float(delta))
            if debug_writer is not None:
                i1_bar_field = _interpolate_scalar_expression(
                    fem, scalar_space, invariants["I1_bar"], "I1_bar"
                )
                i2_bar_field = _interpolate_scalar_expression(
                    fem, scalar_space, invariants["I2_bar"], "I2_bar"
                )
                w_field = _interpolate_scalar_expression(
                    fem, scalar_space, strain_energy_density, "W"
                )
                i3_field = _interpolate_scalar_expression(
                    fem, scalar_space, invariants["I3"], "I3"
                )
                debug_writer.write_function(j_field, float(delta))
                debug_writer.write_function(i1_bar_field, float(delta))
                debug_writer.write_function(i2_bar_field, float(delta))
                debug_writer.write_function(i3_field, float(delta))
                debug_writer.write_function(w_field, float(delta))

            row = {
                "load_step": step_index,
                "delta": float(delta),
                "load_substeps": int(used_substeps),
                "Rx_right": reactions["Rx_right"],
                "Ry_top": reactions["Ry_top"],
                "Rx_left": reactions["Rx_left"],
                "Ry_bottom": reactions["Ry_bottom"],
                "converged": True,
                "newton_iterations": int(iterations),
                "max_displacement": max_displacement,
                "J_min": j_min,
            }
            rows.append(row)
            nodal_displacements.append(displacement_xy)
            displacement_max_values.append(max_displacement)
            j_min_values.append(j_min)
            newton_iterations.append(int(iterations))
            load_substeps_used.append(int(used_substeps))
            converged_steps.append(True)
            step_directories.append(str(step_dir))
            bc_force_rows.append(bc_forces.tolist())
            step_file_rows.append(
                {
                    "nodes_csv": str(nodes_csv_path),
                    "elements_csv": str(elements_csv_path),
                    "integrator_csv": str(integrator_csv_path),
                    "reactions_csv": str(reactions_step_csv_path),
                }
            )
    finally:
        displacement_writer.close()
        if debug_writer is not None:
            debug_writer.close()

    _write_reaction_csv(reactions_csv_path, rows)

    right_reactions = [row["Rx_right"] for row in rows]
    top_reactions = [row["Ry_top"] for row in rows]
    left_reactions = [row["Rx_left"] for row in rows]
    bottom_reactions = [row["Ry_bottom"] for row in rows]

    displacement_monotonic = bool(
        np.all(np.diff(np.asarray(displacement_max_values, dtype=float)) >= -1e-10)
    )
    reaction_magnitude = np.sqrt(
        np.square(np.asarray(right_reactions, dtype=float))
        + np.square(np.asarray(top_reactions, dtype=float))
    )
    reaction_monotonic = bool(np.all(np.diff(reaction_magnitude) >= -1e-10))
    jacobian_positive = bool(np.all(np.asarray(j_min_values, dtype=float) > 0.0))

    config_payload = asdict(config)
    if isinstance(config_payload.get("output_root"), Path):
        config_payload["output_root"] = str(config_payload["output_root"])
    if isinstance(config_payload.get("output_dir"), Path):
        config_payload["output_dir"] = str(config_payload["output_dir"])
    if isinstance(config_payload.get("input_msh_path"), Path):
        config_payload["input_msh_path"] = str(config_payload["input_msh_path"])

    results = {
        "config": config_payload,
        "mesh_info": {
            "num_nodes": num_nodes,
            "num_cells": num_cells,
            "cell_type": "triangle",
            "gdim": int(domain.geometry.dim),
            "cell_tag": int(config.domain_tag),
            "boundary_tags": boundary_tags.copy(),
            "mesh_msh_path": str(mesh_msh_path),
            "mesh_source": mesh_source,
        },
        "load_steps": [float(delta) for delta in load_steps],
        "reactions": {
            "Rx_right": right_reactions,
            "Ry_top": top_reactions,
            "Rx_left": left_reactions,
            "Ry_bottom": bottom_reactions,
        },
        "bc_forces": {
            "left_x_id1": [row[0] for row in bc_force_rows],
            "right_x_id2": [row[1] for row in bc_force_rows],
            "bottom_y_id3": [row[2] for row in bc_force_rows],
            "top_y_id4": [row[3] for row in bc_force_rows],
        },
        "newton_iterations": newton_iterations,
        "load_substeps": load_substeps_used,
        "converged": converged_steps,
        "max_displacement": displacement_max_values,
        "jacobian_min": j_min_values,
        "checks": {
            "mesh_has_all_boundary_tags": True,
            "solver_converged_all_steps": bool(all(converged_steps)),
            "displacement_monotonic_increasing": displacement_monotonic,
            "reaction_magnitude_monotonic_increasing": reaction_monotonic,
            "jacobian_positive_everywhere": jacobian_positive,
        },
        "files": {
            "output_dir": str(output_root),
            "step_directories": step_directories,
            "step_files": step_file_rows,
            "displacement_xdmf": str(displacement_xdmf_path),
            "debug_xdmf": str(debug_xdmf_path) if config.save_debug_fields else None,
            "reactions_csv": str(reactions_csv_path),
            "summary_json": str(summary_json_path),
            "mesh_msh": str(mesh_msh_path),
        },
        "rows": rows,
        "nodal_displacements": nodal_displacements,
        "nodal_coordinates": nodal_coordinates,
        "triangles": triangles,
    }

    with summary_json_path.open("w", encoding="utf-8") as handle:
        serializable = {
            key: value
            for key, value in results.items()
            if key not in {"nodal_displacements", "nodal_coordinates", "triangles"}
        }
        json.dump(serializable, handle, indent=2)

    return results


__all__ = [
    "BENCHMARK_BOUNDARY_TAGS",
    "BENCHMARK_CELL_TAG",
    "ForwardFEMBenchmarkConfig",
    "SUPPORTED_FORWARD_BENCHMARK_MODELS",
    "plot_forward_benchmark_loadsteps",
    "run_forward_hyperelastic_benchmark",
]
