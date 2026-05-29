from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

SUPPORTED_BOX_BOUNDARIES = ("left", "right", "bottom", "top", "front", "back")
SUPPORTED_HYPERELASTIC_MODELS = (
    "neo_hookean_log",
    "st_venant_kirchhoff",
    "mooney_rivlin",
)


@dataclass(frozen=True)
class DolfinxHyperelasticityConfig:
    """Minimal DOLFINx hyperelastic setup."""

    # Structured box mesh mode
    lower_corner: tuple[float, float, float] = (0.0, 0.0, 0.0)
    upper_corner: tuple[float, float, float] = (20.0, 1.0, 1.0)
    cells: tuple[int, int, int] = (20, 5, 5)
    fixed_boundary: str | None = "left"
    traction_boundary: str | None = "right"

    # Gmsh .msh mode
    msh_path: str | None = None
    msh_gdim: int = 2
    use_comm_self: bool = False
    fixed_boundary_tag: int | None = None
    traction_boundary_tag: int | None = None
    fixed_boundary_name: str | None = None
    traction_boundary_name: str | None = None

    element_degree: int = 2
    quadrature_degree: int = 4
    material_model: str = "neo_hookean_log"
    young_modulus: float = 1.0e4
    poisson_ratio: float = 0.3
    mooney_c10: float | None = None
    mooney_c01: float | None = None
    bulk_modulus: float | None = None
    body_force: tuple[float, ...] = (0.0, 0.0, 0.0)
    traction: tuple[float, ...] = (0.0, 0.0, 0.0)
    boundary_tol: float = 1e-8
    solver_atol: float = 1e-8
    solver_rtol: float = 1e-8
    solver_max_it: int = 50

    def __post_init__(self) -> None:
        if self.element_degree < 1:
            raise ValueError("element_degree must be >= 1.")
        if self.quadrature_degree < 1:
            raise ValueError("quadrature_degree must be >= 1.")
        if self.material_model not in SUPPORTED_HYPERELASTIC_MODELS:
            raise ValueError(
                "Unsupported material_model: %s. Supported models: %s"
                % (self.material_model, SUPPORTED_HYPERELASTIC_MODELS)
            )
        if self.young_modulus <= 0.0:
            raise ValueError("young_modulus must be positive.")
        if not (-1.0 < self.poisson_ratio < 0.5):
            raise ValueError("poisson_ratio must lie in (-1, 0.5).")
        if self.mooney_c10 is not None and self.mooney_c10 < 0.0:
            raise ValueError("mooney_c10 must be >= 0.")
        if self.mooney_c01 is not None and self.mooney_c01 < 0.0:
            raise ValueError("mooney_c01 must be >= 0.")
        if self.bulk_modulus is not None and self.bulk_modulus <= 0.0:
            raise ValueError("bulk_modulus must be positive.")
        if self.boundary_tol <= 0.0:
            raise ValueError("boundary_tol must be positive.")
        if self.solver_atol <= 0.0 or self.solver_rtol <= 0.0:
            raise ValueError("solver tolerances must be positive.")
        if self.solver_max_it < 1:
            raise ValueError("solver_max_it must be >= 1.")
        if len(self.body_force) not in (2, 3):
            raise ValueError("body_force must have 2 or 3 components.")
        if len(self.traction) not in (2, 3):
            raise ValueError("traction must have 2 or 3 components.")

        if self.msh_path is None:
            if len(self.lower_corner) != 3 or len(self.upper_corner) != 3:
                raise ValueError("lower_corner and upper_corner must be 3D points.")
            if len(self.cells) != 3:
                raise ValueError("cells must contain exactly three entries.")
            if any(cells < 1 for cells in self.cells):
                raise ValueError("cells entries must be positive integers.")
            if any(upper <= lower for lower, upper in zip(self.lower_corner, self.upper_corner)):
                raise ValueError("upper_corner must be larger than lower_corner in every direction.")
            if self.fixed_boundary not in SUPPORTED_BOX_BOUNDARIES:
                raise ValueError("Unsupported fixed_boundary: %s" % self.fixed_boundary)
            if self.traction_boundary not in SUPPORTED_BOX_BOUNDARIES:
                raise ValueError("Unsupported traction_boundary: %s" % self.traction_boundary)
            if self.fixed_boundary == self.traction_boundary:
                raise ValueError("fixed_boundary and traction_boundary must be different.")
            return

        if self.msh_gdim not in (2, 3):
            raise ValueError("msh_gdim must be 2 or 3.")

        fixed_specified = self.fixed_boundary_tag is not None or self.fixed_boundary_name is not None
        traction_specified = (
            self.traction_boundary_tag is not None or self.traction_boundary_name is not None
        )
        if not fixed_specified:
            raise ValueError(
                "For msh mode, set fixed_boundary_tag or fixed_boundary_name."
            )
        if not traction_specified:
            raise ValueError(
                "For msh mode, set traction_boundary_tag or traction_boundary_name."
            )
        if (
            self.fixed_boundary_tag is not None
            and self.traction_boundary_tag is not None
            and self.fixed_boundary_tag == self.traction_boundary_tag
        ):
            raise ValueError("fixed_boundary_tag and traction_boundary_tag must be different.")
        if (
            self.fixed_boundary_name is not None
            and self.traction_boundary_name is not None
            and self.fixed_boundary_name == self.traction_boundary_name
        ):
            raise ValueError("fixed_boundary_name and traction_boundary_name must be different.")


@dataclass(frozen=True)
class DolfinxSolveResult:
    traction: np.ndarray
    iterations: int
    displacement_norm: float


@dataclass(frozen=True)
class DolfinxSimulationData:
    points: np.ndarray
    topology: np.ndarray
    cell_types: np.ndarray
    displacement: np.ndarray
    displacement_magnitude: np.ndarray
    traction: np.ndarray
    iterations: int | None = None


def plot_simulation_data(
    data: DolfinxSimulationData,
    warp_factor: float = 1.0,
    cmap: str = "viridis",
    title: str | None = None,
    show: bool = True,
    backend: str = "auto",
) -> Any:
    """
    Plot displacement results from `DolfinxSimulationData`.

    `backend` can be:
    - "auto": use pyvista when available, otherwise matplotlib
    - "pyvista": force pyvista
    - "matplotlib": force matplotlib
    """

    if backend not in {"auto", "pyvista", "matplotlib"}:
        raise ValueError("backend must be one of: auto, pyvista, matplotlib")

    use_pyvista = backend in {"auto", "pyvista"}
    if use_pyvista:
        try:
            import pyvista
        except ImportError:
            if backend == "pyvista":
                raise
            pyvista = None

        if pyvista is not None:
            grid = pyvista.UnstructuredGrid(
                np.asarray(data.topology),
                np.asarray(data.cell_types),
                np.asarray(data.points, dtype=float),
            )
            vectors = np.zeros((data.points.shape[0], 3), dtype=float)
            dim = data.displacement.shape[1]
            vectors[:, :dim] = data.displacement
            grid["u"] = vectors
            grid["|u|"] = np.asarray(data.displacement_magnitude, dtype=float)

            warped = grid.warp_by_vector("u", factor=float(warp_factor))
            plotter = pyvista.Plotter()
            plotter.add_mesh(
                warped,
                scalars="|u|",
                cmap=cmap,
                show_edges=True,
            )
            if title:
                plotter.add_title(title)
            if show:
                plotter.show()
            return plotter, warped

    points = np.asarray(data.points, dtype=float)
    displacement = np.asarray(data.displacement, dtype=float)
    displaced = points.copy()
    displaced[:, : displacement.shape[1]] += float(warp_factor) * displacement

    if show:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 5))
    else:
        # Keep plotting headless without mutating the global matplotlib backend.
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure

        fig = Figure(figsize=(7, 5))
        FigureCanvasAgg(fig)
        ax = fig.add_subplot(111)

    scatter = ax.scatter(
        displaced[:, 0],
        displaced[:, 1] if displaced.shape[1] > 1 else np.zeros(displaced.shape[0]),
        c=np.asarray(data.displacement_magnitude, dtype=float),
        s=10,
        cmap=cmap,
    )
    fig.colorbar(scatter, ax=ax, label="|u|")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title or "Displacement magnitude")
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
    return fig, ax


def _import_fenicsx():
    try:
        from mpi4py import MPI
        import ufl
        from dolfinx import fem, io, mesh
        from dolfinx.fem import petsc as fem_petsc

        try:
            from dolfinx.nls import petsc as nls_petsc
        except ImportError:
            nls_petsc = None
    except Exception as exc:  # pragma: no cover - depends on runtime environment
        raise ImportError(
            "DolfinxHyperelasticitySimulation requires a working DOLFINx/FEniCSx environment. "
            "Use the Docker setup from docker-compose.yml or the Conda environment from environment.yml."
        ) from exc

    return MPI, ufl, fem, io, mesh, fem_petsc, nls_petsc


def _create_vector_function_space(fem, domain, degree: int):
    if hasattr(fem, "functionspace"):
        return fem.functionspace(domain, ("Lagrange", degree, (domain.geometry.dim,)))
    return fem.VectorFunctionSpace(domain, ("Lagrange", degree))


def _build_solver(
    domain,
    residual,
    u,
    bcs,
    fem_petsc,
    nls_petsc,
    petsc_options,
    config,
):
    try:
        problem = fem_petsc.NonlinearProblem(
            residual,
            u,
            bcs=bcs,
            petsc_options=petsc_options,
            petsc_options_prefix="hyperelasticity",
        )
    except TypeError:
        problem = fem_petsc.NonlinearProblem(residual, u, bcs=bcs)
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

        krylov_solver = getattr(solver, "krylov_solver", None)
        if krylov_solver is not None:
            from petsc4py import PETSc

            prefix = krylov_solver.getOptionsPrefix()
            petsc_db = PETSc.Options()
            for key, value in petsc_options.items():
                if key.startswith(("ksp_", "pc_")):
                    petsc_db[f"{prefix}{key}"] = value
            krylov_solver.setFromOptions()

        def solve_once() -> int:
            iterations, converged = solver.solve(u)
            if not converged:
                raise RuntimeError("Hyperelastic solve failed to converge.")
            return int(iterations)

        return solve_once

    def solve_once() -> int:
        problem.solve()
        solver = getattr(problem, "solver", None)
        if solver is None:
            return -1
        if solver.getConvergedReason() <= 0:
            raise RuntimeError(
                "Hyperelastic solve failed to converge "
                f"(reason={solver.getConvergedReason()})."
            )
        return int(solver.getIterationNumber())

    return solve_once


def _lame_parameters(config: DolfinxHyperelasticityConfig) -> tuple[float, float]:
    young_modulus = float(config.young_modulus)
    poisson_ratio = float(config.poisson_ratio)
    mu = young_modulus / (2.0 * (1.0 + poisson_ratio))
    lmbda = young_modulus * poisson_ratio / (
        (1.0 + poisson_ratio) * (1.0 - 2.0 * poisson_ratio)
    )
    return mu, lmbda


def _bulk_modulus(config: DolfinxHyperelasticityConfig) -> float:
    if config.bulk_modulus is not None:
        return float(config.bulk_modulus)
    young_modulus = float(config.young_modulus)
    poisson_ratio = float(config.poisson_ratio)
    return young_modulus / (3.0 * (1.0 - 2.0 * poisson_ratio))


def _build_strain_energy_density(ufl, fem, domain, deformation_gradient, config):
    cauchy_green = ufl.variable(deformation_gradient.T * deformation_gradient)
    jacobian = ufl.variable(ufl.det(deformation_gradient))
    mu_value, lmbda_value = _lame_parameters(config)

    if config.material_model == "neo_hookean_log":
        first_invariant = ufl.variable(ufl.tr(cauchy_green))
        mu = fem.Constant(domain, np.float64(mu_value))
        lmbda = fem.Constant(domain, np.float64(lmbda_value))
        return (
            (mu / 2.0) * (first_invariant - 3.0)
            - mu * ufl.ln(jacobian)
            + (lmbda / 2.0) * (ufl.ln(jacobian) ** 2)
        )

    if config.material_model == "st_venant_kirchhoff":
        identity = ufl.Identity(domain.geometry.dim)
        green_lagrange = 0.5 * (cauchy_green - identity)
        mu = fem.Constant(domain, np.float64(mu_value))
        lmbda = fem.Constant(domain, np.float64(lmbda_value))
        return mu * ufl.inner(green_lagrange, green_lagrange) + (lmbda / 2.0) * (
            ufl.tr(green_lagrange) ** 2
        )

    if config.material_model == "mooney_rivlin":
        first_invariant = ufl.variable(ufl.tr(cauchy_green))
        second_invariant = ufl.variable(
            0.5 * (first_invariant**2 - ufl.tr(cauchy_green * cauchy_green))
        )
        c10_value = float(config.mooney_c10) if config.mooney_c10 is not None else (mu_value / 2.0)
        c01_value = float(config.mooney_c01) if config.mooney_c01 is not None else 0.0
        bulk_value = _bulk_modulus(config)

        c10 = fem.Constant(domain, np.float64(c10_value))
        c01 = fem.Constant(domain, np.float64(c01_value))
        bulk_modulus = fem.Constant(domain, np.float64(bulk_value))

        i1_bar = (jacobian ** (-2.0 / 3.0)) * first_invariant
        i2_bar = (jacobian ** (-4.0 / 3.0)) * second_invariant
        return c10 * (i1_bar - 3.0) + c01 * (i2_bar - 3.0) + (bulk_modulus / 2.0) * (
            jacobian - 1.0
        ) ** 2

    raise ValueError("Unsupported material_model: %s" % config.material_model)


def _box_boundary_locator(boundary_name: str, config: DolfinxHyperelasticityConfig):
    lower = np.asarray(config.lower_corner, dtype=float)
    upper = np.asarray(config.upper_corner, dtype=float)
    tol = config.boundary_tol

    if boundary_name == "left":
        return lambda x: np.isclose(x[0], lower[0], atol=tol)
    if boundary_name == "right":
        return lambda x: np.isclose(x[0], upper[0], atol=tol)
    if boundary_name == "bottom":
        return lambda x: np.isclose(x[1], lower[1], atol=tol)
    if boundary_name == "top":
        return lambda x: np.isclose(x[1], upper[1], atol=tol)
    if boundary_name == "front":
        return lambda x: np.isclose(x[2], lower[2], atol=tol)
    if boundary_name == "back":
        return lambda x: np.isclose(x[2], upper[2], atol=tol)
    raise ValueError("Unsupported boundary name: %s" % boundary_name)


def _validate_vector(
    vector: tuple[float, ...] | list[float] | np.ndarray,
    dimension: int,
) -> np.ndarray:
    array = np.asarray(vector, dtype=np.float64)
    if array.shape == (dimension,):
        return array
    if dimension == 2 and array.shape == (3,):
        if not np.isclose(array[2], 0.0):
            raise ValueError("For 2D meshes, 3rd component must be zero.")
        return array[:2]
    raise ValueError(
        "Expected a vector of shape (%d,), got shape %s." % (dimension, array.shape)
    )


def _load_msh_mesh(dolfinx_io, msh_path: str, comm, gdim: int):
    if hasattr(dolfinx_io, "gmsh") and hasattr(dolfinx_io.gmsh, "read_from_msh"):
        mesh_data = dolfinx_io.gmsh.read_from_msh(msh_path, comm, gdim=gdim)
    elif hasattr(dolfinx_io, "gmshio") and hasattr(dolfinx_io.gmshio, "read_from_msh"):
        mesh_data = dolfinx_io.gmshio.read_from_msh(msh_path, comm, gdim=gdim)
    else:  # pragma: no cover - depends on dolfinx version
        raise AttributeError("Could not find dolfinx.io.gmsh.read_from_msh in this DOLFINx build.")

    if hasattr(mesh_data, "mesh"):
        return mesh_data.mesh, mesh_data.cell_tags, mesh_data.facet_tags
    return mesh_data


def read_msh_physical_names(path: str | Path) -> dict[tuple[int, int], str]:
    """
    Parse `$PhysicalNames` from a Gmsh `.msh` file.

    Returns: `{(dimension, tag): name}`.
    """

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    entries: dict[tuple[int, int], str] = {}
    inside_section = False
    num_entries: int | None = None
    parsed_entries = 0

    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped == "$PhysicalNames":
            inside_section = True
            num_entries = None
            parsed_entries = 0
            continue
        if not inside_section:
            continue
        if stripped == "$EndPhysicalNames":
            break
        if num_entries is None:
            num_entries = int(stripped)
            continue
        parts = stripped.split(maxsplit=2)
        if len(parts) < 3:
            continue
        dimension = int(parts[0])
        tag = int(parts[1])
        name = parts[2].strip().strip('"')
        entries[(dimension, tag)] = name
        parsed_entries += 1
        if num_entries is not None and parsed_entries >= num_entries:
            break

    return entries


def _resolve_boundary_tag(
    config: DolfinxHyperelasticityConfig,
    facet_dim: int,
    physical_names: dict[tuple[int, int], str],
    boundary_kind: str,
) -> int:
    tag_attr = f"{boundary_kind}_boundary_tag"
    name_attr = f"{boundary_kind}_boundary_name"
    marker = getattr(config, tag_attr)
    name = getattr(config, name_attr)

    if marker is not None:
        return int(marker)

    assert name is not None
    candidates = {
        tag: physical_name
        for (dimension, tag), physical_name in physical_names.items()
        if dimension == facet_dim
    }
    for tag, physical_name in candidates.items():
        if physical_name == name:
            return int(tag)

    available = sorted(candidates.values())
    raise ValueError(
        f"Could not resolve {boundary_kind}_boundary_name={name!r}. "
        f"Available facet physical names: {available}"
    )


class DolfinxHyperelasticitySimulation:
    """
    Small notebook-style DOLFINx wrapper for setup, solve, and data extraction.

    Example (structured box):
        simulation = setup_hyperelasticity_simulation()

    Example (gmsh .msh):
        simulation = setup_hyperelasticity_from_msh(
            msh_path="dataset/fem_data/plate_hole_fenics/mesh/mesh.msh",
            fixed_boundary_tag=7,
            traction_boundary_tag=9,
            gdim=2,
        )
    """

    def __init__(self, config: DolfinxHyperelasticityConfig | None = None) -> None:
        self.config = config or DolfinxHyperelasticityConfig()

        MPI, ufl, fem, dolfinx_io, mesh, fem_petsc, nls_petsc = _import_fenicsx()
        from dolfinx import plot

        self._plot = plot

        comm = MPI.COMM_SELF if self.config.use_comm_self else MPI.COMM_WORLD
        self.cell_tags = None
        self.msh_physical_names: dict[tuple[int, int], str] = {}

        if self.config.msh_path is None:
            self.domain = mesh.create_box(
                comm,
                [
                    np.asarray(self.config.lower_corner, dtype=float),
                    np.asarray(self.config.upper_corner, dtype=float),
                ],
                list(self.config.cells),
                cell_type=mesh.CellType.hexahedron,
            )
            fdim = self.domain.topology.dim - 1
            boundary_order = (self.config.fixed_boundary, self.config.traction_boundary)
            self.boundary_markers = {
                boundary_name: marker for marker, boundary_name in enumerate(boundary_order, 1)
            }

            marked_facet_chunks = []
            marked_value_chunks = []
            for boundary_name in boundary_order:
                boundary_facets = mesh.locate_entities_boundary(
                    self.domain,
                    fdim,
                    _box_boundary_locator(boundary_name, self.config),
                )
                marked_facet_chunks.append(boundary_facets.astype(np.int32))
                marked_value_chunks.append(
                    np.full(
                        boundary_facets.shape,
                        self.boundary_markers[boundary_name],
                        dtype=np.int32,
                    )
                )

            marked_facets = np.hstack(marked_facet_chunks).astype(np.int32)
            marked_values = np.hstack(marked_value_chunks).astype(np.int32)
            ordering = np.argsort(marked_facets, kind="mergesort")
            self.facet_tag = mesh.meshtags(
                self.domain,
                fdim,
                marked_facets[ordering],
                marked_values[ordering],
            )
            self.fixed_boundary_marker = int(self.boundary_markers[self.config.fixed_boundary])
            self.traction_boundary_marker = int(self.boundary_markers[self.config.traction_boundary])
        else:
            self.domain, self.cell_tags, self.facet_tag = _load_msh_mesh(
                dolfinx_io,
                self.config.msh_path,
                comm,
                self.config.msh_gdim,
            )
            if self.facet_tag is None:
                raise ValueError(
                    "Loaded .msh file does not contain facet tags. "
                    "Define Physical Curve/Surface groups in gmsh."
                )

            self.msh_physical_names = read_msh_physical_names(self.config.msh_path)
            facet_dim = self.domain.topology.dim - 1
            self.fixed_boundary_marker = _resolve_boundary_tag(
                self.config, facet_dim, self.msh_physical_names, "fixed"
            )
            self.traction_boundary_marker = _resolve_boundary_tag(
                self.config, facet_dim, self.msh_physical_names, "traction"
            )

            if self.fixed_boundary_marker == self.traction_boundary_marker:
                raise ValueError("Fixed and traction boundary markers must be different.")

            self.boundary_markers = {
                "fixed": self.fixed_boundary_marker,
                "traction": self.traction_boundary_marker,
            }

        self.function_space = _create_vector_function_space(
            fem,
            self.domain,
            degree=self.config.element_degree,
        )
        fixed_dofs = fem.locate_dofs_topological(
            self.function_space,
            self.facet_tag.dim,
            self.facet_tag.find(self.fixed_boundary_marker),
        )
        zero_displacement = np.zeros(self.domain.geometry.dim, dtype=np.float64)
        self.boundary_conditions = [
            fem.dirichletbc(zero_displacement, fixed_dofs, self.function_space)
        ]

        self.body_force = fem.Constant(
            self.domain,
            _validate_vector(self.config.body_force, self.domain.geometry.dim),
        )
        self.traction = fem.Constant(
            self.domain,
            _validate_vector(self.config.traction, self.domain.geometry.dim),
        )

        test_function = ufl.TestFunction(self.function_space)
        self.displacement = fem.Function(self.function_space)
        identity = ufl.variable(ufl.Identity(self.domain.geometry.dim))
        deformation_gradient = ufl.variable(identity + ufl.grad(self.displacement))
        strain_energy = _build_strain_energy_density(
            ufl=ufl,
            fem=fem,
            domain=self.domain,
            deformation_gradient=deformation_gradient,
            config=self.config,
        )
        self.first_piola = ufl.diff(strain_energy, deformation_gradient)

        metadata = {"quadrature_degree": self.config.quadrature_degree}
        ds = ufl.Measure(
            "ds",
            domain=self.domain,
            subdomain_data=self.facet_tag,
            metadata=metadata,
        )
        dx = ufl.Measure("dx", domain=self.domain, metadata=metadata)
        residual = (
            ufl.inner(ufl.grad(test_function), self.first_piola) * dx
            - ufl.inner(test_function, self.body_force) * dx
            - ufl.inner(test_function, self.traction) * ds(self.traction_boundary_marker)
        )

        petsc_options = {
            "snes_type": "newtonls",
            "snes_linesearch_type": "none",
            "snes_atol": self.config.solver_atol,
            "snes_rtol": self.config.solver_rtol,
            "snes_stol": self.config.solver_rtol,
            "snes_max_it": self.config.solver_max_it,
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
        }
        self._solve_once = _build_solver(
            self.domain,
            residual,
            self.displacement,
            self.boundary_conditions,
            fem_petsc,
            nls_petsc,
            petsc_options,
            self.config,
        )

    def set_traction(self, traction: tuple[float, ...] | list[float] | np.ndarray) -> None:
        self.traction.value = _validate_vector(traction, self.domain.geometry.dim)

    def solve(
        self,
        traction: tuple[float, ...] | list[float] | np.ndarray | None = None,
    ) -> DolfinxSolveResult:
        if traction is not None:
            self.set_traction(traction)

        iterations = self._solve_once()
        return DolfinxSolveResult(
            traction=np.asarray(self.traction.value, dtype=float).copy(),
            iterations=int(iterations),
            displacement_norm=float(np.linalg.norm(self.displacement.x.array)),
        )

    def extract_data(self, iterations: int | None = None) -> DolfinxSimulationData:
        topology, cell_types, points = self._plot.vtk_mesh(self.displacement.function_space)

        if self.displacement.x.array.size % self.domain.geometry.dim != 0:
            raise RuntimeError("Unexpected displacement vector layout for DOLFINx extraction.")

        num_points = self.displacement.x.array.size // self.domain.geometry.dim
        if num_points != points.shape[0]:
            raise RuntimeError(
                "Function-space point count does not match the displacement vector layout."
            )

        displacement = self.displacement.x.array.reshape(points.shape[0], self.domain.geometry.dim).copy()
        return DolfinxSimulationData(
            points=np.asarray(points, dtype=float).copy(),
            topology=np.asarray(topology).copy(),
            cell_types=np.asarray(cell_types).copy(),
            displacement=displacement,
            displacement_magnitude=np.linalg.norm(displacement, axis=1),
            traction=np.asarray(self.traction.value, dtype=float).copy(),
            iterations=iterations,
        )

    def run_load_steps(
        self,
        traction_values: tuple[tuple[float, ...], ...] | list[tuple[float, ...]] | np.ndarray,
    ) -> tuple[DolfinxSimulationData, ...]:
        snapshots = []
        for traction in traction_values:
            result = self.solve(traction)
            snapshots.append(self.extract_data(iterations=result.iterations))
        return tuple(snapshots)

    def get_facet_markers(self) -> np.ndarray:
        """Return sorted unique facet marker ids in the loaded mesh tags."""

        return np.unique(np.asarray(self.facet_tag.values, dtype=np.int64))

    def plot(
        self,
        warp_factor: float = 1.0,
        cmap: str = "viridis",
        title: str | None = None,
        show: bool = True,
        backend: str = "auto",
    ) -> Any:
        """Plot the current simulation displacement field."""

        data = self.extract_data()
        return plot_simulation_data(
            data=data,
            warp_factor=warp_factor,
            cmap=cmap,
            title=title,
            show=show,
            backend=backend,
        )


def setup_hyperelasticity_simulation(
    config: DolfinxHyperelasticityConfig | None = None,
) -> DolfinxHyperelasticitySimulation:
    """Build a minimal DOLFINx hyperelastic simulation from config."""

    return DolfinxHyperelasticitySimulation(config=config)


def setup_hyperelasticity_from_msh(
    msh_path: str,
    fixed_boundary_tag: int | None = None,
    traction_boundary_tag: int | None = None,
    fixed_boundary_name: str | None = None,
    traction_boundary_name: str | None = None,
    gdim: int = 2,
    use_comm_self: bool = False,
    **kwargs,
) -> DolfinxHyperelasticitySimulation:
    """Convenience wrapper for `.msh` mesh mode."""

    config = DolfinxHyperelasticityConfig(
        msh_path=msh_path,
        msh_gdim=gdim,
        use_comm_self=use_comm_self,
        fixed_boundary_tag=fixed_boundary_tag,
        traction_boundary_tag=traction_boundary_tag,
        fixed_boundary_name=fixed_boundary_name,
        traction_boundary_name=traction_boundary_name,
        **kwargs,
    )
    return setup_hyperelasticity_simulation(config=config)


__all__ = [
    "DolfinxHyperelasticityConfig",
    "DolfinxHyperelasticitySimulation",
    "DolfinxSimulationData",
    "DolfinxSolveResult",
    "SUPPORTED_BOX_BOUNDARIES",
    "SUPPORTED_HYPERELASTIC_MODELS",
    "plot_simulation_data",
    "read_msh_physical_names",
    "setup_hyperelasticity_from_msh",
    "setup_hyperelasticity_simulation",
]
