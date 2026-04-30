from __future__ import annotations

import csv
import json
import os
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


SUPPORTED_MATERIAL_MODELS = (
    "notebook_neo_hookean_log",
    "euclid_neo_hookean_j2",
)
SUPPORTED_BOUNDARIES = ("left", "right", "top", "bottom")


@dataclass(frozen=True)
class HyperelasticGenerationConfig:
    """Configuration for generating EUCLID-compatible hyperelastic FEM data."""

    material_model: str = "notebook_neo_hookean_log"
    fixed_boundary: str = "left"
    traction_boundaries: tuple[str, ...] = ("right",)
    length: float = 20.0
    height: float = 1.0
    nx: int = 20
    ny: int = 5
    young_modulus: float = 1.0e4
    poisson_ratio: float = 0.3
    quadrature_degree: int = 4
    solver_atol: float = 1e-8
    solver_rtol: float = 1e-8
    solver_max_it: int = 50
    boundary_tol: float = 1e-8
    load_steps: tuple[int, ...] = (10, 20, 30, 40)
    traction_per_step: float = 25.0
    traction_direction: tuple[float, float] = (1.0, 0.0)
    traction_boundary_vectors: tuple[tuple[float, float], ...] | None = None
    traction_values: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        if self.length <= 0.0 or self.height <= 0.0:
            raise ValueError("length and height must be positive.")
        if self.material_model not in SUPPORTED_MATERIAL_MODELS:
            raise ValueError("Unsupported material_model: %s" % self.material_model)
        if self.fixed_boundary not in SUPPORTED_BOUNDARIES:
            raise ValueError("Unsupported fixed_boundary: %s" % self.fixed_boundary)
        if not self.traction_boundaries:
            raise ValueError("traction_boundaries must contain at least one boundary name.")
        if len(set(self.traction_boundaries)) != len(self.traction_boundaries):
            raise ValueError("traction_boundaries must be unique.")
        if any(boundary not in SUPPORTED_BOUNDARIES for boundary in self.traction_boundaries):
            raise ValueError("Unsupported traction boundary in %s" % (self.traction_boundaries,))
        if self.fixed_boundary in self.traction_boundaries:
            raise ValueError("fixed_boundary cannot also be a traction boundary.")
        if self.nx < 1 or self.ny < 1:
            raise ValueError("nx and ny must be positive integers.")
        if not self.load_steps:
            raise ValueError("load_steps must contain at least one entry.")
        if len(set(self.load_steps)) != len(self.load_steps):
            raise ValueError("load_steps must be unique.")
        if self.quadrature_degree < 1:
            raise ValueError("quadrature_degree must be >= 1.")
        if self.solver_atol <= 0.0 or self.solver_rtol <= 0.0:
            raise ValueError("solver tolerances must be positive.")
        if self.solver_max_it < 1:
            raise ValueError("solver_max_it must be >= 1.")
        if not (-1.0 < self.poisson_ratio < 0.5):
            raise ValueError("poisson_ratio must lie in (-1, 0.5).")
        if np.isclose(np.linalg.norm(self.traction_direction), 0.0):
            raise ValueError("traction_direction must be non-zero.")
        if self.traction_boundary_vectors is not None:
            if len(self.traction_boundary_vectors) != len(self.traction_boundaries):
                raise ValueError(
                    "traction_boundary_vectors must match traction_boundaries in length."
                )
            for vector in self.traction_boundary_vectors:
                if np.isclose(np.linalg.norm(vector), 0.0):
                    raise ValueError("Each traction boundary vector must be non-zero.")
        if self.traction_values is not None and len(self.traction_values) != len(self.load_steps):
            raise ValueError("traction_values must match load_steps in length.")

    @property
    def traction_direction_unit(self) -> np.ndarray:
        direction = np.asarray(self.traction_direction, dtype=float)
        return direction / np.linalg.norm(direction)

    @property
    def resolved_traction_values(self) -> tuple[float, ...]:
        if self.traction_values is not None:
            return self.traction_values
        return tuple(self.traction_per_step * (i + 1) for i in range(len(self.load_steps)))

    @property
    def resolved_traction_boundary_vectors(self) -> tuple[np.ndarray, ...]:
        if self.traction_boundary_vectors is None:
            return tuple(
                self.traction_direction_unit.copy() for _ in range(len(self.traction_boundaries))
            )
        return tuple(np.asarray(vector, dtype=float) for vector in self.traction_boundary_vectors)


@dataclass(frozen=True)
class HyperelasticGenerationResult:
    output_root: Path
    load_steps: tuple[int, ...]
    step_directories: tuple[Path, ...]
    traction_vectors: tuple[tuple[float, float], ...]
    manifest_path: Path


@dataclass(frozen=True)
class HyperelasticSuiteCase:
    name: str
    config: HyperelasticGenerationConfig
    load_step_offset: int = 0


@dataclass(frozen=True)
class HyperelasticSuiteResult:
    output_root: Path
    data_root: Path
    load_steps: tuple[int, ...]
    manifest_path: Path
    case_results: tuple[HyperelasticGenerationResult, ...]


@dataclass(frozen=True)
class HyperelasticExportValidationResult:
    dataset_root: Path
    material_model: str
    load_steps: tuple[int, ...]
    max_abs_error: float
    mean_rmse: float
    per_load_step: tuple[dict[str, float], ...]


def _import_fenicsx():
    try:
        from mpi4py import MPI
        from petsc4py import PETSc
        import ufl
        from dolfinx import fem, mesh
        from dolfinx.fem import petsc as fem_petsc

        try:
            from dolfinx.nls import petsc as nls_petsc
        except ImportError:
            nls_petsc = None
    except Exception as exc:  # pragma: no cover - depends on runtime environment
        raise ImportError(
            "generate_hyperelastic_data requires a working DOLFINx/FEniCSx environment. "
            "Use the Conda environment from environment.yml."
        ) from exc

    return MPI, PETSc, ufl, fem, mesh, fem_petsc, nls_petsc


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
    jit_options=None,
):
    try:
        problem = fem_petsc.NonlinearProblem(
            residual,
            u,
            bcs=bcs,
            jit_options=jit_options,
            petsc_options=petsc_options,
            petsc_options_prefix="hyperelasticity",
        )
    except TypeError:
        legacy_kwargs = {"bcs": bcs}
        if jit_options is not None:
            legacy_kwargs["jit_params"] = jit_options
        try:
            problem = fem_petsc.NonlinearProblem(residual, u, **legacy_kwargs)
        except TypeError:
            legacy_kwargs.pop("jit_params", None)
            problem = fem_petsc.NonlinearProblem(residual, u, **legacy_kwargs)
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
            options = {
                key: value
                for key, value in petsc_options.items()
                if key.startswith(("ksp_", "pc_"))
            }
            prefix = krylov_solver.getOptionsPrefix()
            from petsc4py import PETSc

            petsc_db = PETSc.Options()
            for key, value in options.items():
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


def _boundary_locator(boundary_name: str, config: HyperelasticGenerationConfig):
    if boundary_name == "left":
        return lambda x: np.isclose(x[0], 0.0, atol=config.boundary_tol)
    if boundary_name == "right":
        return lambda x: np.isclose(x[0], config.length, atol=config.boundary_tol)
    if boundary_name == "bottom":
        return lambda x: np.isclose(x[1], 0.0, atol=config.boundary_tol)
    if boundary_name == "top":
        return lambda x: np.isclose(x[1], config.height, atol=config.boundary_tol)
    raise ValueError("Unsupported boundary name: %s" % boundary_name)


def _boundary_measure(boundary_name: str, config: HyperelasticGenerationConfig) -> float:
    if boundary_name in {"left", "right"}:
        return float(config.height)
    if boundary_name in {"top", "bottom"}:
        return float(config.length)
    raise ValueError("Unsupported boundary name: %s" % boundary_name)


def _lame_parameters(config: HyperelasticGenerationConfig) -> tuple[float, float]:
    mu = config.young_modulus / (2.0 * (1.0 + config.poisson_ratio))
    lmbda = config.young_modulus * config.poisson_ratio / (
        (1.0 + config.poisson_ratio) * (1.0 - 2.0 * config.poisson_ratio)
    )
    return mu, lmbda


def _bulk_modulus(config: HyperelasticGenerationConfig) -> float:
    return config.young_modulus / (3.0 * (1.0 - 2.0 * config.poisson_ratio))


def _embed_plane_strain_deformation_gradient(ufl, deformation_gradient):
    return ufl.as_tensor(
        (
            (deformation_gradient[0, 0], deformation_gradient[0, 1], 0.0),
            (deformation_gradient[1, 0], deformation_gradient[1, 1], 0.0),
            (0.0, 0.0, 1.0),
        )
    )


def _build_strain_energy_density(ufl, fem, domain, deformation_gradient, config):
    deformation_gradient_3d = _embed_plane_strain_deformation_gradient(ufl, deformation_gradient)
    cauchy_green = ufl.variable(deformation_gradient_3d.T * deformation_gradient_3d)
    first_invariant = ufl.variable(ufl.tr(cauchy_green))
    jacobian = ufl.variable(ufl.det(deformation_gradient_3d))

    mu_value, lmbda_value = _lame_parameters(config)
    mu = fem.Constant(domain, np.float64(mu_value))

    if config.material_model == "notebook_neo_hookean_log":
        lmbda = fem.Constant(domain, np.float64(lmbda_value))
        return (
            (mu / 2.0) * (first_invariant - 3.0)
            - mu * ufl.ln(jacobian)
            + (lmbda / 2.0) * (ufl.ln(jacobian) ** 2)
        )

    bulk_modulus = fem.Constant(domain, np.float64(_bulk_modulus(config)))
    third_invariant = ufl.variable(ufl.det(cauchy_green))
    reduced_first_invariant = ufl.variable(
        first_invariant * (third_invariant ** (-1.0 / 3.0)) - 3.0
    )
    return (mu / 2.0) * reduced_first_invariant + (bulk_modulus / 2.0) * (
        jacobian - 1.0
    ) ** 2


def _compute_piola_field(
    u_nodes: np.ndarray,
    connectivity: np.ndarray,
    grad_na: np.ndarray,
    config: HyperelasticGenerationConfig,
) -> np.ndarray:
    piola = np.zeros((connectivity.shape[0], 4), dtype=float)

    for cell in range(connectivity.shape[0]):
        local_nodes = connectivity[cell]
        local_u = u_nodes[local_nodes]
        deformation_gradient = np.eye(2, dtype=float)
        for a in range(3):
            deformation_gradient += np.outer(local_u[a], grad_na[cell, a])

        jacobian = np.linalg.det(deformation_gradient)
        if jacobian <= 0.0:
            raise ValueError(
                "Encountered a non-positive Jacobian while exporting Piola stresses."
            )
        stress = _compute_piola_from_deformation_gradient(deformation_gradient, config)
        piola[cell] = np.array(
            [stress[0, 0], stress[0, 1], stress[1, 0], stress[1, 1]],
            dtype=float,
        )

    return piola


def _compute_piola_from_deformation_gradient(
    deformation_gradient: np.ndarray,
    config: HyperelasticGenerationConfig,
) -> np.ndarray:
    F11 = deformation_gradient[0, 0]
    F12 = deformation_gradient[0, 1]
    F21 = deformation_gradient[1, 0]
    F22 = deformation_gradient[1, 1]
    jacobian = F11 * F22 - F12 * F21

    if config.material_model == "notebook_neo_hookean_log":
        mu, lmbda = _lame_parameters(config)
        inverse_transpose = np.linalg.inv(deformation_gradient).T
        return (
            mu * (deformation_gradient - inverse_transpose)
            + lmbda * np.log(jacobian) * inverse_transpose
        )

    mu, _ = _lame_parameters(config)
    bulk_modulus = _bulk_modulus(config)
    I1 = F11**2 + F12**2 + F21**2 + F22**2 + 1.0
    I3 = jacobian**2
    J = jacobian
    dI1dF = 2.0 * np.array([F11, F12, F21, F22], dtype=float)
    dI3dF = 2.0 * jacobian * np.array([F22, -F21, -F12, F11], dtype=float)
    dK1dF = np.power(I3, -1.0 / 3.0) * dI1dF + I1 * (
        -1.0 / 3.0
    ) * np.power(I3, -4.0 / 3.0) * dI3dF
    dJdF = 0.5 * np.power(I3, -0.5) * dI3dF
    stress_vector = (mu / 2.0) * dK1dF + bulk_modulus * (J - 1.0) * dJdF
    return np.array(
        [
            [stress_vector[0], stress_vector[1]],
            [stress_vector[2], stress_vector[3]],
        ],
        dtype=float,
    )


def _write_case_csvs(
    output_dir: Path,
    x_nodes: np.ndarray,
    u_nodes: np.ndarray,
    bcx: np.ndarray,
    bcy: np.ndarray,
    connectivity: np.ndarray,
    grad_na: np.ndarray,
    qp_weights: np.ndarray,
    piola: np.ndarray,
    reaction_forces: np.ndarray,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    nodes_rows = np.column_stack((x_nodes, u_nodes, bcx, bcy))
    _write_csv(
        output_dir / "output_nodes.csv",
        ("x", "y", "ux", "uy", "bcx", "bcy"),
        nodes_rows,
    )

    element_rows = np.column_stack((connectivity, piola))
    _write_csv(
        output_dir / "output_elements.csv",
        ("node1", "node2", "node3", "Pxx", "Pxy", "Pyx", "Pyy"),
        element_rows,
    )

    integrator_rows = np.column_stack(
        (
            grad_na[:, 0, 0],
            grad_na[:, 0, 1],
            grad_na[:, 1, 0],
            grad_na[:, 1, 1],
            grad_na[:, 2, 0],
            grad_na[:, 2, 1],
            qp_weights,
        )
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
        integrator_rows,
    )

    _write_csv(
        output_dir / "output_reactions.csv",
        ("forces",),
        reaction_forces.reshape(-1, 1),
    )


def _write_csv(path: Path, header: tuple[str, ...], rows: np.ndarray) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for row in rows:
            writer.writerow(row.tolist())


def _resolve_validation_entries(
    dataset_root: Path,
    config: HyperelasticGenerationConfig | None,
) -> tuple[tuple[tuple[int, HyperelasticGenerationConfig], ...], str]:
    manifest_path = dataset_root / "generation_manifest.json"
    if config is not None:
        return (
            tuple((int(step), config) for step in config.load_steps),
            config.material_model,
        )

    if not manifest_path.exists():
        raise FileNotFoundError(
            "Validation requires either a config object or a generation_manifest.json under {}.".format(
                dataset_root
            )
        )

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload_config = payload.get("config")
    entries = []
    material_models = set()
    for entry in payload.get("load_steps", []):
        entry_payload_config = entry.get("config", payload_config)
        if entry_payload_config is None:
            raise ValueError(
                "generation_manifest.json is missing a top-level 'config' entry or per-load-step 'config' payloads."
            )
        entry_config = HyperelasticGenerationConfig(**entry_payload_config)
        entries.append((int(entry["load_step"]), entry_config))
        material_models.add(entry_config.material_model)

    if entries:
        material_model = next(iter(material_models)) if len(material_models) == 1 else "mixed"
        return tuple(entries), material_model

    if payload_config is None:
        raise ValueError("generation_manifest.json is missing the 'config' entry.")
    resolved_config = HyperelasticGenerationConfig(**payload_config)
    return (
        tuple((int(step), resolved_config) for step in resolved_config.load_steps),
        resolved_config.material_model,
    )


def validate_hyperelastic_data_export(
    dataset_root: str | Path,
    config: HyperelasticGenerationConfig | None = None,
) -> HyperelasticExportValidationResult:
    """
    Reload exported CSV cases and verify that their Piola stresses match `P(F)`.

    This validates the CSV conversion path independently of the DOLFINx solve.
    """

    from .csv_loader import loadFemData

    dataset_root = Path(dataset_root)
    validation_entries, material_model = _resolve_validation_entries(dataset_root, config)

    per_load_step: list[dict[str, float]] = []
    max_abs_error = 0.0
    rmse_values: list[float] = []

    for load_step, entry_config in validation_entries:
        case_dir = dataset_root / str(load_step)
        data = loadFemData(str(case_dir), AD=True, noiseLevel=0.0, noiseType="displacement")
        data.convertToNumpy()

        if data.P is None:
            raise ValueError(
                "Reference Piola stresses are missing in {}. Export validation cannot continue.".format(
                    case_dir
                )
            )

        reconstructed_piola = np.zeros_like(data.P)
        for element, deformation_gradient_voigt in enumerate(data.F):
            deformation_gradient = np.array(
                [
                    [deformation_gradient_voigt[0], deformation_gradient_voigt[1]],
                    [deformation_gradient_voigt[2], deformation_gradient_voigt[3]],
                ],
                dtype=float,
            )
            stress = _compute_piola_from_deformation_gradient(
                deformation_gradient,
                entry_config,
            )
            reconstructed_piola[element] = np.array(
                [stress[0, 0], stress[0, 1], stress[1, 0], stress[1, 1]],
                dtype=float,
            )

        error = reconstructed_piola - data.P
        step_max_abs_error = float(np.max(np.abs(error)))
        step_rmse = float(np.sqrt(np.mean(np.square(error))))
        max_abs_error = max(max_abs_error, step_max_abs_error)
        rmse_values.append(step_rmse)
        per_load_step.append(
            {
                "load_step": int(load_step),
                "max_abs_error": step_max_abs_error,
                "rmse": step_rmse,
            }
        )

    mean_rmse = float(np.mean(rmse_values)) if rmse_values else 0.0
    return HyperelasticExportValidationResult(
        dataset_root=dataset_root,
        material_model=material_model,
        load_steps=tuple(load_step for load_step, _ in validation_entries),
        max_abs_error=max_abs_error,
        mean_rmse=mean_rmse,
        per_load_step=tuple(per_load_step),
    )


def generate_hyperelastic_data(
    output_root: str | Path,
    config: HyperelasticGenerationConfig | None = None,
) -> HyperelasticGenerationResult:
    """
    Generate a 2D plane-strain hyperelastic dataset in the CSV layout expected by EUCLID.

    The default material model follows the constitutive setup from
    `hyperelasticity.ipynb`. An optional Euclid-compatible J2 variant is also
    available for tutorial and debugging workflows. Both modes export a
    triangle-based 2D dataset that can be consumed directly by `loadFemData`
    and `EuclidWorkflow`.
    """

    config = config or HyperelasticGenerationConfig()
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    cache_root = (output_root / ".cache").resolve()
    fenics_cache_dir = cache_root / "fenics"
    fenics_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["XDG_CACHE_HOME"] = str(cache_root)

    MPI, _, ufl, fem, mesh, fem_petsc, nls_petsc = _import_fenicsx()

    if MPI.COMM_WORLD.size != 1:
        raise NotImplementedError(
            "generate_hyperelastic_data currently supports only single-rank runs."
        )

    domain = mesh.create_rectangle(
        MPI.COMM_WORLD,
        [np.array([0.0, 0.0]), np.array([config.length, config.height])],
        [config.nx, config.ny],
        cell_type=mesh.CellType.triangle,
    )
    space = _create_vector_function_space(fem, domain, degree=1)

    fdim = domain.topology.dim - 1
    boundary_order = (config.fixed_boundary,) + tuple(config.traction_boundaries)
    boundary_markers = {boundary_name: marker for marker, boundary_name in enumerate(boundary_order, 1)}

    marked_facet_chunks = []
    marked_value_chunks = []
    for boundary_name in boundary_order:
        boundary_facets = mesh.locate_entities_boundary(
            domain,
            fdim,
            _boundary_locator(boundary_name, config),
        )
        marked_facet_chunks.append(boundary_facets.astype(np.int32))
        marked_value_chunks.append(
            np.full(boundary_facets.shape, boundary_markers[boundary_name], dtype=np.int32)
        )
    marked_facets = np.hstack(marked_facet_chunks).astype(np.int32)
    marked_values = np.hstack(marked_value_chunks).astype(np.int32)
    ordering = np.argsort(marked_facets, kind="mergesort")
    facet_tag = mesh.meshtags(
        domain,
        fdim,
        marked_facets[ordering],
        marked_values[ordering],
    )

    fixed_dofs = fem.locate_dofs_topological(
        space,
        facet_tag.dim,
        facet_tag.find(boundary_markers[config.fixed_boundary]),
    )
    zero_displacement = np.zeros(domain.geometry.dim, dtype=np.float64)
    bcs = [fem.dirichletbc(zero_displacement, fixed_dofs, space)]

    body_force = fem.Constant(domain, np.zeros(domain.geometry.dim, dtype=np.float64))
    traction_constants = {
        boundary_name: fem.Constant(domain, np.zeros(domain.geometry.dim, dtype=np.float64))
        for boundary_name in config.traction_boundaries
    }

    test_function = ufl.TestFunction(space)
    displacement = fem.Function(space)
    identity = ufl.variable(ufl.Identity(domain.geometry.dim))
    deformation_gradient = ufl.variable(identity + ufl.grad(displacement))
    strain_energy = _build_strain_energy_density(
        ufl=ufl,
        fem=fem,
        domain=domain,
        deformation_gradient=deformation_gradient,
        config=config,
    )
    first_piola = ufl.diff(strain_energy, deformation_gradient)
    metadata = {"quadrature_degree": config.quadrature_degree}
    ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_tag, metadata=metadata)
    dx = ufl.Measure("dx", domain=domain, metadata=metadata)
    traction_terms = [
        ufl.inner(test_function, traction_constants[boundary_name]) * ds(boundary_markers[boundary_name])
        for boundary_name in config.traction_boundaries
    ]
    residual = ufl.inner(ufl.grad(test_function), first_piola) * dx - ufl.inner(
        test_function, body_force
    ) * dx
    for traction_term in traction_terms:
        residual -= traction_term

    petsc_options = {
        "snes_type": "newtonls",
        "snes_linesearch_type": "none",
        "snes_atol": config.solver_atol,
        "snes_rtol": config.solver_rtol,
        "snes_stol": config.solver_rtol,
        "snes_max_it": config.solver_max_it,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
    }
    solve_once = _build_solver(
        domain,
        residual,
        displacement,
        bcs,
        fem_petsc,
        nls_petsc,
        petsc_options,
        config,
        jit_options={"cache_dir": str(fenics_cache_dir)},
    )

    tdim = domain.topology.dim
    domain.topology.create_connectivity(tdim, 0)
    num_cells = domain.topology.index_map(tdim).size_local
    connectivity = np.asarray(domain.geometry.dofmap[:num_cells], dtype=int)

    x_nodes = np.asarray(domain.geometry.x[:, :2], dtype=float)
    fixed_nodes = _boundary_locator(config.fixed_boundary, config)(x_nodes.T)
    bcx = np.zeros(x_nodes.shape[0], dtype=int)
    bcy = np.zeros(x_nodes.shape[0], dtype=int)
    bcx[fixed_nodes] = 1
    bcy[fixed_nodes] = 2

    grad_na = np.zeros((num_cells, 3, 2), dtype=float)
    qp_weights = np.zeros(num_cells, dtype=float)
    for cell in range(num_cells):
        local_vertices = x_nodes[connectivity[cell]]
        grad_na[cell], qp_weights[cell] = _compute_triangle_gradients(local_vertices)

    manifest = {
        "generator": "generate_hyperelastic_data",
        "config": asdict(config),
        "load_steps": [],
    }

    if displacement.x.array.size != x_nodes.shape[0] * domain.geometry.dim:
        raise RuntimeError(
            "Unexpected displacement vector layout. The hyperelastic exporter assumes "
            "a first-order vector Lagrange space with one displacement vector per mesh node."
        )

    step_directories: list[Path] = []
    traction_vectors: list[tuple[float, float]] = []
    boundary_vectors = config.resolved_traction_boundary_vectors
    for load_step, traction_scale in zip(config.load_steps, config.resolved_traction_values):
        boundary_traction_vectors = {}
        for boundary_name, boundary_vector in zip(config.traction_boundaries, boundary_vectors):
            traction_vector = traction_scale * boundary_vector
            traction_constants[boundary_name].value = traction_vector
            boundary_traction_vectors[boundary_name] = traction_vector
        iterations = solve_once()

        u_nodes = displacement.x.array.reshape(x_nodes.shape[0], domain.geometry.dim).copy()
        piola = _compute_piola_field(
            u_nodes=u_nodes,
            connectivity=connectivity,
            grad_na=grad_na,
            config=config,
        )
        reaction_forces = np.zeros(domain.geometry.dim, dtype=float)
        for boundary_name, traction_vector in boundary_traction_vectors.items():
            reaction_forces -= traction_vector * _boundary_measure(boundary_name, config)
        net_traction_vector = np.sum(
            np.vstack(tuple(boundary_traction_vectors.values())),
            axis=0,
        )

        step_dir = output_root / str(load_step)
        _write_case_csvs(
            output_dir=step_dir,
            x_nodes=x_nodes,
            u_nodes=u_nodes,
            bcx=bcx,
            bcy=bcy,
            connectivity=connectivity,
            grad_na=grad_na,
            qp_weights=qp_weights,
            piola=piola,
            reaction_forces=reaction_forces,
        )

        step_directories.append(step_dir)
        traction_vectors.append((float(net_traction_vector[0]), float(net_traction_vector[1])))
        manifest["load_steps"].append(
            {
                "load_step": int(load_step),
                "traction": [float(net_traction_vector[0]), float(net_traction_vector[1])],
                "boundary_tractions": {
                    boundary_name: [float(vector[0]), float(vector[1])]
                    for boundary_name, vector in boundary_traction_vectors.items()
                },
                "reaction_forces": [float(reaction_forces[0]), float(reaction_forces[1])],
                "solver_iterations": iterations,
                "path": str(step_dir),
            }
        )

    manifest_path = output_root / "generation_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    return HyperelasticGenerationResult(
        output_root=output_root,
        load_steps=config.load_steps,
        step_directories=tuple(step_directories),
        traction_vectors=tuple(traction_vectors),
        manifest_path=manifest_path,
    )


def generate_hyperelastic_suite(
    output_root: str | Path,
    cases: tuple[HyperelasticSuiteCase, ...] | list[HyperelasticSuiteCase],
) -> HyperelasticSuiteResult:
    """
    Generate multiple hyperelastic FEM cases and combine them into one Euclid dataset root.
    """

    if not cases:
        raise ValueError("cases must contain at least one suite case.")

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    case_root = output_root / "cases"
    case_root.mkdir(parents=True, exist_ok=True)

    seen_case_names = set()
    seen_load_steps = set()
    case_results: list[HyperelasticGenerationResult] = []
    combined_load_steps: list[int] = []
    manifest = {
        "generator": "generate_hyperelastic_suite",
        "suite_cases": [],
        "load_steps": [],
    }

    for suite_case in cases:
        if suite_case.name in seen_case_names:
            raise ValueError("Duplicate suite case name: %s" % suite_case.name)
        seen_case_names.add(suite_case.name)

        case_output_root = case_root / suite_case.name
        case_result = generate_hyperelastic_data(case_output_root, suite_case.config)
        case_results.append(case_result)
        manifest["suite_cases"].append(
            {
                "name": suite_case.name,
                "load_step_offset": int(suite_case.load_step_offset),
                "output_root": str(case_output_root),
                "config": asdict(suite_case.config),
            }
        )

        case_manifest = json.loads(case_result.manifest_path.read_text(encoding="utf-8"))
        case_entries = case_manifest.get("load_steps", [])
        case_entries_by_load_step = {
            int(entry["load_step"]): entry for entry in case_entries
        }

        for load_step, step_dir in zip(case_result.load_steps, case_result.step_directories):
            combined_load_step = int(load_step) + int(suite_case.load_step_offset)
            if combined_load_step in seen_load_steps:
                raise ValueError("Duplicate combined load step: %s" % combined_load_step)
            seen_load_steps.add(combined_load_step)
            combined_load_steps.append(combined_load_step)

            combined_step_dir = output_root / str(combined_load_step)
            combined_step_dir.mkdir(parents=True, exist_ok=True)
            for filename in (
                "output_nodes.csv",
                "output_elements.csv",
                "output_integrator.csv",
                "output_reactions.csv",
            ):
                shutil.copy2(step_dir / filename, combined_step_dir / filename)

            case_entry = case_entries_by_load_step[int(load_step)]
            manifest["load_steps"].append(
                {
                    "load_step": combined_load_step,
                    "case_name": suite_case.name,
                    "case_load_step": int(load_step),
                    "config": asdict(suite_case.config),
                    "traction": case_entry.get("traction"),
                    "boundary_tractions": case_entry.get("boundary_tractions"),
                    "reaction_forces": case_entry.get("reaction_forces"),
                    "solver_iterations": case_entry.get("solver_iterations"),
                    "path": str(combined_step_dir),
                    "source_path": str(step_dir),
                }
            )

    combined_load_steps.sort()
    manifest["load_steps"].sort(key=lambda entry: int(entry["load_step"]))
    manifest_path = output_root / "generation_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    return HyperelasticSuiteResult(
        output_root=output_root,
        data_root=output_root,
        load_steps=tuple(combined_load_steps),
        manifest_path=manifest_path,
        case_results=tuple(case_results),
    )


__all__ = [
    "HyperelasticGenerationConfig",
    "HyperelasticExportValidationResult",
    "HyperelasticGenerationResult",
    "HyperelasticSuiteCase",
    "HyperelasticSuiteResult",
    "generate_hyperelastic_data",
    "generate_hyperelastic_suite",
    "validate_hyperelastic_data_export",
]
