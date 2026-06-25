from __future__ import annotations

import contextlib
import csv
import io
import json
import shutil
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse import linalg as sparse_linalg
from sklearn.kernel_ridge import KernelRidge

from sym_modeling.domains.fem.io.csv_loader import loadFemData
from sym_modeling.domains.fem.methods.common.stress_data import resolve_loadsteps


DEFAULT_ALPHAS = (1e-10, 1e-8, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1.0)
DEFAULT_GAMMAS = (0.1, 1.0, 10.0, 30.0, 100.0)
DEFAULT_BLENDS = (0.25, 0.5, 0.75, 1.0)
DEFAULT_LAPLACIAN_LAMBDAS = (0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)
METHODS = ("krr", "mesh-laplacian")
OBJECTIVES = ("u_rmse", "F_rmse", "J_rmse", "I1_rmse", "I2_rmse", "I3_rmse")


@dataclass(frozen=True)
class DenoiseCandidate:
    alpha: float
    gamma: float
    blend: float = 1.0


@dataclass(frozen=True)
class MeshLaplacianCandidate:
    lambda_smooth: float
    blend: float = 1.0


@dataclass(frozen=True)
class DenoiseSearchConfig:
    data_dir: str | Path
    output_dir: str | Path
    loadsteps: Sequence[int] | None = None
    noise_level: float = 0.0
    seed: int = 20260623
    objective: str = "F_rmse"
    alphas: Sequence[float] = DEFAULT_ALPHAS
    gammas: Sequence[float] = DEFAULT_GAMMAS
    lambdas: Sequence[float] = DEFAULT_LAPLACIAN_LAMBDAS
    blends: Sequence[float] = DEFAULT_BLENDS
    boundary_weight: float = 500.0
    preserve_dirichlet: bool = True
    overwrite: bool = False
    method: str = "krr"

    def __post_init__(self) -> None:
        if self.method not in METHODS:
            raise ValueError("method must be one of: %s." % ", ".join(METHODS))
        if self.objective not in OBJECTIVES:
            raise ValueError("objective must be one of: %s." % ", ".join(OBJECTIVES))
        if self.noise_level < 0.0:
            raise ValueError("noise_level must be non-negative.")
        if self.boundary_weight <= 0.0:
            raise ValueError("boundary_weight must be positive.")
        for name, values in (
            ("alphas", self.alphas),
            ("gammas", self.gammas),
            ("lambdas", self.lambdas),
            ("blends", self.blends),
        ):
            if not values:
                raise ValueError("%s must contain at least one value." % name)
        for lambda_smooth in self.lambdas:
            if float(lambda_smooth) < 0.0:
                raise ValueError("lambda_smooth values must be non-negative.")
        for blend in self.blends:
            if float(blend) < 0.0 or float(blend) > 1.0:
                raise ValueError("blend values must be in [0, 1].")


@dataclass(frozen=True)
class DenoiseResult:
    config: DenoiseSearchConfig
    loadsteps: tuple[int, ...]
    selected_candidate: DenoiseCandidate | MeshLaplacianCandidate
    selected_score: float
    baseline_metrics: dict[str, float]
    search_rows: list[dict]
    loadstep_metrics: list[dict]
    output_dir: str
    summary_path: str
    search_csv_path: str
    loadstep_metrics_csv_path: str


def add_displacement_noise(
    u_nodes: np.ndarray,
    dirichlet_nodes: np.ndarray,
    noise_level: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Add synthetic displacement noise while preserving constrained DOFs."""
    u_nodes = np.asarray(u_nodes, dtype=float)
    dirichlet_nodes = np.asarray(dirichlet_nodes, dtype=bool)
    if noise_level == 0.0:
        return np.array(u_nodes, copy=True)
    noise = float(noise_level) * rng.standard_normal(u_nodes.shape)
    noise[dirichlet_nodes] = 0.0
    return u_nodes + noise


def denoise_displacements_krr(
    x_nodes: np.ndarray,
    u_nodes: np.ndarray,
    dirichlet_nodes: np.ndarray,
    candidate: DenoiseCandidate,
    *,
    boundary_weight: float = 500.0,
    preserve_dirichlet: bool = True,
) -> np.ndarray:
    """Denoise 2D nodal displacement with a shared multi-output RBF KRR model."""
    x_nodes = np.asarray(x_nodes, dtype=float)
    u_nodes = np.asarray(u_nodes, dtype=float)
    dirichlet_nodes = np.asarray(dirichlet_nodes, dtype=bool)
    if x_nodes.ndim != 2 or x_nodes.shape[0] != u_nodes.shape[0]:
        raise ValueError("x_nodes and u_nodes have incompatible shapes.")
    if u_nodes.ndim != 2 or u_nodes.shape[1] != 2:
        raise ValueError("u_nodes must have shape (num_nodes, 2).")
    if dirichlet_nodes.shape != u_nodes.shape:
        raise ValueError("dirichlet_nodes must have the same shape as u_nodes.")

    constrained_node_mask = np.any(dirichlet_nodes, axis=1)
    sample_weight = np.ones(x_nodes.shape[0], dtype=float)
    sample_weight[constrained_node_mask] = float(boundary_weight)

    model = KernelRidge(alpha=float(candidate.alpha), kernel="rbf", gamma=float(candidate.gamma))
    model.fit(x_nodes, u_nodes, sample_weight=sample_weight)
    predicted = np.asarray(model.predict(x_nodes), dtype=float)
    denoised = (1.0 - float(candidate.blend)) * u_nodes + float(candidate.blend) * predicted
    if preserve_dirichlet:
        denoised[dirichlet_nodes] = u_nodes[dirichlet_nodes]
    return denoised


def assemble_scalar_fem_laplacian(
    connectivity: Sequence[np.ndarray],
    grad_na: Sequence[np.ndarray],
    qp_weights: np.ndarray,
    num_nodes: int,
) -> sparse.csr_matrix:
    """Assemble the scalar P1 triangular FEM H1 stiffness matrix."""
    connectivity_by_element = _connectivity_by_element(connectivity)
    grad_by_element = _grad_na_by_element(grad_na)
    qp_weights = np.asarray(qp_weights, dtype=float)
    if connectivity_by_element.shape[0] != qp_weights.shape[0]:
        raise ValueError("connectivity and qp_weights have incompatible element counts.")
    if grad_by_element.shape != (qp_weights.shape[0], 3, 2):
        raise ValueError("grad_na must describe three 2D gradients per element.")

    rows = []
    cols = []
    values = []
    for element, nodes in enumerate(connectivity_by_element):
        local_grad = grad_by_element[element]
        local_k = float(qp_weights[element]) * (local_grad @ local_grad.T)
        for a in range(3):
            for b in range(3):
                rows.append(int(nodes[a]))
                cols.append(int(nodes[b]))
                values.append(float(local_k[a, b]))
    matrix = sparse.coo_matrix((values, (rows, cols)), shape=(int(num_nodes), int(num_nodes)))
    return matrix.tocsr()


def assemble_lumped_mass_diagonal(
    connectivity: Sequence[np.ndarray],
    qp_weights: np.ndarray,
    num_nodes: int,
) -> np.ndarray:
    """Assemble a lumped nodal mass/data-fidelity diagonal from element weights."""
    connectivity_by_element = _connectivity_by_element(connectivity)
    qp_weights = np.asarray(qp_weights, dtype=float)
    if connectivity_by_element.shape[0] != qp_weights.shape[0]:
        raise ValueError("connectivity and qp_weights have incompatible element counts.")
    mass = np.zeros(int(num_nodes), dtype=float)
    for element, nodes in enumerate(connectivity_by_element):
        mass[nodes.astype(int)] += float(qp_weights[element]) / 3.0
    return mass


def denoise_displacements_mesh_laplacian(
    u_nodes: np.ndarray,
    dirichlet_nodes: np.ndarray,
    connectivity: Sequence[np.ndarray],
    grad_na: Sequence[np.ndarray],
    qp_weights: np.ndarray,
    candidate: MeshLaplacianCandidate,
    *,
    preserve_dirichlet: bool = True,
) -> np.ndarray:
    """Denoise nodal displacements with lumped-mass fidelity plus FEM H1 smoothness."""
    u_nodes = np.asarray(u_nodes, dtype=float)
    dirichlet_nodes = np.asarray(dirichlet_nodes, dtype=bool)
    if u_nodes.ndim != 2 or u_nodes.shape[1] != 2:
        raise ValueError("u_nodes must have shape (num_nodes, 2).")
    if dirichlet_nodes.shape != u_nodes.shape:
        raise ValueError("dirichlet_nodes must have the same shape as u_nodes.")

    num_nodes = u_nodes.shape[0]
    mass = assemble_lumped_mass_diagonal(connectivity, qp_weights, num_nodes)
    stiffness = assemble_scalar_fem_laplacian(connectivity, grad_na, qp_weights, num_nodes)
    mass_matrix, stiffness = _normalize_mass_and_stiffness(mass, stiffness)
    system = mass_matrix + float(candidate.lambda_smooth) * stiffness

    smoothed = np.empty_like(u_nodes)
    for component in range(u_nodes.shape[1]):
        constrained = dirichlet_nodes[:, component] if preserve_dirichlet else np.zeros(num_nodes, dtype=bool)
        smoothed[:, component] = _solve_scalar_smoothing(system, mass_matrix, u_nodes[:, component], constrained)

    denoised = (1.0 - float(candidate.blend)) * u_nodes + float(candidate.blend) * smoothed
    if preserve_dirichlet:
        denoised[dirichlet_nodes] = u_nodes[dirichlet_nodes]
    return denoised


def search_denoise_hyperparameters(config: DenoiseSearchConfig) -> DenoiseResult:
    """Run global denoising hyperparameter search and write the selected dataset."""
    data_dir = Path(config.data_dir)
    output_dir = Path(config.output_dir)
    steps = tuple(int(step) for step in (config.loadsteps if config.loadsteps is not None else resolve_loadsteps(data_dir)))
    candidates = _build_candidates(config)

    step_inputs = []
    baseline_rows = []
    for step_index, step in enumerate(steps):
        clean = _load_fem_quiet(data_dir / str(step))
        rng = np.random.default_rng(_loadstep_seed(config.seed, step_index, step, config.noise_level))
        noisy_u = add_displacement_noise(clean.u_nodes, clean.dirichlet_nodes, config.noise_level, rng)
        noisy = _load_fem_quiet(data_dir / str(step), denoised_displacements=noisy_u)
        baseline = _metrics(clean, noisy)
        baseline_rows.append({"method": "noisy", "loadstep": int(step), **baseline})
        step_inputs.append((int(step), clean, noisy))

    search_rows = []
    for candidate in candidates:
        step_scores = []
        for step, clean, noisy in step_inputs:
            denoised_u = _denoise_candidate(noisy, candidate, config)
            denoised = _load_fem_quiet(data_dir / str(step), denoised_displacements=denoised_u)
            metrics = _metrics(clean, denoised)
            physics = _physics_checks(denoised)
            step_scores.append(metrics[config.objective])
            search_rows.append(
                {
                    "method": config.method,
                    "loadstep": int(step),
                    **_candidate_row(candidate),
                    **metrics,
                    **physics,
                }
            )

        mean_score = float(np.mean(step_scores))
        for row in search_rows[-len(step_inputs) :]:
            row["mean_%s" % config.objective] = mean_score

    aggregate_rows = _aggregate_search_rows(search_rows, config.objective, config.method)
    selectable_rows = _valid_physics_rows_or_all(aggregate_rows)
    selected_aggregate = min(selectable_rows, key=lambda row: row["mean_%s" % config.objective])
    selected = _candidate_from_row(selected_aggregate, config.method)

    loadstep_metrics = []
    denoised_by_step = {}
    for step, clean, noisy in step_inputs:
        denoised_u = _denoise_candidate(noisy, selected, config)
        denoised_by_step[int(step)] = denoised_u
        denoised = _load_fem_quiet(data_dir / str(step), denoised_displacements=denoised_u)
        loadstep_metrics.append(
            {
                "method": config.method,
                "loadstep": int(step),
                **_candidate_row(selected),
                **_metrics(clean, denoised),
                **_physics_checks(denoised),
            }
        )

    output_paths = write_denoised_fem_dataset(
        data_dir,
        output_dir,
        denoised_by_step,
        overwrite=config.overwrite,
    )
    search_csv_path = output_dir / "denoise_search.csv"
    loadstep_metrics_csv_path = output_dir / "denoise_loadstep_metrics.csv"
    summary_path = output_dir / "denoise_summary.json"
    _write_dict_csv(search_csv_path, search_rows)
    _write_dict_csv(loadstep_metrics_csv_path, loadstep_metrics)

    baseline_metrics = _mean_metrics(baseline_rows)
    summary = {
        "method": config.method,
        "config": _jsonable_config(config),
        "loadsteps": list(steps),
        "selected_candidate": asdict(selected),
        "selected_score": float(selected_aggregate["mean_%s" % config.objective]),
        "objective": config.objective,
        "baseline_metrics": baseline_metrics,
        "selected_metrics": _mean_metrics(loadstep_metrics),
        "output_paths": {
            **output_paths,
            "summary_json": str(summary_path),
            "search_csv": str(search_csv_path),
            "loadstep_metrics_csv": str(loadstep_metrics_csv_path),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return DenoiseResult(
        config=config,
        loadsteps=steps,
        selected_candidate=selected,
        selected_score=float(selected_aggregate["mean_%s" % config.objective]),
        baseline_metrics=baseline_metrics,
        search_rows=search_rows,
        loadstep_metrics=loadstep_metrics,
        output_dir=str(output_dir),
        summary_path=str(summary_path),
        search_csv_path=str(search_csv_path),
        loadstep_metrics_csv_path=str(loadstep_metrics_csv_path),
    )


def search_krr_hyperparameters(config: DenoiseSearchConfig) -> DenoiseResult:
    """Backward-compatible wrapper for RBF-KRR denoising search."""
    return search_denoise_hyperparameters(replace(config, method="krr"))


def write_denoised_fem_dataset(
    data_dir: str | Path,
    output_dir: str | Path,
    denoised_displacements_by_step: dict[int, np.ndarray],
    *,
    overwrite: bool = False,
) -> dict[str, str]:
    """Write a SGEPPY-ready FEM dataset with smoothed nodal displacements."""
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError("Output directory already exists: %s" % output_dir)
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for child in data_dir.iterdir():
        if child.is_file():
            shutil.copy2(child, output_dir / child.name)

    step_paths = {}
    for step, denoised_u in denoised_displacements_by_step.items():
        source_step_dir = data_dir / str(int(step))
        target_step_dir = output_dir / str(int(step))
        shutil.copytree(source_step_dir, target_step_dir)
        nodes_path = target_step_dir / "output_nodes.csv"
        nodes = pd.read_csv(nodes_path)
        denoised_u = np.asarray(denoised_u, dtype=float)
        if denoised_u.shape != (len(nodes), 2):
            raise ValueError("Denoised displacement shape does not match %s." % nodes_path)
        nodes["ux"] = denoised_u[:, 0]
        nodes["uy"] = denoised_u[:, 1]
        nodes.to_csv(nodes_path, index=False)
        step_paths[str(int(step))] = str(target_step_dir)

    manifest = {
        "generator": "sym_modeling.domains.fem.methods.common.denoising",
        "source_data_dir": str(data_dir),
        "load_steps": [
            {"load_step": int(step), "path": str(output_dir / str(int(step)))}
            for step in sorted(denoised_displacements_by_step)
        ],
    }
    manifest_path = output_dir / "denoise_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {"output_dir": str(output_dir), "manifest_json": str(manifest_path), "step_dirs": step_paths}


def _load_fem_quiet(path: Path, denoised_displacements: np.ndarray | None = None):
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        return loadFemData(
            str(path),
            AD=True,
            noiseLevel=0.0,
            noiseType="displacement",
            denoisedDisplacements=denoised_displacements,
        )


def _loadstep_seed(seed: int, step_index: int, step: int, noise_level: float) -> int:
    return int(seed) + 1009 * int(step_index) + 9176 * int(step) + int(round(float(noise_level) * 1e12))


def _metrics(clean, candidate) -> dict[str, float]:
    return {
        "u_rmse": _rmse(candidate.u_nodes, clean.u_nodes),
        "F_rmse": _rmse(candidate.F, clean.F),
        "J_rmse": _rmse(candidate.J, clean.J),
        "I1_rmse": _rmse(candidate.I1, clean.I1),
        "I2_rmse": _rmse(candidate.I2, clean.I2),
        "I3_rmse": _rmse(candidate.I3, clean.I3),
    }


def _rmse(values: np.ndarray, reference: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(values, dtype=float) - np.asarray(reference, dtype=float)) ** 2)))


def _build_candidates(config: DenoiseSearchConfig) -> list[DenoiseCandidate | MeshLaplacianCandidate]:
    if config.method == "krr":
        return [
            DenoiseCandidate(alpha=float(alpha), gamma=float(gamma), blend=float(blend))
            for alpha in config.alphas
            for gamma in config.gammas
            for blend in config.blends
        ]
    if config.method == "mesh-laplacian":
        return [
            MeshLaplacianCandidate(lambda_smooth=float(lambda_smooth), blend=float(blend))
            for lambda_smooth in config.lambdas
            for blend in config.blends
        ]
    raise ValueError("Unsupported denoising method: %s" % config.method)


def _denoise_candidate(clean_or_noisy, candidate, config: DenoiseSearchConfig) -> np.ndarray:
    if isinstance(candidate, DenoiseCandidate):
        return denoise_displacements_krr(
            clean_or_noisy.x_nodes,
            clean_or_noisy.u_nodes,
            clean_or_noisy.dirichlet_nodes,
            candidate,
            boundary_weight=config.boundary_weight,
            preserve_dirichlet=config.preserve_dirichlet,
        )
    if isinstance(candidate, MeshLaplacianCandidate):
        return denoise_displacements_mesh_laplacian(
            clean_or_noisy.u_nodes,
            clean_or_noisy.dirichlet_nodes,
            clean_or_noisy.connectivity,
            clean_or_noisy.gradNa,
            clean_or_noisy.qpWeights,
            candidate,
            preserve_dirichlet=config.preserve_dirichlet,
        )
    raise TypeError("Unsupported denoise candidate: %r" % (candidate,))


def _candidate_row(candidate: DenoiseCandidate | MeshLaplacianCandidate) -> dict[str, float]:
    if isinstance(candidate, DenoiseCandidate):
        return {
            "alpha": float(candidate.alpha),
            "gamma": float(candidate.gamma),
            "blend": float(candidate.blend),
        }
    return {
        "lambda_smooth": float(candidate.lambda_smooth),
        "blend": float(candidate.blend),
    }


def _candidate_from_row(row: dict, method: str) -> DenoiseCandidate | MeshLaplacianCandidate:
    if method == "krr":
        return DenoiseCandidate(
            alpha=float(row["alpha"]),
            gamma=float(row["gamma"]),
            blend=float(row["blend"]),
        )
    if method == "mesh-laplacian":
        return MeshLaplacianCandidate(
            lambda_smooth=float(row["lambda_smooth"]),
            blend=float(row["blend"]),
        )
    raise ValueError("Unsupported denoising method: %s" % method)


def _aggregate_search_rows(rows: list[dict], objective: str, method: str) -> list[dict]:
    groups: dict[tuple[float, ...], list[dict]] = {}
    for row in rows:
        key = _candidate_key(row, method)
        groups.setdefault(key, []).append(row)
    aggregate = []
    for key, group_rows in groups.items():
        row = _candidate_aggregate_key_row(key, method)
        row.update(
            {
                "mean_%s" % objective: float(np.mean([item[objective] for item in group_rows])),
                "J_min": float(np.min([item["J_min"] for item in group_rows])),
                "J_nonpositive_count": int(np.sum([item["J_nonpositive_count"] for item in group_rows])),
            }
        )
        aggregate.append(row)
    return aggregate


def _mean_metrics(rows: list[dict]) -> dict[str, float]:
    metric_keys = [key for key in OBJECTIVES if rows and key in rows[0]]
    metrics = {key: float(np.mean([row[key] for row in rows])) for key in metric_keys}
    if rows and "J_min" in rows[0]:
        metrics["J_min"] = float(np.min([row["J_min"] for row in rows]))
    if rows and "J_nonpositive_count" in rows[0]:
        metrics["J_nonpositive_count"] = int(np.sum([row["J_nonpositive_count"] for row in rows]))
    return metrics


def _write_dict_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _jsonable_config(config: DenoiseSearchConfig) -> dict:
    payload = asdict(config)
    payload["data_dir"] = str(config.data_dir)
    payload["output_dir"] = str(config.output_dir)
    payload["loadsteps"] = None if config.loadsteps is None else [int(step) for step in config.loadsteps]
    payload["alphas"] = [float(value) for value in config.alphas]
    payload["gammas"] = [float(value) for value in config.gammas]
    payload["lambdas"] = [float(value) for value in config.lambdas]
    payload["blends"] = [float(value) for value in config.blends]
    return payload


def _connectivity_by_element(connectivity: Sequence[np.ndarray]) -> np.ndarray:
    if len(connectivity) != 3:
        raise ValueError("connectivity must contain three node-index arrays.")
    by_element = np.column_stack([np.asarray(nodes, dtype=int) for nodes in connectivity])
    if by_element.ndim != 2 or by_element.shape[1] != 3:
        raise ValueError("connectivity must have shape (num_elements, 3).")
    return by_element


def _grad_na_by_element(grad_na: Sequence[np.ndarray]) -> np.ndarray:
    if len(grad_na) != 3:
        raise ValueError("grad_na must contain three gradient arrays.")
    return np.stack([np.asarray(grad, dtype=float) for grad in grad_na], axis=1)


def _normalize_mass_and_stiffness(mass: np.ndarray, stiffness: sparse.csr_matrix) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
    mass = np.asarray(mass, dtype=float)
    positive_mass = mass[mass > 0.0]
    if positive_mass.size == 0:
        raise ValueError("lumped mass diagonal is zero everywhere.")
    mass_scale = float(np.mean(positive_mass))
    stiffness_diag = np.asarray(stiffness.diagonal(), dtype=float)
    positive_stiffness_diag = stiffness_diag[stiffness_diag > 0.0]
    stiffness_scale = float(np.mean(positive_stiffness_diag)) if positive_stiffness_diag.size else 1.0
    mass_matrix = sparse.diags(mass / mass_scale, format="csr")
    return mass_matrix, (stiffness / stiffness_scale).tocsr()


def _solve_scalar_smoothing(
    system: sparse.csr_matrix,
    mass_matrix: sparse.csr_matrix,
    values: np.ndarray,
    constrained: np.ndarray,
) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    constrained = np.asarray(constrained, dtype=bool)
    free = np.logical_not(constrained)
    result = np.array(values, copy=True)
    if not np.any(free):
        return result

    rhs = mass_matrix @ values
    free_index = np.flatnonzero(free)
    if np.any(constrained):
        constrained_index = np.flatnonzero(constrained)
        rhs_free = rhs[free_index] - system[free_index][:, constrained_index] @ values[constrained_index]
    else:
        rhs_free = rhs[free_index]
    result[free_index] = sparse_linalg.spsolve(system[free_index][:, free_index].tocsc(), rhs_free)
    result[constrained] = values[constrained]
    return result


def _physics_checks(dataset) -> dict[str, float | int]:
    j_values = np.asarray(dataset.J, dtype=float)
    return {
        "J_min": float(np.min(j_values)),
        "J_nonpositive_count": int(np.count_nonzero(j_values <= 0.0)),
    }


def _candidate_key(row: dict, method: str) -> tuple[float, ...]:
    if method == "krr":
        return (float(row["alpha"]), float(row["gamma"]), float(row["blend"]))
    if method == "mesh-laplacian":
        return (float(row["lambda_smooth"]), float(row["blend"]))
    raise ValueError("Unsupported denoising method: %s" % method)


def _candidate_aggregate_key_row(key: tuple[float, ...], method: str) -> dict[str, float]:
    if method == "krr":
        alpha, gamma, blend = key
        return {"alpha": alpha, "gamma": gamma, "blend": blend}
    if method == "mesh-laplacian":
        lambda_smooth, blend = key
        return {"lambda_smooth": lambda_smooth, "blend": blend}
    raise ValueError("Unsupported denoising method: %s" % method)


def _valid_physics_rows_or_all(rows: list[dict]) -> list[dict]:
    valid = [row for row in rows if int(row.get("J_nonpositive_count", 0)) == 0]
    return valid if valid else rows
