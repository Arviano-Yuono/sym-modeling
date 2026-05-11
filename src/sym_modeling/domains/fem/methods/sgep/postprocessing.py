from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np

from sym_modeling.domains.fem.io.csv_loader import loadFemData
from sym_modeling.domains.fem.methods.euclid.postprocessing import (
    DEFAULT_PLOT_QUANTITIES,
    getFieldLabel,
    getFieldValues,
    plotFieldComparison,
)
from sym_modeling.domains.fem.methods.sgep.hyperelastic_eval import (
    build_stress_dataset_from_fem_data,
    resolve_loadsteps,
    stress_feature_for_gene,
)


def save_json(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_history_csv(path: str | Path, history: Sequence[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not history:
        path.write_text("", encoding="utf-8")
        return
    keys = list(history[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(history)


def plot_predicted_vs_true(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: str | Path,
) -> str:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(5, 5), constrained_layout=True)
    axis.scatter(y_true, y_pred, s=8, alpha=0.45)
    lower = float(min(np.min(y_true), np.min(y_pred)))
    upper = float(max(np.max(y_true), np.max(y_pred)))
    if np.isclose(lower, upper):
        lower -= 1.0
        upper += 1.0
    axis.plot([lower, upper], [lower, upper], color="black", linewidth=1.0)
    axis.set_xlabel("True Piola component")
    axis.set_ylabel("Predicted Piola component")
    axis.set_title("SGEP stress fit")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return str(output_path)


def plot_history(history: Sequence[dict], output_path: str | Path) -> str:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    generations = [entry["generation"] for entry in history]
    rmse = [entry["best_rmse"] for entry in history]
    aicc = [entry["best_aicc"] for entry in history]

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), constrained_layout=True)
    axes[0].plot(generations, rmse, marker="o")
    axes[0].set_xlabel("Generation")
    axes[0].set_ylabel("Best RMSE")
    axes[1].plot(generations, aicc, marker="o")
    axes[1].set_xlabel("Generation")
    axes[1].set_ylabel("Best AICc")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return str(output_path)


def compute_sgep_piola(
    data,
    genes,
    theta: np.ndarray,
    config=None,
    variable_names: Sequence[str] | None = None,
    derivative_step: float | None = None,
    value_limit: float | None = None,
    threshold: float | None = None,
) -> np.ndarray:
    if config is not None:
        variable_names = variable_names or config.variable_names
        derivative_step = derivative_step if derivative_step is not None else config.derivative_step
        value_limit = value_limit if value_limit is not None else config.invalid_value_limit
        threshold = threshold if threshold is not None else config.sparsity_threshold
    if variable_names is None:
        variable_names = ("K1", "K2", "Jm1")
    if derivative_step is None:
        derivative_step = 1e-6
    if value_limit is None:
        value_limit = 1e8
    if threshold is None:
        threshold = 0.0

    dataset = build_stress_dataset_from_fem_data(data)
    modeled = np.zeros((dataset.num_points, 4), dtype=float)
    for coefficient, gene in zip(np.asarray(theta, dtype=float), genes):
        if abs(float(coefficient)) <= threshold:
            continue
        stress = stress_feature_for_gene(
            gene,
            dataset,
            variable_names,
            derivative_step=derivative_step,
            value_limit=value_limit,
        )
        if stress is None:
            raise ValueError("Active SGEP gene became invalid on plotting dataset: %s" % gene.expression())
        modeled += float(coefficient) * stress
    return modeled


def plot_sgep_piola_field_comparisons(
    data_dir: str | Path,
    loadsteps: Sequence[int] | None,
    genes,
    theta: np.ndarray,
    config,
    output_dir: str | Path,
    quantities: Sequence[str] = DEFAULT_PLOT_QUANTITIES,
) -> tuple[dict[str, list[str]], list[dict], list[str]]:
    data_root = Path(data_dir)
    steps = list(loadsteps) if loadsteps is not None else resolve_loadsteps(data_root)
    plot_root = Path(output_dir)
    plot_paths: dict[str, list[str]] = {}
    metrics: list[dict] = []
    warnings: list[str] = []

    for loadstep in steps:
        case_dir = data_root / str(loadstep)
        data = loadFemData(
            str(case_dir),
            AD=True,
            noiseLevel=0.0,
            noiseType="displacement",
        )
        data.convertToNumpy()
        if data.P is None:
            warnings.append("Skipping load step %s because reference Piola stresses are missing." % loadstep)
            continue

        try:
            modeled = compute_sgep_piola(data, genes, theta, config=config)
        except ValueError as exc:
            warnings.append("Skipping load step %s: %s" % (loadstep, exc))
            continue

        error = modeled - data.P
        metrics.append(
            {
                "loadstep": int(loadstep),
                "rmse": float(np.sqrt(np.mean(np.square(error)))),
                "mae": float(np.mean(np.abs(error))),
                "max_abs_error": float(np.max(np.abs(error))),
            }
        )

        loadstep_dir = plot_root / ("loadstep_%s" % loadstep)
        paths = []
        for quantity in quantities:
            output_path = loadstep_dir / ("loadstep_%s_%s.png" % (loadstep, quantity))
            plotFieldComparison(
                data,
                getFieldValues(data.P, quantity),
                getFieldValues(modeled, quantity),
                quantity,
                str(output_path),
                title="Load step %s | %s" % (loadstep, getFieldLabel(quantity)),
            )
            paths.append(str(output_path))
        plot_paths[str(loadstep)] = paths

    return plot_paths, metrics, warnings
