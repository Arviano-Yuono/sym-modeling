from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from sym_modeling.domains.fem.methods.common.weak_form import weak_residual_vector
from sym_modeling.domains.fem.methods.euclid.config import (
    FORWARD_MODEL_NAMES,
    EuclidConfig,
    normalize_euclid_model_name,
)
from sym_modeling.domains.fem.methods.euclid.feature_library import (
    formatFeatureExpression,
)
from sym_modeling.domains.fem.methods.euclid.workflow import EuclidWorkflow


DEFAULT_DATA_ROOT = "dataset/fem_data/plate_hole_fenics"
DEFAULT_OUTPUT_ROOT = "output/euclid_results/plate_hole_fenics"
DEFAULT_MODELS = ",".join(FORWARD_MODEL_NAMES)


def _parse_csv_strings(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def _infer_loadsteps(model_dir: Path) -> list[int]:
    loadsteps = sorted(
        int(path.name)
        for path in model_dir.iterdir()
        if path.is_dir() and path.name.isdigit()
    )
    if not loadsteps:
        raise FileNotFoundError("No numeric load-step folders found under %s" % model_dir)
    return loadsteps


def _count_active_terms(theta: np.ndarray, active_threshold: float) -> int:
    return int(np.sum(np.abs(theta) > active_threshold))


def _weak_form_metrics(
    workflow: EuclidWorkflow,
    theta: np.ndarray,
    active_threshold: float,
) -> dict:
    residual = weak_residual_vector(workflow.datasets, theta, workflow.config)
    rss = float(np.dot(residual, residual))
    residual_count = int(residual.size)
    rmse = math.sqrt(rss / residual_count) if residual_count else 0.0
    penalty = float(workflow.config.penaltyLp_init) * float(
        np.sum(np.power(np.abs(theta), float(workflow.config.p)))
    )
    return {
        "rss": rss,
        "rmse": rmse,
        "lp_penalty": penalty,
        "cost": rss + penalty,
        "residual_count": residual_count,
        "num_parameters": _count_active_terms(theta, active_threshold),
    }


def _system_diagnostics(lhs: np.ndarray, rhs: np.ndarray) -> dict:
    return {
        "lhs_shape": list(lhs.shape),
        "rhs_shape": list(rhs.shape),
        "rank": int(np.linalg.matrix_rank(lhs)),
        "condition_number": float(np.linalg.cond(lhs)),
    }


def _write_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=True) + "\n", encoding="utf-8")


def run_model(
    model: str,
    data_root: Path,
    output_root: Path,
    noise_level: float,
    active_threshold: float,
) -> dict:
    model_name = normalize_euclid_model_name(model)
    model_dir = data_root / model_name
    if not model_dir.is_dir():
        raise FileNotFoundError("Missing FEM data directory for %s: %s" % (model_name, model_dir))

    loadsteps = _infer_loadsteps(model_dir)
    model_output_dir = output_root / model_name.lower()
    legacy_results_dir = model_output_dir / "legacy_results"
    config = EuclidConfig(
        str_model=model_name,
        str_mesh=data_root.name,
        noiseLevel=noise_level,
        femDataPathOverride=str(model_dir),
        loadstepsOverride=loadsteps,
        resultsDir=str(legacy_results_dir),
        appendResults=False,
    )

    print("[euclid] running %s with load steps %s" % (model_name, loadsteps))
    start_wall = time.perf_counter()
    start_cpu = time.process_time()
    workflow = EuclidWorkflow(config=config)
    result = workflow.train()
    wall_seconds = time.perf_counter() - start_wall
    cpu_seconds = time.process_time() - start_cpu

    theta = np.asarray(result.theta, dtype=float).reshape(-1)
    expression = formatFeatureExpression(theta, active_threshold=active_threshold)
    summary_path = model_output_dir / "summary.json"
    legacy_results_file = legacy_results_dir / ("results_%s.txt" % model_name)
    payload = {
        "method": "euclid",
        "model": model_name,
        "experimental": model_name == "AB",
        "dataset": str(model_dir),
        "data_dir": str(model_dir),
        "loadsteps": loadsteps,
        "best_expression": expression,
        "theta": theta.tolist(),
        "metrics": _weak_form_metrics(workflow, theta, active_threshold),
        "system_diagnostics": _system_diagnostics(result.lhs, result.rhs),
        "timing": {
            "wall_seconds": wall_seconds,
            "cpu_seconds": cpu_seconds,
        },
        "config": {
            "balance": config.balance,
            "penalty_lp": config.penaltyLp_init,
            "p": config.p,
            "num_increments": config.numIncrements,
            "factor_increments": config.factorIncrements,
            "num_guesses": config.numGuesses,
            "num_iterations": config.numIterations,
            "threshold_iter": config.threshold_iter,
            "threshold": config.threshold,
            "noise_level": noise_level,
        },
        "output_paths": {
            "summary_json": str(summary_path),
            "legacy_results_txt": str(legacy_results_file),
        },
    }
    if model_name == "AB":
        payload["notes"] = [
            "AB is fitted with the current fixed EUCLID feature library as an experimental comparison.",
        ]

    _write_json(payload, summary_path)
    print("[euclid] %s W = %s" % (model_name, expression))
    print("[euclid] wrote %s" % summary_path)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run weak-form EUCLID discovery on FEM dataset folders."
    )
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--models",
        default=DEFAULT_MODELS,
        help="Comma-separated model folders to run.",
    )
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--noise-level", type=float, default=0.0)
    parser.add_argument(
        "--active-threshold",
        type=float,
        default=1e-12,
        help="Absolute coefficient threshold used only when formatting expressions.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    data_root = Path(args.data_root)
    output_root = Path(args.output_root)
    models = _parse_csv_strings(args.models)
    if not models:
        raise ValueError("At least one EUCLID model must be provided.")

    output_root.mkdir(parents=True, exist_ok=True)
    summaries = []
    for model in models:
        summaries.append(
            run_model(
                model=model,
                data_root=data_root,
                output_root=output_root,
                noise_level=float(args.noise_level),
                active_threshold=float(args.active_threshold),
            )
        )

    index_path = output_root / "summary.json"
    _write_json(
        {
            "method": "euclid",
            "data_root": str(data_root),
            "models": [summary["model"] for summary in summaries],
            "summary_paths": [
                summary["output_paths"]["summary_json"] for summary in summaries
            ],
        },
        index_path,
    )
    print("[euclid] wrote batch summary: %s" % index_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
