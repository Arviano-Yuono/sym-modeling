"""
uv run python scripts/run_euclid_thermocorr.py \
  --data-dir dataset/fem_data/thermocorr/formatted_sampled_3k \
  --loadsteps 10,20,30,40,50 \
  --output-dir output/euclid/thermocorr
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from sym_modeling.domains.fem.methods.common.weak_form import weak_residual_vector  # noqa: E402
from sym_modeling.domains.fem.methods.euclid import EuclidConfig, EuclidWorkflow  # noqa: E402
from sym_modeling.domains.fem.methods.euclid.feature_library import formatFeatureExpression  # noqa: E402


DEFAULT_DATA_DIR = "dataset/fem_data/thermocorr/formatted_sampled_3k"
DEFAULT_OUTPUT_DIR = "output/euclid/thermocorr"
DEFAULT_LOADSTEPS = (10, 20, 30, 40, 50)


def _parse_csv_ints(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def _infer_loadsteps(data_dir: Path) -> list[int]:
    loadsteps = sorted(
        int(path.name)
        for path in data_dir.iterdir()
        if path.is_dir() and path.name.isdigit()
    )
    if not loadsteps:
        raise FileNotFoundError("No numeric load-step folders found under %s" % data_dir)
    return loadsteps


def _loadsteps_from_arg(value: str, data_dir: Path) -> list[int]:
    if value.strip().lower() == "all":
        return _infer_loadsteps(data_dir)
    loadsteps = _parse_csv_ints(value)
    if not loadsteps:
        raise ValueError("--loadsteps must contain at least one integer or 'all'.")
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run EUCLID weak-form discovery on the formatted ThermoCorr FEM dataset."
    )
    parser.add_argument(
        "--data-dir",
        default=DEFAULT_DATA_DIR,
        help="ThermoCorr dataset root containing numeric load-step folders.",
    )
    parser.add_argument(
        "--loadsteps",
        default=",".join(str(step) for step in DEFAULT_LOADSTEPS),
        help="Comma-separated load steps, or 'all' to use every numeric folder.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for summary.json and EUCLID legacy result text.",
    )
    parser.add_argument(
        "--model-label",
        default="NH2",
        help="Valid EUCLID label used for result naming. The feature library is fixed.",
    )
    parser.add_argument("--noise-level", type=float, default=0.0)
    parser.add_argument("--balance", type=float, default=100.0)
    parser.add_argument("--penalty-lp", type=float, default=1e-4)
    parser.add_argument("--p", type=float, default=1.0 / 4.0)
    parser.add_argument("--num-increments", type=int, default=5)
    parser.add_argument("--factor-increments", type=float, default=5.0)
    parser.add_argument("--num-guesses", type=int, default=1)
    parser.add_argument("--num-iterations", type=int, default=200)
    parser.add_argument(
        "--active-threshold",
        type=float,
        default=1e-12,
        help="Absolute coefficient threshold used when formatting the expression.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved run settings without fitting EUCLID.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    loadsteps = _loadsteps_from_arg(args.loadsteps, data_dir)

    config = EuclidConfig(
        str_model=args.model_label,
        str_mesh="thermocorr",
        noiseLevel=float(args.noise_level),
        femDataPathOverride=str(data_dir),
        loadstepsOverride=loadsteps,
        resultsDir=str(output_dir / "legacy_results"),
        appendResults=False,
        balance=float(args.balance),
        penaltyLp=float(args.penalty_lp),
        p=float(args.p),
        numIncrements=int(args.num_increments),
        factorIncrements=float(args.factor_increments),
        numGuesses=int(args.num_guesses),
        numIterations=int(args.num_iterations),
    )

    print(
        "[euclid-thermocorr] data=%s loadsteps=%s output=%s"
        % (data_dir, loadsteps, output_dir),
        flush=True,
    )
    if args.dry_run:
        return 0

    start_wall = time.perf_counter()
    start_cpu = time.process_time()
    workflow = EuclidWorkflow(config=config)
    result = workflow.train()
    wall_seconds = time.perf_counter() - start_wall
    cpu_seconds = time.process_time() - start_cpu

    theta = np.asarray(result.theta, dtype=float).reshape(-1)
    expression = formatFeatureExpression(theta, active_threshold=float(args.active_threshold))
    summary_path = output_dir / "summary.json"
    legacy_results_file = output_dir / "legacy_results" / ("results_%s.txt" % config.saveResultsName)
    payload = {
        "method": "euclid",
        "dataset": str(data_dir),
        "data_dir": str(data_dir),
        "loadsteps": loadsteps,
        "best_expression": expression,
        "theta": theta.tolist(),
        "metrics": _weak_form_metrics(workflow, theta, float(args.active_threshold)),
        "system_diagnostics": _system_diagnostics(result.lhs, result.rhs),
        "timing": {
            "wall_seconds": wall_seconds,
            "cpu_seconds": cpu_seconds,
        },
        "config": {
            "model_label": config.str_model,
            "balance": config.balance,
            "penalty_lp": config.penaltyLp_init,
            "p": config.p,
            "num_increments": config.numIncrements,
            "factor_increments": config.factorIncrements,
            "num_guesses": config.numGuesses,
            "num_iterations": config.numIterations,
            "noise_level": config.noiseLevel,
        },
        "output_paths": {
            "summary_json": str(summary_path),
            "legacy_results_txt": str(legacy_results_file),
        },
    }
    _write_json(payload, summary_path)

    print("[euclid-thermocorr] W = %s" % expression, flush=True)
    print("[euclid-thermocorr] wrote %s" % summary_path, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
