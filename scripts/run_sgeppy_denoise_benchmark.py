"""Benchmark clean, artificially noisy, and denoised FEM data with matched SGEPPY seeds.

The script deliberately materializes the noisy displacement field before running
SGEPPY.  Every condition therefore reads a fixed dataset with ``noise_level=0``.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import sympy as sp


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from sym_modeling.domains.fem.methods.common.denoising import (  # noqa: E402
    DEFAULT_ALPHAS,
    DEFAULT_BLENDS,
    DEFAULT_GAMMAS,
    DEFAULT_LAPLACIAN_LAMBDAS,
    DenoiseSearchConfig,
    search_denoise_hyperparameters,
    write_artificially_noised_fem_dataset,
)
from sym_modeling.domains.fem.methods.sgeppy.run_gep_sparse import config_from_file  # noqa: E402
from sym_modeling.domains.fem.methods.sgeppy.workflow import SGEPWorkflow  # noqa: E402


DEFAULT_MODELS = ("gt", "hw", "ih", "nh2", "nh4", "ab")
DEFAULT_NOISES = ("1e-4", "1e-3")
DEFAULT_BENCHMARK_ALPHAS = tuple(value for value in DEFAULT_ALPHAS if value >= 1e-8)
DEFAULT_BENCHMARK_WEIGHTS = {
    "F_rmse": 4.0,
    "J_rmse": 1.0,
    "I1_rmse": 0.5,
    "I2_rmse": 0.5,
    "I3_rmse": 0.5,
    "u_rmse": 0.25,
}
MODEL_DIRS = {
    "gt": "GT",
    "hw": "HW",
    "ih": "IH",
    "nh2": "NH2",
    "nh4": "NH4",
    "ab": "AB",
}
STAGES = ("datasets", "sgeppy")


def _parse_csv_strings(value: str) -> list[str]:
    return [part.strip().lower() for part in value.split(",") if part.strip()]


def _parse_csv_ints(value: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def _parse_csv_floats(value: str) -> tuple[float, ...]:
    return tuple(float(part.strip()) for part in value.split(",") if part.strip())


def _noise_label(value: str) -> str:
    text = value.strip().lower()
    return "0" if float(text) == 0.0 else text


def _parse_objective_weights(value: str) -> dict[str, float]:
    weights = {}
    for item in value.split(","):
        key, separator, raw_value = item.strip().partition("=")
        if not separator:
            raise ValueError("Objective weights must use metric=value entries.")
        weights[key.strip()] = float(raw_value.strip())
    return weights


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _dataset_dir(dataset_root: Path, condition: str, noise: str, model_dir: str, noise_seed: int) -> Path:
    if condition == "clean":
        raise ValueError("Clean datasets are read directly from --clean-data-root.")
    return dataset_root / ("noise_seed_%d" % noise_seed) / condition / noise / model_dir


def _run_output_dir(
    output_root: Path,
    condition: str,
    model_name: str,
    sgeppy_seed: int,
    *,
    noise: str | None = None,
    noise_seed: int | None = None,
) -> Path:
    if condition == "clean":
        return output_root / "clean" / model_name / ("seed_%d" % sgeppy_seed)
    assert noise is not None and noise_seed is not None
    return (
        output_root
        / ("noise_seed_%d" % noise_seed)
        / condition
        / noise
        / model_name
        / ("seed_%d" % sgeppy_seed)
    )


def _expression_terms(expression: str, variable_names: tuple[str, ...]) -> dict[str, float] | None:
    """Return additive term coefficients after treating protected operators algebraically."""
    symbols = {name: sp.Symbol(name) for name in variable_names}
    parser_locals = {
        **symbols,
        "protected_div": lambda left, right: left / right,
        "protected_sqrt": sp.sqrt,
        "protected_log": sp.log,
        "protected_exp": sp.exp,
        "sin": sp.sin,
        "cos": sp.cos,
    }
    try:
        parsed = sp.expand(sp.simplify(sp.sympify(expression, locals=parser_locals)))
    except Exception:
        return None

    terms = {}
    for term in sp.Add.make_args(parsed):
        coefficient, basis = term.as_coeff_Mul()
        try:
            value = float(coefficient)
        except (TypeError, ValueError):
            return None
        key = str(sp.simplify(basis))
        terms[key] = terms.get(key, 0.0) + value
    return {key: value for key, value in terms.items() if abs(value) > 1e-12}


def compare_expressions(
    clean_expression: str,
    candidate_expression: str,
    variable_names: tuple[str, ...],
) -> dict[str, object]:
    clean_terms = _expression_terms(clean_expression, variable_names)
    candidate_terms = _expression_terms(candidate_expression, variable_names)
    if clean_terms is None or candidate_terms is None:
        return {"same_structure": False, "coefficient_relative_l2": ""}

    same_structure = set(clean_terms) == set(candidate_terms)
    coefficient_error: float | str = ""
    if same_structure:
        keys = sorted(clean_terms)
        clean_coefficients = np.asarray([clean_terms[key] for key in keys], dtype=float)
        candidate_coefficients = np.asarray([candidate_terms[key] for key in keys], dtype=float)
        scale = max(float(np.linalg.norm(clean_coefficients)), 1e-12)
        coefficient_error = float(np.linalg.norm(candidate_coefficients - clean_coefficients) / scale)
    return {
        "same_structure": same_structure,
        "coefficient_relative_l2": coefficient_error,
        "clean_terms": json.dumps(clean_terms, sort_keys=True),
        "candidate_terms": json.dumps(candidate_terms, sort_keys=True),
    }


def _summary_row(
    payload: dict,
    *,
    model_name: str,
    condition: str,
    dataset_dir: Path,
    output_dir: Path,
    sgeppy_seed: int,
    noise: str | None,
    noise_seed: int | None,
) -> dict:
    metrics = payload["metrics"]
    return {
        "status": "ok",
        "model": model_name,
        "condition": condition,
        "noise_level": "" if noise is None else float(noise),
        "noise_seed": "" if noise_seed is None else int(noise_seed),
        "sgeppy_seed": int(sgeppy_seed),
        "dataset_dir": str(dataset_dir),
        "output_dir": str(output_dir),
        "best_expression": payload["best_expression"],
        "rmse": float(metrics["rmse"]),
        "rss": float(metrics["rss"]),
        "num_parameters": int(metrics["num_parameters"]),
        "wall_seconds": float(payload.get("timing", {}).get("wall_seconds", 0.0)),
    }


def _load_existing_run_row(**kwargs) -> dict:
    output_dir = Path(kwargs["output_dir"])
    payload = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    return _summary_row(payload, **kwargs)


def _comparison_rows(run_rows: list[dict], variable_names_by_model: dict[str, tuple[str, ...]]) -> list[dict]:
    clean_rows = {
        (str(row["model"]), int(row["sgeppy_seed"])): row
        for row in run_rows
        if row.get("status") == "ok" and row["condition"] == "clean"
    }
    comparisons = []
    for row in run_rows:
        if row.get("status") != "ok" or row["condition"] == "clean":
            continue
        clean = clean_rows.get((str(row["model"]), int(row["sgeppy_seed"])))
        if clean is None:
            continue
        comparison = compare_expressions(
            str(clean["best_expression"]),
            str(row["best_expression"]),
            variable_names_by_model[str(row["model"])],
        )
        comparisons.append(
            {
                "model": row["model"],
                "condition": row["condition"],
                "noise_level": row["noise_level"],
                "noise_seed": row["noise_seed"],
                "sgeppy_seed": row["sgeppy_seed"],
                "same_structure": comparison["same_structure"],
                "coefficient_relative_l2": comparison["coefficient_relative_l2"],
                "rmse_ratio_to_clean": float(row["rmse"]) / max(float(clean["rmse"]), 1e-15),
                "clean_expression": clean["best_expression"],
                "candidate_expression": row["best_expression"],
                "clean_terms": comparison.get("clean_terms", ""),
                "candidate_terms": comparison.get("candidate_terms", ""),
            }
        )
    return comparisons


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate deterministic noisy/denoised FEM datasets and benchmark them with matched SGEPPY seeds."
    )
    parser.add_argument("--config-dir", default="configs/sgeppy")
    parser.add_argument("--clean-data-root", default="dataset/fem_data/plate/0")
    parser.add_argument("--dataset-root", default="tmp/sgeppy_denoise_benchmark/datasets")
    parser.add_argument("--output-root", default="tmp/sgeppy_denoise_benchmark/runs")
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS))
    parser.add_argument("--noises", default=",".join(DEFAULT_NOISES))
    parser.add_argument("--noise-seeds", default="20260623")
    parser.add_argument("--sgeppy-seeds", default="0")
    parser.add_argument("--stages", default=",".join(STAGES), help="Comma-separated stages: datasets,sgeppy.")
    parser.add_argument("--method", choices=("krr", "mesh-laplacian"), default="krr")
    parser.add_argument("--objective", choices=("F_rmse", "composite"), default="composite")
    parser.add_argument(
        "--objective-weights",
        default=",".join("%s=%g" % item for item in DEFAULT_BENCHMARK_WEIGHTS.items()),
    )
    parser.add_argument("--selection-scope", choices=("global", "per-loadstep"), default="per-loadstep")
    parser.add_argument("--alphas", default=",".join("%g" % value for value in DEFAULT_BENCHMARK_ALPHAS))
    parser.add_argument("--gammas", default=",".join("%g" % value for value in DEFAULT_GAMMAS))
    parser.add_argument("--lambdas", default=",".join("%g" % value for value in DEFAULT_LAPLACIAN_LAMBDAS))
    parser.add_argument("--blends", default=",".join("%g" % value for value in DEFAULT_BLENDS))
    parser.add_argument("--backend", choices=("jax", "torch"), default=None)
    parser.add_argument("--generations", type=int, default=None)
    parser.add_argument("--population-size", type=int, default=None)
    parser.add_argument("--quiet", action="store_true", default=False)
    parser.add_argument("--overwrite-datasets", action="store_true", default=False)
    parser.add_argument("--skip-existing", action="store_true", default=False)
    parser.add_argument("--keep-going", action="store_true", default=False)
    parser.add_argument("--dry-run", action="store_true", default=False)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config_dir = Path(args.config_dir)
    clean_data_root = Path(args.clean_data_root)
    dataset_root = Path(args.dataset_root)
    output_root = Path(args.output_root)
    models = _parse_csv_strings(args.models)
    noises = [_noise_label(value) for value in _parse_csv_strings(args.noises)]
    noise_seeds = _parse_csv_ints(args.noise_seeds)
    sgeppy_seeds = _parse_csv_ints(args.sgeppy_seeds)
    stages = set(_parse_csv_strings(args.stages))
    unknown_stages = sorted(stages - set(STAGES))
    if unknown_stages:
        raise ValueError("Unknown stages: %s" % ", ".join(unknown_stages))
    unknown_models = sorted(set(models) - set(MODEL_DIRS))
    if unknown_models:
        raise ValueError("Unknown models: %s" % ", ".join(unknown_models))

    objective_weights = _parse_objective_weights(args.objective_weights)
    alphas = _parse_csv_floats(args.alphas)
    gammas = _parse_csv_floats(args.gammas)
    lambdas = _parse_csv_floats(args.lambdas)
    blends = _parse_csv_floats(args.blends)
    configs = {model: config_from_file(config_dir / (model + ".json")) for model in models}
    variable_names_by_model = {model: tuple(configs[model].model.variable_names) for model in models}
    dataset_rows = []
    run_rows = []

    if "datasets" in stages:
        for noise_seed in noise_seeds:
            for noise in noises:
                for model_name in models:
                    model_dir = MODEL_DIRS[model_name]
                    clean_dir = clean_data_root / model_dir
                    noisy_dir = _dataset_dir(dataset_root, "noisy", noise, model_dir, noise_seed)
                    denoised_dir = _dataset_dir(dataset_root, "denoised", noise, model_dir, noise_seed)
                    loadsteps = configs[model_name].loadsteps
                    print(
                        "[sgeppy-denoise-benchmark] datasets model=%s noise=%s noise_seed=%d method=%s"
                        % (model_name, noise, noise_seed, args.method),
                        flush=True,
                    )
                    if args.dry_run:
                        continue
                    try:
                        if not (args.skip_existing and (noisy_dir / "noise_manifest.json").is_file()):
                            write_artificially_noised_fem_dataset(
                                clean_dir,
                                noisy_dir,
                                loadsteps=loadsteps,
                                noise_level=float(noise),
                                seed=noise_seed,
                                overwrite=bool(args.overwrite_datasets),
                            )
                        if not (args.skip_existing and (denoised_dir / "denoise_summary.json").is_file()):
                            search_denoise_hyperparameters(
                                DenoiseSearchConfig(
                                    data_dir=clean_dir,
                                    output_dir=denoised_dir,
                                    loadsteps=loadsteps,
                                    noise_level=float(noise),
                                    seed=noise_seed,
                                    objective=str(args.objective),
                                    selection_scope=str(args.selection_scope),
                                    objective_weights=objective_weights,
                                    alphas=alphas,
                                    gammas=gammas,
                                    lambdas=lambdas,
                                    blends=blends,
                                    method=str(args.method),
                                    overwrite=bool(args.overwrite_datasets),
                                )
                            )
                        denoise_payload = json.loads(
                            (denoised_dir / "denoise_summary.json").read_text(encoding="utf-8")
                        )
                        baseline_metrics = denoise_payload["baseline_metrics"]
                        selected_metrics = denoise_payload["selected_metrics"]
                        dataset_rows.append(
                            {
                                "status": "ok",
                                "model": model_name,
                                "noise_level": float(noise),
                                "noise_seed": noise_seed,
                                "clean_dir": str(clean_dir),
                                "noisy_dir": str(noisy_dir),
                                "denoised_dir": str(denoised_dir),
                                "selected_score": denoise_payload["selected_score"],
                                **{"noisy_%s" % key: value for key, value in baseline_metrics.items()},
                                **{"denoised_%s" % key: value for key, value in selected_metrics.items()},
                                "F_error_reduction_fraction": 1.0
                                - float(selected_metrics["F_rmse"])
                                / max(float(baseline_metrics["F_rmse"]), 1e-15),
                            }
                        )
                    except Exception as exc:
                        if not args.keep_going:
                            raise
                        print("[sgeppy-denoise-benchmark] dataset failure: %s" % exc, flush=True)
                        dataset_rows.append(
                            {
                                "status": "failed",
                                "model": model_name,
                                "noise_level": float(noise),
                                "noise_seed": noise_seed,
                                "error": str(exc),
                            }
                        )

    if "sgeppy" in stages:
        run_specs = []
        for model_name in models:
            clean_dir = clean_data_root / MODEL_DIRS[model_name]
            for sgeppy_seed in sgeppy_seeds:
                run_specs.append((model_name, "clean", None, None, sgeppy_seed, clean_dir))
        for noise_seed in noise_seeds:
            for noise in noises:
                for model_name in models:
                    model_dir = MODEL_DIRS[model_name]
                    for condition in ("noisy", "denoised"):
                        data_dir = _dataset_dir(dataset_root, condition, noise, model_dir, noise_seed)
                        for sgeppy_seed in sgeppy_seeds:
                            run_specs.append((model_name, condition, noise, noise_seed, sgeppy_seed, data_dir))

        for model_name, condition, noise, noise_seed, sgeppy_seed, data_dir in run_specs:
            output_dir = _run_output_dir(
                output_root,
                condition,
                model_name,
                sgeppy_seed,
                noise=noise,
                noise_seed=noise_seed,
            )
            print(
                "[sgeppy-denoise-benchmark] sgeppy model=%s condition=%s noise=%s noise_seed=%s seed=%d"
                % (model_name, condition, noise or "0", noise_seed if noise_seed is not None else "-", sgeppy_seed),
                flush=True,
            )
            if args.dry_run:
                continue
            row_kwargs = {
                "model_name": model_name,
                "condition": condition,
                "dataset_dir": data_dir,
                "output_dir": output_dir,
                "sgeppy_seed": sgeppy_seed,
                "noise": noise,
                "noise_seed": noise_seed,
            }
            try:
                if args.skip_existing and (output_dir / "summary.json").is_file():
                    run_rows.append(_load_existing_run_row(**row_kwargs))
                    continue
                if not data_dir.is_dir():
                    raise FileNotFoundError("Missing benchmark dataset: %s" % data_dir)
                base_config = configs[model_name]
                model_config = replace(
                    base_config.model,
                    random_seed=sgeppy_seed,
                    n_generations=(
                        base_config.model.n_generations if args.generations is None else int(args.generations)
                    ),
                    population_size=(
                        base_config.model.population_size
                        if args.population_size is None
                        else int(args.population_size)
                    ),
                    verbose=False if args.quiet else base_config.model.verbose,
                )
                config = replace(
                    base_config,
                    model=model_config,
                    data_dir=str(data_dir),
                    noise_level=0.0,
                    output_dir=str(output_dir),
                    backend=base_config.backend if args.backend is None else str(args.backend),
                    progress_log=False if args.quiet else base_config.progress_log,
                )
                result = SGEPWorkflow(config).train()
                payload = {
                    "best_expression": result.best_expression,
                    "metrics": result.metrics,
                    "timing": result.timing,
                }
                run_rows.append(_summary_row(payload, **row_kwargs))
            except Exception as exc:
                if not args.keep_going:
                    raise
                print("[sgeppy-denoise-benchmark] SGEPPY failure: %s" % exc, flush=True)
                run_rows.append(
                    {
                        "status": "failed",
                        "model": model_name,
                        "condition": condition,
                        "noise_level": "" if noise is None else float(noise),
                        "noise_seed": "" if noise_seed is None else noise_seed,
                        "sgeppy_seed": sgeppy_seed,
                        "dataset_dir": str(data_dir),
                        "output_dir": str(output_dir),
                        "error": str(exc),
                    }
                )

    if not args.dry_run:
        _write_csv(output_root / "benchmark_datasets.csv", dataset_rows)
        _write_csv(output_root / "benchmark_runs.csv", run_rows)
        comparison_rows = _comparison_rows(run_rows, variable_names_by_model)
        _write_csv(output_root / "benchmark_comparisons.csv", comparison_rows)
        print("[sgeppy-denoise-benchmark] reports: %s" % output_root, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
