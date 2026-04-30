from __future__ import print_function

import os
import sys


MIN_PYTHON = (3, 10)


def _print_version_error():
    message = (
        "tutorial.py requires Python {major}.{minor}+.\n"
        "Current interpreter: {executable} ({version})\n\n"
        "Use the FEniCSx Conda environment from environment.yml, for example:\n"
        "  conda activate sym-modeling-fenicsx\n"
        "  python tutorial.py --workspace output/my_euclid_run\n"
    ).format(
        major=MIN_PYTHON[0],
        minor=MIN_PYTHON[1],
        executable=sys.executable,
        version=sys.version.split()[0],
    )
    sys.stderr.write(message)


if sys.version_info < MIN_PYTHON:
    _print_version_error()
    raise SystemExit(1)


import argparse
import csv
import json
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from sym_modeling.domains.fem import (  # noqa: E402
    HyperelasticGenerationConfig,
    HyperelasticSuiteCase,
    generate_hyperelastic_data,
    generate_hyperelastic_suite,
    loadFemData,
    validate_hyperelastic_data_export,
)
from sym_modeling.domains.fem.methods.euclid import EuclidConfig, EuclidWorkflow  # noqa: E402
from sym_modeling.domains.fem.methods.euclid.postprocessing import (  # noqa: E402
    plotPiolaFieldComparison,
)
from sym_modeling.domains.fem.methods.euclid.weak_form import (  # noqa: E402
    computeFirstPiolaTheta,
)


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Executable Euclid tutorial: generate hyperelastic FEM data, run discovery, "
            "and evaluate the discovered model against the reference Piola field."
        )
    )
    parser.add_argument(
        "--workspace",
        default="output/tutorial_euclid",
        help="Directory used for generated data, Euclid results, and evaluation artifacts.",
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help=(
            "Use an existing FEM dataset root instead of generating tutorial data. "
            "The directory should contain numeric load-step folders with EUCLID CSV files."
        ),
    )
    parser.add_argument(
        "--skip-generate",
        action="store_true",
        help="Reuse an existing FEM dataset under <workspace>/fem_data instead of generating it.",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip Piola field comparison plots and only write scalar evaluation metrics.",
    )
    parser.add_argument(
        "--noise-level",
        type=float,
        default=0.0,
        help="Additional displacement noise passed into Euclid during loading.",
    )
    parser.add_argument(
        "--material-model",
        choices=("euclid_neo_hookean_j2", "notebook_neo_hookean_log"),
        default="euclid_neo_hookean_j2",
        help=(
            "Material model used when generating tutorial data. "
            "'euclid_neo_hookean_j2' matches Euclid's feature basis. "
            "'notebook_neo_hookean_log' matches hyperelasticity.ipynb more closely."
        ),
    )
    parser.add_argument(
        "--single-case",
        action="store_true",
        help="Generate only the original single traction case instead of the richer tutorial suite.",
    )
    return parser


def load_generation_manifest(data_dir):
    manifest_path = data_dir / "generation_manifest.json"
    if not manifest_path.exists():
        return None
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def build_tutorial_suite(material_model):
    return (
        HyperelasticSuiteCase(
            name="uniaxial_x",
            load_step_offset=0,
            config=HyperelasticGenerationConfig(
                material_model=material_model,
                fixed_boundary="left",
                traction_boundaries=("right",),
                traction_per_step=25.0,
                traction_direction=(1.0, 0.0),
            ),
        ),
        HyperelasticSuiteCase(
            name="simple_shear",
            load_step_offset=100,
            config=HyperelasticGenerationConfig(
                material_model=material_model,
                fixed_boundary="bottom",
                traction_boundaries=("top",),
                traction_per_step=1.25,
                traction_direction=(1.0, 0.0),
            ),
        ),
        HyperelasticSuiteCase(
            name="vertical_extension",
            load_step_offset=200,
            config=HyperelasticGenerationConfig(
                material_model=material_model,
                fixed_boundary="bottom",
                traction_boundaries=("top",),
                traction_per_step=1.25,
                traction_direction=(0.0, 1.0),
            ),
        ),
        HyperelasticSuiteCase(
            name="mixed_extension",
            load_step_offset=300,
            config=HyperelasticGenerationConfig(
                material_model=material_model,
                fixed_boundary="left",
                traction_boundaries=("right", "top"),
                traction_per_step=15.0,
                traction_boundary_vectors=((1.0, 0.0), (0.0, 0.05)),
            ),
        ),
    )


def resolve_loadsteps(data_dir):
    manifest_path = data_dir / "generation_manifest.json"
    if manifest_path.exists():
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        return [int(entry["load_step"]) for entry in payload.get("load_steps", [])]

    loadsteps = []
    for child in data_dir.iterdir():
        if child.is_dir():
            try:
                loadsteps.append(int(child.name))
            except ValueError:
                continue
    loadsteps.sort()
    if not loadsteps:
        raise FileNotFoundError(
            "Could not find any numeric load-step directories under {}.".format(data_dir)
        )
    return loadsteps


def evaluate_discovery(theta, data_dir, loadsteps, evaluation_dir, make_plots):
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    plot_root = evaluation_dir / "plots"
    metrics = []

    for loadstep in loadsteps:
        case_dir = data_dir / str(loadstep)
        data = loadFemData(str(case_dir), AD=True, noiseLevel=0.0, noiseType="displacement")
        data.convertToNumpy()

        if data.P is None:
            raise ValueError(
                "Reference Piola stresses are missing in {}. Evaluation cannot continue.".format(
                    case_dir
                )
            )

        modeled_piola = computeFirstPiolaTheta(data, theta)
        error = modeled_piola - data.P
        step_metrics = {
            "loadstep": int(loadstep),
            "rmse": float(np.sqrt(np.mean(np.square(error)))),
            "mae": float(np.mean(np.abs(error))),
            "max_abs_error": float(np.max(np.abs(error))),
            "component_rmse": {
                "Pxx": float(np.sqrt(np.mean(np.square(error[:, 0])))),
                "Pxy": float(np.sqrt(np.mean(np.square(error[:, 1])))),
                "Pyx": float(np.sqrt(np.mean(np.square(error[:, 2])))),
                "Pyy": float(np.sqrt(np.mean(np.square(error[:, 3])))),
            },
        }

        if make_plots:
            plot_dir = plot_root / "loadstep_{}".format(loadstep)
            plot_paths = plotPiolaFieldComparison(
                data=data,
                theta=theta,
                output_dir=str(plot_dir),
                prefix="loadstep_{}".format(loadstep),
                title_prefix="Load step {}".format(loadstep),
            )
            step_metrics["plot_paths"] = [str(Path(path)) for path in plot_paths]

        metrics.append(step_metrics)

    rmse_values = [entry["rmse"] for entry in metrics]
    mae_values = [entry["mae"] for entry in metrics]
    return {
        "num_loadsteps": len(metrics),
        "mean_rmse": float(np.mean(rmse_values)),
        "max_rmse": float(np.max(rmse_values)),
        "mean_mae": float(np.mean(mae_values)),
        "per_loadstep": metrics,
    }


def summarize_linear_system(matrix):
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    if singular_values.size == 0:
        return {
            "rank": 0,
            "num_rows": int(matrix.shape[0]),
            "num_cols": int(matrix.shape[1]),
            "condition_number": float("inf"),
        }

    tolerance = (
        max(matrix.shape) * np.finfo(singular_values.dtype).eps * singular_values[0]
    )
    active_singular_values = singular_values[singular_values > tolerance]
    if active_singular_values.size == 0:
        condition_number = float("inf")
        rank = 0
    else:
        condition_number = float(active_singular_values[0] / active_singular_values[-1])
        rank = int(active_singular_values.size)

    return {
        "rank": rank,
        "num_rows": int(matrix.shape[0]),
        "num_cols": int(matrix.shape[1]),
        "condition_number": condition_number,
    }


def validate_external_dataset_kinematics(data_dir, loadsteps, material_model):
    per_load_step = []
    rmse_values = []
    max_abs_error = 0.0

    for loadstep in loadsteps:
        case_dir = data_dir / str(loadstep)
        data = loadFemData(str(case_dir), AD=True, noiseLevel=0.0, noiseType="displacement")
        data.convertToNumpy()

        with (case_dir / "output_elements.csv").open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = reader.fieldnames or []
            required = ("Fxx", "Fxy", "Fyx", "Fyy")
            if any(name not in fieldnames for name in required):
                raise ValueError(
                    "External dataset {} is missing Fxx/Fxy/Fyx/Fyy columns, so kinematic validation cannot run.".format(
                        case_dir
                    )
                )
            exported_F = np.array(
                [
                    [float(row["Fxx"]), float(row["Fxy"]), float(row["Fyx"]), float(row["Fyy"])]
                    for row in reader
                ],
                dtype=float,
            )

        error = data.F - exported_F
        step_max_abs_error = float(np.max(np.abs(error)))
        step_rmse = float(np.sqrt(np.mean(np.square(error))))
        max_abs_error = max(max_abs_error, step_max_abs_error)
        rmse_values.append(step_rmse)
        per_load_step.append(
            {
                "load_step": int(loadstep),
                "max_abs_error": step_max_abs_error,
                "rmse": step_rmse,
            }
        )

    return {
        "mode": "kinematic",
        "material_model": material_model,
        "max_abs_error": max_abs_error,
        "mean_rmse": float(np.mean(rmse_values)),
        "per_load_step": per_load_step,
    }


def run(args):
    workspace = Path(args.workspace).resolve()
    external_data_dir = Path(args.data_dir).resolve() if args.data_dir else None
    data_dir = external_data_dir if external_data_dir is not None else workspace / "fem_data"
    results_dir = workspace / "euclid_results"
    evaluation_dir = workspace / "evaluation"

    workspace.mkdir(parents=True, exist_ok=True)

    generation_config = None
    suite_cases = None
    if external_data_dir is not None:
        if not data_dir.exists():
            raise FileNotFoundError("--data-dir does not exist: {}".format(data_dir))
        print("[1/3] Using external FEM data in {}".format(data_dir))
    elif args.skip_generate:
        if not data_dir.exists():
            raise FileNotFoundError(
                "--skip-generate was set, but {} does not exist.".format(data_dir)
            )
        print("[1/3] Reusing FEM data in {}".format(data_dir))
    else:
        if args.single_case:
            print("[1/3] Generating tutorial FEM data in {}".format(data_dir))
            generation_config = HyperelasticGenerationConfig(material_model=args.material_model)
            generation_result = generate_hyperelastic_data(data_dir, generation_config)
            print("      Wrote load steps: {}".format(list(generation_result.load_steps)))
        else:
            print("[1/3] Generating tutorial FEM suite in {}".format(data_dir))
            suite_cases = build_tutorial_suite(args.material_model)
            suite_result = generate_hyperelastic_suite(data_dir, suite_cases)
            print("      Cases: {}".format([case.name for case in suite_cases]))
            print("      Wrote combined load steps: {}".format(list(suite_result.load_steps)))

    loadsteps = resolve_loadsteps(data_dir)
    manifest = load_generation_manifest(data_dir)
    if generation_config is None and manifest is None:
        validation = validate_external_dataset_kinematics(data_dir, loadsteps, args.material_model)
    else:
        validation = validate_hyperelastic_data_export(data_dir, generation_config)
        validation = {
            "mode": "constitutive",
            "material_model": validation.material_model,
            "max_abs_error": validation.max_abs_error,
            "mean_rmse": validation.mean_rmse,
            "per_load_step": list(validation.per_load_step),
        }
    print(
        "      Validation: mode={} material_model={} max_error={:.3e}".format(
            validation["mode"],
            validation["material_model"],
            validation["max_abs_error"],
        )
    )
    if validation["mode"] == "constitutive" and validation["material_model"] != "euclid_neo_hookean_j2":
        print(
            "      Warning: tutorial discovery uses NeoHookeanJ2 features, so notebook_neo_hookean_log "
            "data will not be recovered exactly."
        )

    print("[2/3] Running Euclid discovery")
    euclid_config = EuclidConfig(
        str_model="NeoHookeanJ2",
        str_mesh=(
            data_dir.name
            if external_data_dir is not None
            else ("tutorial_suite" if not args.single_case else "tutorial_plate")
        ),
        femDataPathOverride=str(data_dir),
        loadstepsOverride=loadsteps,
        resultsDir=str(results_dir),
        appendResults=False,
        noiseLevel=float(args.noise_level),
    )
    workflow = EuclidWorkflow(config=euclid_config)
    result = workflow.train()
    workflow.evaluate()
    system_diagnostics = summarize_linear_system(result.lhs)
    if system_diagnostics["rank"] < min(result.lhs.shape):
        print(
            "      Warning: discovery system is rank-deficient "
            "(rank={rank}, size={rows}x{cols}, cond={cond:.3e}).".format(
                rank=system_diagnostics["rank"],
                rows=system_diagnostics["num_rows"],
                cols=system_diagnostics["num_cols"],
                cond=system_diagnostics["condition_number"],
            )
        )

    print("[3/3] Evaluating the discovered model")
    evaluation = evaluate_discovery(
        theta=result.theta,
        data_dir=data_dir,
        loadsteps=loadsteps,
        evaluation_dir=evaluation_dir,
        make_plots=not args.skip_plots,
    )

    summary = {
        "workspace": str(workspace),
        "data_dir": str(data_dir),
        "results_dir": str(results_dir),
        "evaluation_dir": str(evaluation_dir),
        "loadsteps": loadsteps,
        "generation_mode": (
            "external_data"
            if external_data_dir is not None
            else ("single_case" if args.single_case else "suite")
        ),
        "suite_cases": manifest.get("suite_cases", []) if manifest is not None else [],
        "material_model": validation["material_model"],
        "theta": result.theta.tolist(),
        "num_active_terms": int(np.count_nonzero(np.abs(result.theta) > euclid_config.threshold)),
        "lhs_shape": list(result.lhs.shape),
        "rhs_shape": list(result.rhs.shape),
        "system_diagnostics": system_diagnostics,
        "csv_validation": {
            "mode": validation["mode"],
            "max_abs_error": validation["max_abs_error"],
            "mean_rmse": validation["mean_rmse"],
            "per_load_step": validation["per_load_step"],
        },
        "evaluation": evaluation,
        "results_file": str(results_dir / "results_{}.txt".format(euclid_config.saveResultsName)),
    }

    summary_path = workspace / "tutorial_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("")
    print("Tutorial run complete.")
    print("Summary: {}".format(summary_path))
    print("Results text file: {}".format(summary["results_file"]))
    print("Generation mode: {}".format(summary["generation_mode"]))
    print("Material model: {}".format(summary["material_model"]))
    print(
        "Validation ({}) max error: {:.6e}".format(
            summary["csv_validation"]["mode"],
            summary["csv_validation"]["max_abs_error"],
        )
    )
    print(
        "Discovery system rank: {}/{} (cond={:.6e})".format(
            summary["system_diagnostics"]["rank"],
            summary["system_diagnostics"]["num_rows"],
            summary["system_diagnostics"]["condition_number"],
        )
    )
    print("Mean Piola RMSE: {:.6e}".format(evaluation["mean_rmse"]))
    print("Active terms: {}".format(summary["num_active_terms"]))
    if args.skip_plots:
        print("Plots: skipped")
    else:
        print("Plots: {}".format(evaluation_dir / "plots"))

    return summary_path


def main():
    parser = build_parser()
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
