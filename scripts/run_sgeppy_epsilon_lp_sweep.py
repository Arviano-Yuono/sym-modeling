"""
uv run python scripts/run_sgeppy_epsilon_lp_sweep.py --config-dir configs/sgeppy --output-root output/sgeppy_epsilon_lp --epsilons 1,2,3,4,5 --lambda-lps 1e-4,1e-3,1e-2
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from sym_modeling.domains.fem.methods.sgeppy.run_gep_sparse import config_from_file  # noqa: E402
from sym_modeling.domains.fem.methods.sgeppy.workflow import SGEPWorkflow  # noqa: E402


DEFAULT_MODELS = ("gt", "hw", "ih", "nh2", "nh4", "ab")
DEFAULT_EPSILONS = (1, 2, 3, 4, 5)


def _parse_csv_ints(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def _parse_csv_floats(value: str | None) -> list[float] | None:
    if value is None or value.strip() == "":
        return None
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def _parse_csv_strings(value: str) -> list[str]:
    return [part.strip().lower() for part in value.split(",") if part.strip()]


def _value_label(value: float) -> str:
    text = "%.12g" % float(value)
    return text.replace("+", "").replace("-", "m").replace(".", "p")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run SGEPPY weak-form epsilon-constrained Pareto sweeps at fixed LP lambda values."
    )
    parser.add_argument(
        "--config-dir",
        default="configs/sgeppy",
        help="Directory containing per-model SGEPPY JSON configs.",
    )
    parser.add_argument(
        "--models",
        default=",".join(DEFAULT_MODELS),
        help="Comma-separated model names to run, e.g. gt,hw,ih,nh2,nh4.",
    )
    parser.add_argument(
        "--epsilons",
        default=",".join(str(value) for value in DEFAULT_EPSILONS),
        help="Comma-separated active-term epsilon constraints.",
    )
    parser.add_argument(
        "--lambda-lps",
        "--penalty-lps",
        dest="lambda_lps",
        default=None,
        help="Optional comma-separated fixed LP lambda values. Defaults to each config weak_form.penalty_lp.",
    )
    parser.add_argument(
        "--num-increments",
        type=int,
        default=1,
        help="LP penalty increments to run. Default 1 keeps each lambda fixed in lp_solver.py.",
    )
    parser.add_argument(
        "--output-root",
        default="output/sgeppy_epsilon_lp",
        help="Root directory for sweep outputs.",
    )
    parser.add_argument(
        "--backend",
        choices=("jax", "torch"),
        default=None,
        help="Override the weak-form backend from the loaded config.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Disable workflow and per-generation progress logs.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned runs without executing SGEPPY.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config_dir = Path(args.config_dir)
    output_root = Path(args.output_root)
    models = _parse_csv_strings(args.models)
    epsilons = _parse_csv_ints(args.epsilons)
    lambda_lps = _parse_csv_floats(args.lambda_lps)

    for model_name in models:
        config_path = config_dir / ("%s.json" % model_name)
        if not config_path.exists():
            raise FileNotFoundError("Missing SGEPPY config for model '%s': %s" % (model_name, config_path))

        base_config = config_from_file(config_path)
        lambdas = (
            lambda_lps
            if lambda_lps is not None
            else [float(base_config.weak_form.penalty_lp)]
        )
        backend = args.backend if args.backend is not None else base_config.backend
        for epsilon in epsilons:
            for lambda_lp in lambdas:
                output_dir = (
                    output_root
                    / model_name
                    / ("eps_%d" % epsilon)
                    / ("lambda_lp_%s" % _value_label(lambda_lp))
                )
                model_config = replace(
                    base_config.model,
                    fitness_metrics=("active_terms", "rmse"),
                    epsilons=(float(epsilon), None),
                    verbose=False if args.quiet else base_config.model.verbose,
                )
                weak_form = replace(
                    base_config.weak_form,
                    penalty_lp=float(lambda_lp),
                    num_increments=int(args.num_increments),
                )
                config = replace(
                    base_config,
                    model=model_config,
                    weak_form=weak_form,
                    backend=backend,
                    output_dir=str(output_dir),
                    progress_log=False if args.quiet else base_config.progress_log,
                )

                print(
                    "[sgeppy-epsilon-lp] model=%s epsilon=%d lambda_lp=%s increments=%d backend=%s output=%s"
                    % (
                        model_name,
                        epsilon,
                        "%.12g" % float(lambda_lp),
                        int(args.num_increments),
                        config.backend,
                        output_dir,
                    ),
                    flush=True,
                )
                if args.dry_run:
                    continue

                result = SGEPWorkflow(config).train()
                print(
                    "[sgeppy-epsilon-lp] done model=%s epsilon=%d lambda_lp=%s rmse=%.6e active_terms=%d weak_cost=%.6e lp_cost=%.6e total_cost=%.6e wall=%.3fs"
                    % (
                        model_name,
                        epsilon,
                        "%.12g" % float(lambda_lp),
                        result.metrics["rmse"],
                        result.metrics["num_parameters"],
                        result.metrics["weak_accuracy_cost"],
                        result.metrics["lp_sparsity_cost"],
                        result.metrics["lp_total_cost"],
                        result.timing["wall_seconds"],
                    ),
                    flush=True,
                )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
