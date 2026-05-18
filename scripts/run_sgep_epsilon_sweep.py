from __future__ import annotations

import argparse
import sys
from dataclasses import fields
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from sym_modeling.domains.fem.methods.sgep.run_gep_sparse import config_from_file  # noqa: E402
from sym_modeling.domains.fem.methods.sgep.workflow import SGEPConfig, SGEPWorkflow  # noqa: E402


DEFAULT_MODELS = ("nh2", "nh4", "ih", "hw", "gt")
DEFAULT_EPSILONS = (2, 3, 4, 5)


def _parse_csv_ints(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def _parse_csv_strings(value: str) -> list[str]:
    return [part.strip().lower() for part in value.split(",") if part.strip()]


def _config_with_overrides(config: SGEPConfig, **overrides) -> SGEPConfig:
    values = {field.name: getattr(config, field.name) for field in fields(SGEPConfig)}
    values.update(overrides)
    return SGEPConfig(**values)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run SGEP direct-Piola epsilon sweeps for the FEM benchmark configs."
    )
    parser.add_argument(
        "--config-dir",
        default="configs/sgep",
        help="Directory containing per-model SGEP JSON configs.",
    )
    parser.add_argument(
        "--models",
        default=",".join(DEFAULT_MODELS),
        help="Comma-separated model names to run, e.g. nh2,nh4,ih.",
    )
    parser.add_argument(
        "--epsilons",
        default=",".join(str(value) for value in DEFAULT_EPSILONS),
        help="Comma-separated active-term epsilon values.",
    )
    parser.add_argument(
        "--output-root",
        default="output/sgep_direct_piola",
        help="Root directory for sweep outputs.",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Disable plots for each sweep run.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Disable SGEP per-generation progress logs.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned runs without executing SGEP.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config_dir = Path(args.config_dir)
    output_root = Path(args.output_root)
    models = _parse_csv_strings(args.models)
    epsilons = _parse_csv_ints(args.epsilons)

    for model in models:
        config_path = config_dir / ("%s.json" % model)
        if not config_path.exists():
            raise FileNotFoundError("Missing SGEP config for model '%s': %s" % (model, config_path))

        base_config = config_from_file(config_path)
        for epsilon in epsilons:
            output_dir = output_root / model / ("eps%d" % epsilon)
            config = _config_with_overrides(
                base_config,
                fitting_mode="direct_stress",
                selection_objective="epsilon_rmse",
                active_terms_epsilon=epsilon,
                output_dir=str(output_dir),
                save_plots=False if args.skip_plots else base_config.save_plots,
                progress_log=False if args.quiet else base_config.progress_log,
            )
            print(
                "[sgep-sweep] model=%s epsilon=%d output=%s"
                % (model, epsilon, output_dir),
                flush=True,
            )
            if args.dry_run:
                continue

            result = SGEPWorkflow(config).train()
            print(
                "[sgep-sweep] done model=%s epsilon=%d rmse=%.6e active_terms=%d expression=%s"
                % (
                    model,
                    epsilon,
                    result.metrics["rmse"],
                    result.metrics["num_parameters"],
                    result.best_expression,
                ),
                flush=True,
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
