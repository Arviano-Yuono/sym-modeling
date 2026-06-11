"""
uv run python scripts/run_sgeppy_active_terms_sweep.py \
  --config-dir configs/sgeppy_test \
  --output-root output/sgeppy_better_sweep \
  --quiet
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


DEFAULT_MODELS = ("gt", "hw", "ih", "nh2", "nh4")
DEFAULT_ACTIVE_TERMS = (1, 2, 3, 4, 5)


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
        description="Run SGEPPY weak-form active-term epsilon sweeps for the FEM benchmark configs."
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
        "--active-terms",
        default=",".join(str(value) for value in DEFAULT_ACTIVE_TERMS),
        help="Comma-separated active-term epsilon values.",
    )
    parser.add_argument(
        "--penalty-lps",
        default=None,
        help="Optional comma-separated weak-form penalty_lp values. Defaults to each config value.",
    )
    parser.add_argument(
        "--output-root",
        default="output/sgeppy_active_terms",
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
    active_terms = _parse_csv_ints(args.active_terms)
    penalty_lps = _parse_csv_floats(args.penalty_lps)

    for model_name in models:
        config_path = config_dir / ("%s.json" % model_name)
        if not config_path.exists():
            raise FileNotFoundError("Missing SGEPPY config for model '%s': %s" % (model_name, config_path))

        base_config = config_from_file(config_path)
        penalties = penalty_lps if penalty_lps is not None else [float(base_config.weak_form.penalty_lp)]
        backend = args.backend if args.backend is not None else base_config.backend
        for k in active_terms:
            for penalty_lp in penalties:
                output_dir = output_root / model_name / ("k_%d" % k) / ("lp_%s" % _value_label(penalty_lp))
                model_config = replace(
                    base_config.model,
                    fitness_metrics=("active_terms", "rmse"),
                    epsilons=(float(k), None),
                    verbose=False if args.quiet else base_config.model.verbose,
                )
                weak_form = replace(base_config.weak_form, penalty_lp=float(penalty_lp))
                config = replace(
                    base_config,
                    model=model_config,
                    weak_form=weak_form,
                    backend=backend,
                    output_dir=str(output_dir),
                    progress_log=False if args.quiet else base_config.progress_log,
                )

                print(
                    "[sgeppy-active-sweep] model=%s k=%d penalty_lp=%s backend=%s output=%s"
                    % (model_name, k, "%.12g" % float(penalty_lp), config.backend, output_dir),
                    flush=True,
                )
                if args.dry_run:
                    continue

                result = SGEPWorkflow(config).train()
                print(
                    "[sgeppy-active-sweep] done model=%s k=%d penalty_lp=%s rmse=%.6e active_terms=%d wall=%.3fs"
                    % (
                        model_name,
                        k,
                        "%.12g" % float(penalty_lp),
                        result.metrics["rmse"],
                        result.metrics["num_parameters"],
                        result.timing["wall_seconds"],
                    ),
                    flush=True,
                )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
