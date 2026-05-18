from __future__ import annotations

import argparse
import json
from dataclasses import fields
from pathlib import Path
from typing import Any

from sym_modeling.domains.fem.methods.sgep.workflow import SGEPConfig, SGEPWorkflow


def _parse_loadsteps(value: str | None) -> list[int] | None:
    if value is None or value.strip() == "":
        return None
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def _config_payload(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if "sgep" in payload:
        payload = payload["sgep"]
    if not isinstance(payload, dict):
        raise ValueError("SGEP config must be a JSON object.")
    return payload


def config_from_file(path: str | Path) -> SGEPConfig:
    payload = _config_payload(path)
    field_names = {field.name for field in fields(SGEPConfig)}
    unknown = sorted(set(payload) - field_names)
    if unknown:
        raise ValueError("Unknown SGEP config keys: %s" % ", ".join(unknown))

    values = dict(payload)
    for tuple_key in ("variable_names", "unary_operators", "binary_operators"):
        if tuple_key in values and values[tuple_key] is not None:
            values[tuple_key] = tuple(values[tuple_key])
    if "loadsteps" in values and isinstance(values["loadsteps"], str):
        values["loadsteps"] = _parse_loadsteps(values["loadsteps"])
    return SGEPConfig(**values)


def _apply_cli_overrides(config: SGEPConfig, args: argparse.Namespace) -> SGEPConfig:
    values = {field.name: getattr(config, field.name) for field in fields(SGEPConfig)}
    overrides = {
        "data_dir": args.data_dir,
        "loadsteps": _parse_loadsteps(args.loadsteps),
        "output_dir": args.output_dir,
        "generations": args.generations,
        "num_models": args.num_models,
        "genes_per_model": args.genes_per_model,
        "max_depth": args.max_depth,
        "random_seed": args.seed,
        "sparsity_threshold": args.threshold,
        "regression_method": args.regression_method,
        "fitting_mode": args.fitting_mode,
        "selection_objective": args.selection_objective,
        "active_terms_epsilon": args.active_terms_epsilon,
    }
    for key, value in overrides.items():
        if value is not None:
            values[key] = value
    if args.max_elements_per_loadstep is not None:
        values["max_elements_per_loadstep"] = args.max_elements_per_loadstep
    if args.all_elements is True:
        values["max_elements_per_loadstep"] = None
    if args.skip_plots is True:
        values["save_plots"] = False
    if args.compare_euclid is True:
        values["compare_euclid"] = True
    if args.quiet is True:
        values["progress_log"] = False
    return SGEPConfig(**values)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run SGEP sparse hyperelastic discovery.")
    parser.add_argument("--config", default=None, help="Path to an SGEP JSON config file.")
    parser.add_argument("--data-dir", default=None, help="EUCLID-compatible dataset root.")
    parser.add_argument("--loadsteps", default=None, help="Comma-separated load steps.")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--generations", type=int, default=None)
    parser.add_argument("--num-models", type=int, default=None)
    parser.add_argument("--genes-per-model", type=int, default=None)
    parser.add_argument("--max-depth", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--max-elements-per-loadstep",
        type=int,
        default=None,
        help="Maximum elements sampled per load step for fitting. Use 0 to disable the cap.",
    )
    parser.add_argument(
        "--all-elements",
        action="store_true",
        default=False,
        help="Use every element from every load step, overriding any configured cap.",
    )
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--regression-method", default=None)
    parser.add_argument(
        "--fitting-mode",
        choices=("auto", "weak_form", "direct_stress"),
        default=None,
        help="Fitting target: auto, weak_form, or direct_stress.",
    )
    parser.add_argument(
        "--selection-objective",
        choices=("aicc", "epsilon_rmse"),
        default=None,
        help="Candidate selection objective.",
    )
    parser.add_argument(
        "--active-terms-epsilon",
        type=int,
        default=None,
        help="Maximum active terms for epsilon_rmse selection.",
    )
    parser.add_argument("--skip-plots", action="store_true", default=None)
    parser.add_argument("--compare-euclid", action="store_true", default=None)
    parser.add_argument("--quiet", action="store_true", default=False, help="Disable SGEP progress logs.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = config_from_file(args.config) if args.config is not None else SGEPConfig()
    config = _apply_cli_overrides(config, args)
    result = SGEPWorkflow(config).train()
    print("Best SGEP expression:")
    print(result.best_expression)
    print("RMSE: %.6e | RSS: %.6e | AICc: %.6e" % (
        result.metrics["rmse"],
        result.metrics["rss"],
        result.metrics["aicc"],
    ))
    print("Summary: %s" % result.output_paths.get("summary_json", config.output_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
