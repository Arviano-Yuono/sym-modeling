from __future__ import annotations

import argparse
import json
from dataclasses import fields
from pathlib import Path

from .sgep import SGEPConfig
from .workflow import SGEPWorkflow, SGEPWorkflowConfig, WeakFormConfig


def _parse_csv(value: str | None) -> tuple[str, ...] | None:
    if value is None or value.strip() == "":
        return None
    return tuple(part.strip() for part in value.split(",") if part.strip())


def _parse_epsilons(value: str | None) -> tuple[float | None, ...] | None:
    parts = _parse_csv(value)
    if parts is None:
        return None
    return tuple(None if part.lower() in {"none", "null"} else float(part) for part in parts)


def _parse_loadsteps(value: str | None) -> list[int] | None:
    parts = _parse_csv(value)
    if parts is None:
        return None
    return [int(part) for part in parts]


def config_from_file(path: str | Path) -> SGEPWorkflowConfig:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if "sgeppy" not in payload:
        raise ValueError("sgeppy config files must contain a top-level 'sgeppy' object.")
    values = dict(payload["sgeppy"])
    model_values = dict(values.pop("model", {}))
    weak_form_values = dict(values.pop("weak_form", {}))

    workflow_fields = {field.name for field in fields(SGEPWorkflowConfig)}
    model_fields = {field.name for field in fields(SGEPConfig)}
    weak_form_fields = {field.name for field in fields(WeakFormConfig)}
    unknown_workflow = sorted(set(values) - workflow_fields)
    unknown_model = sorted(set(model_values) - model_fields)
    unknown_weak_form = sorted(set(weak_form_values) - weak_form_fields)
    if unknown_workflow:
        raise ValueError("Unknown sgeppy workflow config keys: %s" % ", ".join(unknown_workflow))
    if unknown_model:
        raise ValueError("Unknown sgeppy model config keys: %s" % ", ".join(unknown_model))
    if unknown_weak_form:
        raise ValueError("Unknown sgeppy weak_form config keys: %s" % ", ".join(unknown_weak_form))

    for key in ("variable_names", "unary_operators", "binary_operators", "fitness_metrics"):
        if key in model_values and model_values[key] is not None:
            model_values[key] = tuple(model_values[key])
    if "epsilons" in model_values and model_values["epsilons"] is not None:
        model_values["epsilons"] = tuple(model_values["epsilons"])
    values["model"] = SGEPConfig(**model_values)
    values["weak_form"] = WeakFormConfig(**weak_form_values)
    return SGEPWorkflowConfig(**values)


def _apply_overrides(config: SGEPWorkflowConfig, args: argparse.Namespace) -> SGEPWorkflowConfig:
    workflow_values = {field.name: getattr(config, field.name) for field in fields(SGEPWorkflowConfig)}
    model_values = {field.name: getattr(config.model, field.name) for field in fields(SGEPConfig)}

    for key in ("data_dir", "loadsteps", "output_dir"):
        value = getattr(args, key)
        if value is not None:
            workflow_values[key] = value
    if args.fitting_mode is not None:
        workflow_values["fitting_mode"] = args.fitting_mode
    if args.loadsteps is not None:
        workflow_values["loadsteps"] = _parse_loadsteps(args.loadsteps)
    if args.quiet:
        workflow_values["progress_log"] = False
        model_values["verbose"] = False

    overrides = {
        "n_generations": args.generations,
        "population_size": args.population_size,
        "n_genes": args.n_genes,
        "random_seed": args.seed,
        "fitness_metrics": _parse_csv(args.fitness_metrics),
        "epsilons": _parse_epsilons(args.epsilons),
    }
    for key, value in overrides.items():
        if value is not None:
            model_values[key] = value

    workflow_values["model"] = SGEPConfig(**model_values)
    return SGEPWorkflowConfig(**workflow_values)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run geppy-backed SGEP direct-stress discovery.")
    parser.add_argument("--config", required=True, help="Path to a JSON file with a top-level 'sgeppy' object.")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--loadsteps", default=None, help="Comma-separated load steps.")
    parser.add_argument("--fitting-mode", choices=("direct_stress", "weak_form"), default=None)
    parser.add_argument("--generations", type=int, default=None)
    parser.add_argument("--population-size", type=int, default=None)
    parser.add_argument("--n-genes", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--fitness-metrics", default=None, help="Comma-separated metrics, e.g. rmse,aic.")
    parser.add_argument("--epsilons", default=None, help="Comma-separated epsilons, e.g. none,10.")
    parser.add_argument("--quiet", action="store_true", default=False)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = _apply_overrides(config_from_file(args.config), args)
    result = SGEPWorkflow(config).train()
    print("Best SGEPPY expression:")
    print(result.best_expression)
    print(
        "RMSE: %.6e | RSS: %.6e | AIC: %.6e | AICc: %.6e | active terms: %d"
        % (
            result.metrics["rmse"],
            result.metrics["rss"],
            result.metrics["aic"],
            result.metrics["aicc"],
            result.metrics["num_parameters"],
        )
    )
    print(
        "Time: wall %.3fs | CPU %.3fs"
        % (result.timing["wall_seconds"], result.timing["cpu_seconds"])
    )
    print("Outputs: %s" % result.output_paths["summary_json"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
