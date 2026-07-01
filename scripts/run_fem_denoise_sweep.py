from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from sym_modeling.domains.fem.methods.common.denoising import (  # noqa: E402
    DEFAULT_OBJECTIVE_WEIGHTS,
    METHODS,
    OBJECTIVES,
    SELECTION_SCOPES,
    DenoiseSearchConfig,
    search_denoise_hyperparameters,
)


DEFAULT_MODELS = ("gt", "hw", "ih", "nh2", "nh4", "ab")
DEFAULT_NOISES = ("1e-5", "1e-4", "1e-3")
BROAD_KRR_ALPHAS = (1e-10, 1e-8, 1e-7, 3e-7, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 1e-3, 1e-2)
BROAD_KRR_GAMMAS = (0.1, 1.0, 3.0, 10.0, 20.0, 30.0, 50.0, 80.0, 100.0)
BROAD_BLENDS = (0.25, 0.5, 0.75, 0.85, 0.9, 0.95, 1.0)
BROAD_LAPLACIAN_LAMBDAS = (
    0.0,
    1e-8,
    3e-8,
    1e-7,
    3e-7,
    1e-6,
    3e-6,
    1e-5,
    3e-5,
    1e-4,
    3e-4,
    1e-3,
    3e-3,
    1e-2,
    3e-2,
    1e-1,
    3e-1,
    1.0,
    3.0,
    10.0,
    30.0,
    100.0,
)
MODEL_DIRS = {
    "gt": "GT",
    "hw": "HW",
    "ih": "IH",
    "nh2": "NH2",
    "nh4": "NH4",
    "ab": "AB",
}


def _parse_csv_strings(value: str) -> list[str]:
    return [part.strip().lower() for part in value.split(",") if part.strip()]


def _parse_csv_floats(value: str) -> tuple[float, ...]:
    return tuple(float(part.strip()) for part in value.split(",") if part.strip())


def _parse_csv_ints(value: str | None) -> list[int] | None:
    if value is None or value.strip() == "":
        return None
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def _noise_label(value: str) -> str:
    text = value.strip().lower()
    return "0" if float(text) == 0.0 else text


def _method_label(method: str) -> str:
    return method.replace("-", "_")


def _parse_objective_weights(value: str | None) -> dict[str, float] | None:
    if value is None or value.strip() == "":
        return None
    weights = {}
    for part in value.split(","):
        text = part.strip()
        if not text:
            continue
        if "=" in text:
            key, raw_weight = text.split("=", 1)
        elif ":" in text:
            key, raw_weight = text.split(":", 1)
        else:
            raise ValueError("Objective weights must use metric=value entries.")
        weights[key.strip()] = float(raw_weight.strip())
    return weights


def _output_dir(
    output_root: Path,
    method: str,
    noise: str,
    model_dir: str | None,
    layout: str,
    num_methods: int,
) -> Path:
    if layout == "auto":
        layout = "method-first" if num_methods > 1 else "noise-first"
    if layout == "method-first":
        if model_dir is None:
            return output_root / _method_label(method) / noise
        return output_root / _method_label(method) / noise / model_dir
    if layout == "noise-first":
        if model_dir is None:
            return output_root / noise
        return output_root / noise / model_dir
    raise ValueError("Unknown output layout: %s" % layout)


def _candidate_fields(candidate) -> dict[str, float | str]:
    if candidate is None:
        return {}
    fields = {}
    for name in ("alpha", "gamma", "lambda_smooth", "blend"):
        if hasattr(candidate, name):
            fields[name] = float(getattr(candidate, name))
    return fields


def _write_summary_csv(path: Path, rows: list[dict]) -> None:
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


def _resolve_source_dirs(data_root: Path, models: list[str]) -> list[tuple[str, Path, str | None]]:
    if not models:
        return [(data_root.name, data_root, None)]

    sources = []
    for model_name in models:
        if model_name not in MODEL_DIRS:
            raise ValueError("Unknown model '%s'. Known models: %s" % (model_name, ", ".join(MODEL_DIRS)))
        model_dir = MODEL_DIRS[model_name]
        sources.append((model_dir, data_root / model_dir, model_dir))
    return sources


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate denoised FEM datasets over material models, noise levels, and denoising methods."
    )
    parser.add_argument(
        "--data-root",
        default="dataset/fem_data/plate_hole_fenics",
        help="Root containing clean FEM model directories such as NH2, NH4, GT.",
    )
    parser.add_argument(
        "--output-root",
        default="dataset/fem_data/fenics_denoised",
        help="Root directory for generated denoised datasets.",
    )
    parser.add_argument(
        "--models",
        help=(
            "Comma-separated material models, e.g. gt,hw,ih,nh2,nh4,ab. "
            "Omit to denoise --data-root directly as a single dataset."
        ),
    )
    parser.add_argument(
        "--noises",
        default=",".join(DEFAULT_NOISES),
        help="Comma-separated artificial displacement noise levels.",
    )
    parser.add_argument(
        "--methods",
        default="krr,mesh-laplacian",
        help="Comma-separated methods: krr,mesh-laplacian. Use both to compare denoisers.",
    )
    parser.add_argument(
        "--layout",
        choices=("auto", "noise-first", "method-first"),
        default="auto",
        help="Output layout. auto uses noise/model for one method and method/noise/model for multiple methods.",
    )
    parser.add_argument("--loadsteps", default=None, help="Optional comma-separated load steps applied to every model.")
    parser.add_argument("--seed", type=int, default=20260623, help="Base seed for deterministic artificial noise.")
    parser.add_argument(
        "--objective",
        choices=OBJECTIVES,
        default="composite",
        help="Metric minimized during denoising search.",
    )
    parser.add_argument(
        "--selection-scope",
        choices=SELECTION_SCOPES,
        default="per-loadstep",
        help="Select one candidate globally or independently per load step.",
    )
    parser.add_argument(
        "--objective-weights",
        default=",".join("%s=%g" % (key, value) for key, value in DEFAULT_OBJECTIVE_WEIGHTS.items()),
        help="Comma-separated metric weights for --objective composite.",
    )
    parser.add_argument(
        "--alphas",
        default=",".join("%g" % value for value in BROAD_KRR_ALPHAS),
        help="Comma-separated KRR alpha values.",
    )
    parser.add_argument(
        "--gammas",
        default=",".join("%g" % value for value in BROAD_KRR_GAMMAS),
        help="Comma-separated KRR RBF gamma values.",
    )
    parser.add_argument(
        "--lambdas",
        default=",".join("%g" % value for value in BROAD_LAPLACIAN_LAMBDAS),
        help="Comma-separated mesh-Laplacian smoothness values.",
    )
    parser.add_argument(
        "--blends",
        default=",".join("%g" % value for value in BROAD_BLENDS),
        help="Comma-separated blend values between noisy and denoised displacement.",
    )
    parser.add_argument("--boundary-weight", type=float, default=500.0, help="KRR sample weight for constrained nodes.")
    parser.add_argument("--no-preserve-dirichlet", action="store_true", default=False)
    parser.add_argument("--overwrite", action="store_true", default=False)
    parser.add_argument("--skip-existing", action="store_true", default=False)
    parser.add_argument("--keep-going", action="store_true", default=False, help="Continue after a failed run.")
    parser.add_argument("--dry-run", action="store_true", default=False, help="Print planned runs without writing data.")
    parser.add_argument(
        "--summary-csv",
        default=None,
        help="Optional sweep summary CSV path. Defaults to <output-root>/denoise_sweep_summary.csv.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    data_root = Path(args.data_root)
    output_root = Path(args.output_root)
    summary_csv = Path(args.summary_csv) if args.summary_csv is not None else output_root / "denoise_sweep_summary.csv"
    models = _parse_csv_strings(args.models) if args.models else []
    noises = [_noise_label(value) for value in _parse_csv_strings(args.noises)]
    methods = _parse_csv_strings(args.methods)
    loadsteps = _parse_csv_ints(args.loadsteps)
    alphas = _parse_csv_floats(args.alphas)
    gammas = _parse_csv_floats(args.gammas)
    lambdas = _parse_csv_floats(args.lambdas)
    blends = _parse_csv_floats(args.blends)
    objective_weights = _parse_objective_weights(args.objective_weights)
    summary_rows = []

    invalid_methods = [method for method in methods if method not in METHODS]
    if invalid_methods:
        raise ValueError("Unknown denoising methods: %s" % ", ".join(invalid_methods))

    sources = _resolve_source_dirs(data_root, models)

    for method in methods:
        for noise in noises:
            for dataset_name, source_dir, model_dir in sources:
                target_dir = _output_dir(output_root, method, noise, model_dir, args.layout, len(methods))
                if not source_dir.exists():
                    raise FileNotFoundError("Missing FEM data directory for dataset '%s': %s" % (dataset_name, source_dir))

                print(
                    "[fem-denoise-sweep] method=%s dataset=%s noise=%s source=%s output=%s"
                    % (method, dataset_name, noise, source_dir, target_dir),
                    flush=True,
                )
                if args.dry_run:
                    continue
                if args.skip_existing and (target_dir / "denoise_summary.json").exists():
                    print("[fem-denoise-sweep] skip existing: %s" % target_dir, flush=True)
                    summary_rows.append(
                        {
                            "status": "skipped",
                            "method": method,
                            "dataset": dataset_name,
                            "model": model_dir or "",
                            "noise_level": float(noise),
                            "selection_scope": str(args.selection_scope),
                            "output_dir": str(target_dir),
                        }
                    )
                    continue

                config = DenoiseSearchConfig(
                    data_dir=source_dir,
                    output_dir=target_dir,
                    loadsteps=loadsteps,
                    noise_level=float(noise),
                    seed=int(args.seed),
                    objective=str(args.objective),
                    selection_scope=str(args.selection_scope),
                    objective_weights=objective_weights,
                    alphas=alphas,
                    gammas=gammas,
                    lambdas=lambdas,
                    blends=blends,
                    boundary_weight=float(args.boundary_weight),
                    preserve_dirichlet=not bool(args.no_preserve_dirichlet),
                    overwrite=bool(args.overwrite),
                    method=method,
                )

                try:
                    result = search_denoise_hyperparameters(config)
                except Exception as exc:
                    if not args.keep_going:
                        raise
                    print(
                        "[fem-denoise-sweep] failed method=%s dataset=%s noise=%s error=%s"
                        % (method, dataset_name, noise, exc),
                        flush=True,
                    )
                    summary_rows.append(
                        {
                            "status": "failed",
                            "method": method,
                            "dataset": dataset_name,
                            "model": model_dir or "",
                            "noise_level": float(noise),
                            "selection_scope": str(args.selection_scope),
                            "output_dir": str(target_dir),
                            "error": str(exc),
                        }
                    )
                    continue

                selected_metrics = result.loadstep_metrics
                mean_metrics = {}
                if selected_metrics:
                    for key in ("u_rmse", "F_rmse", "J_rmse", "I1_rmse", "I2_rmse", "I3_rmse"):
                        mean_metrics[key] = float(
                            sum(float(row[key]) for row in selected_metrics) / len(selected_metrics)
                        )
                    mean_metrics["J_min"] = min(float(row["J_min"]) for row in selected_metrics)
                    mean_metrics["J_nonpositive_count"] = sum(
                        int(row["J_nonpositive_count"]) for row in selected_metrics
                    )

                row = {
                    "status": "ok",
                    "method": method,
                    "dataset": dataset_name,
                    "model": model_dir or "",
                    "noise_level": float(noise),
                    "seed": int(args.seed),
                    "objective": str(args.objective),
                    "selection_scope": str(args.selection_scope),
                    "selected_score": float(result.selected_score),
                    "selected_candidates_by_loadstep": json.dumps(
                        result.selected_candidates_by_loadstep,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    "selected_scores_by_loadstep": json.dumps(
                        result.selected_scores_by_loadstep,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    "output_dir": result.output_dir,
                    "summary_path": result.summary_path,
                    **_candidate_fields(result.selected_candidate),
                    **mean_metrics,
                }
                summary_rows.append(row)
                print(
                    "[fem-denoise-sweep] done method=%s dataset=%s noise=%s %s=%.6e output=%s"
                    % (method, dataset_name, noise, args.objective, result.selected_score, result.output_dir),
                    flush=True,
                )

    if not args.dry_run:
        _write_summary_csv(summary_csv, summary_rows)
        print("[fem-denoise-sweep] summary: %s" % summary_csv, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
