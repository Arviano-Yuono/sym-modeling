from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from sym_modeling.domains.fem.methods.sgeppy.run_gep_sparse import config_from_file  # noqa: E402
from sym_modeling.domains.fem.methods.sgeppy.torch_backend import require_torch_backend  # noqa: E402
from sym_modeling.domains.fem.methods.sgeppy.workflow import SGEPWorkflow  # noqa: E402


def _parse_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def _parse_loadsteps(value: str | None) -> list[int] | None:
    if value is None:
        return None
    return [int(part) for part in _parse_csv(value)]


def _safe_json(value: Any):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _safe_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_json(item) for item in value]
    return value


def _backend_device(backend: str) -> str:
    if backend == "torch":
        try:
            torch = require_torch_backend()
        except ImportError:
            return "unavailable"
        return str(torch.device("cuda"))
    if backend == "jax":
        try:
            import jax
        except ModuleNotFoundError:
            return "unavailable"
        try:
            return str(jax.default_backend())
        except Exception:
            return "default"
    return "unknown"


def _run_backend(base_config, backend: str, args: argparse.Namespace) -> dict:
    if backend == "torch":
        try:
            require_torch_backend()
        except ImportError as exc:
            return {
                "backend": backend,
                "device": "unavailable",
                "precision": base_config.precision,
                "status": "skipped",
                "reason": str(exc),
            }

    model = base_config.model
    model_overrides = {}
    if args.generations is not None:
        model_overrides["n_generations"] = args.generations
    if args.population_size is not None:
        model_overrides["population_size"] = args.population_size
    if args.n_genes is not None:
        model_overrides["n_genes"] = args.n_genes
    if args.quiet:
        model_overrides["verbose"] = False
    if model_overrides:
        model = replace(model, **model_overrides)

    output_dir = Path(args.output_root) / backend
    config = replace(
        base_config,
        backend=backend,
        loadsteps=_parse_loadsteps(args.loadsteps) if args.loadsteps is not None else base_config.loadsteps,
        output_dir=str(output_dir),
        model=model,
        progress_log=False if args.quiet else base_config.progress_log,
    )

    wall_start = time.perf_counter()
    cpu_start = time.process_time()
    result = SGEPWorkflow(config).train()
    outer_wall = time.perf_counter() - wall_start
    outer_cpu = time.process_time() - cpu_start
    return {
        "backend": backend,
        "device": _backend_device(backend),
        "precision": config.precision,
        "status": "ok",
        "best_expression": result.best_expression,
        "theta": result.theta,
        "metrics": result.metrics,
        "wall_seconds": outer_wall,
        "cpu_seconds": outer_cpu,
        "timing": result.timing,
        "output_paths": result.output_paths,
    }


def _comparison(results: dict[str, dict]) -> dict:
    jax_result = results.get("jax", {})
    torch_result = results.get("torch", {})
    if jax_result.get("status") != "ok" or torch_result.get("status") != "ok":
        return {"status": "skipped", "reason": "Both jax and torch must complete for comparison."}

    jax_theta = np.asarray(jax_result.get("theta", []), dtype=float)
    torch_theta = np.asarray(torch_result.get("theta", []), dtype=float)
    n_common = min(jax_theta.size, torch_theta.size)
    theta_linf = None
    if n_common:
        theta_linf = float(np.max(np.abs(jax_theta[:n_common] - torch_theta[:n_common])))
    rmse_delta = float(torch_result["metrics"]["rmse"] - jax_result["metrics"]["rmse"])
    wall_ratio = float(torch_result["wall_seconds"] / jax_result["wall_seconds"]) if jax_result["wall_seconds"] else None
    return {
        "status": "ok",
        "theta_linf": theta_linf,
        "rmse_delta": rmse_delta,
        "wall_ratio_torch_over_jax": wall_ratio,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="A/B benchmark SGEPPY JAX and Torch-CUDA backends.")
    parser.add_argument("--config", required=True, help="Path to a SGEPPY config JSON.")
    parser.add_argument("--backends", default="jax,torch", help="Comma-separated backends to run.")
    parser.add_argument("--loadsteps", default=None, help="Comma-separated load steps.")
    parser.add_argument("--generations", type=int, default=None)
    parser.add_argument("--population-size", type=int, default=None)
    parser.add_argument("--n-genes", type=int, default=None)
    parser.add_argument("--output-root", default="tmp/sgeppy_backend_ab")
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--quiet", action="store_true", default=False)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    base_config = config_from_file(args.config)
    backends = _parse_csv(args.backends)
    unknown = sorted(set(backends) - {"jax", "torch"})
    if unknown:
        raise ValueError("Unknown backend(s): %s" % ", ".join(unknown))

    results = {}
    for backend in backends:
        print("[sgeppy-backend-ab] running backend=%s" % backend, flush=True)
        results[backend] = _safe_json(_run_backend(base_config, backend, args))
        status = results[backend]["status"]
        if status == "ok":
            print(
                "[sgeppy-backend-ab] done backend=%s rmse=%.6e wall=%.3fs"
                % (
                    backend,
                    results[backend]["metrics"]["rmse"],
                    results[backend]["wall_seconds"],
                ),
                flush=True,
            )
        else:
            print(
                "[sgeppy-backend-ab] skipped backend=%s reason=%s"
                % (backend, results[backend].get("reason", "")),
                flush=True,
            )

    report = {
        "config": str(args.config),
        "results": results,
        "comparison": _comparison(results),
    }
    if args.output_json is not None:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(_safe_json(report), indent=2) + "\n", encoding="utf-8")
        print("[sgeppy-backend-ab] wrote %s" % output_path, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
