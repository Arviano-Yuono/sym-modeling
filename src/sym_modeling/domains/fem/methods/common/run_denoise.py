from __future__ import annotations

import argparse

from sym_modeling.domains.fem.methods.common.denoising import (
    DEFAULT_ALPHAS,
    DEFAULT_BLENDS,
    DEFAULT_GAMMAS,
    DEFAULT_LAPLACIAN_LAMBDAS,
    METHODS,
    OBJECTIVES,
    DenoiseSearchConfig,
    MeshLaplacianCandidate,
    search_denoise_hyperparameters,
)


def _parse_int_csv(value: str | None) -> list[int] | None:
    if value is None or value.strip() == "":
        return None
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def _parse_float_csv(value: str | None) -> tuple[float, ...]:
    if value is None or value.strip() == "":
        raise ValueError("CSV value must not be empty.")
    return tuple(float(part.strip()) for part in value.split(",") if part.strip())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a denoised FEM CSV dataset.")
    parser.add_argument("--method", choices=METHODS, default="krr", help="Denoising method. Defaults to KRR.")
    parser.add_argument("--data-dir", required=True, help="Clean FEM dataset root with numeric load-step folders.")
    parser.add_argument("--output-dir", required=True, help="Output dataset root to write.")
    parser.add_argument("--loadsteps", default=None, help="Comma-separated load steps. Defaults to dataset discovery.")
    parser.add_argument("--noise-level", type=float, default=0.0, help="Artificial displacement noise level to add first.")
    parser.add_argument("--seed", type=int, default=20260623, help="Base seed for deterministic artificial noise.")
    parser.add_argument("--objective", choices=OBJECTIVES, default="F_rmse", help="Metric minimized during search.")
    parser.add_argument(
        "--alphas",
        default=",".join("%g" % value for value in DEFAULT_ALPHAS),
        help="Comma-separated KernelRidge alpha values.",
    )
    parser.add_argument(
        "--gammas",
        default=",".join("%g" % value for value in DEFAULT_GAMMAS),
        help="Comma-separated RBF gamma values.",
    )
    parser.add_argument(
        "--lambdas",
        default=",".join("%g" % value for value in DEFAULT_LAPLACIAN_LAMBDAS),
        help="Comma-separated mesh-Laplacian smoothness values.",
    )
    parser.add_argument(
        "--blends",
        default=",".join("%g" % value for value in DEFAULT_BLENDS),
        help="Comma-separated blend values between noisy and denoised prediction.",
    )
    parser.add_argument("--boundary-weight", type=float, default=500.0)
    parser.add_argument("--no-preserve-dirichlet", action="store_true", default=False)
    parser.add_argument("--overwrite", action="store_true", default=False)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = DenoiseSearchConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        loadsteps=_parse_int_csv(args.loadsteps),
        method=str(args.method),
        noise_level=float(args.noise_level),
        seed=int(args.seed),
        objective=str(args.objective),
        alphas=_parse_float_csv(args.alphas),
        gammas=_parse_float_csv(args.gammas),
        lambdas=_parse_float_csv(args.lambdas),
        blends=_parse_float_csv(args.blends),
        boundary_weight=float(args.boundary_weight),
        preserve_dirichlet=not bool(args.no_preserve_dirichlet),
        overwrite=bool(args.overwrite),
    )
    result = search_denoise_hyperparameters(config)
    candidate = result.selected_candidate
    if isinstance(candidate, MeshLaplacianCandidate):
        message = "[fem-denoise] selected method=%s lambda_smooth=%g blend=%g %s=%.6e"
        values = (config.method, candidate.lambda_smooth, candidate.blend, config.objective, result.selected_score)
    else:
        message = "[fem-denoise] selected method=%s alpha=%g gamma=%g blend=%g %s=%.6e"
        values = (config.method, candidate.alpha, candidate.gamma, candidate.blend, config.objective, result.selected_score)
    print(message % values, flush=True)
    print("[fem-denoise] output: %s" % result.output_dir, flush=True)
    print("[fem-denoise] summary: %s" % result.summary_path, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
