from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from sym_modeling.domains.fem import (  # noqa: E402
    ForwardFEMBenchmarkConfig,
    run_forward_hyperelastic_benchmark,
)


DEFAULT_INPUT_MSH_PATH = "dataset/fem_data/plate_hole_fenics/mesh/mesh_3k.msh"
DEFAULT_OUTPUT_DIR = "dataset/fem_data/plate_hole_fenics/AB"
DEFAULT_LOAD_STEPS = tuple(0.05 * step for step in range(1, 11))


def _parse_load_steps(value: str) -> tuple[float, ...]:
    load_steps = tuple(float(part.strip()) for part in value.split(",") if part.strip())
    if not load_steps:
        raise argparse.ArgumentTypeError("load steps must contain at least one value")
    return load_steps


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate an EUCLID-compatible Arruda-Boyce plate-hole dataset with DOLFINx."
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Dataset output directory.",
    )
    parser.add_argument(
        "--input-msh-path",
        default=DEFAULT_INPUT_MSH_PATH,
        help="Tagged Gmsh mesh to reuse.",
    )
    parser.add_argument(
        "--load-steps",
        type=_parse_load_steps,
        default=DEFAULT_LOAD_STEPS,
        help="Comma-separated prescribed displacement steps, e.g. 0.05,0.10,0.15.",
    )
    parser.add_argument("--mu", type=float, default=1.0, help="Arruda-Boyce energy scale.")
    parser.add_argument(
        "--lambda-m",
        type=float,
        default=3.0,
        help="Arruda-Boyce limiting chain stretch.",
    )
    parser.add_argument(
        "--bulk-modulus",
        type=float,
        default=3.0,
        help="Volumetric penalty bulk modulus.",
    )
    parser.add_argument("--solver-atol", type=float, default=1e-12)
    parser.add_argument("--solver-rtol", type=float, default=1e-12)
    parser.add_argument("--solver-max-it", type=int, default=50)
    parser.add_argument("--initial-load-subdivisions", type=int, default=1)
    parser.add_argument("--max-load-subdivisions", type=int, default=32)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    input_msh_path = Path(args.input_msh_path)
    if not input_msh_path.is_file():
        raise FileNotFoundError("Input mesh does not exist: %s" % input_msh_path)

    config = ForwardFEMBenchmarkConfig(
        material_model="AB",
        output_dir=args.output_dir,
        load_steps=args.load_steps,
        input_msh_path=input_msh_path,
        solver_atol=args.solver_atol,
        solver_rtol=args.solver_rtol,
        solver_max_it=args.solver_max_it,
        initial_load_subdivisions=args.initial_load_subdivisions,
        max_load_subdivisions=args.max_load_subdivisions,
        arruda_boyce_mu=args.mu,
        arruda_boyce_lambda_m=args.lambda_m,
        arruda_boyce_bulk_modulus=args.bulk_modulus,
        left_tag=7,
        bottom_tag=10,
        right_tag=9,
        top_tag=8,
        hole_tag=6,
        domain_tag=11,
    )
    print(
        "[arruda-boyce] generating %d load steps in %s"
        % (len(config.resolved_load_steps), config.resolved_output_dir),
        flush=True,
    )
    result = run_forward_hyperelastic_benchmark(config)
    print("[arruda-boyce] summary: %s" % result["files"]["summary_json"], flush=True)
    print("[arruda-boyce] checks: %s" % result["checks"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
