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
DEFAULT_NOISES = ("0", "1e-4", "1e-3")


def _parse_csv_strings(value: str) -> list[str]:
    return [part.strip().lower() for part in value.split(",") if part.strip()]


def _noise_label(value: str) -> str:
    text = value.strip().lower()
    return "0" if float(text) == 0.0 else text


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run SGEPPY weak-form FEM sweeps over model and noise configs."
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
        "--noises",
        default=",".join(DEFAULT_NOISES),
        help="Comma-separated noise levels, e.g. 0,1e-4,1e-3.",
    )
    parser.add_argument(
        "--output-root",
        default="output/sgep_weak",
        help="Root directory for sweep outputs.",
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
    noises = [_noise_label(value) for value in _parse_csv_strings(args.noises)]

    for noise in noises:
        noise_level = float(noise)
        for model_name in models:
            config_path = config_dir / ("%s.json" % model_name)
            if not config_path.exists():
                raise FileNotFoundError("Missing SGEPPY config for model '%s': %s" % (model_name, config_path))

            base_config = config_from_file(config_path)
            output_dir = output_root / noise / model_name
            model_config = replace(base_config.model, verbose=False) if args.quiet else base_config.model
            config = replace(
                base_config,
                model=model_config,
                fitting_mode="weak_form",
                noise_level=noise_level,
                output_dir=str(output_dir),
                progress_log=False if args.quiet else base_config.progress_log,
            )

            print(
                "[sgeppy-weak-sweep] model=%s noise=%s output=%s"
                % (model_name, noise, output_dir),
                flush=True,
            )
            if args.dry_run:
                continue

            result = SGEPWorkflow(config).train()
            print(
                "[sgeppy-weak-sweep] done model=%s noise=%s rmse=%.6e active_terms=%d wall=%.3fs"
                % (
                    model_name,
                    noise,
                    result.metrics["rmse"],
                    result.metrics["num_parameters"],
                    result.timing["wall_seconds"],
                ),
                flush=True,
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
