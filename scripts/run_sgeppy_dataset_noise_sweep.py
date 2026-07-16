"""
Run SGEPPY over pre-noised or pre-denoised FEM datasets.

Example:
uv run python scripts/run_sgeppy_dataset_noise_sweep.py \
  --config-dir configs/sgeppy \
  --noises 0,1e-4,1e-3 \
  --output-root output/run/sgeppy_final \
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


DEFAULT_MODELS = ("gt", "hw", "ih", "nh2", "nh4", "ab")
DEFAULT_NOISES = ("0", "1e-4", "1e-3")


def _parse_csv_strings(value: str) -> list[str]:
    return [part.strip().lower() for part in value.split(",") if part.strip()]


def _noise_label(value: str) -> str:
    text = value.strip().lower()
    return "0" if float(text) == 0.0 else text


def _replace_path_part(path: str | Path, old: str, new: str) -> str:
    parts = list(Path(path).parts)
    try:
        index = parts.index(old)
    except ValueError as exc:
        raise ValueError("Could not find path segment '%s' in %s" % (old, path)) from exc
    parts[index] = new
    return str(Path(*parts))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run SGEPPY sweeps where noise is represented by dataset/output "
            "directories, not by SGEPPY's noise_level input."
        )
    )
    parser.add_argument(
        "--config-dir",
        default="configs/sgeppy",
        help="Directory containing per-model SGEPPY JSON configs.",
    )
    parser.add_argument(
        "--models",
        default=",".join(DEFAULT_MODELS),
        help="Comma-separated model names to run, e.g. gt,hw,ih,nh2,nh4,ab.",
    )
    parser.add_argument(
        "--noises",
        default=",".join(DEFAULT_NOISES),
        help="Comma-separated dataset noise directory labels, e.g. 0,1e-4,1e-3.",
    )
    parser.add_argument(
        "--reference-noise",
        default="0",
        help="Path segment in the config data_dir/output_dir to replace.",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help=(
            "Optional output root. If omitted, each config output_dir has the "
            "reference-noise path segment replaced by the sweep noise label."
        ),
    )
    parser.add_argument(
        "--backend",
        choices=("jax", "torch"),
        default=None,
        help="Override the weak-form backend from the loaded config.",
    )
    parser.add_argument(
        "--skip-missing",
        action="store_true",
        help="Skip runs whose rewritten data_dir does not exist.",
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
    output_root = None if args.output_root is None else Path(args.output_root)
    reference_noise = _noise_label(args.reference_noise)
    models = _parse_csv_strings(args.models)
    noises = [_noise_label(value) for value in _parse_csv_strings(args.noises)]

    for noise in noises:
        for model_name in models:
            config_path = config_dir / ("%s.json" % model_name)
            if not config_path.exists():
                raise FileNotFoundError("Missing SGEPPY config for model '%s': %s" % (model_name, config_path))

            base_config = config_from_file(config_path)
            data_dir = _replace_path_part(base_config.data_dir, reference_noise, noise)
            if output_root is None:
                output_dir = _replace_path_part(base_config.output_dir, reference_noise, noise)
            else:
                output_dir = str(output_root / noise / model_name)

            if args.skip_missing and not Path(data_dir).is_dir():
                print(
                    "[sgeppy-dataset-noise-sweep] skip missing data model=%s noise=%s data=%s"
                    % (model_name, noise, data_dir),
                    flush=True,
                )
                continue

            model_config = replace(base_config.model, verbose=False) if args.quiet else base_config.model
            backend = args.backend if args.backend is not None else base_config.backend
            config = replace(
                base_config,
                model=model_config,
                backend=backend,
                data_dir=data_dir,
                noise_level=0.0,
                output_dir=output_dir,
                progress_log=False if args.quiet else base_config.progress_log,
            )

            print(
                "[sgeppy-dataset-noise-sweep] model=%s noise=%s backend=%s data=%s output=%s"
                % (model_name, noise, config.backend, data_dir, output_dir),
                flush=True,
            )
            if args.dry_run:
                continue

            result = SGEPWorkflow(config).train()
            print(
                "[sgeppy-dataset-noise-sweep] done model=%s noise=%s rmse=%.6e active_terms=%d wall=%.3fs"
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
