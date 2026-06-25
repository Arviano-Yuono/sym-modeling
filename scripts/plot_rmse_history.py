"""
Plot RMSE history from one or more SGEP history.csv files.

Examples:
  uv run python scripts/plot_rmse_history.py output/sgep/nh4/history.csv
  uv run python scripts/plot_rmse_history.py output/sgep_direct_piola --output output/sgep_direct_piola/rmse_history.png
  uv run python scripts/plot_rmse_history.py output/sgep/nh2/history.csv output/sgep/nh4/history.csv --log-y
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import LogFormatterSciNotation  # noqa: E402


GENERATION_COLUMNS = ("generation", "gen")
RMSE_COLUMNS = ("best_rmse", "rmse", "rmse_normalized", "rmse_original_units", "min")


def _first_existing_column(fieldnames: list[str], candidates: tuple[str, ...], kind: str) -> str:
    for candidate in candidates:
        if candidate in fieldnames:
            return candidate
    raise ValueError(
        "Could not find a %s column. Expected one of: %s. Found: %s"
        % (kind, ", ".join(candidates), ", ".join(fieldnames))
    )


def _history_files(inputs: list[Path]) -> list[Path]:
    files: list[Path] = []
    for path in inputs:
        if path.is_dir():
            files.extend(sorted(path.rglob("history.csv")))
        elif path.is_file():
            files.append(path)
        else:
            raise FileNotFoundError(path)
    if not files:
        raise FileNotFoundError("No history.csv files found in the given inputs.")
    return files


def _series_label(path: Path, all_paths: list[Path]) -> str:
    if len(all_paths) == 1:
        return path.parent.name
    common_root = Path(os.path.commonpath([str(item.parent) for item in all_paths]))
    try:
        label = path.parent.relative_to(common_root)
    except ValueError:
        label = path.parent
    return str(label) if str(label) != "." else path.parent.name


def read_history(
    path: Path,
    rmse_column: str | None = None,
    require_positive: bool = False,
) -> tuple[list[float], list[float], str, int]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        generation_column = _first_existing_column(fieldnames, GENERATION_COLUMNS, "generation")
        value_column = rmse_column or _first_existing_column(fieldnames, RMSE_COLUMNS, "RMSE")
        if value_column not in fieldnames:
            raise ValueError("Column %r does not exist in %s" % (value_column, path))

        generations: list[float] = []
        rmse_values: list[float] = []
        skipped_nonpositive = 0
        for row in reader:
            try:
                generation = float(row[generation_column])
                rmse = float(row[value_column])
            except (TypeError, ValueError):
                continue
            if math.isfinite(generation) and math.isfinite(rmse):
                if require_positive and rmse <= 0.0:
                    skipped_nonpositive += 1
                    continue
                generations.append(generation)
                rmse_values.append(rmse)

    if not generations:
        raise ValueError("No numeric history rows found in %s" % path)
    return generations, rmse_values, value_column, skipped_nonpositive


def default_output_path(inputs: list[Path]) -> Path:
    if len(inputs) == 1:
        path = inputs[0]
        if path.is_file():
            return path.with_name("rmse_history.png")
        return path / "rmse_history.png"
    return Path("rmse_history.png")


def plot_histories(
    history_paths: list[Path],
    output_path: Path,
    rmse_column: str | None = None,
    yscale: str = "linear",
) -> set[str]:
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    detected_columns: set[str] = set()
    use_markers = len(history_paths) <= 8
    log_y = yscale == "log"

    for path in history_paths:
        generations, rmse_values, detected_column, skipped_nonpositive = read_history(
            path,
            rmse_column,
            require_positive=log_y,
        )
        detected_columns.add(detected_column)
        plot = ax.semilogy if log_y else ax.plot
        plot(
            generations,
            rmse_values,
            marker="o" if use_markers else None,
            markersize=3,
            linewidth=1.7,
            label=_series_label(path, history_paths),
        )
        if skipped_nonpositive:
            print("Skipped %d nonpositive values for log scale in %s" % (skipped_nonpositive, path))

    ax.set_xlabel("Generation")
    ylabel = rmse_column or (next(iter(detected_columns)) if len(detected_columns) == 1 else "RMSE")
    ax.set_ylabel(ylabel.replace("_", " ").upper() if ylabel == "rmse" else ylabel.replace("_", " "))
    title_suffix = " (log y-axis)" if log_y else ""
    ax.set_title("RMSE vs Generation%s" % title_suffix)
    ax.grid(True, alpha=0.3)
    if log_y:
        ax.yaxis.set_major_formatter(LogFormatterSciNotation())
        ax.grid(True, which="minor", alpha=0.15)
    if len(history_paths) <= 20:
        ax.legend(fontsize=8)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return detected_columns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot RMSE versus generation from SGEP history.csv files.")
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="history.csv files or directories to search recursively for history.csv.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="PNG output path. Defaults beside the input file, inside the input directory, or ./rmse_history.png.",
    )
    parser.add_argument(
        "--rmse-column",
        help="Column to plot on the y-axis. Defaults to auto-detecting best_rmse/rmse/rmse_normalized/min.",
    )
    parser.add_argument("--log-y", "--logy", action="store_true", help="Use a logarithmic y-axis.")
    parser.add_argument(
        "--yscale",
        choices=("linear", "log"),
        default="linear",
        help="Y-axis scale. --log-y is a shortcut for --yscale log.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    history_paths = _history_files(args.inputs)
    output_path = args.output or default_output_path(args.inputs)
    yscale = "log" if args.log_y else args.yscale
    detected_columns = plot_histories(history_paths, output_path, args.rmse_column, yscale)
    print("Saved RMSE history plot to %s" % output_path)
    print("Y-axis scale: %s" % yscale)
    print("Y-axis column(s): %s" % ", ".join(sorted(detected_columns)))


if __name__ == "__main__":
    main()
