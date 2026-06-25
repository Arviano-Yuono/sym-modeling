from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np


DEFAULT_PLOT_QUANTITIES = (
    "umag",
    "ux",
    "uy",
    "Fnorm",
    "Fxx",
    "Fxy",
    "Fyx",
    "Fyy",
    "Pnorm",
    "Pxx",
    "Pxy",
    "Pyx",
    "Pyy",
)
DISPLACEMENT_QUANTITIES = {"umag", "ux", "uy"}
F_QUANTITIES = {"Fnorm", "Fxx", "Fxy", "Fyx", "Fyy"}
P_QUANTITIES = {"Pnorm", "Pxx", "Pxy", "Pyx", "Pyy"}
F_COLUMNS = ("Fxx", "Fxy", "Fyx", "Fyy")
P_COLUMNS = ("Pxx", "Pxy", "Pyx", "Pyy")


@dataclass(frozen=True)
class ForwardComparisonConfig:
    """Configuration for comparing generated FEM forward output to a reference dataset."""

    forward_root: str | Path
    reference_root: str | Path
    output_dir: str | Path | None = None
    load_steps: Sequence[int | float | str] | None = None
    plot_quantities: Sequence[str] | str = "all"

    @property
    def resolved_forward_root(self) -> Path:
        return Path(self.forward_root)

    @property
    def resolved_reference_root(self) -> Path:
        return Path(self.reference_root)

    @property
    def resolved_output_dir(self) -> Path:
        if self.output_dir is not None:
            return Path(self.output_dir)
        return self.resolved_forward_root / "comparison"

    @property
    def resolved_plot_quantities(self) -> tuple[str, ...]:
        return _normalize_plot_quantities(self.plot_quantities)


@dataclass(frozen=True)
class ForwardComparisonResult:
    """Machine-readable result for a forward FEM comparison run."""

    forward_root: Path
    reference_root: Path
    output_dir: Path
    compared_steps: tuple[str, ...]
    missing_forward_steps: tuple[str, ...]
    missing_reference_steps: tuple[str, ...]
    metrics: tuple[dict[str, Any], ...]
    summary_path: Path
    metrics_csv_path: Path
    plot_paths: tuple[Path, ...]
    summary: dict[str, Any]


@dataclass(frozen=True)
class _StepData:
    step: str
    node_ids: np.ndarray
    coordinates: np.ndarray
    displacement: np.ndarray
    connectivity: np.ndarray
    triangles: np.ndarray
    deformation_gradient: np.ndarray
    piola: np.ndarray | None
    reactions: np.ndarray


def compare_forward_results(config: ForwardComparisonConfig) -> ForwardComparisonResult:
    """Compare shared FEM load-step folders and write metrics plus diagnostic plots."""

    forward_root = config.resolved_forward_root
    reference_root = config.resolved_reference_root
    output_dir = config.resolved_output_dir
    plot_quantities = config.resolved_plot_quantities

    forward_steps = _numeric_step_dirs(forward_root)
    reference_steps = _numeric_step_dirs(reference_root)
    step_pairs = _resolve_compared_steps(config.load_steps, forward_steps, reference_steps)
    if not step_pairs:
        raise ValueError(
            "No shared numeric load-step folders found between forward_root=%s and reference_root=%s."
            % (forward_root, reference_root)
        )
    compared_steps = [_comparison_step_label(step_value, forward_step, reference_step) for step_value, forward_step, reference_step in step_pairs]

    output_dir.mkdir(parents=True, exist_ok=True)

    metrics: list[dict[str, Any]] = []
    plot_paths: list[Path] = []
    piola_missing_steps: list[str] = []
    reaction_reference: list[np.ndarray] = []
    reaction_forward: list[np.ndarray] = []
    reaction_step_values: list[float] = []

    for step, (step_value, forward_step, reference_step) in zip(compared_steps, step_pairs):
        reference = _load_step(reference_root, reference_step)
        forward = _load_step(forward_root, forward_step)
        _validate_matching_mesh(reference, forward)

        reaction_reference.append(reference.reactions)
        reaction_forward.append(forward.reactions)
        reaction_step_values.append(step_value)

        metrics.extend(_metric_rows(step, "reaction", "forces", reference.reactions, forward.reactions))
        metrics.extend(_displacement_metric_rows(step, reference.displacement, forward.displacement))
        metrics.extend(
            _component_metric_rows(
                step,
                "F",
                F_COLUMNS,
                reference.deformation_gradient,
                forward.deformation_gradient,
            )
        )

        has_reference_piola = reference.piola is not None
        has_forward_piola = forward.piola is not None
        if has_reference_piola and has_forward_piola:
            metrics.extend(
                _component_metric_rows(
                    step,
                    "P",
                    P_COLUMNS,
                    reference.piola,
                    forward.piola,
                )
            )
        else:
            piola_missing_steps.append(step)

        for quantity in plot_quantities:
            if quantity in P_QUANTITIES and not (has_reference_piola and has_forward_piola):
                continue
            path = _plot_step_quantity(output_dir, step, quantity, reference, forward)
            plot_paths.append(path)

    reaction_plot = _plot_reaction_comparison(
        output_dir,
        np.asarray(reaction_step_values, dtype=float),
        reaction_reference,
        reaction_forward,
    )
    plot_paths.insert(0, reaction_plot)

    if reaction_reference:
        metrics.extend(
            _metric_rows(
                "all",
                "reaction_curve",
                "forces",
                np.concatenate(reaction_reference),
                np.concatenate(reaction_forward),
            )
        )

    metrics_csv_path = output_dir / "comparison_metrics.csv"
    _write_metrics_csv(metrics_csv_path, metrics)

    summary_path = output_dir / "comparison_summary.json"
    summary = _build_summary(
        config=config,
        compared_steps=compared_steps,
        forward_steps=forward_steps,
        reference_steps=reference_steps,
        metrics=metrics,
        piola_missing_steps=piola_missing_steps,
        plot_paths=plot_paths,
        metrics_csv_path=metrics_csv_path,
        summary_path=summary_path,
    )
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    return ForwardComparisonResult(
        forward_root=forward_root,
        reference_root=reference_root,
        output_dir=output_dir,
        compared_steps=tuple(compared_steps),
        missing_forward_steps=tuple(_missing_steps(reference_steps, forward_steps)),
        missing_reference_steps=tuple(_missing_steps(forward_steps, reference_steps)),
        metrics=tuple(metrics),
        summary_path=summary_path,
        metrics_csv_path=metrics_csv_path,
        plot_paths=tuple(plot_paths),
        summary=summary,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare a generated FEM forward result directory against a reference FEM dataset."
    )
    parser.add_argument("--forward-root", required=True, help="Generated forward FEM output root.")
    parser.add_argument("--reference-root", required=True, help="Reference FEM dataset root.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Comparison output directory. Defaults to <forward-root>/comparison.",
    )
    parser.add_argument(
        "--load-step",
        dest="load_steps",
        action="append",
        default=None,
        help="Numeric load-step folder to compare. May be passed multiple times.",
    )
    parser.add_argument(
        "--plot-quantities",
        default="all",
        help="Comma-separated plot quantities, or 'all'.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = compare_forward_results(
        ForwardComparisonConfig(
            forward_root=args.forward_root,
            reference_root=args.reference_root,
            output_dir=args.output_dir,
            load_steps=args.load_steps,
            plot_quantities=args.plot_quantities,
        )
    )
    print("Compared load steps: %s" % ", ".join(result.compared_steps))
    print("Wrote comparison summary: %s" % result.summary_path)
    print("Wrote comparison metrics: %s" % result.metrics_csv_path)
    return 0


def _normalize_plot_quantities(plot_quantities: Sequence[str] | str) -> tuple[str, ...]:
    if plot_quantities == "all":
        return DEFAULT_PLOT_QUANTITIES
    if isinstance(plot_quantities, str):
        values = tuple(item.strip() for item in plot_quantities.split(",") if item.strip())
    else:
        values = tuple(str(item).strip() for item in plot_quantities if str(item).strip())
    if not values:
        raise ValueError("plot_quantities must contain at least one quantity, or be 'all'.")

    allowed = set(DEFAULT_PLOT_QUANTITIES)
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError("Unknown plot quantities: %s" % ", ".join(unknown))
    return values


def _numeric_step_dirs(root: Path) -> dict[float, str]:
    if not root.exists():
        raise FileNotFoundError("FEM result root does not exist: %s" % root)

    steps: dict[float, str] = {}
    for path in root.iterdir():
        if not path.is_dir():
            continue
        try:
            key = _step_sort_key(path.name)
        except ValueError:
            continue
        if key in steps:
            raise ValueError("Duplicate numeric load-step folders under %s: %s and %s" % (root, steps[key], path.name))
        steps[key] = path.name
    return steps


def _resolve_compared_steps(
    requested_steps: Sequence[int | float | str] | None,
    forward_steps: dict[float, str],
    reference_steps: dict[float, str],
) -> list[tuple[float, str, str]]:
    if requested_steps is None:
        shared = sorted(set(forward_steps) & set(reference_steps))
        return [(key, forward_steps[key], reference_steps[key]) for key in shared]

    steps: list[tuple[float, str, str]] = []
    missing: list[str] = []
    for requested in requested_steps:
        label = str(requested)
        key = _step_sort_key(label)
        if key not in forward_steps or key not in reference_steps:
            missing.append(label)
            continue
        steps.append((key, forward_steps[key], reference_steps[key]))
    if missing:
        raise ValueError("Requested load steps are not present in both roots: %s" % ", ".join(missing))
    return steps


def _comparison_step_label(step_value: float, forward_step: str, reference_step: str) -> str:
    if forward_step == reference_step:
        return forward_step
    if float(step_value).is_integer():
        return str(int(step_value))
    return ("%g" % step_value)


def _missing_steps(source: dict[float, str], target: dict[float, str]) -> list[str]:
    return [source[key] for key in sorted(set(source) - set(target))]


def _step_sort_key(step: str) -> float:
    value = float(step)
    if not np.isfinite(value):
        raise ValueError("Load-step folder is not finite: %s" % step)
    return value


def _load_step(root: Path, step: str) -> _StepData:
    step_dir = root / step
    nodes = _read_named_csv(step_dir / "output_nodes.csv")
    elements = _read_named_csv(step_dir / "output_elements.csv")
    reactions = _read_reactions(step_dir / "output_reactions.csv")

    required_node_columns = ("id", "x", "y", "ux", "uy")
    _require_columns(nodes, required_node_columns, step_dir / "output_nodes.csv")
    _require_columns(elements, ("node1", "node2", "node3", *F_COLUMNS), step_dir / "output_elements.csv")

    node_ids = _column(nodes, "id").astype(np.int64)
    coordinates = np.column_stack((_column(nodes, "x"), _column(nodes, "y")))
    displacement = np.column_stack((_column(nodes, "ux"), _column(nodes, "uy")))
    connectivity = np.column_stack(
        (_column(elements, "node1"), _column(elements, "node2"), _column(elements, "node3"))
    ).astype(np.int64)
    triangles = _connectivity_to_row_indices(node_ids, connectivity, step_dir / "output_elements.csv")
    deformation_gradient = _columns(elements, F_COLUMNS)
    piola = _columns(elements, P_COLUMNS) if _has_columns(elements, P_COLUMNS) else None

    return _StepData(
        step=step,
        node_ids=node_ids,
        coordinates=coordinates,
        displacement=displacement,
        connectivity=connectivity,
        triangles=triangles,
        deformation_gradient=deformation_gradient,
        piola=piola,
        reactions=reactions,
    )


def _read_named_csv(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError("Missing required FEM CSV: %s" % path)
    data = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding=None)
    return np.atleast_1d(data)


def _read_reactions(path: Path) -> np.ndarray:
    data = _read_named_csv(path)
    if data.dtype.names is None or len(data.dtype.names) == 0:
        raise ValueError("Reaction CSV must have a header: %s" % path)
    if "forces" in data.dtype.names:
        values = _column(data, "forces")
    else:
        values = _column(data, data.dtype.names[0])
    return np.asarray(values, dtype=float).reshape(-1)


def _require_columns(data: np.ndarray, columns: Sequence[str], path: Path) -> None:
    missing = [column for column in columns if data.dtype.names is None or column not in data.dtype.names]
    if missing:
        raise ValueError("Missing columns in %s: %s" % (path, ", ".join(missing)))


def _has_columns(data: np.ndarray, columns: Sequence[str]) -> bool:
    return data.dtype.names is not None and all(column in data.dtype.names for column in columns)


def _column(data: np.ndarray, name: str) -> np.ndarray:
    return np.asarray(data[name], dtype=float).reshape(-1)


def _columns(data: np.ndarray, names: Sequence[str]) -> np.ndarray:
    return np.column_stack([_column(data, name) for name in names])


def _connectivity_to_row_indices(node_ids: np.ndarray, connectivity: np.ndarray, path: Path) -> np.ndarray:
    id_to_row = {int(node_id): index for index, node_id in enumerate(node_ids)}
    triangles = np.empty_like(connectivity, dtype=np.int64)
    for row_index in range(connectivity.shape[0]):
        for col_index in range(connectivity.shape[1]):
            node_id = int(connectivity[row_index, col_index])
            if node_id not in id_to_row:
                raise ValueError("Connectivity in %s references missing node id %s." % (path, node_id))
            triangles[row_index, col_index] = id_to_row[node_id]
    return triangles


def _validate_matching_mesh(reference: _StepData, forward: _StepData) -> None:
    if not np.array_equal(reference.node_ids, forward.node_ids):
        raise ValueError("Node ids/order do not match for load step %s." % reference.step)
    if reference.coordinates.shape != forward.coordinates.shape or not np.allclose(
        reference.coordinates, forward.coordinates, rtol=0.0, atol=1e-12
    ):
        raise ValueError("Node coordinates/order do not match for load step %s." % reference.step)
    if not np.array_equal(reference.connectivity, forward.connectivity):
        raise ValueError("Element connectivity/order does not match for load step %s." % reference.step)
    if reference.reactions.shape != forward.reactions.shape:
        raise ValueError("Reaction vector length does not match for load step %s." % reference.step)


def _metric_rows(
    step: str,
    field: str,
    quantity: str,
    reference_values: np.ndarray,
    forward_values: np.ndarray,
) -> list[dict[str, Any]]:
    metrics = _compute_metrics(reference_values, forward_values)
    return [
        {
            "step": step,
            "field": field,
            "quantity": quantity,
            **metrics,
        }
    ]


def _displacement_metric_rows(
    step: str,
    reference_displacement: np.ndarray,
    forward_displacement: np.ndarray,
) -> list[dict[str, Any]]:
    rows = _metric_rows(step, "u", "u", reference_displacement, forward_displacement)
    rows.extend(_metric_rows(step, "u", "ux", reference_displacement[:, 0], forward_displacement[:, 0]))
    rows.extend(_metric_rows(step, "u", "uy", reference_displacement[:, 1], forward_displacement[:, 1]))
    rows.extend(
        _metric_rows(
            step,
            "u",
            "umag",
            np.linalg.norm(reference_displacement, axis=1),
            np.linalg.norm(forward_displacement, axis=1),
        )
    )
    return rows


def _component_metric_rows(
    step: str,
    field: str,
    column_names: Sequence[str],
    reference_values: np.ndarray,
    forward_values: np.ndarray,
) -> list[dict[str, Any]]:
    rows = _metric_rows(step, field, field, reference_values, forward_values)
    norm_name = "%snorm" % field
    rows.extend(
        _metric_rows(
            step,
            field,
            norm_name,
            np.linalg.norm(reference_values.reshape(-1, 2, 2), axis=(1, 2)),
            np.linalg.norm(forward_values.reshape(-1, 2, 2), axis=(1, 2)),
        )
    )
    for index, column_name in enumerate(column_names):
        rows.extend(_metric_rows(step, field, column_name, reference_values[:, index], forward_values[:, index]))
    return rows


def _compute_metrics(reference_values: np.ndarray, forward_values: np.ndarray) -> dict[str, float]:
    reference = np.asarray(reference_values, dtype=float).reshape(-1)
    forward = np.asarray(forward_values, dtype=float).reshape(-1)
    if reference.shape != forward.shape:
        raise ValueError("Metric arrays have different shapes: %s != %s" % (reference.shape, forward.shape))
    diff = forward - reference
    reference_norm = float(np.linalg.norm(reference))
    diff_norm = float(np.linalg.norm(diff))
    if reference_norm == 0.0:
        relative_l2 = 0.0 if diff_norm == 0.0 else float("inf")
    else:
        relative_l2 = diff_norm / reference_norm
    return {
        "rmse": float(np.sqrt(np.mean(np.square(diff)))),
        "mae": float(np.mean(np.abs(diff))),
        "max_abs": float(np.max(np.abs(diff))),
        "relative_l2": float(relative_l2),
    }


def _plot_step_quantity(output_dir: Path, step: str, quantity: str, reference: _StepData, forward: _StepData) -> Path:
    import matplotlib

    matplotlib.use("Agg", force=True)
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.colors import Normalize, TwoSlopeNorm
    from matplotlib.figure import Figure
    import matplotlib.tri as mtri

    reference_values = _quantity_values(quantity, reference)
    forward_values = _quantity_values(quantity, forward)
    error_values = forward_values - reference_values
    triangulation = mtri.Triangulation(reference.coordinates[:, 0], reference.coordinates[:, 1], reference.triangles)

    field_min = float(np.min(np.concatenate([reference_values, forward_values])))
    field_max = float(np.max(np.concatenate([reference_values, forward_values])))
    if np.isclose(field_min, field_max):
        margin = max(1.0, abs(field_min)) * 1e-12
        field_min -= margin
        field_max += margin
    error_abs_max = float(np.max(np.abs(error_values)))
    if np.isclose(error_abs_max, 0.0):
        error_abs_max = 1e-12

    fig = Figure(figsize=(13.0, 4.0), constrained_layout=True)
    FigureCanvasAgg(fig)
    axes = [fig.add_subplot(1, 3, index + 1) for index in range(3)]
    field_norm = Normalize(vmin=field_min, vmax=field_max)
    error_norm = TwoSlopeNorm(vmin=-error_abs_max, vcenter=0.0, vmax=error_abs_max)

    is_element_quantity = quantity in F_QUANTITIES or quantity in P_QUANTITIES
    artists = []
    for ax, values, title, cmap, norm in (
        (axes[0], reference_values, "reference", "viridis", field_norm),
        (axes[1], forward_values, "forward", "viridis", field_norm),
        (axes[2], error_values, "forward - reference", "coolwarm", error_norm),
    ):
        if is_element_quantity:
            artist = ax.tripcolor(
                triangulation,
                facecolors=values,
                shading="flat",
                cmap=cmap,
                norm=norm,
            )
        else:
            artist = ax.tripcolor(
                triangulation,
                values,
                shading="gouraud",
                cmap=cmap,
                norm=norm,
            )
        ax.set_title(title)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        artists.append(artist)

    fig.suptitle("step %s: %s" % (step, quantity))
    fig.colorbar(artists[0], ax=axes[:2], label=quantity)
    fig.colorbar(artists[2], ax=axes[2], label="error")

    output_path = output_dir / ("step_%s_%s.png" % (step, _plot_quantity_filename(quantity)))
    fig.savefig(output_path, dpi=160)
    return output_path


def _plot_quantity_filename(quantity: str) -> str:
    if quantity in DISPLACEMENT_QUANTITIES:
        return "displacement_%s" % quantity
    return quantity


def _quantity_values(quantity: str, step: _StepData) -> np.ndarray:
    if quantity == "ux":
        return step.displacement[:, 0]
    if quantity == "uy":
        return step.displacement[:, 1]
    if quantity == "umag":
        return np.linalg.norm(step.displacement, axis=1)

    if quantity in F_QUANTITIES:
        return _matrix_quantity_values(quantity, "F", step.deformation_gradient)

    if quantity in P_QUANTITIES:
        if step.piola is None:
            raise ValueError("Piola columns are missing for load step %s." % step.step)
        return _matrix_quantity_values(quantity, "P", step.piola)

    raise ValueError("Unknown plot quantity: %s" % quantity)


def _matrix_quantity_values(quantity: str, prefix: str, values: np.ndarray) -> np.ndarray:
    if quantity == "%snorm" % prefix:
        return np.linalg.norm(values.reshape(-1, 2, 2), axis=(1, 2))
    names = F_COLUMNS if prefix == "F" else P_COLUMNS
    return values[:, names.index(quantity)]


def _plot_reaction_comparison(
    output_dir: Path,
    steps: np.ndarray,
    reference_reactions: Sequence[np.ndarray],
    forward_reactions: Sequence[np.ndarray],
) -> Path:
    import matplotlib

    matplotlib.use("Agg", force=True)
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    reference = np.vstack(reference_reactions)
    forward = np.vstack(forward_reactions)
    labels = ("left_x", "right_x", "bottom_y", "top_y")
    if reference.shape[1] != len(labels):
        labels = tuple("force_%d" % index for index in range(reference.shape[1]))

    fig = Figure(figsize=(7.0, 4.8), constrained_layout=True)
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(1, 1, 1)
    for index, label in enumerate(labels):
        ax.plot(steps, reference[:, index], "-", label="reference %s" % label)
        ax.plot(steps, forward[:, index], "--", label="forward %s" % label)
    ax.set_xlabel("load step")
    ax.set_ylabel("reaction force")
    ax.set_title("Reaction comparison")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncols=2)
    output_path = output_dir / "reaction_comparison.png"
    fig.savefig(output_path, dpi=160)
    return output_path


def _write_metrics_csv(path: Path, metrics: Sequence[dict[str, Any]]) -> None:
    fieldnames = ("step", "field", "quantity", "rmse", "mae", "max_abs", "relative_l2")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metrics)


def _build_summary(
    config: ForwardComparisonConfig,
    compared_steps: Sequence[str],
    forward_steps: dict[float, str],
    reference_steps: dict[float, str],
    metrics: Sequence[dict[str, Any]],
    piola_missing_steps: Sequence[str],
    plot_paths: Sequence[Path],
    metrics_csv_path: Path,
    summary_path: Path,
) -> dict[str, Any]:
    forward_summary = _read_json_if_exists(config.resolved_forward_root / "summary.json")
    return {
        "forward_root": str(config.resolved_forward_root),
        "reference_root": str(config.resolved_reference_root),
        "output_dir": str(config.resolved_output_dir),
        "compared_steps": list(compared_steps),
        "missing_forward_steps": _missing_steps(reference_steps, forward_steps),
        "missing_reference_steps": _missing_steps(forward_steps, reference_steps),
        "requested_load_steps": None if config.load_steps is None else [str(step) for step in config.load_steps],
        "plot_quantities": list(config.resolved_plot_quantities),
        "piola": {
            "available_for_all_compared_steps": len(piola_missing_steps) == 0,
            "missing_steps": list(piola_missing_steps),
        },
        "forward_checks": None if forward_summary is None else forward_summary.get("checks"),
        "metrics": list(metrics),
        "files": {
            "comparison_summary_json": str(summary_path),
            "comparison_metrics_csv": str(metrics_csv_path),
            "plots": [str(path) for path in plot_paths],
        },
    }


def _read_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


__all__ = [
    "DEFAULT_PLOT_QUANTITIES",
    "ForwardComparisonConfig",
    "ForwardComparisonResult",
    "build_parser",
    "compare_forward_results",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
