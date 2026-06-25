from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_DATA_DIR = Path("dataset/fem_data/thermocorr/formatted_sampled_3k")
DEFAULT_OUTPUT_PATH = Path("output/mesh/thermocorr_mesh.png")


@dataclass(frozen=True)
class ThermoCorrMeshPlotData:
    step_dir: Path
    nodes: np.ndarray
    connectivity: np.ndarray
    bcx: np.ndarray
    bcy: np.ndarray


def _numeric_step_dirs(data_dir: Path) -> list[Path]:
    step_dirs = []
    for child in data_dir.iterdir():
        if not child.is_dir():
            continue
        try:
            int(child.name)
        except ValueError:
            continue
        step_dirs.append(child)
    return sorted(step_dirs, key=lambda path: int(path.name))


def _resolve_step_dir(data_dir: str | Path, step: int | None) -> Path:
    data_dir = Path(data_dir)
    if (data_dir / "output_nodes.csv").is_file() and (data_dir / "output_elements.csv").is_file():
        return data_dir

    if step is not None:
        step_dir = data_dir / str(int(step))
        if not step_dir.is_dir():
            raise FileNotFoundError("ThermoCorr load-step directory does not exist: %s" % step_dir)
        return step_dir

    step_dirs = _numeric_step_dirs(data_dir)
    if not step_dirs:
        raise FileNotFoundError("No numeric ThermoCorr load-step directories found in %s." % data_dir)
    return step_dirs[0]


def load_thermocorr_mesh(data_dir: str | Path = DEFAULT_DATA_DIR, step: int | None = None) -> ThermoCorrMeshPlotData:
    step_dir = _resolve_step_dir(data_dir, step)
    nodes = pd.read_csv(step_dir / "output_nodes.csv")
    elements = pd.read_csv(step_dir / "output_elements.csv")

    required_node_columns = {"x", "y", "bcx", "bcy"}
    required_element_columns = {"node1", "node2", "node3"}
    missing_nodes = sorted(required_node_columns.difference(nodes.columns))
    missing_elements = sorted(required_element_columns.difference(elements.columns))
    if missing_nodes:
        raise ValueError("Missing node columns in %s: %s" % (step_dir / "output_nodes.csv", missing_nodes))
    if missing_elements:
        raise ValueError(
            "Missing element columns in %s: %s" % (step_dir / "output_elements.csv", missing_elements)
        )

    points = nodes[["x", "y"]].to_numpy(dtype=float)
    connectivity = elements[["node1", "node2", "node3"]].round().astype(int).to_numpy()
    if connectivity.size and (connectivity.min() < 0 or connectivity.max() >= points.shape[0]):
        raise ValueError("Element connectivity references nodes outside output_nodes.csv.")

    return ThermoCorrMeshPlotData(
        step_dir=step_dir,
        nodes=points,
        connectivity=connectivity,
        bcx=nodes["bcx"].round().astype(int).to_numpy(),
        bcy=nodes["bcy"].round().astype(int).to_numpy(),
    )


def _triangle_areas(nodes: np.ndarray, connectivity: np.ndarray) -> np.ndarray:
    p0 = nodes[connectivity[:, 0]]
    p1 = nodes[connectivity[:, 1]]
    p2 = nodes[connectivity[:, 2]]
    return 0.5 * np.abs((p1[:, 0] - p0[:, 0]) * (p2[:, 1] - p0[:, 1]) - (p2[:, 0] - p0[:, 0]) * (p1[:, 1] - p0[:, 1]))


def save_thermocorr_mesh_plot(
    data_dir: str | Path = DEFAULT_DATA_DIR,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    step: int | None = None,
    show_nodes: bool = False,
    show_boundary: bool = True,
    line_width: float = 0.25,
    node_size: float = 1.0,
    boundary_size: float = 12.0,
    title: str | None = None,
) -> Path:
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri
    from matplotlib.colors import ListedColormap

    plot_data = load_thermocorr_mesh(data_dir, step=step)
    output_path = Path(output_path)
    triangulation = mtri.Triangulation(
        plot_data.nodes[:, 0],
        plot_data.nodes[:, 1],
        plot_data.connectivity,
    )
    areas = _triangle_areas(plot_data.nodes, plot_data.connectivity)

    fig, ax = plt.subplots(figsize=(7.5, 10.0), facecolor="white")
    ax.tripcolor(
        triangulation,
        facecolors=np.zeros(plot_data.connectivity.shape[0], dtype=float),
        cmap=ListedColormap(["#eef1f4"]),
        edgecolors="#293241",
        linewidth=line_width,
        alpha=1.0,
    )

    if show_nodes:
        ax.scatter(
            plot_data.nodes[:, 0],
            plot_data.nodes[:, 1],
            s=node_size,
            c="#1d3557",
            linewidths=0.0,
            alpha=0.65,
            label="nodes",
        )

    boundary_nodes = (plot_data.bcx != 0) | (plot_data.bcy != 0)
    if show_boundary and np.any(boundary_nodes):
        ax.scatter(
            plot_data.nodes[boundary_nodes, 0],
            plot_data.nodes[boundary_nodes, 1],
            s=boundary_size,
            c="#d62828",
            linewidths=0.0,
            label="reaction/bc nodes",
        )
        ax.legend(loc="upper left", frameon=False)

    if title is None:
        dataset_name = plot_data.step_dir.parent.name
        title = (
            "ThermoCorr %s step %s: %d nodes, %d triangles, area %.4g"
            % (
                dataset_name,
                plot_data.step_dir.name,
                plot_data.nodes.shape[0],
                plot_data.connectivity.shape[0],
                float(np.sum(areas)),
            )
        )
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("x (normalized)")
    ax.set_ylabel("y (normalized)")
    ax.set_aspect("equal", adjustable="box")
    ax.margins(0.02)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=250, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot a formatted ThermoCorr FEM CSV mesh.")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR), help="Formatted ThermoCorr dataset root or step directory.")
    parser.add_argument("--step", type=int, default=None, help="Load step to plot. Defaults to the first numeric step.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH), help="Output image path.")
    parser.add_argument("--show-nodes", action="store_true", help="Overlay all mesh nodes.")
    parser.add_argument("--hide-boundary", action="store_true", help="Do not highlight constrained/reaction nodes.")
    parser.add_argument("--line-width", type=float, default=0.25, help="Triangle edge line width.")
    parser.add_argument("--node-size", type=float, default=1.0, help="Node marker size when --show-nodes is set.")
    parser.add_argument("--boundary-size", type=float, default=12.0, help="Highlighted boundary marker size.")
    parser.add_argument("--title", default=None, help="Optional plot title.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_path = save_thermocorr_mesh_plot(
        data_dir=args.data_dir,
        output_path=args.output,
        step=args.step,
        show_nodes=args.show_nodes,
        show_boundary=not args.hide_boundary,
        line_width=args.line_width,
        node_size=args.node_size,
        boundary_size=args.boundary_size,
        title=args.title,
    )
    print("Wrote ThermoCorr mesh image: %s" % output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
