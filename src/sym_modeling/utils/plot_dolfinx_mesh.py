from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


DEFAULT_MSH_PATH = "dataset/fem_data/plate_hole_fenics/mesh/mesh_3k.msh"
DEFAULT_OUTPUT_PATH = "output/mesh/plate_hole_mesh.png"


def _read_from_msh(dolfinx_io, msh_path: Path, comm, gdim: int):
    if hasattr(dolfinx_io, "gmsh") and hasattr(dolfinx_io.gmsh, "read_from_msh"):
        mesh_data = dolfinx_io.gmsh.read_from_msh(str(msh_path), comm, gdim=gdim)
    elif hasattr(dolfinx_io, "gmshio") and hasattr(dolfinx_io.gmshio, "read_from_msh"):
        mesh_data = dolfinx_io.gmshio.read_from_msh(str(msh_path), comm, gdim=gdim)
    else:  # pragma: no cover - depends on dolfinx version
        raise AttributeError("Could not find read_from_msh in this DOLFINx build.")

    if hasattr(mesh_data, "mesh"):
        return mesh_data.mesh, getattr(mesh_data, "cell_tags", None)

    domain, cell_tags, _facet_tags = mesh_data
    return domain, cell_tags


TRIANGLE_ELEMENT_NODE_COUNTS = {
    2: 3,
    9: 6,
    21: 10,
}
QUAD_ELEMENT_NODE_COUNTS = {
    3: 4,
    10: 9,
    36: 16,
}


def read_gmsh_41_mesh(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Read 2D cells from a Gmsh 4.1 ASCII `.msh` file.

    Returns `(points, triangles, cell_tags)`.
    """
    path = Path(path)
    lines = path.read_text(encoding="utf-8").splitlines()
    nodes_start = lines.index("$Nodes") + 1
    num_node_blocks, _num_nodes, _min_node_tag, _max_node_tag = (
        int(value) for value in lines[nodes_start].split()
    )
    cursor = nodes_start + 1
    nodes: dict[int, tuple[float, float, float]] = {}
    for _ in range(num_node_blocks):
        _entity_dim, _entity_tag, parametric, num_nodes = (
            int(value) for value in lines[cursor].split()
        )
        cursor += 1
        tags = [int(lines[cursor + idx].split()[0]) for idx in range(num_nodes)]
        cursor += num_nodes
        for tag in tags:
            parts = [float(value) for value in lines[cursor].split()]
            cursor += 1
            nodes[tag] = (parts[0], parts[1], parts[2])
            if parametric:
                cursor += 0

    ordered_tags = sorted(nodes)
    tag_to_index = {tag: idx for idx, tag in enumerate(ordered_tags)}
    points = np.asarray([nodes[tag] for tag in ordered_tags], dtype=float)

    elements_start = lines.index("$Elements") + 1
    num_element_blocks, _num_elements, _min_element_tag, _max_element_tag = (
        int(value) for value in lines[elements_start].split()
    )
    cursor = elements_start + 1
    triangles = []
    cell_tags = []
    for _ in range(num_element_blocks):
        entity_dim, entity_tag, element_type, num_elements = (
            int(value) for value in lines[cursor].split()
        )
        cursor += 1
        if entity_dim != 2:
            cursor += num_elements
            continue

        for _element_idx in range(num_elements):
            values = [int(value) for value in lines[cursor].split()]
            cursor += 1
            node_tags = values[1:]
            if element_type in TRIANGLE_ELEMENT_NODE_COUNTS:
                corners = node_tags[:3]
                triangles.append([tag_to_index[tag] for tag in corners])
                cell_tags.append(entity_tag)
            elif element_type in QUAD_ELEMENT_NODE_COUNTS:
                corners = node_tags[:4]
                i0, i1, i2, i3 = [tag_to_index[tag] for tag in corners]
                triangles.append([i0, i1, i2])
                triangles.append([i0, i2, i3])
                cell_tags.extend([entity_tag, entity_tag])

    if not triangles:
        raise ValueError("No 2D triangle or quad cells found in %s." % path)
    return points, np.asarray(triangles, dtype=int), np.asarray(cell_tags, dtype=int)


def save_mesh_plot_matplotlib(
    msh_path: str | Path = DEFAULT_MSH_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    show_markers: bool = False,
    surface_color: str = "white",
    edge_color: str = "black",
    point_color: str = "black",
    background: str = "white",
    line_width: float = 0.6,
    point_size: float = 0.0,
    title: str | None = None,
    show_axes: bool = False,
) -> Path:
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    import matplotlib.tri as mtri

    msh_path = Path(msh_path)
    output_path = Path(output_path)
    points, triangles, cell_tags = read_gmsh_41_mesh(msh_path)
    triangulation = mtri.Triangulation(points[:, 0], points[:, 1], triangles)

    fig, ax = plt.subplots(figsize=(9, 8), facecolor=background)
    ax.set_facecolor(background)
    if show_markers:
        ax.tripcolor(
            triangulation,
            facecolors=cell_tags,
            cmap="tab20",
            edgecolors=edge_color,
            linewidth=line_width,
        )
    else:
        ax.tripcolor(
            triangulation,
            facecolors=np.zeros(triangles.shape[0], dtype=float),
            cmap=ListedColormap([surface_color]),
            edgecolors=edge_color,
            linewidth=line_width,
            alpha=1.0,
        )

    if point_size > 0.0:
        ax.scatter(points[:, 0], points[:, 1], s=point_size, c=point_color, linewidths=0.0)
    if title:
        ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    if not show_axes:
        ax.set_axis_off()
    fig.tight_layout(pad=0.05)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=250, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return output_path


def save_mesh_plot_pyvista(
    msh_path: str | Path = DEFAULT_MSH_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    gdim: int = 2,
    show_markers: bool = False,
    surface_color: str = "white",
    edge_color: str = "black",
    point_color: str = "black",
    background: str = "white",
    line_width: float = 1.0,
    point_size: float = 5.0,
    window_size: tuple[int, int] = (1400, 1000),
    title: str | None = None,
) -> Path:
    """
    Render a DOLFINx-readable Gmsh mesh to an image file.
    """
    from mpi4py import MPI

    import dolfinx
    import dolfinx.fem
    import dolfinx.io
    import dolfinx.plot
    import pyvista

    msh_path = Path(msh_path)
    output_path = Path(output_path)
    if not msh_path.is_file():
        raise FileNotFoundError(msh_path)

    pyvista.OFF_SCREEN = True
    domain, cell_tags = _read_from_msh(dolfinx.io, msh_path, MPI.COMM_WORLD, gdim)
    rank = MPI.COMM_WORLD.rank
    if rank != 0:
        return output_path

    V_linear = dolfinx.fem.functionspace(domain, ("Lagrange", 1))
    linear_grid = pyvista.UnstructuredGrid(*dolfinx.plot.vtk_mesh(V_linear))

    plotter = pyvista.Plotter(off_screen=True, window_size=window_size)
    plotter.set_background(background)
    if domain.geometry.cmap.degree > 1:
        curved_grid = pyvista.UnstructuredGrid(*dolfinx.plot.vtk_mesh(domain))
        plotter.add_mesh(
            curved_grid,
            style="points",
            color=point_color,
            point_size=point_size,
        )
        curved_grid = curved_grid.tessellate()
        if show_markers and cell_tags is not None and len(cell_tags.values) == curved_grid.n_cells:
            curved_grid.cell_data["Marker"] = cell_tags.values
            plotter.add_mesh(curved_grid, scalars="Marker", cmap="tab20", show_edges=False)
        else:
            plotter.add_mesh(curved_grid, color=surface_color, show_edges=False)
        plotter.add_mesh(
            linear_grid,
            style="wireframe",
            color=edge_color,
            line_width=line_width,
        )
    else:
        if show_markers and cell_tags is not None and len(cell_tags.values) == linear_grid.n_cells:
            linear_grid.cell_data["Marker"] = cell_tags.values
            plotter.add_mesh(
                linear_grid,
                scalars="Marker",
                cmap="tab20",
                show_edges=True,
                edge_color=edge_color,
                line_width=line_width,
            )
        else:
            plotter.add_mesh(
                linear_grid,
                color=surface_color,
                show_edges=True,
                edge_color=edge_color,
                line_width=line_width,
            )

    if title:
        plotter.add_title(title)
    plotter.view_xy()
    plotter.camera.zoom(1.15)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plotter.screenshot(str(output_path))
    plotter.close()
    return output_path


def save_mesh_plot(
    msh_path: str | Path = DEFAULT_MSH_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    gdim: int = 2,
    show_markers: bool = False,
    surface_color: str = "white",
    edge_color: str = "black",
    point_color: str = "black",
    background: str = "white",
    line_width: float = 0.6,
    point_size: float = 0.0,
    window_size: tuple[int, int] = (1400, 1000),
    title: str | None = None,
    backend: str = "matplotlib",
    show_axes: bool = False,
) -> Path:
    if backend == "matplotlib":
        return save_mesh_plot_matplotlib(
            msh_path=msh_path,
            output_path=output_path,
            show_markers=show_markers,
            surface_color=surface_color,
            edge_color=edge_color,
            point_color=point_color,
            background=background,
            line_width=line_width,
            point_size=point_size,
            title=title,
            show_axes=show_axes,
        )
    if backend == "pyvista":
        return save_mesh_plot_pyvista(
            msh_path=msh_path,
            output_path=output_path,
            gdim=gdim,
            show_markers=show_markers,
            surface_color=surface_color,
            edge_color=edge_color,
            point_color=point_color,
            background=background,
            line_width=line_width,
            point_size=point_size,
            window_size=window_size,
            title=title,
        )
    raise ValueError("backend must be one of: matplotlib, pyvista")


def _parse_window_size(value: str) -> tuple[int, int]:
    parts = value.lower().replace("x", ",").split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("window size must look like 1400x1000")
    try:
        width, height = int(parts[0]), int(parts[1])
    except ValueError as exc:
        raise argparse.ArgumentTypeError("window size must contain integers") from exc
    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError("window size values must be positive")
    return width, height


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render a clean DOLFINx/Gmsh mesh image from the command line."
    )
    parser.add_argument("--msh", default=DEFAULT_MSH_PATH, help="Input Gmsh .msh file.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH, help="Output image path.")
    parser.add_argument("--backend", default="matplotlib", choices=("matplotlib", "pyvista"))
    parser.add_argument("--gdim", type=int, default=2, choices=(2, 3))
    parser.add_argument("--show-markers", action="store_true", help="Color cells by mesh marker tags.")
    parser.add_argument("--surface-color", default="white")
    parser.add_argument("--edge-color", default="black")
    parser.add_argument("--point-color", default="black")
    parser.add_argument("--background", default="white")
    parser.add_argument("--line-width", type=float, default=0.6)
    parser.add_argument("--point-size", type=float, default=0.0)
    parser.add_argument("--window-size", type=_parse_window_size, default=(1400, 1000))
    parser.add_argument("--title", default=None)
    parser.add_argument("--show-axes", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_path = save_mesh_plot(
        msh_path=args.msh,
        output_path=args.output,
        gdim=args.gdim,
        show_markers=args.show_markers,
        surface_color=args.surface_color,
        edge_color=args.edge_color,
        point_color=args.point_color,
        background=args.background,
        line_width=args.line_width,
        point_size=args.point_size,
        window_size=args.window_size,
        title=args.title,
        backend=args.backend,
        show_axes=args.show_axes,
    )
    print("Wrote mesh image: %s" % output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
