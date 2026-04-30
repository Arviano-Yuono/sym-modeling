from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np


def plot_structured_no_interp(
    y: np.ndarray,
    z: np.ndarray,
    val: np.ndarray,
    ax=None,
    cmap: str = "RdBu_r",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
):
    """
    Plot structured (y,z,val) data without interpolation.

    Works with random ordering of points as long as the point set is a full
    tensor-product grid in y-z (ny * nz == N).

    Returns:
        fig, ax, mappable
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    y = np.asarray(y, dtype=float).reshape(-1)
    z = np.asarray(z, dtype=float).reshape(-1)
    val = np.asarray(val, dtype=float).reshape(-1)

    if y.shape[0] != z.shape[0] or y.shape[0] != val.shape[0]:
        raise ValueError(
            "y, z, and val must have the same length. "
            f"Got y={y.shape}, z={z.shape}, val={val.shape}"
        )

    y_vec = np.unique(y)
    z_vec = np.unique(z)
    ny, nz = len(y_vec), len(z_vec)

    if ny * nz != val.shape[0]:
        raise ValueError(
            "Data is not a full tensor-product y-z grid "
            f"(ny*nz={ny*nz} != N={val.shape[0]})."
        )

    iy = np.searchsorted(y_vec, y)
    iz = np.searchsorted(z_vec, z)

    V = np.empty((ny, nz), dtype=float)
    V[iy, iz] = val

    Zg, Yg = np.meshgrid(z_vec, y_vec, indexing="xy")
    mappable = ax.pcolormesh(
        Zg,
        Yg,
        V,
        shading="nearest",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    return fig, ax, mappable


def plot_2d_field_triplet(
    y: np.ndarray,
    z: np.ndarray,
    dns_val: np.ndarray,
    baseline_val: np.ndarray,
    sparta_val: np.ndarray,
    field_label: str,
    output_path: Path,
    cmap: str = "RdBu_r",
    figsize: Tuple[float, float] = (15.0, 4.8),
) -> Path:
    """
    Save a 3-panel 2D field comparison (DNS / Baseline / SpaRTA).
    """
    import matplotlib.pyplot as plt

    y = np.asarray(y, dtype=float).reshape(-1)
    z = np.asarray(z, dtype=float).reshape(-1)
    dns_val = np.asarray(dns_val, dtype=float).reshape(-1)
    baseline_val = np.asarray(baseline_val, dtype=float).reshape(-1)
    sparta_val = np.asarray(sparta_val, dtype=float).reshape(-1)

    all_values = np.concatenate([dns_val, baseline_val, sparta_val])
    vmin = float(np.nanmin(all_values))
    vmax = float(np.nanmax(all_values))

    fig, axes = plt.subplots(1, 3, figsize=figsize, constrained_layout=True)
    panels = [
        ("DNS", dns_val),
        ("Baseline", baseline_val),
        ("SpaRTA", sparta_val),
    ]

    for ax, (title, values) in zip(axes, panels):
        _, _, mappable = plot_structured_no_interp(
            y=y,
            z=z,
            val=values,
            ax=ax,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        fig.colorbar(mappable, ax=ax, label=field_label)
        ax.set_title(f"{title} {field_label}")
        ax.set_xlabel("z/h")
        ax.set_ylabel("y/h")
        ax.set_xlim(float(np.min(z)), float(np.max(z)))
        ax.set_ylim(float(np.min(y)), float(np.max(y)))
        ax.set_aspect("equal")

    fig.suptitle(f"{field_label} in y-z Plane", fontsize=13)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=170)
    plt.close(fig)
    return output_path


def plot_2d_baseline_sparta_difference(
    y: np.ndarray,
    z: np.ndarray,
    baseline_val: np.ndarray,
    sparta_val: np.ndarray,
    field_label: str,
    output_path: Path,
    value_cmap: str = "RdBu_r",
    diff_cmap: str = "coolwarm",
    figsize: Tuple[float, float] = (15.0, 4.8),
) -> Path:
    """
    Save a 3-panel 2D field comparison (Baseline / SpaRTA / SpaRTA-Baseline).
    """
    import matplotlib.pyplot as plt

    y = np.asarray(y, dtype=float).reshape(-1)
    z = np.asarray(z, dtype=float).reshape(-1)
    baseline_val = np.asarray(baseline_val, dtype=float).reshape(-1)
    sparta_val = np.asarray(sparta_val, dtype=float).reshape(-1)

    if baseline_val.shape[0] != sparta_val.shape[0]:
        raise ValueError(
            "baseline_val and sparta_val must have the same length. "
            f"Got baseline={baseline_val.shape}, sparta={sparta_val.shape}"
        )

    delta_val = sparta_val - baseline_val

    value_all = np.concatenate([baseline_val, sparta_val])
    value_vmin = float(np.nanmin(value_all))
    value_vmax = float(np.nanmax(value_all))
    delta_abs = float(np.nanmax(np.abs(delta_val)))
    delta_vmin = -delta_abs
    delta_vmax = delta_abs

    fig, axes = plt.subplots(1, 3, figsize=figsize, constrained_layout=True)
    panels = [
        ("Baseline", baseline_val, value_cmap, value_vmin, value_vmax),
        ("SpaRTA", sparta_val, value_cmap, value_vmin, value_vmax),
        ("SpaRTA - Baseline", delta_val, diff_cmap, delta_vmin, delta_vmax),
    ]

    for ax, (title, values, cmap, vmin, vmax) in zip(axes, panels):
        _, _, mappable = plot_structured_no_interp(
            y=y,
            z=z,
            val=values,
            ax=ax,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        fig.colorbar(mappable, ax=ax, label=field_label)
        ax.set_title(f"{title} {field_label}")
        ax.set_xlabel("z/h")
        ax.set_ylabel("y/h")
        ax.set_xlim(float(np.min(z)), float(np.max(z)))
        ax.set_ylim(float(np.min(y)), float(np.max(y)))
        ax.set_aspect("equal")

    fig.suptitle(f"Baseline vs SpaRTA Difference: {field_label}", fontsize=13)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=170)
    plt.close(fig)
    return output_path


def extract_field_components(
    values: np.ndarray,
    field_name: str,
) -> Sequence[Tuple[str, np.ndarray]]:
    """
    Convert a possibly vector/tensor field into named scalar components for plotting.
    """
    values = np.asarray(values, dtype=float)

    if values.ndim == 1:
        return [(field_name, values)]

    if field_name == "U" and values.ndim == 2 and values.shape[1] == 3:
        return [
            ("Ux", values[:, 0]),
            ("Uy", values[:, 1]),
            ("Uz", values[:, 2]),
        ]

    if field_name == "Rij" and values.ndim == 3 and values.shape[1:] == (3, 3):
        return [
            ("Rxx", values[:, 0, 0]),
            ("Ryy", values[:, 1, 1]),
            ("Rzz", values[:, 2, 2]),
            ("Rxy", values[:, 0, 1]),
            ("Rxz", values[:, 0, 2]),
            ("Ryz", values[:, 1, 2]),
        ]

    flat = values.reshape(values.shape[0], -1)
    return [(f"{field_name}[{idx}]", flat[:, idx]) for idx in range(flat.shape[1])]
