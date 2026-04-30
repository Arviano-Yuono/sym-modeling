from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class StructuredGrid:
    """Axis-aligned grid description for CFD point or cell data."""

    x: np.ndarray = field(default_factory=lambda: np.empty((0,), dtype=float))
    y: np.ndarray = field(default_factory=lambda: np.empty((0,), dtype=float))
    z: np.ndarray = field(default_factory=lambda: np.empty((0,), dtype=float))

    @property
    def shape(self) -> tuple[int, int, int]:
        return (int(self.x.shape[0]), int(self.y.shape[0]), int(self.z.shape[0]))


@dataclass
class TriangularMesh:
    """Minimal 2D FEM mesh container."""

    nodes: np.ndarray = field(default_factory=lambda: np.empty((0, 2), dtype=float))
    connectivity: list[np.ndarray] = field(default_factory=list)
