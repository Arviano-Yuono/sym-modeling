from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from sym_modeling.domains.fem import (  # noqa: E402
    BENCHMARK_BOUNDARY_TAGS,
    BENCHMARK_CELL_TAG,
    ForwardFEMBenchmarkConfig,
    SUPPORTED_FORWARD_BENCHMARK_MODELS,
    plot_forward_benchmark_loadsteps,
    run_forward_hyperelastic_benchmark,
)


class ForwardBenchmarkApiTests(unittest.TestCase):
    def test_public_symbols_are_available(self):
        self.assertEqual(ForwardFEMBenchmarkConfig.__name__, "ForwardFEMBenchmarkConfig")
        self.assertEqual(run_forward_hyperelastic_benchmark.__name__, "run_forward_hyperelastic_benchmark")
        self.assertEqual(plot_forward_benchmark_loadsteps.__name__, "plot_forward_benchmark_loadsteps")
        self.assertIn("NH2", SUPPORTED_FORWARD_BENCHMARK_MODELS)
        self.assertIn("GT", SUPPORTED_FORWARD_BENCHMARK_MODELS)
        self.assertEqual(BENCHMARK_CELL_TAG, 11)

    def test_boundary_tag_map_contains_all_boundaries(self):
        self.assertEqual(set(BENCHMARK_BOUNDARY_TAGS.keys()), {"LEFT", "BOTTOM", "RIGHT", "TOP", "HOLE"})
        self.assertEqual(len(set(BENCHMARK_BOUNDARY_TAGS.values())), 5)

    def test_default_load_steps_depend_on_material(self):
        nh2 = ForwardFEMBenchmarkConfig(material_model="NH2")
        hw = ForwardFEMBenchmarkConfig(material_model="HW")
        self.assertEqual(len(nh2.resolved_load_steps), 4)
        self.assertEqual(len(hw.resolved_load_steps), 8)
        self.assertAlmostEqual(nh2.resolved_load_steps[-1], 0.4)
        self.assertAlmostEqual(hw.resolved_load_steps[-1], 0.8)

    def test_custom_load_steps_override_default(self):
        config = ForwardFEMBenchmarkConfig(material_model="IH", load_steps=(0.1, 0.25, 0.5))
        self.assertEqual(config.resolved_load_steps, (0.1, 0.25, 0.5))

    def test_config_accepts_input_msh_path(self):
        config = ForwardFEMBenchmarkConfig(
            input_msh_path="dataset/fem_data/plate_hole_fenics/mesh/mesh.msh",
            left_tag=7,
            bottom_tag=10,
            right_tag=9,
            top_tag=8,
            hole_tag=6,
            domain_tag=11,
        )
        self.assertEqual(config.boundary_tags["LEFT"], 7)
        self.assertEqual(config.domain_tag, 11)

    def test_output_dir_overrides_output_root(self):
        config = ForwardFEMBenchmarkConfig(
            output_root="generated/fem/forward_benchmark",
            output_dir="generated/fem/custom_output",
        )
        self.assertEqual(str(config.resolved_output_dir), "generated/fem/custom_output")

    def test_config_rejects_unknown_material(self):
        with self.assertRaises(ValueError):
            ForwardFEMBenchmarkConfig(material_model="FOO")

    def test_config_rejects_invalid_geometry(self):
        with self.assertRaises(ValueError):
            ForwardFEMBenchmarkConfig(hole_radius=1.0, outer_size=1.0)

    def test_config_rejects_invalid_load_steps(self):
        with self.assertRaises(ValueError):
            ForwardFEMBenchmarkConfig(load_steps=())
        with self.assertRaises(ValueError):
            ForwardFEMBenchmarkConfig(load_steps=(0.2, 0.1))
        with self.assertRaises(ValueError):
            ForwardFEMBenchmarkConfig(load_steps=(0.1, 0.1))

    def test_plot_forward_benchmark_loadsteps(self):
        coordinates = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=float,
        )
        displacements = [
            np.array(
                [
                    [0.0, 0.0],
                    [0.02, 0.0],
                    [0.0, 0.02],
                ],
                dtype=float,
            ),
            np.array(
                [
                    [0.0, 0.0],
                    [0.05, 0.0],
                    [0.0, 0.05],
                ],
                dtype=float,
            ),
        ]
        results = {
            "load_steps": [0.1, 0.2],
            "nodal_coordinates": coordinates,
            "nodal_displacements": displacements,
            "triangles": np.array([[0, 1, 2]], dtype=np.int32),
        }
        fig, axes = plot_forward_benchmark_loadsteps(results, show=False)
        self.assertIsNotNone(fig)
        self.assertEqual(axes.shape[0], 1)


if __name__ == "__main__":
    unittest.main()
