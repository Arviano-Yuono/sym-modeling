from __future__ import annotations

import json
import math
import sys
import tempfile
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
from sym_modeling.domains.fem.forward_benchmark import (  # noqa: E402
    _arruda_boyce_energy_density,
    _load_sgeppy_expression,
    _sgeppy_energy_density_from_expression,
)


class _FakeUFL:
    @staticmethod
    def sqrt(value):
        return math.sqrt(value)

    @staticmethod
    def ln(value):
        return math.log(value)

    @staticmethod
    def exp(value):
        return math.exp(value)

    @staticmethod
    def sin(value):
        return math.sin(value)

    @staticmethod
    def cos(value):
        return math.cos(value)

    @staticmethod
    def conditional(condition, true_value, false_value):
        return true_value if condition else false_value

    @staticmethod
    def lt(left, right):
        return left < right

    @staticmethod
    def min_value(left, right):
        return min(left, right)

    @staticmethod
    def max_value(left, right):
        return max(left, right)


def _fake_invariants():
    return {
        "I1": 3.2,
        "I2": 3.4,
        "I3": 1.1,
        "J": 1.05,
        "I1_bar": 3.2,
        "I2_bar": 3.4,
    }


class ForwardBenchmarkApiTests(unittest.TestCase):
    def test_public_symbols_are_available(self):
        self.assertEqual(ForwardFEMBenchmarkConfig.__name__, "ForwardFEMBenchmarkConfig")
        self.assertEqual(run_forward_hyperelastic_benchmark.__name__, "run_forward_hyperelastic_benchmark")
        self.assertEqual(plot_forward_benchmark_loadsteps.__name__, "plot_forward_benchmark_loadsteps")
        self.assertIn("NH2", SUPPORTED_FORWARD_BENCHMARK_MODELS)
        self.assertIn("GT", SUPPORTED_FORWARD_BENCHMARK_MODELS)
        self.assertIn("AB", SUPPORTED_FORWARD_BENCHMARK_MODELS)
        self.assertIn("SGEPPY", SUPPORTED_FORWARD_BENCHMARK_MODELS)
        self.assertEqual(BENCHMARK_CELL_TAG, 11)

    def test_boundary_tag_map_contains_all_boundaries(self):
        self.assertEqual(set(BENCHMARK_BOUNDARY_TAGS.keys()), {"LEFT", "BOTTOM", "RIGHT", "TOP", "HOLE"})
        self.assertEqual(len(set(BENCHMARK_BOUNDARY_TAGS.values())), 5)

    def test_default_load_steps_depend_on_material(self):
        nh2 = ForwardFEMBenchmarkConfig(material_model="NH2")
        hw = ForwardFEMBenchmarkConfig(material_model="HW")
        ab = ForwardFEMBenchmarkConfig(material_model="AB")
        self.assertEqual(len(nh2.resolved_load_steps), 4)
        self.assertEqual(len(hw.resolved_load_steps), 8)
        self.assertEqual(len(ab.resolved_load_steps), 10)
        self.assertAlmostEqual(nh2.resolved_load_steps[-1], 0.4)
        self.assertAlmostEqual(hw.resolved_load_steps[-1], 0.8)
        self.assertAlmostEqual(ab.resolved_load_steps[-1], 0.5)

    def test_arruda_boyce_energy_is_zero_at_reference_state(self):
        energy = _arruda_boyce_energy_density(
            I1_bar=3.0,
            J=1.0,
            mu=1.0,
            lambda_m=3.0,
            bulk_modulus=3.0,
        )
        self.assertAlmostEqual(energy, 0.0)

    def test_config_rejects_invalid_arruda_boyce_parameters(self):
        with self.assertRaises(ValueError):
            ForwardFEMBenchmarkConfig(arruda_boyce_mu=0.0)
        with self.assertRaises(ValueError):
            ForwardFEMBenchmarkConfig(arruda_boyce_lambda_m=1.0)
        with self.assertRaises(ValueError):
            ForwardFEMBenchmarkConfig(arruda_boyce_bulk_modulus=0.0)

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

    def test_sgeppy_config_requires_one_expression_source(self):
        with self.assertRaises(ValueError):
            ForwardFEMBenchmarkConfig(material_model="SGEPPY")
        with self.assertRaises(ValueError):
            ForwardFEMBenchmarkConfig(
                material_model="SGEPPY",
                sgeppy_expression="K1",
                sgeppy_expression_path="best_so_far.json",
            )
        with self.assertRaises(ValueError):
            ForwardFEMBenchmarkConfig(material_model="NH2", sgeppy_expression="K1")

        config = ForwardFEMBenchmarkConfig(material_model="SGEPPY", sgeppy_expression="K1")
        self.assertEqual(len(config.resolved_load_steps), 4)
        self.assertAlmostEqual(config.resolved_load_steps[-1], 0.4)

    def test_sgeppy_expression_can_be_loaded_from_best_json(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "best_so_far.json"
            path.write_text(json.dumps({"best_expression": "0.5*K1 + 1.5*Jm1**2"}), encoding="utf-8")
            self.assertEqual(_load_sgeppy_expression(path), "0.5*K1 + 1.5*Jm1**2")

    def test_sgeppy_expression_parser_supports_generated_functions(self):
        expression = (
            "-0.280479773763*Jm1*K2 + 0.280479773763*protected_exp(K1) "
            "+ protected_div(K1, 0.0) - protected_sqrt(K2) "
            "+ square(logI13) + cube(logI23)"
        )
        value = _sgeppy_energy_density_from_expression(
            expression,
            ufl=_FakeUFL(),
            invariants=_fake_invariants(),
        )
        self.assertTrue(np.isfinite(value))

    def test_sgeppy_expression_parser_rejects_unsafe_syntax(self):
        with self.assertRaises(ValueError):
            _sgeppy_energy_density_from_expression(
                "__import__('os').system('echo nope')",
                ufl=_FakeUFL(),
                invariants=_fake_invariants(),
            )

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
