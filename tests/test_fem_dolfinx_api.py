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
    DolfinxHyperelasticityConfig,
    DolfinxHyperelasticitySimulation,
    SUPPORTED_HYPERELASTIC_MODELS,
    DolfinxSimulationData,
    plot_simulation_data,
    read_msh_physical_names,
    setup_hyperelasticity_from_msh,
    setup_hyperelasticity_simulation,
)
from sym_modeling.domains.fem.dolfinx import SUPPORTED_BOX_BOUNDARIES  # noqa: E402


class DolfinxApiTests(unittest.TestCase):
    def test_public_api_imports_are_available(self):
        self.assertEqual(DolfinxHyperelasticitySimulation.__name__, "DolfinxHyperelasticitySimulation")
        self.assertEqual(setup_hyperelasticity_from_msh.__name__, "setup_hyperelasticity_from_msh")
        self.assertEqual(read_msh_physical_names.__name__, "read_msh_physical_names")
        self.assertEqual(plot_simulation_data.__name__, "plot_simulation_data")
        self.assertEqual(setup_hyperelasticity_simulation.__name__, "setup_hyperelasticity_simulation")
        self.assertIn("left", SUPPORTED_BOX_BOUNDARIES)
        self.assertIn("right", SUPPORTED_BOX_BOUNDARIES)
        self.assertIn("neo_hookean_log", SUPPORTED_HYPERELASTIC_MODELS)
        self.assertIn("st_venant_kirchhoff", SUPPORTED_HYPERELASTIC_MODELS)
        self.assertIn("mooney_rivlin", SUPPORTED_HYPERELASTIC_MODELS)

    def test_config_defaults_match_notebook_shape(self):
        config = DolfinxHyperelasticityConfig()

        self.assertEqual(config.lower_corner, (0.0, 0.0, 0.0))
        self.assertEqual(config.upper_corner, (20.0, 1.0, 1.0))
        self.assertEqual(config.cells, (20, 5, 5))
        self.assertEqual(config.element_degree, 2)
        self.assertEqual(config.material_model, "neo_hookean_log")
        self.assertEqual(config.fixed_boundary, "left")
        self.assertEqual(config.traction_boundary, "right")

    def test_config_rejects_invalid_boundary_pair(self):
        with self.assertRaises(ValueError):
            DolfinxHyperelasticityConfig(
                fixed_boundary="left",
                traction_boundary="left",
            )

    def test_config_rejects_invalid_vector_length(self):
        with self.assertRaises(ValueError):
            DolfinxHyperelasticityConfig(body_force=(0.0,))

        with self.assertRaises(ValueError):
            DolfinxHyperelasticityConfig(traction=(0.0,))

    def test_config_rejects_invalid_material_model(self):
        with self.assertRaises(ValueError):
            DolfinxHyperelasticityConfig(material_model="arruda_boyce")

    def test_config_rejects_invalid_mooney_parameters(self):
        with self.assertRaises(ValueError):
            DolfinxHyperelasticityConfig(material_model="mooney_rivlin", mooney_c10=-1.0)
        with self.assertRaises(ValueError):
            DolfinxHyperelasticityConfig(material_model="mooney_rivlin", mooney_c01=-1.0)
        with self.assertRaises(ValueError):
            DolfinxHyperelasticityConfig(material_model="mooney_rivlin", bulk_modulus=0.0)

    def test_msh_mode_requires_boundary_specs(self):
        with self.assertRaises(ValueError):
            DolfinxHyperelasticityConfig(
                msh_path="dataset/fem_data/plate_hole_fenics/mesh/mesh.msh",
                traction_boundary_tag=9,
            )

        with self.assertRaises(ValueError):
            DolfinxHyperelasticityConfig(
                msh_path="dataset/fem_data/plate_hole_fenics/mesh/mesh.msh",
                fixed_boundary_tag=7,
            )

    def test_msh_mode_accepts_tag_spec(self):
        config = DolfinxHyperelasticityConfig(
            msh_path="dataset/fem_data/plate_hole_fenics/mesh/mesh.msh",
            msh_gdim=2,
            fixed_boundary_tag=7,
            traction_boundary_tag=9,
            body_force=(0.0, 0.0),
            traction=(1.0, 0.0),
        )
        self.assertEqual(config.msh_gdim, 2)
        self.assertEqual(config.fixed_boundary_tag, 7)
        self.assertEqual(config.traction_boundary_tag, 9)

    def test_read_msh_physical_names(self):
        names = read_msh_physical_names("dataset/fem_data/plate_hole_fenics/mesh/mesh.msh")
        self.assertEqual(names[(1, 7)], "left")
        self.assertEqual(names[(1, 9)], "right")
        self.assertEqual(names[(2, 11)], "surface")

    def test_plot_simulation_data_matplotlib_backend(self):
        data = DolfinxSimulationData(
            points=np.array(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                ],
                dtype=float,
            ),
            topology=np.array([3, 0, 1, 2], dtype=np.int32),
            cell_types=np.array([5], dtype=np.uint8),
            displacement=np.array(
                [
                    [0.0, 0.0],
                    [0.1, 0.0],
                    [0.0, 0.1],
                ],
                dtype=float,
            ),
            displacement_magnitude=np.array([0.0, 0.1, 0.1], dtype=float),
            traction=np.array([1.0, 0.0], dtype=float),
        )
        fig, ax = plot_simulation_data(data, backend="matplotlib", show=False)
        self.assertIsNotNone(fig)
        self.assertIsNotNone(ax)


if __name__ == "__main__":
    unittest.main()
