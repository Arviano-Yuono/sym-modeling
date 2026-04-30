from __future__ import annotations

import json
import sys
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from sym_modeling.domains.fem import (  # noqa: E402
    HyperelasticGenerationConfig,
    loadFemData,
    validate_hyperelastic_data_export,
)
from sym_modeling.domains.fem.io.hyperelastic import (  # noqa: E402
    _compute_piola_field,
    _compute_triangle_gradients,
    _write_case_csvs,
)


class HyperelasticExportTests(unittest.TestCase):
    def _write_single_element_dataset(
        self,
        dataset_root: Path,
        config: HyperelasticGenerationConfig,
        write_manifest: bool = True,
    ):
        x_nodes = np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
            ],
            dtype=float,
        )
        target_F = np.array(
            [
                [1.08, 0.12],
                [0.04, 0.97],
            ],
            dtype=float,
        )
        u_nodes = ((target_F - np.eye(2, dtype=float)) @ x_nodes.T).T
        connectivity = np.array([[0, 1, 2]], dtype=int)
        grad_na = np.zeros((1, 3, 2), dtype=float)
        grad_na[0], area = _compute_triangle_gradients(x_nodes)
        qp_weights = np.array([area], dtype=float)
        piola = _compute_piola_field(u_nodes, connectivity, grad_na, config)

        bcx = np.array([1, 0, 0], dtype=int)
        bcy = np.array([2, 0, 0], dtype=int)
        step_dir = dataset_root / str(config.load_steps[0])
        _write_case_csvs(
            output_dir=step_dir,
            x_nodes=x_nodes,
            u_nodes=u_nodes,
            bcx=bcx,
            bcy=bcy,
            connectivity=connectivity,
            grad_na=grad_na,
            qp_weights=qp_weights,
            piola=piola,
            reaction_forces=np.array([0.0, 0.0], dtype=float),
        )
        if write_manifest:
            manifest = {
                "generator": "unit_test",
                "config": asdict(config),
                "load_steps": [
                    {
                        "load_step": int(config.load_steps[0]),
                        "path": str(step_dir),
                    }
                ],
            }
            (dataset_root / "generation_manifest.json").write_text(
                json.dumps(manifest, indent=2),
                encoding="utf-8",
            )
        return step_dir, target_F, piola[0]

    def test_csv_export_reconstructs_deformation_gradient_and_piola(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_root = Path(tmp_dir)
            for material_model in ("notebook_neo_hookean_log", "euclid_neo_hookean_j2"):
                with self.subTest(material_model=material_model):
                    config = HyperelasticGenerationConfig(
                        material_model=material_model,
                        load_steps=(10,),
                    )
                    step_dir, target_F, target_piola = self._write_single_element_dataset(
                        dataset_root / material_model,
                        config,
                    )
                    data = loadFemData(str(step_dir), AD=True, noiseLevel=0.0, noiseType="displacement")
                    data.convertToNumpy()

                    self.assertTrue(np.allclose(data.F[0], target_F.reshape(-1), atol=1e-12))
                    self.assertTrue(np.allclose(data.P[0], target_piola, atol=1e-12))

                    validation = validate_hyperelastic_data_export(dataset_root / material_model)
                    self.assertLess(validation.max_abs_error, 1e-12)
                    self.assertLess(validation.mean_rmse, 1e-12)

    def test_validation_supports_per_load_step_configs(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_root = Path(tmp_dir) / "suite_like_dataset"
            dataset_root.mkdir(parents=True, exist_ok=True)

            config_x = HyperelasticGenerationConfig(
                material_model="euclid_neo_hookean_j2",
                load_steps=(10,),
            )
            config_y = HyperelasticGenerationConfig(
                material_model="euclid_neo_hookean_j2",
                fixed_boundary="bottom",
                traction_boundaries=("top",),
                traction_direction=(0.0, 1.0),
                load_steps=(110,),
            )

            self._write_single_element_dataset(dataset_root, config_x, write_manifest=False)
            self._write_single_element_dataset(dataset_root, config_y, write_manifest=False)

            manifest = {
                "generator": "unit_test_suite",
                "suite_cases": [
                    {"name": "x_case", "config": asdict(config_x)},
                    {"name": "y_case", "config": asdict(config_y)},
                ],
                "load_steps": [
                    {
                        "load_step": 10,
                        "case_name": "x_case",
                        "config": asdict(config_x),
                        "path": str(dataset_root / "10"),
                    },
                    {
                        "load_step": 110,
                        "case_name": "y_case",
                        "config": asdict(config_y),
                        "path": str(dataset_root / "110"),
                    },
                ],
            }
            (dataset_root / "generation_manifest.json").write_text(
                json.dumps(manifest, indent=2),
                encoding="utf-8",
            )

            validation = validate_hyperelastic_data_export(dataset_root)
            self.assertEqual(validation.material_model, "euclid_neo_hookean_j2")
            self.assertEqual(validation.load_steps, (10, 110))
            self.assertLess(validation.max_abs_error, 1e-12)


if __name__ == "__main__":
    unittest.main()
