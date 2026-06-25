from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
SCRIPTS_DIR = REPO_ROOT / "scripts"
for path in (SRC_DIR, SCRIPTS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


from format_thermocorr_dataset import format_thermocorr_dataset, main as format_thermocorr_main  # noqa: E402
from sym_modeling.domains.fem.io.csv_loader import loadFemData  # noqa: E402


def _write_force_csv(path: Path, values: tuple[float, ...]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("force [N]",))
        for value in values:
            writer.writerow((value,))


def _raw_displacement_from_F(raw_nodes: np.ndarray, deformation_gradient: np.ndarray) -> np.ndarray:
    x_min = float(np.min(raw_nodes[:, 0]))
    y_min = float(np.min(raw_nodes[:, 1]))
    y_max = float(np.max(raw_nodes[:, 1]))
    scale = y_max - y_min
    x_nodes = np.column_stack(
        (
            (raw_nodes[:, 0] - x_min) / scale,
            (y_max - raw_nodes[:, 1]) / scale,
        )
    )
    u_nodes = ((deformation_gradient - np.eye(2, dtype=float)) @ x_nodes.T).T
    return np.column_stack((u_nodes[:, 0] * scale, -u_nodes[:, 1] * scale, np.zeros(len(u_nodes))))


def _write_thermocorr_fixture(input_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    input_dir.mkdir(parents=True, exist_ok=True)
    xs = np.array([10.0, 20.0, 30.0], dtype=float)
    ys = np.array([100.0, 110.0, 120.0], dtype=float)
    raw_nodes = np.asarray([[x, y, 1.0] for y in ys for x in xs], dtype=float)

    step1_F = np.array([[1.10, 0.20], [0.05, 0.90]], dtype=float)
    step2_F = np.array([[1.20, 0.10], [0.02, 0.95]], dtype=float)
    with h5py.File(input_dir / "TPS_2.hdf5", "w") as handle:
        handle.attrs["dim"] = "2d"
        region = handle.create_group("region")
        region.attrs["type"] = "quad"
        nodes = region.create_dataset("nodes", data=raw_nodes)
        nodes.attrs["type"] = "quad"

        output = handle.create_group("output")
        displacements = output.create_group("u")
        displacements.attrs["components"] = "x, y, z"
        displacements.attrs["dim"] = 3
        displacements.attrs["type"] = "t1"
        for frame, field in (
            ("00000", np.zeros((len(raw_nodes), 3), dtype=float)),
            ("00001", _raw_displacement_from_F(raw_nodes, step1_F)),
            ("00002", _raw_displacement_from_F(raw_nodes, step2_F)),
        ):
            dataset = displacements.create_dataset(frame, data=field)
            dataset.attrs["time"] = float(int(frame))
            dataset.attrs["ref"] = 0

    _write_force_csv(input_dir / "force_vfm.csv", (10.0, 20.0))
    np.savez(
        input_dir / "e1.npz",
        e1=np.array([[0.0, 0.1, 0.2], [1.0, 1.5, 2.0]], dtype=float),
        e2=np.array([[0.0, 0.3], [2.0, 3.0]], dtype=float),
    )
    return step1_F, step2_F


def _fixture_boundary_source_indices() -> set[int]:
    return {0, 1, 2, 3, 5, 6, 7, 8}


def _fixture_top_reaction_source_indices() -> set[int]:
    return {0, 1, 2}


class ThermoCorrFormatterTests(unittest.TestCase):
    def test_formatter_writes_fem_csvs_and_reloads_affine_field(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            input_dir = root / "thermocorr"
            output_dir = root / "formatted"
            step1_F, step2_F = _write_thermocorr_fixture(input_dir)

            result = format_thermocorr_dataset(input_dir, output_dir, overwrite=True)

            self.assertEqual(result.load_steps, (1, 2))
            self.assertEqual(result.num_nodes, 9)
            self.assertEqual(result.num_quads, 4)
            self.assertEqual(result.num_elements, 8)

            nodes = pd.read_csv(output_dir / "1" / "output_nodes.csv")
            self.assertEqual(list(nodes.columns), ["x", "y", "ux", "uy", "bcx", "bcy"])
            self.assertAlmostEqual(float(nodes["x"].min()), 0.0)
            self.assertAlmostEqual(float(nodes["x"].max()), 1.0)
            self.assertAlmostEqual(float(nodes["y"].min()), 0.0)
            self.assertAlmostEqual(float(nodes["y"].max()), 1.0)
            self.assertTrue((nodes.loc[np.isclose(nodes["y"], 1.0), "bcy"] == 1).all())
            self.assertTrue((nodes.loc[~np.isclose(nodes["y"], 1.0), "bcy"] == 0).all())
            self.assertTrue((nodes["bcx"] == 0).all())

            elements = pd.read_csv(output_dir / "1" / "output_elements.csv")
            self.assertEqual(list(elements.columns), ["node1", "node2", "node3"])
            self.assertEqual(len(elements), 8)

            integrator = pd.read_csv(output_dir / "1" / "output_integrator.csv")
            self.assertTrue((integrator["qpWeight"] > 0.0).all())
            self.assertAlmostEqual(float(integrator["qpWeight"].sum()), 1.0)

            reactions_1 = pd.read_csv(output_dir / "1" / "output_reactions.csv")
            reactions_2 = pd.read_csv(output_dir / "2" / "output_reactions.csv")
            self.assertEqual(float(reactions_1["forces"].iloc[0]), 0.5)
            self.assertEqual(float(reactions_2["forces"].iloc[0]), 1.0)

            data_1 = loadFemData(str(output_dir / "1"), AD=True, noiseLevel=0.0)
            data_1.convertToNumpy()
            data_2 = loadFemData(str(output_dir / "2"), AD=True, noiseLevel=0.0)
            data_2.convertToNumpy()
            self.assertTrue(np.allclose(data_1.F, step1_F.reshape(1, 4), atol=1e-12))
            self.assertTrue(np.allclose(data_2.F, step2_F.reshape(1, 4), atol=1e-12))
            self.assertIsNone(data_1.P)
            self.assertEqual(len(data_1.reactions), 1)
            self.assertAlmostEqual(data_1.reactions[0].force, 0.5)

            manifest = json.loads((output_dir / "generation_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["load_steps"][0]["frame"], "00001")
            self.assertEqual(manifest["load_steps"][1]["frame"], "00002")
            self.assertEqual(manifest["load_steps"][0]["force_N"], 10.0)
            self.assertEqual(manifest["load_steps"][0]["reaction_force"], 0.5)
            self.assertEqual(manifest["reaction_scaling"]["mode"], "pixel_height")
            self.assertAlmostEqual(manifest["reaction_scaling"]["factor"], 0.05)
            self.assertEqual(manifest["mesh"]["num_dropped_points"], 0)
            self.assertFalse(manifest["sampling"]["enabled"])
            self.assertEqual(manifest["sampling"]["method"], "none")
            self.assertEqual(manifest["sampling"]["actual_node_count"], 9)

    def test_formatter_exports_uniaxial_npz_curves(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            input_dir = root / "thermocorr"
            output_dir = root / "formatted"
            _write_thermocorr_fixture(input_dir)

            result = format_thermocorr_dataset(input_dir, output_dir, overwrite=True)

            self.assertEqual(result.num_uniaxial_curves, 2)
            curve = pd.read_csv(output_dir / "uniaxial" / "e1_e1.csv")
            self.assertEqual(list(curve.columns), ["strain", "stress"])
            self.assertTrue(np.allclose(curve["strain"], [0.0, 0.1, 0.2]))
            self.assertTrue(np.allclose(curve["stress"], [1.0, 1.5, 2.0]))

    def test_boundary_coarsening_preserves_boundary_and_affine_field(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            input_dir = root / "thermocorr"
            output_dir = root / "sampled"
            step1_F, step2_F = _write_thermocorr_fixture(input_dir)

            result = format_thermocorr_dataset(
                input_dir,
                output_dir,
                overwrite=True,
                sample_nodes=5,
                sample_seed=7,
            )

            self.assertEqual(result.num_nodes, 8)
            self.assertGreater(result.num_nodes, 5)
            self.assertEqual(result.num_quads, 0)
            self.assertGreater(result.num_elements, 0)

            nodes_1 = pd.read_csv(output_dir / "1" / "output_nodes.csv")
            nodes_2 = pd.read_csv(output_dir / "2" / "output_nodes.csv")
            elements_1 = pd.read_csv(output_dir / "1" / "output_elements.csv")
            elements_2 = pd.read_csv(output_dir / "2" / "output_elements.csv")
            self.assertTrue(np.allclose(nodes_1[["x", "y"]], nodes_2[["x", "y"]]))
            self.assertTrue(np.array_equal(elements_1.values, elements_2.values))
            top_nodes = nodes_1.loc[np.isclose(nodes_1["y"], 1.0)]
            self.assertEqual(len(top_nodes), 3)
            self.assertTrue(np.allclose(sorted(top_nodes["x"].tolist()), [0.0, 0.5, 1.0]))
            self.assertTrue((top_nodes["bcy"] == 1).all())

            data_1 = loadFemData(str(output_dir / "1"), AD=True, noiseLevel=0.0)
            data_1.convertToNumpy()
            data_2 = loadFemData(str(output_dir / "2"), AD=True, noiseLevel=0.0)
            data_2.convertToNumpy()
            self.assertTrue(np.allclose(data_1.F, step1_F.reshape(1, 4), atol=1e-12))
            self.assertTrue(np.allclose(data_2.F, step2_F.reshape(1, 4), atol=1e-12))

            manifest = json.loads((output_dir / "generation_manifest.json").read_text(encoding="utf-8"))
            self.assertTrue(manifest["sampling"]["enabled"])
            self.assertEqual(manifest["sampling"]["method"], "boundary_coarsen")
            self.assertEqual(manifest["sampling"]["sample_nodes"], 5)
            self.assertEqual(manifest["sampling"]["requested_node_count"], 5)
            self.assertEqual(manifest["sampling"]["actual_node_count"], 8)
            self.assertEqual(manifest["sampling"]["sample_seed"], 7)
            self.assertEqual(manifest["sampling"]["boundary_nodes_preserved"], 8)
            self.assertEqual(manifest["sampling"]["interior_nodes_selected"], 0)
            selected_sources = set(manifest["sampling"]["selected_source_node_indices"])
            self.assertTrue(_fixture_boundary_source_indices().issubset(selected_sources))
            self.assertTrue(_fixture_top_reaction_source_indices().issubset(selected_sources))

    def test_random_sample_method_keeps_exact_count_for_comparison(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            input_dir = root / "thermocorr"
            output_dir = root / "sampled"
            _write_thermocorr_fixture(input_dir)

            result = format_thermocorr_dataset(
                input_dir,
                output_dir,
                overwrite=True,
                sample_nodes=5,
                sample_seed=7,
                sample_method="random",
            )

            self.assertEqual(result.num_nodes, 5)
            self.assertEqual(result.num_quads, 0)
            self.assertGreater(result.num_elements, 0)

            nodes = pd.read_csv(output_dir / "1" / "output_nodes.csv")
            self.assertEqual(len(nodes), 5)
            self.assertTrue((nodes["bcy"] == 1).any())

            manifest = json.loads((output_dir / "generation_manifest.json").read_text(encoding="utf-8"))
            self.assertTrue(manifest["sampling"]["enabled"])
            self.assertEqual(manifest["sampling"]["method"], "random")
            self.assertEqual(manifest["sampling"]["requested_node_count"], 5)
            self.assertEqual(manifest["sampling"]["actual_node_count"], 5)
            self.assertEqual(len(manifest["sampling"]["selected_source_node_indices"]), 5)

    def test_formatter_raw_force_cli_writes_unscaled_reactions(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            input_dir = root / "thermocorr"
            output_dir = root / "formatted"
            _write_thermocorr_fixture(input_dir)

            status = format_thermocorr_main(
                [
                    "--input-dir",
                    str(input_dir),
                    "--output-dir",
                    str(output_dir),
                    "--overwrite",
                    "--raw-force",
                ]
            )

            self.assertEqual(status, 0)
            reactions_1 = pd.read_csv(output_dir / "1" / "output_reactions.csv")
            reactions_2 = pd.read_csv(output_dir / "2" / "output_reactions.csv")
            self.assertEqual(float(reactions_1["forces"].iloc[0]), 10.0)
            self.assertEqual(float(reactions_2["forces"].iloc[0]), 20.0)

            manifest = json.loads((output_dir / "generation_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["reaction_scaling"]["mode"], "raw")
            self.assertEqual(manifest["reaction_scaling"]["factor"], 1.0)

    def test_formatter_supports_custom_reaction_scale(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            input_dir = root / "thermocorr"
            output_dir = root / "formatted"
            _write_thermocorr_fixture(input_dir)

            format_thermocorr_dataset(
                input_dir,
                output_dir,
                overwrite=True,
                reaction_scale=0.25,
            )

            reactions_1 = pd.read_csv(output_dir / "1" / "output_reactions.csv")
            reactions_2 = pd.read_csv(output_dir / "2" / "output_reactions.csv")
            self.assertEqual(float(reactions_1["forces"].iloc[0]), 2.5)
            self.assertEqual(float(reactions_2["forces"].iloc[0]), 5.0)

            manifest = json.loads((output_dir / "generation_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["reaction_scaling"]["mode"], "custom")
            self.assertEqual(manifest["reaction_scaling"]["factor"], 0.25)

    def test_formatter_supports_physical_reaction_scaling(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            input_dir = root / "thermocorr"
            output_dir = root / "formatted"
            _write_thermocorr_fixture(input_dir)

            format_thermocorr_dataset(
                input_dir,
                output_dir,
                overwrite=True,
                physical_height=50.0,
                thickness=2.0,
            )

            reactions_1 = pd.read_csv(output_dir / "1" / "output_reactions.csv")
            self.assertEqual(float(reactions_1["forces"].iloc[0]), 0.1)

            manifest = json.loads((output_dir / "generation_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["reaction_scaling"]["mode"], "physical_stress")
            self.assertEqual(manifest["reaction_scaling"]["physical_height"], 50.0)
            self.assertEqual(manifest["reaction_scaling"]["thickness"], 2.0)
            self.assertEqual(manifest["reaction_scaling"]["factor"], 0.01)


if __name__ == "__main__":
    unittest.main()
