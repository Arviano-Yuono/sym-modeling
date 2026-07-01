from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from sym_modeling.domains.fem.io.csv_loader import loadFemData  # noqa: E402
from sym_modeling.domains.fem.io.hyperelastic import _compute_triangle_gradients, _write_case_csvs  # noqa: E402
from sym_modeling.domains.fem.methods.common.denoising import (  # noqa: E402
    DenoiseCandidate,
    DenoiseSearchConfig,
    MeshLaplacianCandidate,
    _selection_score,
    add_displacement_noise,
    assemble_scalar_fem_laplacian,
    denoise_displacements_krr,
    denoise_displacements_mesh_laplacian,
    search_denoise_hyperparameters,
    search_krr_hyperparameters,
    write_denoised_fem_dataset,
)
from sym_modeling.domains.fem.methods.common.run_denoise import build_parser  # noqa: E402
from scripts.run_fem_denoise_sweep import build_parser as build_sweep_parser  # noqa: E402


def _write_square_case(root: Path, step: int, target_F: np.ndarray) -> None:
    x_nodes = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ],
        dtype=float,
    )
    u_nodes = ((target_F - np.eye(2, dtype=float)) @ x_nodes.T).T
    connectivity = np.array([[0, 1, 2], [1, 3, 2]], dtype=int)
    grad_na = np.zeros((2, 3, 2), dtype=float)
    qp_weights = np.zeros(2, dtype=float)
    for element, nodes in enumerate(connectivity):
        grad_na[element], qp_weights[element] = _compute_triangle_gradients(x_nodes[nodes])
    _write_case_csvs(
        output_dir=root / str(step),
        x_nodes=x_nodes,
        u_nodes=u_nodes,
        bcx=np.array([1, 0, 0, 0], dtype=int),
        bcy=np.array([2, 0, 0, 0], dtype=int),
        connectivity=connectivity,
        grad_na=grad_na,
        qp_weights=qp_weights,
        piola=np.zeros((2, 4), dtype=float),
        reaction_forces=np.zeros(2, dtype=float),
    )


class FEMDenoisingTests(unittest.TestCase):
    def test_add_displacement_noise_preserves_constrained_dofs_and_seed(self):
        u_nodes = np.zeros((4, 2), dtype=float)
        dirichlet_nodes = np.array(
            [
                [True, False],
                [False, False],
                [False, True],
                [False, False],
            ],
            dtype=bool,
        )

        noisy_a = add_displacement_noise(u_nodes, dirichlet_nodes, 1e-3, np.random.default_rng(7))
        noisy_b = add_displacement_noise(u_nodes, dirichlet_nodes, 1e-3, np.random.default_rng(7))

        self.assertTrue(np.allclose(noisy_a, noisy_b))
        self.assertEqual(noisy_a[0, 0], 0.0)
        self.assertEqual(noisy_a[2, 1], 0.0)
        self.assertGreater(np.linalg.norm(noisy_a[np.logical_not(dirichlet_nodes)]), 0.0)

    def test_krr_denoising_uses_multi_output_shape_and_preserves_dirichlet(self):
        x_nodes = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=float)
        u_nodes = np.array([[0.0, 0.0], [0.1, 0.02], [0.03, -0.04], [0.12, -0.01]], dtype=float)
        dirichlet_nodes = np.array([[True, True], [False, False], [False, False], [False, False]], dtype=bool)

        denoised = denoise_displacements_krr(
            x_nodes,
            u_nodes,
            dirichlet_nodes,
            DenoiseCandidate(alpha=1e-4, gamma=1.0, blend=1.0),
            preserve_dirichlet=True,
        )

        self.assertEqual(denoised.shape, u_nodes.shape)
        self.assertTrue(np.allclose(denoised[dirichlet_nodes], u_nodes[dirichlet_nodes]))

    def test_fem_laplacian_assembly_is_symmetric_with_zero_row_sums(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir) / "input"
            _write_square_case(root, 10, np.array([[1.08, 0.02], [0.03, 0.97]], dtype=float))
            data = loadFemData(str(root / "10"), AD=True, noiseLevel=0.0)

            laplacian = assemble_scalar_fem_laplacian(
                data.connectivity,
                data.gradNa,
                data.qpWeights,
                data.numNodes,
            )

            self.assertEqual(laplacian.shape, (data.numNodes, data.numNodes))
            self.assertTrue(np.allclose((laplacian - laplacian.T).toarray(), 0.0))
            self.assertTrue(np.allclose(np.asarray(laplacian.sum(axis=1)).ravel(), 0.0))

    def test_mesh_laplacian_denoising_preserves_dirichlet_and_smooths_components(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir) / "input"
            _write_square_case(root, 10, np.eye(2, dtype=float))
            data = loadFemData(str(root / "10"), AD=True, noiseLevel=0.0)
            noisy_u = np.array(
                [
                    [0.0, 0.0],
                    [0.2, -0.1],
                    [-0.1, 0.25],
                    [0.15, -0.2],
                ],
                dtype=float,
            )

            denoised = denoise_displacements_mesh_laplacian(
                noisy_u,
                data.dirichlet_nodes,
                data.connectivity,
                data.gradNa,
                data.qpWeights,
                MeshLaplacianCandidate(lambda_smooth=1.0, blend=1.0),
            )

            free = np.logical_not(data.dirichlet_nodes)
            self.assertEqual(denoised.shape, noisy_u.shape)
            self.assertTrue(np.allclose(denoised[data.dirichlet_nodes], noisy_u[data.dirichlet_nodes]))
            self.assertGreater(np.linalg.norm(denoised[free[:, 0], 0] - noisy_u[free[:, 0], 0]), 0.0)
            self.assertGreater(np.linalg.norm(denoised[free[:, 1], 1] - noisy_u[free[:, 1], 1]), 0.0)

    def test_write_denoised_fem_dataset_outputs_loadable_dataset(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir) / "input"
            _write_square_case(root, 10, np.array([[1.08, 0.02], [0.03, 0.97]], dtype=float))
            clean = loadFemData(str(root / "10"), AD=True, noiseLevel=0.0)
            denoised_u = np.array(clean.u_nodes, copy=True)
            denoised_u[1:, :] += 0.01

            output_root = Path(tmp_dir) / "output"
            paths = write_denoised_fem_dataset(root, output_root, {10: denoised_u})

            self.assertTrue((output_root / "10" / "output_nodes.csv").is_file())
            self.assertTrue(Path(paths["manifest_json"]).is_file())
            reloaded = loadFemData(str(output_root / "10"), AD=True, noiseLevel=0.0)
            self.assertTrue(np.allclose(reloaded.u_nodes, denoised_u))
            self.assertEqual(
                (root / "10" / "output_elements.csv").read_text(encoding="utf-8"),
                (output_root / "10" / "output_elements.csv").read_text(encoding="utf-8"),
            )

    def test_composite_selection_score_normalizes_by_noisy_baseline(self):
        config = DenoiseSearchConfig(
            data_dir="input",
            output_dir="output",
            objective="composite",
            objective_weights={"F_rmse": 1.0, "J_rmse": 3.0},
        )
        metrics = {"F_rmse": 4.0, "J_rmse": 10.0}
        baseline = {"F_rmse": 2.0, "J_rmse": 5.0}

        score = _selection_score(config, metrics, baseline)

        self.assertAlmostEqual(score, 2.0)

    def test_per_loadstep_selection_chooses_different_candidates_and_records_summary(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir) / "input"
            output_root = Path(tmp_dir) / "denoised"
            for step in (10, 20):
                (root / str(step)).mkdir(parents=True)

            def metric_row(value: float) -> dict[str, float]:
                return {
                    "u_rmse": value,
                    "F_rmse": value,
                    "J_rmse": value,
                    "I1_rmse": value,
                    "I2_rmse": value,
                    "I3_rmse": value,
                }

            def fake_load(path: Path, denoised_displacements=None):
                step = int(Path(path).name)
                if denoised_displacements is None:
                    return SimpleNamespace(
                        step=step,
                        kind="clean",
                        candidate_alpha=None,
                        u_nodes=np.zeros((1, 2), dtype=float),
                        dirichlet_nodes=np.zeros((1, 2), dtype=bool),
                    )
                displacements = np.asarray(denoised_displacements, dtype=float)
                alpha = float(displacements[0, 0])
                kind = "noisy" if alpha == 0.0 else "denoised"
                return SimpleNamespace(
                    step=step,
                    kind=kind,
                    candidate_alpha=None if kind == "noisy" else alpha,
                    u_nodes=displacements,
                    dirichlet_nodes=np.zeros((1, 2), dtype=bool),
                )

            def fake_denoise(noisy, candidate, config):
                return np.array([[candidate.alpha, 0.0]], dtype=float)

            def fake_metrics(clean, candidate):
                if candidate.kind == "noisy":
                    return metric_row(10.0)
                values = {
                    (10, 1.0): 1.0,
                    (10, 2.0): 4.0,
                    (20, 1.0): 5.0,
                    (20, 2.0): 1.0,
                }
                return metric_row(values[(int(clean.step), float(candidate.candidate_alpha))])

            def fake_write(data_dir, output_dir, denoised_displacements_by_step, *, overwrite=False):
                Path(output_dir).mkdir(parents=True, exist_ok=True)
                return {
                    "output_dir": str(output_dir),
                    "manifest_json": str(Path(output_dir) / "denoise_manifest.json"),
                    "step_dirs": {str(step): str(Path(output_dir) / str(step)) for step in denoised_displacements_by_step},
                }

            with (
                patch(
                    "sym_modeling.domains.fem.methods.common.denoising._load_fem_quiet",
                    side_effect=fake_load,
                ),
                patch(
                    "sym_modeling.domains.fem.methods.common.denoising._denoise_candidate",
                    side_effect=fake_denoise,
                ),
                patch(
                    "sym_modeling.domains.fem.methods.common.denoising._metrics",
                    side_effect=fake_metrics,
                ),
                patch(
                    "sym_modeling.domains.fem.methods.common.denoising._physics_checks",
                    return_value={"J_min": 1.0, "J_nonpositive_count": 0},
                ),
                patch(
                    "sym_modeling.domains.fem.methods.common.denoising.write_denoised_fem_dataset",
                    side_effect=fake_write,
                ),
            ):
                result = search_denoise_hyperparameters(
                    DenoiseSearchConfig(
                        data_dir=root,
                        output_dir=output_root,
                        loadsteps=[10, 20],
                        noise_level=0.0,
                        objective="composite",
                        objective_weights={"F_rmse": 1.0},
                        selection_scope="per-loadstep",
                        alphas=(1.0, 2.0),
                        gammas=(1.0,),
                        blends=(1.0,),
                    )
                )

            self.assertIsNone(result.selected_candidate)
            self.assertEqual(result.selection_scope, "per-loadstep")
            self.assertEqual(result.selected_candidates_by_loadstep["10"]["alpha"], 1.0)
            self.assertEqual(result.selected_candidates_by_loadstep["20"]["alpha"], 2.0)
            self.assertAlmostEqual(result.selected_scores_by_loadstep["10"], 0.1)
            self.assertAlmostEqual(result.selected_scores_by_loadstep["20"], 0.1)

            summary = json.loads(Path(result.summary_path).read_text(encoding="utf-8"))
            self.assertEqual(summary["selection_scope"], "per-loadstep")
            self.assertEqual(summary["selected_candidate"], None)
            self.assertEqual(summary["selected_candidates_by_loadstep"]["10"]["alpha"], 1.0)
            self.assertEqual(summary["selected_candidates_by_loadstep"]["20"]["alpha"], 2.0)
            self.assertEqual(summary["selected_scores_by_loadstep"], {"10": 0.1, "20": 0.1})

    def test_search_writes_summary_and_selects_global_candidate(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir) / "input"
            _write_square_case(root, 10, np.array([[1.08, 0.02], [0.03, 0.97]], dtype=float))
            _write_square_case(root, 20, np.array([[1.12, -0.01], [0.02, 0.95]], dtype=float))
            output_root = Path(tmp_dir) / "denoised"

            result = search_krr_hyperparameters(
                DenoiseSearchConfig(
                    data_dir=root,
                    output_dir=output_root,
                    loadsteps=[10, 20],
                    noise_level=1e-4,
                    seed=11,
                    alphas=(1e-6, 1e-3),
                    gammas=(1.0,),
                    blends=(0.5, 1.0),
                )
            )

            self.assertTrue(Path(result.summary_path).is_file())
            self.assertTrue(Path(result.search_csv_path).is_file())
            self.assertTrue(Path(result.loadstep_metrics_csv_path).is_file())
            self.assertIn(result.selected_candidate.alpha, (1e-6, 1e-3))
            summary = json.loads(Path(result.summary_path).read_text(encoding="utf-8"))
            self.assertEqual(summary["objective"], "F_rmse")
            self.assertEqual(summary["selection_scope"], "global")
            self.assertEqual(summary["config"]["noise_level"], 1e-4)
            self.assertIn("selected_candidate", summary)
            self.assertTrue((output_root / "10" / "output_nodes.csv").is_file())

            clean_nodes = pd.read_csv(root / "10" / "output_nodes.csv")
            denoised_nodes = pd.read_csv(output_root / "10" / "output_nodes.csv")
            self.assertFalse(np.allclose(clean_nodes[["ux", "uy"]].values, denoised_nodes[["ux", "uy"]].values))

    def test_mesh_laplacian_search_writes_method_summary_and_loadable_dataset(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir) / "input"
            _write_square_case(root, 10, np.array([[1.08, 0.02], [0.03, 0.97]], dtype=float))
            _write_square_case(root, 20, np.array([[1.12, -0.01], [0.02, 0.95]], dtype=float))
            output_root = Path(tmp_dir) / "denoised_laplacian"

            result = search_denoise_hyperparameters(
                DenoiseSearchConfig(
                    data_dir=root,
                    output_dir=output_root,
                    loadsteps=[10, 20],
                    method="mesh-laplacian",
                    noise_level=1e-4,
                    seed=11,
                    lambdas=(0.0, 1e-3),
                    blends=(0.5, 1.0),
                )
            )

            self.assertIn(result.selected_candidate.lambda_smooth, (0.0, 1e-3))
            self.assertIn(result.selected_candidate.blend, (0.5, 1.0))
            summary = json.loads(Path(result.summary_path).read_text(encoding="utf-8"))
            self.assertEqual(summary["method"], "mesh-laplacian")
            self.assertEqual(summary["selected_candidate"]["lambda_smooth"], result.selected_candidate.lambda_smooth)
            self.assertEqual(summary["selected_candidate"]["blend"], result.selected_candidate.blend)
            self.assertEqual(summary["objective"], "F_rmse")
            self.assertEqual(summary["config"]["noise_level"], 1e-4)
            self.assertEqual(summary["config"]["seed"], 11)
            self.assertIn("J_min", summary["selected_metrics"])
            self.assertIn("J_nonpositive_count", summary["selected_metrics"])

            reloaded = loadFemData(str(output_root / "10"), AD=True, noiseLevel=0.0)
            self.assertEqual(reloaded.u_nodes.shape, (4, 2))

    def test_sym_fem_denoise_default_method_remains_krr(self):
        parser = build_parser()
        args = parser.parse_args(["--data-dir", "input", "--output-dir", "output"])

        self.assertEqual(args.method, "krr")
        self.assertEqual(args.objective, "F_rmse")
        self.assertEqual(args.selection_scope, "global")

    def test_sweep_defaults_use_stronger_per_loadstep_composite_search(self):
        parser = build_sweep_parser()
        args = parser.parse_args([])

        self.assertEqual(args.methods, "krr,mesh-laplacian")
        self.assertEqual(args.objective, "composite")
        self.assertEqual(args.selection_scope, "per-loadstep")
        self.assertIn("1e-10", args.alphas)
        self.assertIn("80", args.gammas)
        self.assertIn("0.95", args.blends)
        self.assertIn("100", args.lambdas)


if __name__ == "__main__":
    unittest.main()
