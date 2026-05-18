from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import json
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from sym_modeling.domains.fem.methods.sgep import (  # noqa: E402
    Gene,
    SGEPConfig,
    SGEPWorkflow,
    aicc,
    assemble_stress_feature_matrix,
    compute_sgep_piola,
    fit_sparse_regression,
    plot_sgep_piola_field_comparisons,
    synthetic_neo_hookean_dataset,
)
from sym_modeling.domains.fem.methods.sgep.gep_gene import (  # noqa: E402
    protected_binary,
    protected_unary,
)
from sym_modeling.domains.fem.methods.sgep.hyperelastic_eval import (  # noqa: E402
    invariant_variables,
    normalized_gene_values,
    reference_variables,
)
from sym_modeling.domains.fem.methods.sgep.run_gep_sparse import (  # noqa: E402
    _apply_cli_overrides,
    build_parser,
    config_from_file,
)
from sym_modeling.domains.fem.methods.sgep.gep_population import (  # noqa: E402
    CandidateModel,
    PopulationEngine,
    selection_rank,
)
from sym_modeling.domains.fem.methods.common.weak_form import (  # noqa: E402
    assemble_B_matrix,
    assemble_feature_derivative_wrt_F,
    compute_first_piola,
    extend_lhs_rhs,
)
from sym_modeling.domains.fem.methods.euclid.feature_library import getNumberOfFeatures  # noqa: E402
from sym_modeling.domains.fem.methods.euclid.weak_form import (  # noqa: E402
    computeFirstPiolaTheta,
    extendLHSRHS,
)
from sym_modeling.domains.fem.io.csv_loader import loadFemData  # noqa: E402
from sym_modeling.domains.fem.io.hyperelastic import (  # noqa: E402
    _compute_triangle_gradients,
    _write_case_csvs,
)


class SGEPTests(unittest.TestCase):
    def _write_single_triangle_known_law(self, root: Path) -> tuple[tuple[Gene, ...], np.ndarray]:
        x_nodes = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=float)
        target_F = np.array([[1.08, 0.12], [0.04, 0.97]], dtype=float)
        u_nodes = ((target_F - np.eye(2, dtype=float)) @ x_nodes.T).T
        connectivity = np.array([[0, 1, 2]], dtype=int)
        grad_na = np.zeros((1, 3, 2), dtype=float)
        grad_na[0], area = _compute_triangle_gradients(x_nodes)
        qp_weights = np.array([area], dtype=float)
        genes = (
            Gene.variable("K1"),
            Gene.unary("square", Gene.variable("Jm1")),
        )
        theta = np.array([0.5, 5.0], dtype=float)
        piola = compute_sgep_piola(
            SimpleNamespace(F=target_F.reshape(1, 4), P=np.zeros((1, 4)), path="single"),
            genes,
            theta,
            config=SGEPConfig(save_plots=False),
        )

        fake_data = SimpleNamespace(
            gradNa=[grad_na[:, local_node, :] for local_node in range(3)]
        )
        weak_config = SimpleNamespace(dim=2, numNodesPerElement=3)
        internal_force = assemble_B_matrix(fake_data, 0, weak_config).T.dot(piola[0]) * area
        _write_case_csvs(
            output_dir=root / "10",
            x_nodes=x_nodes,
            u_nodes=u_nodes,
            bcx=np.array([1, 3, 5], dtype=int),
            bcy=np.array([2, 4, 6], dtype=int),
            connectivity=connectivity,
            grad_na=grad_na,
            qp_weights=qp_weights,
            piola=piola,
            reaction_forces=internal_force,
        )
        return genes, theta

    def test_protected_operators_return_finite_values(self):
        values = np.array([-2.0, -1e-14, 0.0, 1e-14, 2.0])
        for op in ("sqrt", "log", "exp", "sin", "cos"):
            self.assertTrue(np.all(np.isfinite(protected_unary(op, values))))
        self.assertTrue(np.all(np.isfinite(protected_binary("div", values, values * 0.0))))

    def test_gene_reference_normalization_zeroes_undeformed_state(self):
        gene = Gene.binary("add", Gene.variable("K1"), Gene.unary("square", Gene.variable("Jm1")))
        reference = reference_variables(("K1", "Jm1"))
        values = normalized_gene_values(gene, reference, reference)
        self.assertTrue(np.allclose(values, 0.0))

    def test_sparse_fit_recovers_synthetic_neo_hookean_coefficients(self):
        dataset = synthetic_neo_hookean_dataset(num_samples=80, seed=11, mu=2.0, bulk=7.0)
        genes = (
            Gene.variable("K1"),
            Gene.unary("square", Gene.variable("Jm1")),
            Gene.variable("K2"),
        )
        X, valid, reasons = assemble_stress_feature_matrix(
            genes,
            dataset,
            ("K1", "K2", "Jm1"),
        )
        self.assertEqual(reasons[0], "ok")
        self.assertEqual(reasons[1], "ok")
        self.assertTrue(valid[0])
        fit = fit_sparse_regression(X, dataset.target_vector, method="stlsq", threshold=1e-7)
        theta = np.zeros(len(genes))
        theta[valid] = fit.theta
        self.assertLess(fit.metrics.rmse, 1e-5)
        self.assertAlmostEqual(theta[0], 1.0, places=4)
        self.assertAlmostEqual(theta[1], 3.5, places=4)

    def test_aicc_invalid_when_too_many_parameters(self):
        self.assertTrue(np.isinf(aicc(rss=1.0, num_samples=3, num_parameters=2)))

    def test_workflow_runs_tiny_seeded_loop(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = SGEPConfig(
                generations=2,
                num_models=3,
                genes_per_model=3,
                synthetic_samples=40,
                random_seed=5,
                output_dir=tmp_dir,
                save_plots=False,
                progress_log=False,
            )
            result = SGEPWorkflow(config).train()
            self.assertEqual(len(result.history), 2)
            self.assertTrue(np.isfinite(result.metrics["rmse"]))
            self.assertNotEqual(result.best_expression, "")
            self.assertEqual(result.history[0]["selection_objective"], "epsilon_rmse")
            self.assertEqual(result.history[0]["active_terms_epsilon"], 2)
            self.assertIn("best_feasible", result.history[0])
            self.assertTrue((Path(tmp_dir) / "summary.json").exists())

    def test_invariant_variables_reuse_euclid_kinematics_shape(self):
        dataset = synthetic_neo_hookean_dataset(num_samples=5, seed=1)
        variables = invariant_variables(dataset, ("K1", "K2", "Jm1"))
        self.assertEqual(set(variables), {"K1", "K2", "Jm1"})
        self.assertEqual(variables["K1"].shape, (5,))

    def test_config_file_can_drive_sgep_inputs(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "sgep.json"
            config_path.write_text(
                json.dumps(
                    {
                        "sgep": {
                            "data_dir": "dataset/fem_data/plate_hole_fenics/NH2",
                            "loadsteps": [10],
                            "variable_names": ["K1", "Jm1"],
                            "unary_operators": ["square"],
                            "binary_operators": ["add", "mul"],
                            "generations": 2,
                            "num_models": 3,
                            "genes_per_model": 2,
                            "fitting_mode": "direct_stress",
                            "selection_objective": "epsilon_rmse",
                            "active_terms_epsilon": 2,
                            "output_dir": "output/from_config",
                            "save_plots": False,
                        }
                    }
                ),
                encoding="utf-8",
            )

            config = config_from_file(config_path)
            self.assertEqual(config.loadsteps, [10])
            self.assertEqual(config.variable_names, ("K1", "Jm1"))
            self.assertEqual(config.unary_operators, ("square",))
            self.assertEqual(config.binary_operators, ("add", "mul"))
            self.assertEqual(config.generations, 2)
            self.assertEqual(config.fitting_mode, "direct_stress")
            self.assertEqual(config.selection_objective, "epsilon_rmse")
            self.assertEqual(config.active_terms_epsilon, 2)

            args = build_parser().parse_args(
                [
                    "--config",
                    str(config_path),
                    "--generations",
                    "4",
                    "--fitting-mode",
                    "weak_form",
                    "--selection-objective",
                    "aicc",
                    "--active-terms-epsilon",
                    "3",
                ]
            )
            updated = _apply_cli_overrides(config, args)
            self.assertEqual(updated.generations, 4)
            self.assertEqual(updated.num_models, 3)
            self.assertEqual(updated.fitting_mode, "weak_form")
            self.assertEqual(updated.selection_objective, "aicc")
            self.assertEqual(updated.active_terms_epsilon, 3)

    def test_population_engine_uses_configured_operator_set(self):
        engine = PopulationEngine(
            variable_names=("K1", "Jm1"),
            genes_per_model=3,
            num_models=2,
            max_depth=3,
            random_seed=4,
            unary_operators=("square",),
            binary_operators=("add",),
        )
        allowed = {"var", "const", "square", "add"}
        for gene in engine.initial_population():
            for path in gene.paths():
                self.assertIn(gene.subtree(path).op, allowed)

        with self.assertRaises(ValueError):
            SGEPConfig(unary_operators=("unsupported",))

        with self.assertRaises(ValueError):
            SGEPConfig(fitting_mode="unsupported")

        with self.assertRaises(ValueError):
            SGEPConfig(selection_objective="unsupported")

        with self.assertRaises(ValueError):
            SGEPConfig(active_terms_epsilon=0)

    def test_selection_rank_supports_aicc_and_epsilon_rmse(self):
        self.assertLess(
            selection_rank("aicc", aicc=1.0, rmse=10.0, active_terms=3, complexity=20, active_terms_epsilon=2),
            selection_rank("aicc", aicc=2.0, rmse=1.0, active_terms=1, complexity=1, active_terms_epsilon=2),
        )
        feasible_high_error = selection_rank(
            "epsilon_rmse",
            aicc=100.0,
            rmse=2.0,
            active_terms=2,
            complexity=10,
            active_terms_epsilon=2,
        )
        feasible_low_error = selection_rank(
            "epsilon_rmse",
            aicc=1000.0,
            rmse=1.0,
            active_terms=2,
            complexity=20,
            active_terms_epsilon=2,
        )
        infeasible_low_error = selection_rank(
            "epsilon_rmse",
            aicc=-1000.0,
            rmse=0.1,
            active_terms=3,
            complexity=1,
            active_terms_epsilon=2,
        )
        failed_feasible = selection_rank(
            "epsilon_rmse",
            aicc=float("inf"),
            rmse=float("inf"),
            active_terms=0,
            complexity=0,
            active_terms_epsilon=2,
        )

        self.assertLess(feasible_low_error, feasible_high_error)
        self.assertLess(feasible_high_error, infeasible_low_error)
        self.assertLess(infeasible_low_error, failed_feasible)

    def test_fitting_mode_auto_preserves_existing_defaults(self):
        self.assertEqual(SGEPWorkflow(SGEPConfig())._resolved_fitting_mode(), "direct_stress")
        self.assertEqual(
            SGEPWorkflow(SGEPConfig(data_dir="dataset/fem_data/plate_hole_fenics/NH2"))._resolved_fitting_mode(),
            "weak_form",
        )

        with self.assertRaisesRegex(ValueError, "requires data_dir"):
            SGEPWorkflow(SGEPConfig(fitting_mode="weak_form"))._resolved_fitting_mode()

    def test_max_elements_per_loadstep_can_be_disabled(self):
        self.assertIsNone(SGEPConfig(max_elements_per_loadstep=None).max_elements_per_loadstep)
        self.assertIsNone(SGEPConfig(max_elements_per_loadstep=0).max_elements_per_loadstep)
        self.assertEqual(SGEPConfig(max_elements_per_loadstep=12).max_elements_per_loadstep, 12)

        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "sgep.json"
            config_path.write_text(
                json.dumps({"sgep": {"max_elements_per_loadstep": None}}),
                encoding="utf-8",
            )
            self.assertIsNone(config_from_file(config_path).max_elements_per_loadstep)

            config = SGEPConfig(max_elements_per_loadstep=600)
            args = build_parser().parse_args(["--all-elements"])
            self.assertIsNone(_apply_cli_overrides(config, args).max_elements_per_loadstep)

            args = build_parser().parse_args(["--max-elements-per-loadstep", "0"])
            self.assertIsNone(_apply_cli_overrides(config, args).max_elements_per_loadstep)

            args = build_parser().parse_args(["--quiet"])
            self.assertFalse(_apply_cli_overrides(config, args).progress_log)

    def test_compute_sgep_piola_matches_synthetic_known_law(self):
        dataset = synthetic_neo_hookean_dataset(num_samples=20, seed=7, mu=1.0, bulk=10.0)
        data = SimpleNamespace(F=dataset.F, P=dataset.P, path="synthetic")
        genes = (
            Gene.variable("K1"),
            Gene.unary("square", Gene.variable("Jm1")),
        )
        theta = np.array([0.5, 5.0], dtype=float)
        config = SGEPConfig(save_plots=False)

        modeled = compute_sgep_piola(data, genes, theta, config=config)

        self.assertLess(np.sqrt(np.mean(np.square(modeled - dataset.P))), 1e-8)

    def test_sgep_piola_field_plots_are_written_for_csv_loadstep(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir) / "dataset"
            genes, theta = self._write_single_triangle_known_law(root)

            config = SGEPConfig(
                data_dir=str(root),
                loadsteps=[10],
                output_dir=str(Path(tmp_dir) / "out"),
                save_plots=True,
            )
            plot_paths, metrics, warnings = plot_sgep_piola_field_comparisons(
                data_dir=root,
                loadsteps=[10],
                genes=genes,
                theta=theta,
                config=config,
                output_dir=Path(tmp_dir) / "out" / "piola_fields",
            )

            self.assertEqual(warnings, [])
            self.assertEqual(len(metrics), 1)
            self.assertLess(metrics[0]["rmse"], 1e-8)
            self.assertEqual(len(plot_paths["10"]), 5)
            for path in plot_paths["10"]:
                self.assertTrue(Path(path).exists())

    def test_shared_weak_form_matches_euclid_fixed_feature_path(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir) / "dataset"
            self._write_single_triangle_known_law(root)
            data = loadFemData(str(root / "10"), AD=True, noiseLevel=0.0)
            data.convertToNumpy()
            theta = np.linspace(0.01, 0.02, getNumberOfFeatures())
            config = SimpleNamespace(dim=2, numNodesPerElement=3, balance=100.0)

            manual_dQdF = (
                np.outer(data.featureSet.d_features_dI1[0, :], data.dI1dF[0, :])
                + np.outer(data.featureSet.d_features_dI3[0, :], data.dI3dF[0, :])
            )
            self.assertTrue(np.allclose(assemble_feature_derivative_wrt_F(data, 0), manual_dQdF))
            self.assertTrue(np.allclose(compute_first_piola(data, theta), computeFirstPiolaTheta(data, theta)))

            lhs_common = np.zeros((getNumberOfFeatures(), getNumberOfFeatures()), dtype=float)
            rhs_common = np.zeros(getNumberOfFeatures(), dtype=float)
            lhs_euclid = np.zeros_like(lhs_common)
            rhs_euclid = np.zeros_like(rhs_common)
            lhs_common, rhs_common = extend_lhs_rhs(data, config, lhs_common, rhs_common, verbose=False)
            lhs_euclid, rhs_euclid = extendLHSRHS(data, config, lhs_euclid, rhs_euclid)
            self.assertTrue(np.allclose(lhs_common, lhs_euclid))
            self.assertTrue(np.allclose(rhs_common, rhs_euclid))

    def test_sgep_dataset_backed_candidate_uses_weak_form_target(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir) / "dataset"
            genes, expected_theta = self._write_single_triangle_known_law(root)
            config = SGEPConfig(
                data_dir=str(root),
                loadsteps=[10],
                output_dir=str(Path(tmp_dir) / "out"),
                save_plots=False,
                variable_names=("K1", "Jm1"),
                weak_form_penalty_lp=0.0,
                weak_form_num_increments=1,
                weak_form_threshold=1e-8,
                weak_form_threshold_iter=1e-12,
                weak_form_energy_checks=False,
            )
            workflow = SGEPWorkflow(config)
            dataset = workflow._load_dataset()
            evaluation = workflow._fit_candidate(CandidateModel(genes), dataset)

            self.assertLess(evaluation.fit.metrics.rmse, 1e-8)
            self.assertTrue(np.allclose(evaluation.theta_full, expected_theta, atol=1e-6))
            self.assertEqual(evaluation.fit.metrics.num_parameters, 2)

    def test_sgep_dataset_backed_candidate_can_use_direct_stress_target(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir) / "dataset"
            genes, expected_theta = self._write_single_triangle_known_law(root)
            config = SGEPConfig(
                data_dir=str(root),
                loadsteps=[10],
                output_dir=str(Path(tmp_dir) / "out"),
                save_plots=False,
                variable_names=("K1", "Jm1"),
                fitting_mode="direct_stress",
            )
            workflow = SGEPWorkflow(config)
            dataset = workflow._load_dataset()
            evaluation = workflow._fit_candidate(CandidateModel(genes), dataset)

            self.assertEqual(workflow._resolved_fitting_mode(), "direct_stress")
            self.assertLess(evaluation.fit.metrics.rmse, 1e-8)
            self.assertTrue(np.allclose(evaluation.theta_full, expected_theta, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
