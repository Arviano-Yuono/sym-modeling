from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path
from types import SimpleNamespace

import geppy as gep
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from sym_modeling.domains.fem.methods.common.stress_data import (  # noqa: E402
    build_stress_dataset_from_F,
    invariant_variables,
    synthetic_neo_hookean_dataset,
    variable_derivatives_wrt_F,
)
from sym_modeling.domains.fem.io.hyperelastic import (  # noqa: E402
    _compute_triangle_gradients,
    _write_case_csvs,
)
from sym_modeling.domains.fem.methods.common.weak_form import assemble_B_matrix  # noqa: E402
from sym_modeling.domains.fem.methods.sgeppy import SGEP as GeppySGEP  # noqa: E402
from sym_modeling.domains.fem.methods.sgeppy import SGEPConfig as GeppySGEPConfig  # noqa: E402
from sym_modeling.domains.fem.methods.sgeppy.run_gep_sparse import (  # noqa: E402
    _apply_overrides,
    build_parser,
    config_from_file,
)
import sym_modeling.domains.fem.methods.sgeppy.workflow as sgeppy_workflow  # noqa: E402
from sym_modeling.domains.fem.methods.sgeppy.workflow import (  # noqa: E402
    SGEPWorkflow,
    SGEPWorkflowConfig,
    WEAK_FORM_JAX_TIMING_KEYS,
    WeakFormConfig,
    stress_feature_builder,
)
from sym_modeling.domains.fem.methods.sgeppy.jax_backend import (  # noqa: E402
    JaxWeakFormEvaluationCache,
    feature_values_and_dqdf,
    is_jax_fem_backend_available,
    require_jax_fem_backend,
    stress_feature_builder as jax_stress_feature_builder,
)


class SGEPPYTests(unittest.TestCase):
    def _model(
        self,
        variable_names: tuple[str, ...] = ("x", "y"),
        n_genes: int = 2,
        fit_intercept: bool = False,
        binary_operators: tuple[str, ...] = ("add",),
    ) -> GeppySGEP:
        return GeppySGEP(
            GeppySGEPConfig(
                variable_names=variable_names,
                binary_operators=binary_operators,
                unary_operators=(),
                head_length=1,
                n_genes=n_genes,
                population_size=3,
                n_elites=1,
                fit_intercept=fit_intercept,
                verbose=False,
            )
        ).build()

    @staticmethod
    def _terminal_gene(model: GeppySGEP, name: str):
        terminals = {terminal.name: terminal for terminal in model.pset.terminals}
        terminal = terminals[name]
        return gep.Gene.from_genome([terminal, terminal, terminal], head_length=1)

    @staticmethod
    def _binary_gene(model: GeppySGEP, op_name: str, left_name: str, right_name: str):
        functions = {function.name: function for function in model.pset.functions}
        terminals = {terminal.name: terminal for terminal in model.pset.terminals}
        return gep.Gene.from_genome(
            [functions[op_name], terminals[left_name], terminals[right_name]],
            head_length=1,
        )

    @staticmethod
    def _individual(model: GeppySGEP, genes):
        individual = model.toolbox.individual()
        individual[:] = genes
        return individual

    @staticmethod
    def _write_single_triangle_known_law(root: Path) -> np.ndarray:
        x_nodes = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=float)
        target_F = np.array([[1.08, 0.12], [0.04, 0.97]], dtype=float)
        u_nodes = ((target_F - np.eye(2, dtype=float)) @ x_nodes.T).T
        connectivity = np.array([[0, 1, 2]], dtype=int)
        grad_na = np.zeros((1, 3, 2), dtype=float)
        grad_na[0], area = _compute_triangle_gradients(x_nodes)

        stress_dataset = build_stress_dataset_from_F(
            target_F.reshape(1, 4),
            np.zeros((1, 4), dtype=float),
            name="single_triangle",
        )
        dvar_dF = variable_derivatives_wrt_F(stress_dataset, ("K1", "Jm1"))
        theta = np.array([0.5, 5.0], dtype=float)
        piola = theta[0] * dvar_dF["K1"] + theta[1] * dvar_dF["Jm1"]

        fake_data = SimpleNamespace(
            gradNa=[grad_na[:, local_node, :] for local_node in range(3)]
        )
        weak_config = SimpleNamespace(dim=2, numNodesPerElement=3)
        reaction_forces = assemble_B_matrix(fake_data, 0, weak_config).T.dot(piola[0]) * area
        _write_case_csvs(
            output_dir=root / "10",
            x_nodes=x_nodes,
            u_nodes=u_nodes,
            bcx=np.array([1, 3, 5], dtype=int),
            bcy=np.array([2, 4, 6], dtype=int),
            connectivity=connectivity,
            grad_na=grad_na,
            qp_weights=np.array([area], dtype=float),
            piola=piola,
            reaction_forces=reaction_forces,
        )
        return theta

    def test_feature_matrix_uses_one_column_per_gene(self):
        model = self._model()
        individual = self._individual(
            model,
            (
                self._terminal_gene(model, "x"),
                self._terminal_gene(model, "y"),
            ),
        )
        X = np.array([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]], dtype=float)

        features = model.feature_matrix(X, individual)

        self.assertEqual(features.shape, (3, 2))
        self.assertTrue(np.allclose(features[:, 0], X[:, 0]))
        self.assertTrue(np.allclose(features[:, 1], X[:, 1]))

    def test_sparse_fit_recovers_separate_gene_coefficients(self):
        model = self._model()
        individual = self._individual(
            model,
            (
                self._terminal_gene(model, "x"),
                self._terminal_gene(model, "y"),
            ),
        )
        X = np.array([[1.0, -1.0], [2.0, 0.5], [3.0, 2.0], [4.0, 3.5]], dtype=float)
        model._X = X
        model._y = 2.0 * X[:, 0] - 3.0 * X[:, 1]

        fitness = model.evaluate(individual)

        self.assertTrue(np.all(np.isfinite(fitness)))
        self.assertTrue(np.allclose(individual.theta, [2.0, -3.0], atol=1e-8))
        self.assertTrue(np.all(individual.sparse_fit.active_mask))

    def test_fit_stops_early_when_primary_fitness_reaches_threshold(self):
        config = GeppySGEPConfig(
            variable_names=("x",),
            binary_operators=("add",),
            unary_operators=(),
            head_length=1,
            n_genes=1,
            population_size=3,
            n_generations=5,
            n_elites=1,
            fit_intercept=False,
            mut_uniform_pb=0.0,
            mut_invert_pb=0.0,
            mut_is_transpose_pb=0.0,
            mut_ris_transpose_pb=0.0,
            mut_gene_transpose_pb=0.0,
            cx_one_point_pb=0.0,
            cx_two_point_pb=0.0,
            cx_gene_pb=0.0,
            fitness_metrics=("rmse",),
            early_stop_value=1e-12,
            verbose=False,
        )
        X = np.array([[1.0], [2.0], [3.0]], dtype=float)
        y = X[:, 0]

        model = GeppySGEP(config).fit(X, y)
        history = [dict(row) for row in model.logbook]

        self.assertEqual(len(history), 1)
        self.assertTrue(history[0]["early_stop"])
        self.assertLessEqual(history[0]["min"], config.early_stop_value)

    def test_expression_reports_active_gene_terms(self):
        model = self._model()
        individual = self._individual(
            model,
            (
                self._terminal_gene(model, "x"),
                self._terminal_gene(model, "y"),
            ),
        )
        X = np.array([[1.0, -1.0], [2.0, 0.5], [3.0, 2.0], [4.0, 3.5]], dtype=float)
        model._X = X
        model._y = 2.0 * X[:, 0]
        model.evaluate(individual)

        expression = model.expression(individual)

        self.assertIn("(2) * (x)", expression)
        self.assertNotIn("(y)", expression)

    def test_workflow_best_expression_is_reference_normalized(self):
        model = self._model(variable_names=("I1",), n_genes=1)
        individual = self._individual(model, (self._terminal_gene(model, "I1"),))
        individual.theta = np.array([2.0], dtype=float)
        individual.sparse_fit = SimpleNamespace(active_mask=np.array([True], dtype=bool))
        model.best_individual = individual

        expression = sgeppy_workflow._reference_normalized_expression(model)

        self.assertIn("(2) * (I1)", expression)
        self.assertIn("+ (-6)", expression)
        self.assertAlmostEqual(sgeppy_workflow._reference_energy_offset(model), 6.0)

    def test_stress_builder_filters_duplicate_gene_columns(self):
        dataset = synthetic_neo_hookean_dataset(num_samples=6, seed=3)
        variable_names = ("K1", "Jm1")
        model = self._model(variable_names=variable_names, n_genes=3, fit_intercept=True)
        individual = self._individual(
            model,
            (
                self._terminal_gene(model, "K1"),
                self._terminal_gene(model, "K1"),
                self._terminal_gene(model, "Jm1"),
            ),
        )
        variables = invariant_variables(dataset, variable_names)
        builder = stress_feature_builder(dataset, variable_names)

        features, valid = builder(model, individual, model._as_matrix(variables))

        self.assertEqual(features.shape, (dataset.target_vector.size, 4))
        self.assertTrue(valid[0])
        self.assertFalse(valid[1])
        self.assertFalse(valid[-1])
        self.assertGreater(np.linalg.norm(features[:, 0]), 1e-12)
        self.assertTrue(np.allclose(features[:, 1], 0.0))

    def test_config_file_and_cli_overrides_drive_sgeppy_inputs(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "sgeppy.json"
            config_path.write_text(
                json.dumps(
                    {
                        "sgeppy": {
                            "data_dir": "dataset/fem_data/plate_hole_fenics/GT",
                            "fitting_mode": "weak_form_jax",
                            "loadsteps": [10],
                            "max_elements_per_loadstep": None,
                            "jax_precision": "float32",
                            "jax_cache_enabled": True,
                            "jax_cache_size": 32,
                            "jax_cache_device_outputs": False,
                            "weak_form": {
                                "balance": 12.0,
                                "penalty_lp": 0.0,
                                "p": 0.5,
                                "num_increments": 1,
                                "factor_increments": 2.0,
                                "num_guesses": 1,
                                "num_iterations": 10,
                                "threshold_iter": 1e-12,
                                "threshold": 1e-9,
                            },
                            "model": {
                                "variable_names": ["K2", "Jm1"],
                                "unary_operators": ["square"],
                                "binary_operators": ["add", "mul"],
                                "n_generations": 2,
                                "population_size": 5,
                                "n_genes": 3,
                                "early_stop_value": 1e-8,
                                "fitness_metrics": ["rmse", "aicc"],
                                "epsilons": [10.0, None],
                            },
                        }
                    }
                ),
                encoding="utf-8",
            )

            config = config_from_file(config_path)
            self.assertEqual(config.fitting_mode, "weak_form_jax")
            self.assertEqual(SGEPWorkflowConfig().jax_precision, "float64")
            self.assertTrue(SGEPWorkflowConfig().jax_cache_enabled)
            self.assertEqual(SGEPWorkflowConfig().jax_cache_size, 256)
            self.assertTrue(SGEPWorkflowConfig().jax_cache_device_outputs)
            self.assertEqual(config.jax_precision, "float32")
            self.assertTrue(config.jax_cache_enabled)
            self.assertEqual(config.jax_cache_size, 32)
            self.assertFalse(config.jax_cache_device_outputs)
            self.assertEqual(config.loadsteps, [10])
            self.assertIsNone(config.max_elements_per_loadstep)
            self.assertEqual(config.weak_form.balance, 12.0)
            self.assertEqual(config.weak_form.penalty_lp, 0.0)
            self.assertEqual(config.weak_form.threshold, 1e-9)
            self.assertEqual(config.model.variable_names, ("K2", "Jm1"))
            self.assertEqual(config.model.unary_operators, ("square",))
            self.assertEqual(config.model.binary_operators, ("add", "mul"))
            self.assertEqual(config.model.n_generations, 2)
            self.assertEqual(config.model.early_stop_value, 1e-8)
            self.assertEqual(config.model.fitness_metrics, ("rmse", "aicc"))
            self.assertEqual(config.model.epsilons, (10.0, None))

            args = build_parser().parse_args(
                [
                    "--config",
                    str(config_path),
                    "--generations",
                    "4",
                    "--population-size",
                    "7",
                    "--n-genes",
                    "2",
                    "--early-stop-value",
                    "1e-6",
                    "--loadsteps",
                    "20,30",
                    "--fitness-metrics",
                    "aic,rmse",
                    "--epsilons",
                    "none,5",
                    "--fitting-mode",
                    "direct_stress",
                    "--jax-precision",
                    "float64",
                    "--jax-cache-size",
                    "8",
                    "--disable-jax-cache",
                    "--disable-jax-cache-device-outputs",
                    "--quiet",
                ]
            )
            updated = _apply_overrides(config, args)
            self.assertEqual(updated.fitting_mode, "direct_stress")
            self.assertEqual(updated.jax_precision, "float64")
            self.assertEqual(updated.jax_cache_size, 8)
            self.assertFalse(updated.jax_cache_enabled)
            self.assertFalse(updated.jax_cache_device_outputs)
            self.assertEqual(updated.loadsteps, [20, 30])
            self.assertEqual(updated.model.n_generations, 4)
            self.assertEqual(updated.model.population_size, 7)
            self.assertEqual(updated.model.n_genes, 2)
            self.assertEqual(updated.model.early_stop_value, 1e-6)
            self.assertEqual(updated.model.fitness_metrics, ("aic", "rmse"))
            self.assertEqual(updated.model.epsilons, (None, 5.0))
            self.assertFalse(updated.progress_log)
            self.assertFalse(updated.model.verbose)

    def test_weak_form_jax_requires_optional_dependencies(self):
        with mock.patch(
            "sym_modeling.domains.fem.methods.sgeppy.jax_backend.importlib.import_module",
            side_effect=ModuleNotFoundError("No module named 'jax'"),
        ):
            with self.assertRaisesRegex(ImportError, "weak_form_jax.*pip install"):
                require_jax_fem_backend()

    def test_jax_weak_form_cache_is_bounded(self):
        timing = {key: 0.0 for key in WEAK_FORM_JAX_TIMING_KEYS}
        cache = JaxWeakFormEvaluationCache(enabled=True, max_size=2, timing=timing)
        cache.put_artifact(("a",), 1)
        cache.put_artifact(("b",), 2)
        self.assertEqual(timing["weak_form_jax_cache_entries"], 2.0)
        self.assertEqual(cache.get_artifact(("a",)), 1)
        cache.put_artifact(("c",), 3)

        self.assertIsNone(cache.get_artifact(("b",)))
        self.assertEqual(cache.get_artifact(("a",)), 1)
        self.assertEqual(cache.get_artifact(("c",)), 3)
        self.assertEqual(timing["weak_form_jax_cache_entries"], 2.0)

    def test_config_rejects_unknown_weak_form_key(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "sgeppy.json"
            config_path.write_text(
                json.dumps({"sgeppy": {"weak_form": {"not_a_key": 1.0}}}),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "weak_form"):
                config_from_file(config_path)

    def test_weak_form_requires_data_dir(self):
        workflow = SGEPWorkflow(
            SGEPWorkflowConfig(
                fitting_mode="weak_form",
                model=GeppySGEPConfig(
                    binary_operators=("add",),
                    population_size=3,
                    verbose=False,
                ),
            )
        )

        with self.assertRaisesRegex(ValueError, "requires data_dir"):
            workflow.train()

    def test_weak_form_jax_requires_data_dir(self):
        workflow = SGEPWorkflow(
            SGEPWorkflowConfig(
                fitting_mode="weak_form_jax",
                model=GeppySGEPConfig(
                    binary_operators=("add",),
                    population_size=3,
                    verbose=False,
                ),
            )
        )

        with self.assertRaisesRegex(ValueError, "requires data_dir"):
            workflow.train()

    def test_workflow_records_total_and_generation_timing_outputs(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = SGEPWorkflowConfig(
                synthetic_samples=8,
                output_dir=tmp_dir,
                model=GeppySGEPConfig(
                    variable_names=("K1", "Jm1"),
                    binary_operators=("add",),
                    unary_operators=(),
                    head_length=1,
                    n_genes=2,
                    population_size=3,
                    n_generations=1,
                    n_elites=1,
                    fit_intercept=False,
                    mut_uniform_pb=0.0,
                    mut_invert_pb=0.0,
                    mut_is_transpose_pb=0.0,
                    mut_ris_transpose_pb=0.0,
                    mut_gene_transpose_pb=0.0,
                    cx_one_point_pb=0.0,
                    cx_two_point_pb=0.0,
                    cx_gene_pb=0.0,
                    verbose=False,
                ),
                progress_log=False,
            )

            result = SGEPWorkflow(config).train()

            self.assertGreater(result.timing["wall_seconds"], 0.0)
            self.assertGreater(result.timing["cpu_seconds"], 0.0)
            self.assertEqual(len(result.history), 2)
            for row in result.history:
                self.assertIn("wall_seconds", row)
                self.assertIn("cpu_seconds", row)
                self.assertIn("evaluation_wall_seconds", row)
                self.assertIn("evaluation_cpu_seconds", row)
                self.assertGreaterEqual(row["wall_seconds"], 0.0)
                self.assertGreaterEqual(row["cpu_seconds"], 0.0)
                self.assertGreaterEqual(row["evaluation_wall_seconds"], 0.0)
                self.assertGreaterEqual(row["evaluation_cpu_seconds"], 0.0)

            summary = json.loads(Path(result.output_paths["summary_json"]).read_text(encoding="utf-8"))
            self.assertIn("timing", summary)
            self.assertGreater(summary["timing"]["wall_seconds"], 0.0)
            self.assertIn("expression_tree_png", result.output_paths)
            self.assertTrue(Path(result.output_paths["expression_tree_png"]).exists())
            self.assertGreater(Path(result.output_paths["expression_tree_png"]).stat().st_size, 0)
            self.assertEqual(summary["output_paths"]["expression_tree_png"], result.output_paths["expression_tree_png"])

            with Path(result.output_paths["history_csv"]).open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 2)
            self.assertIn("wall_seconds", rows[0])
            self.assertIn("cpu_seconds", rows[0])
            self.assertIn("evaluation_wall_seconds", rows[0])
            self.assertIn("evaluation_cpu_seconds", rows[0])
            self.assertIn("early_stop", rows[0])

    def test_weak_form_cache_reuses_fem_invariant_dataset(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir) / "dataset"
            self._write_single_triangle_known_law(root)
            config = SGEPWorkflowConfig(
                fitting_mode="weak_form",
                data_dir=str(root),
                loadsteps=[10],
                model=GeppySGEPConfig(
                    variable_names=("K1", "Jm1"),
                    binary_operators=("add",),
                    unary_operators=(),
                    head_length=1,
                    n_genes=2,
                    population_size=3,
                    n_elites=1,
                    fit_intercept=False,
                    verbose=False,
                ),
                weak_form=WeakFormConfig(
                    penalty_lp=0.0,
                    num_increments=1,
                    threshold=1e-10,
                    threshold_iter=1e-12,
                ),
                progress_log=False,
            )
            workflow = SGEPWorkflow(config)
            workflow.dataset = workflow._load_dataset()
            workflow.fem_datasets = workflow._load_fem_datasets()
            variables = invariant_variables(workflow.dataset, config.model.variable_names)
            model = self._model(variable_names=config.model.variable_names, n_genes=2, fit_intercept=False)
            individual = self._individual(
                model,
                (
                    self._terminal_gene(model, "K1"),
                    self._terminal_gene(model, "Jm1"),
                ),
            )
            builder = stress_feature_builder(
                workflow.dataset,
                config.model.variable_names,
                derivative_step=config.derivative_step,
                value_limit=config.invalid_value_limit,
                duplicate_correlation=config.duplicate_correlation,
            )

            with mock.patch.object(
                sgeppy_workflow,
                "build_stress_dataset_from_fem_data",
                wraps=sgeppy_workflow.build_stress_dataset_from_fem_data,
            ) as build_dataset:
                workflow.weak_form_cache = workflow._build_weak_form_cache()
                evaluator = workflow._weak_form_evaluator(builder)
                for _ in range(2):
                    evaluator(
                        model,
                        individual,
                        model._as_matrix(variables),
                        workflow.dataset.target_vector,
                    )

            self.assertEqual(build_dataset.call_count, 1)

    @unittest.skipUnless(is_jax_fem_backend_available(), "JAX/JAX-FEM optional dependencies are not installed.")
    def test_jax_feature_derivative_matches_finite_difference_stress_builder(self):
        dataset = synthetic_neo_hookean_dataset(num_samples=5, seed=5)
        variable_names = ("K1", "Jm1")
        model = self._model(
            variable_names=variable_names,
            n_genes=4,
            fit_intercept=False,
            binary_operators=("add", "mul"),
        )
        individual = self._individual(
            model,
            (
                self._terminal_gene(model, "K1"),
                self._terminal_gene(model, "Jm1"),
                self._binary_gene(model, "add", "K1", "Jm1"),
                self._binary_gene(model, "mul", "K1", "Jm1"),
            ),
        )
        variables = invariant_variables(dataset, variable_names)
        finite_features, finite_valid = stress_feature_builder(
            dataset,
            variable_names,
            duplicate_correlation=1.1,
        )(model, individual, model._as_matrix(variables))
        jax_features, jax_valid = jax_stress_feature_builder(
            dataset,
            variable_names,
            duplicate_correlation=1.1,
        )(model, individual, model._as_matrix(variables))

        self.assertTrue(np.array_equal(finite_valid, jax_valid))
        self.assertTrue(np.allclose(jax_features, finite_features, atol=2e-5, rtol=2e-5))

        _, dqdf = feature_values_and_dqdf(model, individual, dataset.F, variable_names)
        self.assertEqual(dqdf.shape, (dataset.num_points, 4, 4))
        self.assertTrue(np.all(np.isfinite(dqdf)))

    def test_weak_form_evaluator_recovers_single_triangle_coefficients(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir) / "dataset"
            expected_theta = self._write_single_triangle_known_law(root)
            config = SGEPWorkflowConfig(
                fitting_mode="weak_form",
                data_dir=str(root),
                loadsteps=[10],
                model=GeppySGEPConfig(
                    variable_names=("K1", "Jm1"),
                    binary_operators=("add",),
                    unary_operators=(),
                    head_length=1,
                    n_genes=2,
                    population_size=3,
                    n_elites=1,
                    fit_intercept=False,
                    verbose=False,
                ),
                weak_form=WeakFormConfig(
                    penalty_lp=0.0,
                    num_increments=1,
                    threshold=1e-10,
                    threshold_iter=1e-12,
                ),
                progress_log=False,
            )
            workflow = SGEPWorkflow(config)
            workflow.dataset = workflow._load_dataset()
            workflow.fem_datasets = workflow._load_fem_datasets()
            variables = invariant_variables(workflow.dataset, config.model.variable_names)
            model = self._model(variable_names=config.model.variable_names, n_genes=2, fit_intercept=False)
            individual = self._individual(
                model,
                (
                    self._terminal_gene(model, "K1"),
                    self._terminal_gene(model, "Jm1"),
                ),
            )
            builder = stress_feature_builder(
                workflow.dataset,
                config.model.variable_names,
                derivative_step=config.derivative_step,
                value_limit=config.invalid_value_limit,
                duplicate_correlation=config.duplicate_correlation,
            )

            fit, valid = workflow._weak_form_evaluator(builder)(
                model,
                individual,
                model._as_matrix(variables),
                workflow.dataset.target_vector,
            )

            self.assertTrue(np.array_equal(valid, [True, True]))
            self.assertLess(fit.metrics.rmse, 1e-8)
            self.assertTrue(np.allclose(fit.theta, expected_theta, atol=1e-6))

    @unittest.skipUnless(is_jax_fem_backend_available(), "JAX/JAX-FEM optional dependencies are not installed.")
    def test_weak_form_jax_evaluator_recovers_single_triangle_coefficients(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir) / "dataset"
            expected_theta = self._write_single_triangle_known_law(root)
            config = SGEPWorkflowConfig(
                fitting_mode="weak_form_jax",
                data_dir=str(root),
                loadsteps=[10],
                model=GeppySGEPConfig(
                    variable_names=("K1", "Jm1"),
                    binary_operators=("add",),
                    unary_operators=(),
                    head_length=1,
                    n_genes=2,
                    population_size=3,
                    n_elites=1,
                    fit_intercept=False,
                    verbose=False,
                ),
                weak_form=WeakFormConfig(
                    penalty_lp=0.0,
                    num_increments=1,
                    threshold=1e-10,
                    threshold_iter=1e-12,
                ),
                progress_log=False,
            )
            workflow = SGEPWorkflow(config)
            workflow.dataset = workflow._load_dataset()
            workflow.fem_datasets = workflow._load_fem_datasets()
            workflow.weak_form_cache = workflow._build_weak_form_cache()
            variables = invariant_variables(workflow.dataset, config.model.variable_names)
            model = self._model(variable_names=config.model.variable_names, n_genes=2, fit_intercept=False)
            individual = self._individual(
                model,
                (
                    self._terminal_gene(model, "K1"),
                    self._terminal_gene(model, "Jm1"),
                ),
            )
            builder = jax_stress_feature_builder(
                workflow.dataset,
                config.model.variable_names,
                value_limit=config.invalid_value_limit,
                duplicate_correlation=config.duplicate_correlation,
            )

            evaluator = workflow._weak_form_jax_evaluator(builder)
            fit, valid = evaluator(
                model,
                individual,
                model._as_matrix(variables),
                workflow.dataset.target_vector,
            )
            cache_misses = workflow._weak_form_jax_timing["weak_form_jax_cache_misses"]
            cached_fit, cached_valid = evaluator(
                model,
                individual,
                model._as_matrix(variables),
                workflow.dataset.target_vector,
            )

            self.assertTrue(np.array_equal(valid, [True, True]))
            self.assertLess(fit.metrics.rmse, 1e-8)
            self.assertTrue(np.allclose(fit.theta, expected_theta, atol=1e-6))
            self.assertTrue(np.array_equal(cached_valid, valid))
            self.assertTrue(np.allclose(cached_fit.theta, fit.theta, atol=1e-12))
            self.assertEqual(workflow._weak_form_jax_timing["weak_form_jax_evaluations"], 2.0)
            self.assertGreater(workflow._weak_form_jax_timing["weak_form_jax_cache_hits"], 0.0)
            self.assertEqual(workflow._weak_form_jax_timing["weak_form_jax_cache_misses"], cache_misses)
            self.assertGreaterEqual(workflow._weak_form_jax_timing["weak_form_jax_gene_derivative_seconds"], 0.0)
            self.assertGreaterEqual(workflow._weak_form_jax_timing["weak_form_jax_weak_lhs_seconds"], 0.0)
            self.assertGreaterEqual(workflow._weak_form_jax_timing["weak_form_jax_transfer_seconds"], 0.0)
            self.assertGreaterEqual(workflow._weak_form_jax_timing["weak_form_jax_lp_seconds"], 0.0)

    @unittest.skipUnless(is_jax_fem_backend_available(), "JAX/JAX-FEM optional dependencies are not installed.")
    def test_weak_form_jax_train_records_timing_outputs(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir) / "dataset"
            self._write_single_triangle_known_law(root)
            config = SGEPWorkflowConfig(
                fitting_mode="weak_form_jax",
                data_dir=str(root),
                loadsteps=[10],
                output_dir=str(Path(tmp_dir) / "out"),
                model=GeppySGEPConfig(
                    variable_names=("K1", "Jm1"),
                    binary_operators=("add",),
                    unary_operators=(),
                    head_length=1,
                    n_genes=2,
                    population_size=3,
                    n_generations=0,
                    n_elites=1,
                    fit_intercept=False,
                    verbose=False,
                ),
                weak_form=WeakFormConfig(
                    penalty_lp=0.0,
                    num_increments=1,
                    threshold=1e-10,
                    threshold_iter=1e-12,
                ),
                progress_log=False,
            )

            result = SGEPWorkflow(config).train()
            summary = json.loads(Path(result.output_paths["summary_json"]).read_text(encoding="utf-8"))

            for key in WEAK_FORM_JAX_TIMING_KEYS:
                self.assertIn(key, result.timing)
                self.assertIn(key, summary["timing"])
                self.assertGreaterEqual(result.timing[key], 0.0)
            self.assertGreater(result.timing["weak_form_jax_evaluations"], 0.0)

    @unittest.skipUnless(is_jax_fem_backend_available(), "JAX/JAX-FEM optional dependencies are not installed.")
    def test_weak_form_jax_cache_can_be_disabled(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir) / "dataset"
            expected_theta = self._write_single_triangle_known_law(root)
            config = SGEPWorkflowConfig(
                fitting_mode="weak_form_jax",
                data_dir=str(root),
                loadsteps=[10],
                jax_cache_enabled=False,
                model=GeppySGEPConfig(
                    variable_names=("K1", "Jm1"),
                    binary_operators=("add",),
                    unary_operators=(),
                    head_length=1,
                    n_genes=2,
                    population_size=3,
                    n_elites=1,
                    fit_intercept=False,
                    verbose=False,
                ),
                weak_form=WeakFormConfig(
                    penalty_lp=0.0,
                    num_increments=1,
                    threshold=1e-10,
                    threshold_iter=1e-12,
                ),
                progress_log=False,
            )
            workflow = SGEPWorkflow(config)
            workflow.dataset = workflow._load_dataset()
            workflow.fem_datasets = workflow._load_fem_datasets()
            workflow.weak_form_cache = workflow._build_weak_form_cache()
            variables = invariant_variables(workflow.dataset, config.model.variable_names)
            model = self._model(variable_names=config.model.variable_names, n_genes=2, fit_intercept=False)
            individual = self._individual(
                model,
                (
                    self._terminal_gene(model, "K1"),
                    self._terminal_gene(model, "Jm1"),
                ),
            )
            builder = jax_stress_feature_builder(
                workflow.dataset,
                config.model.variable_names,
                value_limit=config.invalid_value_limit,
                duplicate_correlation=config.duplicate_correlation,
            )

            fit, valid = workflow._weak_form_jax_evaluator(builder)(
                model,
                individual,
                model._as_matrix(variables),
                workflow.dataset.target_vector,
            )

            self.assertTrue(np.array_equal(valid, [True, True]))
            self.assertLess(fit.metrics.rmse, 1e-8)
            self.assertTrue(np.allclose(fit.theta, expected_theta, atol=1e-6))
            self.assertEqual(workflow._weak_form_jax_timing["weak_form_jax_cache_hits"], 0.0)
            self.assertEqual(workflow._weak_form_jax_timing["weak_form_jax_cache_entries"], 0.0)


if __name__ == "__main__":
    unittest.main()
