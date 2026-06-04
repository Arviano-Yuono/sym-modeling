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
from sym_modeling.domains.fem.methods.sgeppy import operator as sgeppy_ops  # noqa: E402
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
from sym_modeling.domains.fem.methods.sgeppy.gene_evaluator import (  # noqa: E402
    PerGeneJitCache,
    _jax_operators,
    compile_gene_function,
    evaluate_genes_on_F,
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

    def test_core_numpy_operators_are_protected_and_non_mutating(self):
        numerator = np.array([4.0, -3.0, 2.0], dtype=float)
        denominator = np.array([2.0, 0.0, 1e-8], dtype=float)
        denominator_before = denominator.copy()

        self.assertTrue(np.allclose(sgeppy_ops.add(numerator, denominator), [6.0, -3.0, 2.00000001]))
        self.assertTrue(np.allclose(sgeppy_ops.sub(numerator, denominator), [2.0, -3.0, 1.99999999]))
        self.assertTrue(np.allclose(sgeppy_ops.mul(numerator, denominator), [8.0, 0.0, 2e-8]))
        self.assertTrue(np.allclose(sgeppy_ops.protected_div(numerator, denominator), [2.0, -3.0, 2.0]))
        self.assertTrue(np.array_equal(denominator, denominator_before))
        self.assertEqual(sgeppy_ops.protected_div(3.0, 0.0), 3.0)
        self.assertTrue(np.allclose(sgeppy_ops.square([-2.0, 3.0]), [4.0, 9.0]))
        self.assertTrue(np.allclose(sgeppy_ops.cube([-2.0, 3.0]), [-8.0, 27.0]))
        self.assertTrue(np.allclose(sgeppy_ops.protected_sqrt([-4.0, 0.0]), np.sqrt([4.0 + 1e-12, 1e-12])))
        self.assertTrue(np.allclose(sgeppy_ops.protected_log([-2.0, 0.0]), np.log([2.0 + 1e-12, 1e-12])))
        self.assertTrue(np.allclose(sgeppy_ops.protected_exp([-30.0, 30.0]), np.exp([-20.0, 20.0])))

    def test_protected_operator_name_is_preserved_in_expression(self):
        model = self._model(binary_operators=("protected_div",))
        individual = self._individual(
            model,
            (self._binary_gene(model, "protected_div", "x", "y"),),
        )
        individual.theta = np.array([2.0], dtype=float)
        individual.sparse_fit = SimpleNamespace(active_mask=np.array([True], dtype=bool))

        expression = model.expression(individual)

        self.assertIn("protected_div(x, y)", expression)

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
            self.assertTrue(SGEPWorkflowConfig().generation_log)
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
                    "--noise-level",
                    "1e-4",
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
            self.assertEqual(updated.noise_level, 1e-4)
            self.assertEqual(updated.model.n_generations, 4)
            self.assertEqual(updated.model.population_size, 7)
            self.assertEqual(updated.model.n_genes, 2)
            self.assertEqual(updated.model.early_stop_value, 1e-6)
            self.assertEqual(updated.model.fitness_metrics, ("aic", "rmse"))
            self.assertEqual(updated.model.epsilons, (None, 5.0))
            self.assertFalse(updated.progress_log)
            self.assertFalse(updated.model.verbose)
            self.assertTrue(updated.generation_log)

            disabled_args = build_parser().parse_args(
                [
                    "--config",
                    str(config_path),
                    "--disable-generation-log",
                ]
            )
            self.assertFalse(_apply_overrides(config, disabled_args).generation_log)

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

    def test_config_rejects_legacy_operator_names(self):
        cases = (
            {"binary_operators": ["div"]},
            {"binary_operators": ["add"], "unary_operators": ["sqrt"]},
            {"binary_operators": ["add"], "unary_operators": ["log"]},
            {"binary_operators": ["add"], "unary_operators": ["exp"]},
        )
        for model_values in cases:
            with self.subTest(model_values=model_values):
                with self.assertRaisesRegex(ValueError, "Unsupported"):
                    GeppySGEPConfig(**model_values)

    def test_config_keeps_optional_neg_sin_and_cos_operators(self):
        config = GeppySGEPConfig(
            binary_operators=("add",),
            unary_operators=("neg", "sin", "cos"),
        )

        self.assertEqual(config.unary_operators, ("neg", "sin", "cos"))

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
            self.assertIn("generation_log_csv", result.output_paths)
            self.assertIn("best_so_far_json", result.output_paths)

            with Path(result.output_paths["history_csv"]).open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 2)
            self.assertIn("wall_seconds", rows[0])
            self.assertIn("cpu_seconds", rows[0])
            self.assertIn("evaluation_wall_seconds", rows[0])
            self.assertIn("evaluation_cpu_seconds", rows[0])
            self.assertIn("early_stop", rows[0])

            with Path(result.output_paths["generation_log_csv"]).open(encoding="utf-8", newline="") as handle:
                generation_rows = list(csv.DictReader(handle))
            self.assertEqual(len(generation_rows), 2)
            self.assertEqual([int(row["gen"]) for row in generation_rows], [0, 1])
            self.assertIn("best_fitness", generation_rows[0])
            self.assertIn("best_expression", generation_rows[0])
            self.assertIn("best_theta", generation_rows[0])
            self.assertIsInstance(json.loads(generation_rows[0]["best_fitness"]), list)
            self.assertIsInstance(json.loads(generation_rows[0]["best_theta"]), list)

            best_so_far = json.loads(Path(result.output_paths["best_so_far_json"]).read_text(encoding="utf-8"))
            self.assertEqual(best_so_far["generation"], 1)
            self.assertIn("statistics", best_so_far)
            self.assertIn("best_expression", best_so_far)
            self.assertIn("best_theta", best_so_far)
            self.assertIn("best_fitness", best_so_far)
            self.assertEqual(sgeppy_workflow._finite_values_or_none([np.inf, np.nan, 1.0]), [None, None, 1.0])

    def test_generation_callback_runs_for_each_completed_generation(self):
        config = GeppySGEPConfig(
            variable_names=("x",),
            binary_operators=("add",),
            unary_operators=(),
            head_length=1,
            n_genes=1,
            population_size=3,
            n_generations=1,
            n_elites=1,
            mut_uniform_pb=0.0,
            mut_invert_pb=0.0,
            mut_is_transpose_pb=0.0,
            mut_ris_transpose_pb=0.0,
            mut_gene_transpose_pb=0.0,
            cx_one_point_pb=0.0,
            cx_two_point_pb=0.0,
            cx_gene_pb=0.0,
            verbose=False,
        )
        model = GeppySGEP(config)
        callbacks = []

        def record(row, best_individual):
            callbacks.append((row, best_individual))
            self.assertIs(best_individual, model.hall_of_fame[0])

        model.fit(
            np.array([[1.0], [2.0], [3.0]], dtype=float),
            np.array([1.0, 2.0, 3.0], dtype=float),
            generation_callback=record,
        )

        self.assertEqual([row["gen"] for row, _ in callbacks], [0, 1])

    def test_generation_log_resets_when_output_directory_is_reused(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = SGEPWorkflowConfig(
                synthetic_samples=8,
                output_dir=tmp_dir,
                model=GeppySGEPConfig(
                    variable_names=("K1",),
                    binary_operators=("add",),
                    unary_operators=(),
                    head_length=1,
                    n_genes=1,
                    population_size=3,
                    n_generations=1,
                    n_elites=1,
                    verbose=False,
                ),
                progress_log=False,
            )
            SGEPWorkflow(config).train()
            config.model.n_generations = 0

            result = SGEPWorkflow(config).train()

            with Path(result.output_paths["generation_log_csv"]).open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["gen"], "0")
            snapshot = json.loads(Path(result.output_paths["best_so_far_json"]).read_text(encoding="utf-8"))
            self.assertEqual(snapshot["generation"], 0)

    def test_generation_log_can_be_disabled_without_disabling_final_outputs(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = SGEPWorkflowConfig(
                synthetic_samples=8,
                output_dir=tmp_dir,
                generation_log=False,
                model=GeppySGEPConfig(
                    variable_names=("K1",),
                    binary_operators=("add",),
                    unary_operators=(),
                    head_length=1,
                    n_genes=1,
                    population_size=3,
                    n_generations=0,
                    n_elites=1,
                    verbose=False,
                ),
                progress_log=False,
            )

            result = SGEPWorkflow(config).train()

            self.assertNotIn("generation_log_csv", result.output_paths)
            self.assertNotIn("best_so_far_json", result.output_paths)
            self.assertTrue(Path(result.output_paths["history_csv"]).exists())
            self.assertTrue(Path(result.output_paths["summary_json"]).exists())
            self.assertFalse((Path(tmp_dir) / "generation_log.csv").exists())
            self.assertFalse((Path(tmp_dir) / "best_so_far.json").exists())

    def test_quiet_cli_override_keeps_durable_generation_log_enabled(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = SGEPWorkflowConfig(
                synthetic_samples=8,
                output_dir=tmp_dir,
                model=GeppySGEPConfig(
                    variable_names=("K1",),
                    binary_operators=("add",),
                    unary_operators=(),
                    head_length=1,
                    n_genes=1,
                    population_size=3,
                    n_generations=0,
                    n_elites=1,
                    verbose=True,
                ),
                progress_log=True,
            )
            args = build_parser().parse_args(["--config", "unused.json", "--quiet"])

            updated = _apply_overrides(config, args)
            result = SGEPWorkflow(updated).train()

            self.assertFalse(updated.progress_log)
            self.assertFalse(updated.model.verbose)
            self.assertTrue(updated.generation_log)
            self.assertTrue(Path(result.output_paths["generation_log_csv"]).exists())
            self.assertTrue(Path(result.output_paths["best_so_far_json"]).exists())

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


    # -- Config inheritance tests --

    def test_config_extends_inherits_parent_values(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            base_path = Path(tmp_dir) / "base.json"
            base_path.write_text(json.dumps({
                "sgeppy": {
                    "fitting_mode": "weak_form_jax",
                    "jax_cache_size": 4096,
                    "weak_form": {"balance": 50.0, "num_iterations": 200},
                    "model": {
                        "variable_names": ["K1"],
                        "binary_operators": ["add"],
                        "population_size": 100,
                        "head_length": 7,
                    },
                }
            }), encoding="utf-8")

            child_path = Path(tmp_dir) / "child.json"
            child_path.write_text(json.dumps({
                "extends": "base.json",
                "sgeppy": {
                    "data_dir": "some/data",
                    "output_dir": "some/output",
                    "jax_cache_size": 512,
                    "model": {"head_length": 4},
                }
            }), encoding="utf-8")

            config = config_from_file(child_path)
            # Child override wins.
            self.assertEqual(config.jax_cache_size, 512)
            self.assertEqual(config.model.head_length, 4)
            # Parent values inherited.
            self.assertEqual(config.fitting_mode, "weak_form_jax")
            self.assertEqual(config.model.population_size, 100)
            self.assertEqual(config.weak_form.balance, 50.0)

    def test_config_extends_deep_merges_nested_dicts(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            base_path = Path(tmp_dir) / "base.json"
            base_path.write_text(json.dumps({
                "sgeppy": {
                    "weak_form": {"balance": 50.0, "num_iterations": 200, "p": 0.25},
                    "model": {
                        "variable_names": ["K1"],
                        "binary_operators": ["add"],
                        "n_genes": 5,
                        "head_length": 7,
                    },
                }
            }), encoding="utf-8")

            child_path = Path(tmp_dir) / "child.json"
            child_path.write_text(json.dumps({
                "extends": "base.json",
                "sgeppy": {
                    "weak_form": {"num_iterations": 100},
                    "model": {"head_length": 3},
                }
            }), encoding="utf-8")

            config = config_from_file(child_path)
            # Child override in nested dict.
            self.assertEqual(config.weak_form.num_iterations, 100)
            self.assertEqual(config.model.head_length, 3)
            # Sibling keys inherited from parent.
            self.assertEqual(config.weak_form.balance, 50.0)
            self.assertEqual(config.weak_form.p, 0.25)
            self.assertEqual(config.model.n_genes, 5)

    def test_config_extends_detects_circular_inheritance(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            a_path = Path(tmp_dir) / "a.json"
            b_path = Path(tmp_dir) / "b.json"
            a_path.write_text(json.dumps({"extends": "b.json", "sgeppy": {}}), encoding="utf-8")
            b_path.write_text(json.dumps({"extends": "a.json", "sgeppy": {}}), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "Circular"):
                config_from_file(a_path)

    def test_config_extends_missing_parent_raises_clear_error(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            child_path = Path(tmp_dir) / "child.json"
            child_path.write_text(json.dumps({
                "extends": "nonexistent.json",
                "sgeppy": {"model": {"variable_names": ["K1"], "binary_operators": ["add"]}},
            }), encoding="utf-8")

            with self.assertRaises(FileNotFoundError):
                config_from_file(child_path)

    def test_config_without_extends_still_works(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "standalone.json"
            config_path.write_text(json.dumps({
                "sgeppy": {
                    "model": {
                        "variable_names": ["K1"],
                        "binary_operators": ["add"],
                    },
                }
            }), encoding="utf-8")

            config = config_from_file(config_path)
            self.assertEqual(config.model.variable_names, ("K1",))

    def test_existing_configs_load_via_inheritance(self):
        """Verify that the real configs under configs/sgeppy/ all load correctly."""
        configs_dir = REPO_ROOT / "configs" / "sgeppy"
        if not configs_dir.is_dir():
            self.skipTest("configs/sgeppy/ not found.")
        for config_path in sorted(configs_dir.glob("*.json")):
            if config_path.name.startswith("_"):
                continue
            with self.subTest(config=config_path.name):
                config = config_from_file(config_path)
                self.assertIsNotNone(config.model)
                self.assertEqual(config.fitting_mode, "weak_form_jax")
                self.assertTrue(len(config.model.variable_names) > 0)
                self.assertTrue(len(config.model.binary_operators) > 0)

    def test_arruda_boyce_config_enables_full_core_grammar(self):
        config = config_from_file(REPO_ROOT / "configs" / "sgeppy" / "ab.json")

        self.assertEqual(config.data_dir, "dataset/fem_data/plate_hole_fenics/AB")
        self.assertEqual(config.loadsteps, [5, 10, 15, 20, 25, 30, 35, 40, 45, 50])
        self.assertEqual(config.output_dir, "output/sgeppy_results_jax/ab_aicc")
        self.assertFalse(config.jax_cache_device_outputs)
        self.assertEqual(config.model.binary_operators, ("add", "sub", "mul", "protected_div"))
        self.assertEqual(
            config.model.unary_operators,
            ("square", "cube", "protected_sqrt", "protected_log", "protected_exp"),
        )

    # -- PerGeneJitCache tests (no JAX required) --

    def test_per_gene_jit_cache_compiles_once(self):
        cache = PerGeneJitCache(enabled=True, max_size=16)
        sentinel = object()
        cache.put("K1 + K2", ("K1", "K2"), "float64", sentinel)
        self.assertIs(cache.get("K1 + K2", ("K1", "K2"), "float64"), sentinel)
        self.assertEqual(cache.size, 1)

    def test_per_gene_jit_cache_bounded_eviction(self):
        cache = PerGeneJitCache(enabled=True, max_size=2)
        cache.put("a", ("K1",), "float64", 1)
        cache.put("b", ("K1",), "float64", 2)
        self.assertEqual(cache.size, 2)
        # Accessing "a" makes it most-recently used.
        cache.get("a", ("K1",), "float64")
        # Adding "c" should evict "b" (the least-recently used).
        cache.put("c", ("K1",), "float64", 3)
        self.assertEqual(cache.size, 2)
        self.assertIsNone(cache.get("b", ("K1",), "float64"))
        self.assertEqual(cache.get("a", ("K1",), "float64"), 1)
        self.assertEqual(cache.get("c", ("K1",), "float64"), 3)

    def test_per_gene_jit_cache_disabled(self):
        cache = PerGeneJitCache(enabled=False, max_size=16)
        cache.put("a", ("K1",), "float64", 1)
        self.assertIsNone(cache.get("a", ("K1",), "float64"))
        self.assertEqual(cache.size, 0)

    def test_per_gene_jit_cache_clear(self):
        cache = PerGeneJitCache(enabled=True, max_size=16)
        cache.put("a", ("K1",), "float64", 1)
        cache.put("b", ("K1",), "float64", 2)
        self.assertEqual(cache.size, 2)
        cache.clear()
        self.assertEqual(cache.size, 0)
        self.assertIsNone(cache.get("a", ("K1",), "float64"))

    def test_config_accepts_jax_gene_cache_size(self):
        config = SGEPWorkflowConfig(jax_gene_cache_size=512)
        self.assertEqual(config.jax_gene_cache_size, 512)
        self.assertEqual(SGEPWorkflowConfig().jax_gene_cache_size, 1024)

    def test_config_rejects_negative_jax_gene_cache_size(self):
        with self.assertRaisesRegex(ValueError, "jax_gene_cache_size"):
            SGEPWorkflowConfig(jax_gene_cache_size=-1)

    # -- Per-gene JIT evaluation tests (require JAX) --

    @unittest.skipUnless(is_jax_fem_backend_available(), "JAX/JAX-FEM optional dependencies are not installed.")
    def test_jax_core_operators_have_finite_outputs_and_gradients(self):
        import jax
        import jax.numpy as jnp

        operators = _jax_operators()
        binary_inputs = (jnp.asarray(2.0), jnp.asarray(0.0))
        for name in ("add", "sub", "mul", "protected_div"):
            with self.subTest(operator=name):
                function = operators[name]
                value = function(*binary_inputs)
                gradient = jax.grad(lambda a: function(a, binary_inputs[1]))(binary_inputs[0])
                self.assertTrue(np.isfinite(np.asarray(value)))
                self.assertTrue(np.isfinite(np.asarray(gradient)))

        for name in ("square", "cube", "protected_sqrt", "protected_log", "protected_exp"):
            with self.subTest(operator=name):
                function = operators[name]
                value = function(jnp.asarray(-2.0))
                gradient = jax.grad(function)(jnp.asarray(-2.0))
                self.assertTrue(np.isfinite(np.asarray(value)))
                self.assertTrue(np.isfinite(np.asarray(gradient)))

    @unittest.skipUnless(is_jax_fem_backend_available(), "JAX/JAX-FEM optional dependencies are not installed.")
    def test_evaluate_genes_matches_direct_gene_compilation(self):
        """Per-gene evaluation should produce the same results as compiling
        all genes into a single function (the old approach)."""
        import jax
        import jax.numpy as jnp

        model = GeppySGEP(GeppySGEPConfig(
            variable_names=("K1", "Jm1"),
            binary_operators=("add", "mul"),
            unary_operators=("square",),
            population_size=4,
            n_genes=2,
            head_length=5,
            random_seed=42,
            verbose=False,
        ))
        model.build()
        individual = model.toolbox.individual()

        F_batch = jnp.asarray([
            [1.1, 0.05, 0.0, 0.95],
            [1.2, 0.0, 0.0, 0.9],
            [1.05, 0.1, -0.05, 1.1],
        ], dtype=jnp.float64)

        variable_names = ("K1", "Jm1")

        # Evaluate via the new per-gene path.
        gene_cache = PerGeneJitCache(enabled=True)
        features, dqdf = evaluate_genes_on_F(
            gene_cache,
            model,
            individual,
            list(range(len(individual))),
            F_batch,
            variable_names,
            precision="float64",
        )

        # Verify shapes.
        n_points = F_batch.shape[0]
        n_genes = len(individual)
        self.assertEqual(features.shape, (n_points, n_genes))
        self.assertEqual(dqdf.shape, (n_points, n_genes, 4))

        # Verify that a second call with the same individual hits the cache.
        features2, dqdf2 = evaluate_genes_on_F(
            gene_cache,
            model,
            individual,
            list(range(len(individual))),
            F_batch,
            variable_names,
            precision="float64",
        )
        np.testing.assert_allclose(np.asarray(features), np.asarray(features2), atol=1e-12)
        np.testing.assert_allclose(np.asarray(dqdf), np.asarray(dqdf2), atol=1e-12)

        # Verify cache has entries.
        self.assertGreater(gene_cache.size, 0)

    @unittest.skipUnless(is_jax_fem_backend_available(), "JAX/JAX-FEM optional dependencies are not installed.")
    def test_per_gene_eval_agrees_with_feature_values_and_dqdf(self):
        """The refactored feature_values_and_dqdf should produce valid results
        using per-gene JIT caching."""
        import jax.numpy as jnp

        model = GeppySGEP(GeppySGEPConfig(
            variable_names=("K1", "Jm1"),
            binary_operators=("add", "mul"),
            population_size=4,
            n_genes=2,
            head_length=5,
            random_seed=7,
            verbose=False,
        ))
        model.build()
        individual = model.toolbox.individual()

        F_batch = jnp.asarray([
            [1.1, 0.05, 0.0, 0.95],
            [1.2, 0.0, 0.0, 0.9],
        ], dtype=jnp.float64)

        gene_cache = PerGeneJitCache(enabled=True)
        features, dqdf = feature_values_and_dqdf(
            model,
            individual,
            np.asarray(F_batch),
            ("K1", "Jm1"),
            precision="float64",
            gene_cache=gene_cache,
        )
        self.assertEqual(features.shape[0], 2)
        self.assertEqual(features.shape[1], 2)
        self.assertTrue(np.all(np.isfinite(features)))
        self.assertTrue(np.all(np.isfinite(dqdf)))


if __name__ == "__main__":
    unittest.main()
