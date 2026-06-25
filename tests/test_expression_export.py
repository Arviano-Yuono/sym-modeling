from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import sympy as sp


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from sym_modeling.utils.export_sgeppy_expressions import (  # noqa: E402
    expression_to_latex,
    main,
    simplify_expression_for_display,
)


class ExpressionExportTests(unittest.TestCase):
    def test_custom_title_is_used_in_reports(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            model_dir = root / "nh2"
            output_dir = root / "report"
            model_dir.mkdir()
            (model_dir / "summary.json").write_text(
                json.dumps(
                    {
                        "best_expression": "(0.5) * (K1) + (1.5) * (Jm1**2)",
                        "metrics": {
                            "rmse": 1.25e-6,
                            "rss": 2.5e-12,
                            "num_parameters": 2,
                        },
                    }
                ),
                encoding="utf-8",
            )

            exit_code = main(
                [
                    "--input-root",
                    str(root),
                    "--models",
                    "nh2",
                    "--output-dir",
                    str(output_dir),
                    "--title",
                    "EUCLID Best Expressions",
                ]
            )

            self.assertEqual(exit_code, 0)
            markdown = (output_dir / "best_expressions.md").read_text(encoding="utf-8")
            latex = (output_dir / "best_expressions.tex").read_text(encoding="utf-8")
            self.assertIn("# EUCLID Best Expressions", markdown)
            self.assertIn("% EUCLID Best Expressions.", latex)
            self.assertIn("NH2", markdown)
            self.assertIn("Model & RMSE & Active terms & Expression", latex)
            self.assertIn("NH2 & 1.250000e-06 & 2", latex)

    def test_export_simplifies_old_protected_expressions_as_math(self):
        expression = (
            "(1) * (protected_div(K1, K2)) + "
            "(2) * (protected_div(K1, K2)) + "
            "(4) * (protected_sqrt(K1))"
        )

        simplified = simplify_expression_for_display(expression)

        K1, K2 = sp.symbols("K1 K2")
        self.assertEqual(sp.simplify(sp.sympify(simplified) - (3 * K1 / K2 + 4 * sp.sqrt(K1))), 0)

    def test_export_rounds_simplified_numbers_when_requested(self):
        expression = "(1.23456) * (K1) + (0.98765) * (K1) + (0.0004)"

        simplified = simplify_expression_for_display(expression, round_decimals=2)

        self.assertEqual(simplified, "2.22*K1")

    def test_latex_output_uses_sympy_printer_without_extra_grouping(self):
        latex = expression_to_latex("1.5*Jm1**2 + 0.5*K1")

        self.assertEqual(latex, r"1.5 \left(J - 1\right)^{2} + 0.5 K_1")

    def test_reports_use_simplified_math_expression(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            model_dir = root / "ab"
            output_dir = root / "report"
            model_dir.mkdir()
            (model_dir / "best_so_far.json").write_text(
                json.dumps(
                    {
                        "best_expression": (
                            "(1) * (protected_div(K1, K2)) + "
                            "(2) * (protected_div(K1, K2))"
                        ),
                        "metrics": {"rmse": 1.0, "num_parameters": 2},
                    }
                ),
                encoding="utf-8",
            )

            exit_code = main(
                [
                    "--input-root",
                    str(root),
                    "--models",
                    "ab",
                    "--output-dir",
                    str(output_dir),
                ]
            )

            self.assertEqual(exit_code, 0)
            csv_report = (output_dir / "best_expressions.csv").read_text(encoding="utf-8")
            text_report = (output_dir / "best_expressions.txt").read_text(encoding="utf-8")
            self.assertIn("3*K1/K2", csv_report)
            self.assertIn("3*K1/K2", text_report)

    def test_reports_honor_round_decimals_option(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            model_dir = root / "nh2"
            output_dir = root / "report"
            model_dir.mkdir()
            (model_dir / "summary.json").write_text(
                json.dumps(
                    {
                        "best_expression": "(1.23456) * (K1) + (0.98765) * (K1)",
                        "metrics": {"rmse": 1.0, "num_parameters": 2},
                    }
                ),
                encoding="utf-8",
            )

            exit_code = main(
                [
                    "--input-root",
                    str(root),
                    "--models",
                    "nh2",
                    "--output-dir",
                    str(output_dir),
                    "--round-decimals",
                    "2",
                ]
            )

            self.assertEqual(exit_code, 0)
            csv_report = (output_dir / "best_expressions.csv").read_text(encoding="utf-8")
            text_report = (output_dir / "best_expressions.txt").read_text(encoding="utf-8")
            self.assertIn("2.22*K1", csv_report)
            self.assertIn("2.22*K1", text_report)


if __name__ == "__main__":
    unittest.main()
