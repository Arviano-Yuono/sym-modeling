from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from sym_modeling.utils.export_sgeppy_expressions import main  # noqa: E402


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


if __name__ == "__main__":
    unittest.main()
