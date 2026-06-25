from __future__ import annotations

import csv
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from sym_modeling.domains.fem.forward_comparison import (  # noqa: E402
    DEFAULT_PLOT_QUANTITIES,
    ForwardComparisonConfig,
    build_parser,
    compare_forward_results,
)


def _write_step(
    root: Path,
    step: str,
    *,
    ux_offset: float = 0.0,
    uy_offset: float = 0.0,
    f_offset: float = 0.0,
    p_values: tuple[float, float, float, float] | None = (2.0, 2.0, 2.0, 2.0),
    reaction_offset: float = 0.0,
) -> None:
    step_dir = root / step
    step_dir.mkdir(parents=True, exist_ok=True)

    with (step_dir / "output_nodes.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("id", "x", "y", "ux", "uy", "fintx", "finty", "bcx", "bcy"))
        writer.writeheader()
        rows = [
            (0, 0.0, 0.0, 1.0, 2.0),
            (1, 1.0, 0.0, 2.0, 3.0),
            (2, 0.0, 1.0, 3.0, 4.0),
        ]
        for node_id, x, y, ux, uy in rows:
            writer.writerow(
                {
                    "id": node_id,
                    "x": x,
                    "y": y,
                    "ux": ux + ux_offset,
                    "uy": uy + uy_offset,
                    "fintx": 0.0,
                    "finty": 0.0,
                    "bcx": 0,
                    "bcy": 0,
                }
            )

    element_fields = ["node1", "node2", "node3", "Fxx", "Fxy", "Fyx", "Fyy"]
    if p_values is not None:
        element_fields.extend(["Pxx", "Pxy", "Pyx", "Pyy"])
    with (step_dir / "output_elements.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=element_fields)
        writer.writeheader()
        row = {
            "node1": 0,
            "node2": 1,
            "node3": 2,
            "Fxx": 1.0 + f_offset,
            "Fxy": 2.0 + f_offset,
            "Fyx": 3.0 + f_offset,
            "Fyy": 4.0 + f_offset,
        }
        if p_values is not None:
            row.update(dict(zip(("Pxx", "Pxy", "Pyx", "Pyy"), p_values)))
        writer.writerow(row)

    with (step_dir / "output_reactions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("forces",))
        writer.writeheader()
        for value in (1.0, 2.0, 3.0, 4.0):
            writer.writerow({"forces": value + reaction_offset})


class ForwardComparisonTests(unittest.TestCase):
    def test_shared_step_discovery_sorts_numerically_and_records_missing_steps(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            forward = root / "forward"
            reference = root / "reference"
            for step in ("2", "10", "30"):
                _write_step(forward, step)
            for step in ("1", "2", "10.0"):
                _write_step(reference, step)

            result = compare_forward_results(
                ForwardComparisonConfig(
                    forward_root=forward,
                    reference_root=reference,
                    output_dir=root / "comparison",
                    plot_quantities=("ux",),
                )
            )

            self.assertEqual(result.compared_steps, ("2", "10"))
            self.assertEqual(result.missing_forward_steps, ("1",))
            self.assertEqual(result.missing_reference_steps, ("30",))

    def test_metrics_for_one_triangle_known_offsets(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            forward = root / "forward"
            reference = root / "reference"
            _write_step(reference, "10", p_values=(2.0, 2.0, 2.0, 2.0))
            _write_step(
                forward,
                "10",
                ux_offset=1.0,
                f_offset=1.0,
                p_values=(1.0, 2.0, 3.0, 4.0),
                reaction_offset=1.0,
            )

            result = compare_forward_results(
                ForwardComparisonConfig(
                    forward_root=forward,
                    reference_root=reference,
                    output_dir=root / "comparison",
                    plot_quantities=("ux",),
                )
            )
            metrics = {
                (row["step"], row["field"], row["quantity"]): row
                for row in result.metrics
            }

            self.assertAlmostEqual(metrics[("10", "reaction", "forces")]["rmse"], 1.0)
            self.assertAlmostEqual(metrics[("10", "u", "ux")]["rmse"], 1.0)
            self.assertAlmostEqual(metrics[("10", "u", "uy")]["rmse"], 0.0)
            self.assertAlmostEqual(metrics[("10", "F", "F")]["rmse"], 1.0)
            self.assertAlmostEqual(metrics[("10", "P", "Pxx")]["rmse"], 1.0)
            self.assertAlmostEqual(metrics[("10", "P", "P")]["rmse"], math.sqrt(6.0 / 4.0))

    def test_cli_parser_defaults_and_output_path(self):
        args = build_parser().parse_args(
            [
                "--forward-root",
                "output/forward_sgeppy/1e-4/ih",
                "--reference-root",
                "dataset/fem_data/plate_hole_fenics/IH",
            ]
        )
        self.assertIsNone(args.output_dir)
        self.assertEqual(args.plot_quantities, "all")
        self.assertIsNone(args.load_steps)

        config = ForwardComparisonConfig(args.forward_root, args.reference_root)
        self.assertEqual(config.resolved_output_dir, Path("output/forward_sgeppy/1e-4/ih") / "comparison")
        self.assertEqual(config.resolved_plot_quantities, DEFAULT_PLOT_QUANTITIES)

    def test_plot_files_are_produced_for_tiny_one_triangle_case(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            forward = root / "forward"
            reference = root / "reference"
            _write_step(reference, "10")
            _write_step(forward, "10", ux_offset=0.5, f_offset=0.25)

            result = compare_forward_results(
                ForwardComparisonConfig(
                    forward_root=forward,
                    reference_root=reference,
                    output_dir=root / "comparison",
                    plot_quantities=("ux", "Fxx", "Pxx"),
                )
            )

            expected = {
                root / "comparison" / "reaction_comparison.png",
                root / "comparison" / "step_10_displacement_ux.png",
                root / "comparison" / "step_10_Fxx.png",
                root / "comparison" / "step_10_Pxx.png",
            }
            self.assertEqual(set(result.plot_paths), expected)
            for path in expected:
                self.assertTrue(path.exists(), path)
                self.assertGreater(path.stat().st_size, 0)

    def test_missing_piola_columns_are_skipped_and_recorded(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            forward = root / "forward"
            reference = root / "reference"
            _write_step(reference, "10", p_values=None)
            _write_step(forward, "10", p_values=None)

            result = compare_forward_results(
                ForwardComparisonConfig(
                    forward_root=forward,
                    reference_root=reference,
                    output_dir=root / "comparison",
                    plot_quantities=("Pxx", "ux"),
                )
            )

            self.assertEqual(result.summary["piola"]["missing_steps"], ["10"])
            self.assertNotIn(root / "comparison" / "step_10_Pxx.png", set(result.plot_paths))
            self.assertIn(root / "comparison" / "step_10_displacement_ux.png", set(result.plot_paths))
            with result.summary_path.open("r", encoding="utf-8") as handle:
                summary = json.load(handle)
            self.assertFalse(summary["piola"]["available_for_all_compared_steps"])


if __name__ == "__main__":
    unittest.main()
