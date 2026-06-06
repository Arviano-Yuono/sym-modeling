from __future__ import annotations

import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from sym_modeling.utils.plot_dolfinx_mesh import (  # noqa: E402
    DEFAULT_MSH_PATH,
    DEFAULT_OUTPUT_PATH,
    build_parser,
)


class MeshPlotCliTests(unittest.TestCase):
    def test_parser_defaults_to_plate_hole_mesh(self):
        args = build_parser().parse_args([])
        self.assertEqual(args.msh, DEFAULT_MSH_PATH)
        self.assertEqual(args.output, DEFAULT_OUTPUT_PATH)
        self.assertEqual(args.gdim, 2)
        self.assertFalse(args.show_markers)

    def test_parser_accepts_style_options(self):
        args = build_parser().parse_args(
            [
                "--msh",
                "mesh.msh",
                "--output",
                "mesh.png",
                "--show-markers",
                "--line-width",
                "2.5",
                "--window-size",
                "1000x800",
            ]
        )
        self.assertEqual(args.msh, "mesh.msh")
        self.assertEqual(args.output, "mesh.png")
        self.assertTrue(args.show_markers)
        self.assertEqual(args.line_width, 2.5)
        self.assertEqual(args.window_size, (1000, 800))


if __name__ == "__main__":
    unittest.main()
