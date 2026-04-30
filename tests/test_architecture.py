from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


class ArchitectureTests(unittest.TestCase):
    def test_top_level_import_is_lightweight(self):
        import sym_modeling

        self.assertTrue(hasattr(sym_modeling, "SparseRegressionModel"))

    def test_cfd_domain_imports(self):
        from sym_modeling.domains.cfd.data import CFDCaseData, FlowData
        from sym_modeling.domains.cfd.methods.sparta import SpartaConfig

        self.assertIs(FlowData, CFDCaseData)
        self.assertEqual(SpartaConfig.__name__, "SpartaConfig")

    @unittest.skipUnless(importlib.util.find_spec("matplotlib"), "matplotlib is not installed")
    @unittest.skipUnless(importlib.util.find_spec("seaborn"), "seaborn is not installed")
    def test_fem_domain_imports(self):
        from sym_modeling.domains.fem import (
            HyperelasticGenerationConfig,
            generate_hyperelastic_data,
            generate_hyperelastic_suite,
            HyperelasticSuiteCase,
            validate_hyperelastic_data_export,
        )
        from sym_modeling.domains.fem.data import FEMCaseData, FemDataset
        from sym_modeling.domains.fem.methods.euclid import EuclidConfig, EuclidWorkflow

        self.assertIs(FemDataset, FEMCaseData)
        self.assertEqual(generate_hyperelastic_data.__name__, "generate_hyperelastic_data")
        self.assertEqual(generate_hyperelastic_suite.__name__, "generate_hyperelastic_suite")
        self.assertEqual(
            validate_hyperelastic_data_export.__name__,
            "validate_hyperelastic_data_export",
        )
        self.assertEqual(HyperelasticGenerationConfig().load_steps[0], 10)
        self.assertEqual(HyperelasticSuiteCase.__name__, "HyperelasticSuiteCase")

        config = EuclidConfig(femDataPathOverride="generated/fem/demo")
        workflow = EuclidWorkflow(config=config)

        self.assertEqual(workflow.config.caseName, config.caseName)
        self.assertTrue(config.femDataPath.endswith("generated/fem/demo"))


if __name__ == "__main__":
    unittest.main()
