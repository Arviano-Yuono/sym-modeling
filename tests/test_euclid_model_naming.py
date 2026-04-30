from __future__ import annotations

import sys
import unittest
import importlib.util
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


CONFIG_PATH = SRC_DIR / "sym_modeling" / "domains" / "fem" / "methods" / "euclid" / "config.py"
CONFIG_SPEC = importlib.util.spec_from_file_location("euclid_config_test_module", CONFIG_PATH)
if CONFIG_SPEC is None or CONFIG_SPEC.loader is None:
    raise RuntimeError("Failed to load Euclid config module from %s" % CONFIG_PATH)
CONFIG_MODULE = importlib.util.module_from_spec(CONFIG_SPEC)
sys.modules[CONFIG_SPEC.name] = CONFIG_MODULE
CONFIG_SPEC.loader.exec_module(CONFIG_MODULE)

EuclidConfig = CONFIG_MODULE.EuclidConfig
FORWARD_MODEL_NAMES = CONFIG_MODULE.FORWARD_MODEL_NAMES
normalize_euclid_model_name = CONFIG_MODULE.normalize_euclid_model_name


class EuclidModelNamingTests(unittest.TestCase):
    def test_supported_forward_names(self):
        self.assertEqual(FORWARD_MODEL_NAMES, ("NH2", "NH4", "IH", "HW", "GT"))

    def test_normalize_legacy_aliases(self):
        self.assertEqual(normalize_euclid_model_name("NeoHookeanJ2"), "NH2")
        self.assertEqual(normalize_euclid_model_name("neo_hookean_j2"), "NH2")
        self.assertEqual(normalize_euclid_model_name("NeoHookeanJ4"), "NH4")

    def test_default_config_uses_forward_name(self):
        cfg = EuclidConfig()
        self.assertEqual(cfg.str_model, "NH2")
        self.assertEqual(cfg.loadsteps, [10, 20, 30, 40])

    def test_loadstep_defaults_follow_forward_model_family(self):
        self.assertEqual(EuclidConfig(str_model="NH4").loadsteps, [10, 20, 30, 40])
        self.assertEqual(EuclidConfig(str_model="IH").loadsteps, [10, 20, 30, 40, 50, 60, 70, 80])
        self.assertEqual(EuclidConfig(str_model="HW").loadsteps, [10, 20, 30, 40, 50, 60, 70, 80])
        self.assertEqual(EuclidConfig(str_model="GT").loadsteps, [10, 20, 30, 40, 50, 60, 70, 80])

    def test_invalid_name_raises(self):
        with self.assertRaises(ValueError):
            EuclidConfig(str_model="bad-model")


if __name__ == "__main__":
    unittest.main()
