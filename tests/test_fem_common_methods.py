from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from sym_modeling.domains.fem.io.csv_loader import loadFemData  # noqa: E402
from sym_modeling.domains.fem.io.hyperelastic import (  # noqa: E402
    _compute_triangle_gradients,
    _write_case_csvs,
)
from sym_modeling.domains.fem.methods.common.regression import (  # noqa: E402
    aicc,
    fit_sparse_regression,
)
from sym_modeling.domains.fem.methods.common.stress_data import (  # noqa: E402
    invariant_variables,
    synthetic_neo_hookean_dataset,
)
from sym_modeling.domains.fem.methods.common.weak_form import (  # noqa: E402
    assemble_feature_derivative_wrt_F,
    compute_first_piola,
    extend_lhs_rhs,
)
from sym_modeling.domains.fem.methods.euclid.feature_library import getNumberOfFeatures  # noqa: E402
from sym_modeling.domains.fem.methods.euclid.weak_form import (  # noqa: E402
    computeFirstPiolaTheta,
    extendLHSRHS,
)


def _write_single_triangle_case(root: Path) -> None:
    x_nodes = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=float)
    target_F = np.array([[1.08, 0.12], [0.04, 0.97]], dtype=float)
    u_nodes = ((target_F - np.eye(2, dtype=float)) @ x_nodes.T).T
    connectivity = np.array([[0, 1, 2]], dtype=int)
    grad_na = np.zeros((1, 3, 2), dtype=float)
    grad_na[0], area = _compute_triangle_gradients(x_nodes)
    piola = np.array([[0.3, -0.1, 0.05, 0.2]], dtype=float)
    reaction_forces = np.zeros(6, dtype=float)
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


class FEMCommonMethodTests(unittest.TestCase):
    def test_sparse_regression_recovers_known_coefficients(self):
        X = np.array(
            [
                [1.0, -1.0, 0.0],
                [2.0, 0.5, 0.0],
                [3.0, 2.0, 0.0],
                [4.0, 3.5, 0.0],
            ],
            dtype=float,
        )
        y = 2.0 * X[:, 0] - 3.0 * X[:, 1]

        fit = fit_sparse_regression(X, y, method="stlsq", threshold=1e-8)

        self.assertLess(fit.metrics.rmse, 1e-10)
        self.assertTrue(np.allclose(fit.theta, [2.0, -3.0, 0.0], atol=1e-8))
        self.assertTrue(np.array_equal(fit.active_mask, [True, True, False]))
        self.assertTrue(np.isinf(aicc(rss=1.0, num_samples=3, num_parameters=2)))

    def test_invariant_variables_reuse_fem_kinematics_shape(self):
        dataset = synthetic_neo_hookean_dataset(num_samples=5, seed=1)
        variables = invariant_variables(dataset, ("K1", "K2", "Jm1", "logI13", "logI23"))

        self.assertEqual(set(variables), {"K1", "K2", "Jm1", "logI13", "logI23"})
        self.assertEqual(variables["K1"].shape, (5,))
        self.assertEqual(dataset.target_vector.shape, (20,))
        self.assertTrue(np.all(np.isfinite(variables["logI13"])))
        self.assertTrue(np.all(np.isfinite(variables["logI23"])))

    def test_shared_weak_form_matches_euclid_fixed_feature_path(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir) / "dataset"
            _write_single_triangle_case(root)
            data = loadFemData(str(root / "10"), AD=True, noiseLevel=0.0)
            data.convertToNumpy()
            theta = np.linspace(0.01, 0.02, getNumberOfFeatures())
            config = SimpleNamespace(dim=2, numNodesPerElement=3, balance=100.0)

            manual_dQdF = (
                np.outer(data.featureSet.d_features_dI1[0, :], data.dI1dF[0, :])
                + np.outer(data.featureSet.d_features_dI2[0, :], data.dI2dF[0, :])
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


if __name__ == "__main__":
    unittest.main()
