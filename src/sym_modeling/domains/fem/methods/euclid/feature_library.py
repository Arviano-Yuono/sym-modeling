from sym_modeling.domains.fem.imports import *


CONSIDER_GENT_THOMAS = True
POLYNOMIAL_DEGREE = 7
VOLUMETRIC_DEGREE = 7


def _as_column(values):
    array = np.asarray(values, dtype=float)
    if array.ndim == 0:
        array = array.reshape(1, 1)
    elif array.ndim == 1:
        array = array.reshape(-1, 1)
    return array


def _compute_reduced_invariants(I1, I3):
    I1 = _as_column(I1)
    I3 = _as_column(I3)
    K1 = I1 * np.power(I3, -1.0 / 3.0) - 3.0
    K2 = (I1 + I3 - 1.0) * np.power(I3, -2.0 / 3.0) - 3.0
    J = np.sqrt(I3)
    dK1_dI1 = np.power(I3, -1.0 / 3.0)
    dK1_dI3 = (-1.0 / 3.0) * I1 * np.power(I3, -4.0 / 3.0)
    dK2_dI1 = np.power(I3, -2.0 / 3.0)
    dK2_dI3 = np.power(I3, -2.0 / 3.0) - (
        2.0 / 3.0
    ) * (I1 + I3 - 1.0) * np.power(I3, -5.0 / 3.0)
    dJ_dI3 = 0.5 * np.power(I3, -0.5)
    return K1, K2, J, dK1_dI1, dK1_dI3, dK2_dI1, dK2_dI3, dJ_dI3


def computeFeatures(I1, I2, I3):
    """
    Compute the features dependent on the right Cauchy-Green strain invariants.
    Note that the features only depend on I1 and I3.
    """
    K1, K2, J, _, _, _, _, _ = _compute_reduced_invariants(I1, I3)
    num_features = getNumberOfFeatures()
    features = np.zeros((K1.shape[0], num_features), dtype=float)
    idx = -1
    for p in range(1, POLYNOMIAL_DEGREE + 1):
        for q in range(p + 1):
            idx += 1
            features[:, idx : idx + 1] = np.power(K1, p - q) * np.power(K2, q)
    for m in range(1, VOLUMETRIC_DEGREE + 1):
        idx += 1
        features[:, idx : idx + 1] = np.power(J - 1.0, 2 * m)
    if CONSIDER_GENT_THOMAS:
        idx += 1
        features[:, idx : idx + 1] = np.log((K2 + 3.0) / 3.0)
    return features


def computeFeatureDerivatives(I1, I2, I3):
    """
    Compute feature derivatives with respect to the strain invariants.
    """
    K1, K2, J, dK1_dI1, dK1_dI3, dK2_dI1, dK2_dI3, dJ_dI3 = _compute_reduced_invariants(
        I1,
        I3,
    )
    num_features = getNumberOfFeatures()
    d_features_dI1 = np.zeros((K1.shape[0], num_features), dtype=float)
    d_features_dI2 = np.zeros((K1.shape[0], num_features), dtype=float)
    d_features_dI3 = np.zeros((K1.shape[0], num_features), dtype=float)

    idx = -1
    for p in range(1, POLYNOMIAL_DEGREE + 1):
        for q in range(p + 1):
            idx += 1
            a = p - q
            b = q
            term_dI1 = np.zeros_like(K1)
            term_dI3 = np.zeros_like(K1)
            if a > 0:
                term_dI1 += (
                    a
                    * np.power(K1, a - 1)
                    * np.power(K2, b)
                    * dK1_dI1
                )
                term_dI3 += (
                    a
                    * np.power(K1, a - 1)
                    * np.power(K2, b)
                    * dK1_dI3
                )
            if b > 0:
                term_dI1 += (
                    b
                    * np.power(K1, a)
                    * np.power(K2, b - 1)
                    * dK2_dI1
                )
                term_dI3 += (
                    b
                    * np.power(K1, a)
                    * np.power(K2, b - 1)
                    * dK2_dI3
                )
            d_features_dI1[:, idx : idx + 1] = term_dI1
            d_features_dI3[:, idx : idx + 1] = term_dI3

    for m in range(1, VOLUMETRIC_DEGREE + 1):
        idx += 1
        d_features_dI3[:, idx : idx + 1] = (
            2.0
            * m
            * np.power(J - 1.0, 2 * m - 1)
            * dJ_dI3
        )

    if CONSIDER_GENT_THOMAS:
        idx += 1
        d_features_dI1[:, idx : idx + 1] = dK2_dI1 / (K2 + 3.0)
        d_features_dI3[:, idx : idx + 1] = dK2_dI3 / (K2 + 3.0)

    return d_features_dI1, d_features_dI2, d_features_dI3


def getNumberOfFeatures():
    """
    Compute number of features.
    """
    num_features = 0
    for n in range(POLYNOMIAL_DEGREE):
        num_features += n + 2
    for _ in range(VOLUMETRIC_DEGREE):
        num_features += 1
    if CONSIDER_GENT_THOMAS:
        num_features += 1
    return num_features
