from .config import EuclidConfig, FORWARD_MODEL_NAMES, normalize_euclid_model_name
from .constraints import checkEnergyRequirementsRigorous
from .feature_library import computeFeatureDerivatives, computeFeatures, getNumberOfFeatures
from .lp_solver import applyPenaltyLpIteration, saveResultsLp

__all__ = [
    "EuclidConfig",
    "EuclidWorkflow",
    "FORWARD_MODEL_NAMES",
    "applyPenaltyLpIteration",
    "checkEnergyRequirementsRigorous",
    "computeFeatureDerivatives",
    "computeFeatures",
    "getNumberOfFeatures",
    "normalize_euclid_model_name",
    "saveResultsLp",
]


def __getattr__(name):
    if name == "EuclidWorkflow":
        from .workflow import EuclidWorkflow

        return EuclidWorkflow
    raise AttributeError("module %r has no attribute %r" % (__name__, name))
