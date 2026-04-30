import os
import shutil
from contextlib import contextmanager
from distutils.dir_util import copy_tree

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from numpy import inf
from scipy import sparse

try:
    import seaborn as sns
except ModuleNotFoundError:  # pragma: no cover - depends on optional plotting extras
    sns = None

np.random.seed(0)

__all__ = [
    "contextmanager",
    "copy_tree",
    "inf",
    "np",
    "os",
    "pd",
    "plt",
    "scipy",
    "shutil",
    "sns",
    "sparse",
]
