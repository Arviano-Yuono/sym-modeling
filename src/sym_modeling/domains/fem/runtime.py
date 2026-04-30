from sym_modeling.domains.fem.imports import *


def initCUDA(cudaID):
    """
    Setup the FEM runtime.

    The NumPy-based EUCLID port runs on CPU. The legacy CUDA switch is kept only
    for configuration compatibility.
    """
    print("\n-----------------------------------------------------")
    if cudaID >= 0:
        print("CUDA acceleration is not available in the NumPy FEM runtime. Falling back to CPU.")
    print("Setting device to: cpu")
    print("-----------------------------------------------------\n")
