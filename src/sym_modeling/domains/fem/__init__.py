from __future__ import annotations

from importlib import import_module


_SYMBOL_MODULES = {
    "FEMCaseData": (".data", "FEMCaseData"),
    "FemDataset": (".data", "FemDataset"),
    "ReactionForce": (".data", "ReactionForce"),
    "DolfinxHyperelasticityConfig": (".dolfinx", "DolfinxHyperelasticityConfig"),
    "DolfinxHyperelasticitySimulation": (".dolfinx", "DolfinxHyperelasticitySimulation"),
    "DolfinxSimulationData": (".dolfinx", "DolfinxSimulationData"),
    "DolfinxSolveResult": (".dolfinx", "DolfinxSolveResult"),
    "BENCHMARK_BOUNDARY_TAGS": (".forward_benchmark", "BENCHMARK_BOUNDARY_TAGS"),
    "BENCHMARK_CELL_TAG": (".forward_benchmark", "BENCHMARK_CELL_TAG"),
    "ForwardFEMBenchmarkConfig": (".forward_benchmark", "ForwardFEMBenchmarkConfig"),
    "SUPPORTED_HYPERELASTIC_MODELS": (".dolfinx", "SUPPORTED_HYPERELASTIC_MODELS"),
    "SUPPORTED_FORWARD_BENCHMARK_MODELS": (
        ".forward_benchmark",
        "SUPPORTED_FORWARD_BENCHMARK_MODELS",
    ),
    "plot_forward_benchmark_loadsteps": (
        ".forward_benchmark",
        "plot_forward_benchmark_loadsteps",
    ),
    "plot_simulation_data": (".dolfinx", "plot_simulation_data"),
    "read_msh_physical_names": (".dolfinx", "read_msh_physical_names"),
    "HyperelasticGenerationConfig": (".io", "HyperelasticGenerationConfig"),
    "HyperelasticExportValidationResult": (".io", "HyperelasticExportValidationResult"),
    "HyperelasticGenerationResult": (".io", "HyperelasticGenerationResult"),
    "HyperelasticSuiteCase": (".io", "HyperelasticSuiteCase"),
    "HyperelasticSuiteResult": (".io", "HyperelasticSuiteResult"),
    "computeCauchyGreenStrain": (".operators.kinematics", "computeCauchyGreenStrain"),
    "computeJacobian": (".operators.kinematics", "computeJacobian"),
    "computeStrainInvariantDerivatives": (
        ".operators.kinematics",
        "computeStrainInvariantDerivatives",
    ),
    "computeStrainInvariants": (".operators.kinematics", "computeStrainInvariants"),
    "generate_hyperelastic_data": (".io", "generate_hyperelastic_data"),
    "generate_hyperelastic_suite": (".io", "generate_hyperelastic_suite"),
    "loadFemData": (".io", "loadFemData"),
    "load_fem_dataset": (".io", "load_fem_dataset"),
    "run_forward_hyperelastic_benchmark": (
        ".forward_benchmark",
        "run_forward_hyperelastic_benchmark",
    ),
    "setup_hyperelasticity_from_msh": (".dolfinx", "setup_hyperelasticity_from_msh"),
    "setup_hyperelasticity_simulation": (".dolfinx", "setup_hyperelasticity_simulation"),
    "validate_hyperelastic_data_export": (".io", "validate_hyperelastic_data_export"),
}

__all__ = list(_SYMBOL_MODULES)


def __getattr__(name: str):
    if name not in _SYMBOL_MODULES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attribute_name = _SYMBOL_MODULES[name]
    module = import_module(module_name, __name__)
    value = getattr(module, attribute_name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(list(globals().keys()) + __all__)
