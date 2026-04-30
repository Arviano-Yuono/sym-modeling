from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


FORWARD_MODEL_NAMES = ("NH2", "NH4", "IH", "HW", "GT")

_MODEL_ALIASES = {
    "NH2": "NH2",
    "NH4": "NH4",
    "IH": "IH",
    "HW": "HW",
    "GT": "GT",
    "NEOHOOKEANJ2": "NH2",
    "NEO_HOOKEAN_J2": "NH2",
    "NEOHOOKEANJ4": "NH4",
    "NEO_HOOKEAN_J4": "NH4",
}

_LEGACY_DATASET_MODEL_NAME = {
    "NH2": "NeoHookeanJ2",
    "NH4": "NeoHookeanJ4",
}


def _normalize_model_token(name: str) -> str:
    return str(name).replace("-", "").replace("_", "").replace(" ", "").upper()


def normalize_euclid_model_name(name: str) -> str:
    """
    Normalize model naming to forward-benchmark labels: NH2/NH4/IH/HW/GT.
    """
    if not isinstance(name, str) or not name.strip():
        raise ValueError("str_model must be a non-empty string.")

    raw = name.strip()
    direct = _MODEL_ALIASES.get(raw)
    if direct is not None:
        return direct

    normalized = _MODEL_ALIASES.get(_normalize_model_token(raw))
    if normalized is None:
        raise ValueError(
            "Unsupported str_model: %s. Supported labels: %s (legacy aliases are accepted)."
            % (name, FORWARD_MODEL_NAMES)
        )
    return normalized


@dataclass
class EuclidConfig:
    cuda: int = -1
    dim: int = 2
    numNodesPerElement: int = 3
    voigtMap: list[list[int]] = field(default_factory=lambda: [[0, 1], [2, 3]])

    str_model: str = "NH2"
    str_mesh: str = "plate_hole_1k"
    noiseLevelData: str = "0"
    noiseLevel: float = 1e-4

    balance: float = 100.0
    penaltyLp: float = 1e-4
    penaltyLp_init: float | None = None
    p: float = 1.0 / 4.0
    numIncrements: int = 5
    factorIncrements: float = 5.0
    numGuesses: int = 1
    numIterations: int = 200
    lowestCost: float = -1.0
    lowestCostGuessID: int = -1
    threshold_iter: float = 1e-6
    threshold: float = 1e-2

    resultsDir: str | None = None
    appendResults: bool = True
    dataGroup: str = "FEM"
    femDataPathOverride: str | None = None
    loadstepsOverride: list[int] | None = None

    def __post_init__(self) -> None:
        if self.penaltyLp_init is None:
            self.penaltyLp_init = self.penaltyLp
        self.str_model = normalize_euclid_model_name(self.str_model)

    @property
    def str_noise(self) -> str:
        if self.noiseLevelData == "0":
            return self.str_mesh
        if self.noiseLevelData == "1e-4" and self.str_mesh == "plate_hole_60k":
            return "denoised_60k_to_60k_noise=0.0001"
        if self.noiseLevelData == "1e-3" and self.str_mesh == "plate_hole_60k":
            return "denoised_60k_to_60k_noise=0.001"
        return self.str_mesh

    @property
    def femDataPath(self) -> str:
        if self.femDataPathOverride is not None:
            return str(Path(self.femDataPathOverride))
        repo_root = Path(__file__).resolve().parents[6]
        base = repo_root / "FEM_data"
        if not base.exists():
            sibling_base = repo_root.parent / "EUCLID-hyperelasticity" / "FEM_data"
            if sibling_base.exists():
                base = sibling_base
        canonical = base / self.str_noise / f"{self.str_mesh}_{self.str_model}"
        if canonical.exists():
            return str(canonical)

        legacy_name = _LEGACY_DATASET_MODEL_NAME.get(self.str_model)
        if legacy_name is not None:
            legacy = base / self.str_noise / f"{self.str_mesh}_{legacy_name}"
            if legacy.exists():
                return str(legacy)

        return str(canonical)

    @property
    def loadsteps(self) -> list[int]:
        if self.loadstepsOverride is not None:
            return list(self.loadstepsOverride)
        if self.str_model in {"NH2", "NH4"}:
            return [10, 20, 30, 40]
        return [10, 20, 30, 40, 50, 60, 70, 80]

    @property
    def saveResultsName(self) -> str:
        return self.str_model

    @property
    def caseName(self) -> str:
        return f"{self.str_mesh}_{self.str_model}"
