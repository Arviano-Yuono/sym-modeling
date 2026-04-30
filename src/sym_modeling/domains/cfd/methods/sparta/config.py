"""
Configuration utilities.

This module provides:
1) A lightweight generic `Config` wrapper (backward compatible)
2) A typed SpaRTA configuration schema (`SpartaConfig`)
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, TypeVar

TDataclass = TypeVar("TDataclass")


def _resolve_config_path(config_path: str) -> Path:
    """
    Resolve config path from:
    1) absolute path
    2) current working directory
    3) repository root (derived from this module location)
    """
    candidate = Path(config_path)
    if candidate.is_absolute() and candidate.exists():
        return candidate

    cwd_candidate = Path.cwd() / candidate
    if cwd_candidate.exists():
        return cwd_candidate

    repo_root = Path(__file__).resolve().parents[3]
    repo_candidate = repo_root / candidate
    if repo_candidate.exists():
        return repo_candidate

    raise FileNotFoundError(f"Config file not found: {config_path}")


def _build_dataclass(dataclass_type: Type[TDataclass], payload: Dict[str, Any]) -> TDataclass:
    allowed = {f.name for f in fields(dataclass_type)}
    filtered = {k: v for k, v in payload.items() if k in allowed}
    return dataclass_type(**filtered)  # type: ignore[arg-type]


class Config:
    """Simple generic config wrapper around JSON dictionaries."""

    def __init__(self, config_path: str = "configs/config.json"):
        resolved = _resolve_config_path(config_path)
        with resolved.open("r", encoding="utf-8") as handle:
            self.config: Dict[str, Any] = json.load(handle)
        self.config_path = resolved

    def __str__(self) -> str:
        return str(self.config)

    def __repr__(self) -> str:
        return f"Config(path='{self.config_path}', keys={list(self.config.keys())})"

    def __getitem__(self, key: str) -> Any:
        return self.get_config(key)

    def get_config(self, key: str) -> Any:
        if key in self.config:
            return self.config[key]
        raise KeyError(f"Key '{key}' not found in config")


@dataclass(frozen=True)
class SpartaRunConfig:
    run_name: str = "sparta_ar_1_180"
    output_root: str = "output/sparta_runs"
    artifacts_subdir: str = "artifacts"
    plots_subdir: str = "plots"
    logs_subdir: str = "logs"
    cases_subdir: str = "cases"
    use_timestamp_subdir: bool = True


@dataclass(frozen=True)
class SpartaDataConfig:
    dns_case_path: str = "dataset/reference/AR_1_180"
    baseline_case_path: str = "dataset/baseline/AR_1_180"
    frozen_case_path: str = "dataset/test-baseline/frozenAR_1_180"
    baseline_time: str = "3000"
    frozen_time: str = "3000"
    sample_location: str = "point"
    streamwise_average: bool = True
    streamwise_yz_atol: float = 1e-10
    baseline_loader_fields: List[str] = field(default_factory=lambda: ["U", "k", "omega"])
    frozen_loader_fields: List[str] = field(
        default_factory=lambda: ["U", "omega", "residual", "Rij"]
    )
    frozen_field_map: Dict[str, str] = field(
        default_factory=lambda: {"residual": "R", "Rij": "bDelta"}
    )
    interpolation_method: str = "linear"


@dataclass(frozen=True)
class SpartaTrainingConfig:
    library_max_degree: int = 2
    candidate_threshold: float = 1e5
    sparse_model_type: str = "elasticnet"
    b_model_kwargs: Dict[str, Any] = field(default_factory=lambda: {"alpha": 1.0})
    r_model_kwargs: Dict[str, Any] = field(default_factory=lambda: {"alpha": 0.2})


@dataclass(frozen=True)
class SpartaPredictionConfig:
    enabled: bool = True
    run_baseline_reference: bool = True
    output_case_path: Optional[str] = None
    solver_executable: str = "simpleFoam"
    solver_timeout_seconds: int = 0
    turbulence_library: str = "libkOmegaSSTA.so"
    turbulence_model: str = "kOmegaSSTA"
    baseline_turbulence_model: str = "kOmegaSST"
    model_expression_filename: str = "modelExpression.H"
    residual_expression_filename: str = "RModelExpression.H"


@dataclass(frozen=True)
class SpartaEvaluationConfig:
    enabled: bool = True
    metrics: List[str] = field(default_factory=lambda: ["mae", "rmse"])
    save_plots: bool = True
    reference_source: str = "dns"


@dataclass(frozen=True)
class SpartaConfig:
    run: SpartaRunConfig = field(default_factory=SpartaRunConfig)
    data: SpartaDataConfig = field(default_factory=SpartaDataConfig)
    training: SpartaTrainingConfig = field(default_factory=SpartaTrainingConfig)
    prediction: SpartaPredictionConfig = field(default_factory=SpartaPredictionConfig)
    evaluation: SpartaEvaluationConfig = field(default_factory=SpartaEvaluationConfig)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "SpartaConfig":
        # Supports the new schema and tolerates older JSON files with extra keys.
        return cls(
            run=_build_dataclass(SpartaRunConfig, payload.get("run", {})),
            data=_build_dataclass(SpartaDataConfig, payload.get("data", {})),
            training=_build_dataclass(SpartaTrainingConfig, payload.get("training", {})),
            prediction=_build_dataclass(
                SpartaPredictionConfig, payload.get("prediction", {})
            ),
            evaluation=_build_dataclass(
                SpartaEvaluationConfig, payload.get("evaluation", {})
            ),
        )

    @classmethod
    def from_json(cls, config_path: str = "configs/sparta_config.json") -> "SpartaConfig":
        resolved = _resolve_config_path(config_path)
        with resolved.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        cfg = cls.from_dict(payload)
        cfg.validate()
        return cfg

    def validate(self) -> None:
        if self.data.sample_location not in {"cell", "point"}:
            raise ValueError(
                f"data.sample_location must be 'cell' or 'point', got '{self.data.sample_location}'"
            )
        if self.training.library_max_degree < 0:
            raise ValueError(
                f"training.library_max_degree must be >= 0, got {self.training.library_max_degree}"
            )
        if self.training.candidate_threshold <= 0:
            raise ValueError(
                "training.candidate_threshold must be > 0, "
                f"got {self.training.candidate_threshold}"
            )
        if self.data.streamwise_yz_atol <= 0:
            raise ValueError(
                f"data.streamwise_yz_atol must be > 0, got {self.data.streamwise_yz_atol}"
            )
        if self.prediction.solver_timeout_seconds < 0:
            raise ValueError(
                "prediction.solver_timeout_seconds must be >= 0, "
                f"got {self.prediction.solver_timeout_seconds}"
            )
        if not self.prediction.turbulence_model:
            raise ValueError("prediction.turbulence_model must not be empty")
        if not self.prediction.baseline_turbulence_model:
            raise ValueError("prediction.baseline_turbulence_model must not be empty")
        if not self.evaluation.metrics:
            raise ValueError("evaluation.metrics must not be empty")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def load_sparta_config(config_path: str = "configs/sparta_config.json") -> SpartaConfig:
    return SpartaConfig.from_json(config_path=config_path)
