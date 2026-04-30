from __future__ import annotations

import json
import shutil
import subprocess
from csv import DictWriter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from sym_modeling.common.logging import setup_logger
from sym_modeling.common.regression import SparseRegressionModel
from sym_modeling.common.workflow import BaseTrainer
from sym_modeling.domains.cfd.data import FlowData
from sym_modeling.domains.cfd.io.kth import KTHLoader
from sym_modeling.domains.cfd.io.openfoam import (
    run_sparta_feature_postprocess,
    write_flow_data_to_openfoam,
)
from sym_modeling.domains.cfd.io.openfoam_loader import FOAMLoader
from sym_modeling.domains.cfd.methods.sparta.config import SpartaConfig, load_sparta_config
from sym_modeling.domains.cfd.methods.sparta.library import (
    BDeltaCandidateLibrary,
    RCandidateLibrary,
)
from sym_modeling.domains.cfd.methods.sparta.plotting import (
    extract_field_components,
    plot_2d_baseline_sparta_difference,
    plot_2d_field_triplet,
)
from sym_modeling.domains.cfd.preprocessing import Preprocessor
from sym_modeling.domains.cfd.transfer import interpolate_fields_between_flows


@dataclass(frozen=True)
class SpartaRunContext:
    """Immutable run-level context resolved during initialization."""

    run_id: str
    config_path: str
    config: SpartaConfig
    run_dir: Path
    artifacts_dir: Path
    plots_dir: Path
    logs_dir: Path
    cases_dir: Path
    output_case_path: Path


@dataclass
class SpartaTrainingArtifacts:
    """Outputs produced by the training stage."""

    status: str = "pending"
    b_expression: Optional[str] = None
    r_expression: Optional[str] = None
    training_flow_path: Optional[Path] = None
    dns_flow_path: Optional[Path] = None
    baseline_flow_path: Optional[Path] = None
    frozen_flow_path: Optional[Path] = None
    theta_b_path: Optional[Path] = None
    theta_r_path: Optional[Path] = None
    b_features_path: Optional[Path] = None
    r_features_path: Optional[Path] = None
    train_metrics_path: Optional[Path] = None
    model_metadata_path: Optional[Path] = None
    notes: List[str] = field(default_factory=list)


@dataclass
class SpartaPredictionArtifacts:
    """Outputs produced by the prediction stage."""

    status: str = "pending"
    case_path: Optional[Path] = None
    log_path: Optional[Path] = None
    prediction_flow_path: Optional[Path] = None
    baseline_case_path: Optional[Path] = None
    baseline_log_path: Optional[Path] = None
    baseline_flow_path: Optional[Path] = None
    model_expression_path: Optional[Path] = None
    residual_expression_path: Optional[Path] = None
    notes: List[str] = field(default_factory=list)


@dataclass
class SpartaEvaluationArtifacts:
    """Outputs produced by the evaluation stage."""

    status: str = "pending"
    metrics_path: Optional[Path] = None
    metrics_csv_path: Optional[Path] = None
    plots_dir: Optional[Path] = None
    notes: List[str] = field(default_factory=list)


class SpartaTrainer(BaseTrainer):
    """
    End-to-end SpaRTA workflow orchestrator.

    Design goal:
    Keep this class as the stage orchestrator and delegate heavy numerical / CFD
    logic into reusable functions from `data`, `features`, and `symbolic_models`.
    """

    def __init__(self, config_path: str = "configs/sparta_config.json") -> None:
        self.config_path = config_path
        self.config = load_sparta_config(config_path=config_path)
        bootstrap_log_dir = Path(self.config.run.output_root) / "bootstrap_logs"
        self.logger = setup_logger(name="SpaRTA", log_dir=str(bootstrap_log_dir))

        self.context: Optional[SpartaRunContext] = None
        self.train_artifacts: Optional[SpartaTrainingArtifacts] = None
        self.prediction_artifacts: Optional[SpartaPredictionArtifacts] = None
        self.evaluation_artifacts: Optional[SpartaEvaluationArtifacts] = None

    def initialize(self) -> SpartaRunContext:
        """
        Initialize run directories and persist resolved configuration snapshot.
        """
        run_root = Path(self.config.run.output_root)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = (
            f"{self.config.run.run_name}_{timestamp}"
            if self.config.run.use_timestamp_subdir
            else self.config.run.run_name
        )

        run_dir = run_root / run_id
        artifacts_dir = run_dir / self.config.run.artifacts_subdir
        plots_dir = run_dir / self.config.run.plots_subdir
        logs_dir = run_dir / self.config.run.logs_subdir
        cases_dir = run_dir / self.config.run.cases_subdir

        for path in (run_dir, artifacts_dir, plots_dir, logs_dir, cases_dir):
            path.mkdir(parents=True, exist_ok=True)

        output_case_path = (
            Path(self.config.prediction.output_case_path)
            if self.config.prediction.output_case_path
            else cases_dir / f"{self.config.run.run_name}_kOmegaSSTA"
        )

        self.context = SpartaRunContext(
            run_id=run_id,
            config_path=self.config_path,
            config=self.config,
            run_dir=run_dir,
            artifacts_dir=artifacts_dir,
            plots_dir=plots_dir,
            logs_dir=logs_dir,
            cases_dir=cases_dir,
            output_case_path=output_case_path,
        )

        resolved_config_path = artifacts_dir / "resolved_sparta_config.json"
        with resolved_config_path.open("w", encoding="utf-8") as handle:
            json.dump(self.config.to_dict(), handle, indent=2)

        summary_path = artifacts_dir / "workflow_stage_summary.json"
        with summary_path.open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "run_id": run_id,
                    "initialize": "done",
                    "train": "pending",
                    "predict": "pending",
                    "evaluate": "pending",
                },
                handle,
                indent=2,
            )

        self.logger.info("Initialized SpaRTA run: %s", run_id)
        self.logger.info("Run directory: %s", run_dir)
        self.logger.info("Resolved config snapshot: %s", resolved_config_path)
        return self.context

    def train(self) -> SpartaTrainingArtifacts:
        """
        Stage 1: train symbolic models for bΔ and residual R.
        """
        context = self._ensure_initialized()
        self.logger.info("Stage train started: run_id=%s", context.run_id)
        train_dir = context.artifacts_dir / "train"
        train_dir.mkdir(parents=True, exist_ok=True)

        try:
            dns_flow = self._load_dns_flow()
            baseline_flow = self._load_baseline_flow()
            frozen_flow = self._load_frozen_flow()

            dns_flow_path = train_dir / "dns_flow.pkl"
            baseline_flow_path = train_dir / "baseline_flow.pkl"
            frozen_flow_path = train_dir / "frozen_flow.pkl"
            dns_flow.save(str(dns_flow_path))
            baseline_flow.save(str(baseline_flow_path))
            frozen_flow.save(str(frozen_flow_path))

            training_flow = self._build_training_flow(
                dns_flow=dns_flow,
                frozen_flow=frozen_flow,
                train_dir=train_dir,
            )
            training_flow_path = train_dir / "training_flow.pkl"
            training_flow.save(str(training_flow_path))

            b_library = BDeltaCandidateLibrary(
                max_degree=self.config.training.library_max_degree
            )
            r_library = RCandidateLibrary(
                max_degree=self.config.training.library_max_degree
            )

            theta_b = b_library.fit_transform(
                flow_data=training_flow,
                threshold=self.config.training.candidate_threshold,
            )
            theta_r = r_library.fit_transform(
                flow_data=training_flow,
                threshold=self.config.training.candidate_threshold,
            )

            y_b = np.asarray(training_flow.Rij, dtype=float).reshape(-1)
            y_r = np.asarray(training_flow.residual, dtype=float).reshape(-1)

            model_b = SparseRegressionModel(
                model_type=self.config.training.sparse_model_type,
                **self.config.training.b_model_kwargs,
            ).fit(X=theta_b, y=y_b)
            model_r = SparseRegressionModel(
                model_type=self.config.training.sparse_model_type,
                **self.config.training.r_model_kwargs,
            ).fit(X=theta_r, y=y_r)

            y_b_pred = model_b.predict(theta_b)
            y_r_pred = model_r.predict(theta_r)

            b_expression = model_b.get_expression(features=b_library.get_features())
            r_expression = model_r.get_expression(features=r_library.get_features())

            if not b_expression.strip():
                b_expression = "0.0*T1"
                self.logger.warning(
                    "bΔ model returned an empty expression; using fallback '%s'",
                    b_expression,
                )
            if not r_expression.strip():
                r_expression = "0.0*T1 : (T_ij dU_i/dx_j)"
                self.logger.warning(
                    "R model returned an empty expression; using fallback '%s'",
                    r_expression,
                )

            theta_b_path = train_dir / "theta_b.npy"
            theta_r_path = train_dir / "theta_r.npy"
            np.save(theta_b_path, theta_b)
            np.save(theta_r_path, theta_r)

            b_features_path = train_dir / "b_feature_names.json"
            r_features_path = train_dir / "r_feature_names.json"
            with b_features_path.open("w", encoding="utf-8") as handle:
                json.dump(b_library.get_features(), handle, indent=2)
            with r_features_path.open("w", encoding="utf-8") as handle:
                json.dump(r_library.get_features(), handle, indent=2)

            b_expression_path = train_dir / "bModelExpression.txt"
            r_expression_path = train_dir / "RModelExpression.txt"
            b_expression_path.write_text(b_expression, encoding="utf-8")
            r_expression_path.write_text(r_expression, encoding="utf-8")

            train_metrics = {
                "b_delta": self._regression_metrics(y_true=y_b, y_pred=y_b_pred),
                "residual": self._regression_metrics(y_true=y_r, y_pred=y_r_pred),
            }
            train_metrics_path = train_dir / "train_metrics.json"
            with train_metrics_path.open("w", encoding="utf-8") as handle:
                json.dump(train_metrics, handle, indent=2)

            model_metadata = {
                "config": self.config.to_dict(),
                "shapes": {
                    "training_points": int(training_flow.coords.shape[0]),
                    "theta_b": list(theta_b.shape),
                    "theta_r": list(theta_r.shape),
                },
                "models": {
                    "b_delta": {
                        "model_type": self.config.training.sparse_model_type,
                        "kwargs": self.config.training.b_model_kwargs,
                        "intercept": float(model_b.model.intercept_),
                        "n_features": int(len(b_library.get_features())),
                        "n_nonzero": int(np.count_nonzero(model_b.model.coef_)),
                        "coefficients": np.asarray(model_b.model.coef_, dtype=float).tolist(),
                        "expression_path": str(b_expression_path),
                    },
                    "residual": {
                        "model_type": self.config.training.sparse_model_type,
                        "kwargs": self.config.training.r_model_kwargs,
                        "intercept": float(model_r.model.intercept_),
                        "n_features": int(len(r_library.get_features())),
                        "n_nonzero": int(np.count_nonzero(model_r.model.coef_)),
                        "coefficients": np.asarray(model_r.model.coef_, dtype=float).tolist(),
                        "expression_path": str(r_expression_path),
                    },
                },
            }
            model_metadata_path = train_dir / "model_metadata.json"
            with model_metadata_path.open("w", encoding="utf-8") as handle:
                json.dump(model_metadata, handle, indent=2)

            artifacts = SpartaTrainingArtifacts(
                status="done",
                b_expression=b_expression,
                r_expression=r_expression,
                training_flow_path=training_flow_path,
                dns_flow_path=dns_flow_path,
                baseline_flow_path=baseline_flow_path,
                frozen_flow_path=frozen_flow_path,
                theta_b_path=theta_b_path,
                theta_r_path=theta_r_path,
                b_features_path=b_features_path,
                r_features_path=r_features_path,
                train_metrics_path=train_metrics_path,
                model_metadata_path=model_metadata_path,
                notes=[
                    "Phase 1 complete: DNS/baseline/frozen data loaded.",
                    "Phase 1 complete: merged training FlowData assembled.",
                    "Phase 1 complete: bΔ and R sparse models fitted and serialized.",
                ],
            )
        except Exception:
            self._update_stage_summary(stage="train", status="failed")
            self.logger.exception("Stage train failed")
            raise

        self.train_artifacts = artifacts
        self._update_stage_summary(stage="train", status=artifacts.status)
        self.logger.info("Stage train finished: artifacts in %s", train_dir)
        return artifacts

    def predict(self) -> SpartaPredictionArtifacts:
        """
        Stage 2: write model expressions to a copied baseline case, run CFD, and
        load prediction fields.
        """
        context = self._ensure_initialized()
        self.logger.info("Stage predict started: run_id=%s", context.run_id)
        predict_dir = context.artifacts_dir / "predict"
        predict_dir.mkdir(parents=True, exist_ok=True)

        try:
            if self.train_artifacts is None:
                self.logger.info("Prediction requires training artifacts; running train() first.")
                self.train()

            if self.train_artifacts is None:
                raise RuntimeError("Training artifacts are unavailable after train().")

            b_expression = (self.train_artifacts.b_expression or "").strip()
            r_expression = (self.train_artifacts.r_expression or "").strip()

            if not b_expression:
                raise ValueError("Cannot run prediction: b_expression is empty.")
            if not r_expression:
                raise ValueError("Cannot run prediction: r_expression is empty.")

            source_case = Path(self.config.data.baseline_case_path)
            output_case = context.output_case_path

            baseline_case_path: Optional[Path] = None
            baseline_log_path: Optional[Path] = None
            baseline_flow_path: Optional[Path] = None

            if self.config.prediction.run_baseline_reference:
                baseline_case_name = (
                    f"{self.config.run.run_name}_"
                    f"{self.config.prediction.baseline_turbulence_model}_reference"
                )
                baseline_case_path = context.cases_dir / baseline_case_name
                if baseline_case_path.resolve() == output_case.resolve():
                    baseline_case_path = context.cases_dir / f"{baseline_case_name}_case"

                self._prepare_prediction_case(
                    source_case=source_case,
                    output_case=baseline_case_path,
                )
                self._configure_baseline_reference_case(case_path=baseline_case_path)

                baseline_log_path = context.logs_dir / "simpleFoam_baseline_reference.log"
                baseline_flow_path = predict_dir / "baseline_reference_flow.pkl"
                self._run_case_and_save_flow(
                    case_path=baseline_case_path,
                    log_path=baseline_log_path,
                    artifact_path=baseline_flow_path,
                    skip_message=(
                        "Baseline reference solver execution skipped: "
                        "prediction.enabled=false\n"
                    ),
                )

            self._prepare_prediction_case(
                source_case=source_case,
                output_case=output_case,
            )

            model_expression_path, residual_expression_path = self._write_prediction_expressions(
                case_path=output_case,
                b_expression=b_expression,
                r_expression=r_expression,
            )
            self._configure_prediction_case(case_path=output_case)

            log_path = context.logs_dir / "simpleFoam_predict.log"
            prediction_flow_path = predict_dir / "prediction_flow.pkl"
            self._run_case_and_save_flow(
                case_path=output_case,
                log_path=log_path,
                artifact_path=prediction_flow_path,
                skip_message="Prediction solver execution skipped: prediction.enabled=false\n",
            )

            artifacts = SpartaPredictionArtifacts(
                status="done",
                case_path=output_case,
                log_path=log_path,
                prediction_flow_path=prediction_flow_path,
                baseline_case_path=baseline_case_path,
                baseline_log_path=baseline_log_path,
                baseline_flow_path=baseline_flow_path,
                model_expression_path=model_expression_path,
                residual_expression_path=residual_expression_path,
                notes=[
                    "Phase 2 complete: baseline case copied to run output case.",
                    (
                        "Phase 2 complete: baseline reference case rerun with "
                        f"{self.config.prediction.baseline_turbulence_model}."
                        if baseline_flow_path is not None
                        else "Phase 2 note: baseline reference rerun disabled."
                    ),
                    "Phase 2 complete: kOmegaSSTA model expressions injected.",
                    "Phase 2 complete: solver run completed and result loaded.",
                ],
            )
        except Exception:
            self._update_stage_summary(stage="predict", status="failed")
            self.logger.exception("Stage predict failed")
            raise

        self.prediction_artifacts = artifacts
        self._update_stage_summary(stage="predict", status=artifacts.status)
        self.logger.info("Stage predict finished: case=%s", artifacts.case_path)
        return artifacts

    def evaluate(self) -> SpartaEvaluationArtifacts:
        """
        Stage 3: compute metrics/plots for DNS vs baseline vs SpaRTA.
        """
        context = self._ensure_initialized()
        self.logger.info("Stage evaluate started: run_id=%s", context.run_id)
        eval_dir = context.artifacts_dir / "evaluate"
        eval_dir.mkdir(parents=True, exist_ok=True)

        try:
            if self.train_artifacts is None:
                self.logger.info("Evaluation requires training artifacts; running train() first.")
                self.train()
            if self.prediction_artifacts is None:
                self.logger.info(
                    "Evaluation requires prediction artifacts; running predict() first."
                )
                self.predict()

            if self.train_artifacts is None or self.prediction_artifacts is None:
                raise RuntimeError("Evaluation prerequisites are unavailable.")

            dns_flow = self._load_flow_from_artifact(
                self.train_artifacts.dns_flow_path, stage="train", name="dns_flow.pkl"
            )
            (
                baseline_artifact_path,
                baseline_stage,
                baseline_name,
            ) = self._resolve_baseline_eval_artifact()

            baseline_flow = self._load_flow_from_artifact(
                baseline_artifact_path,
                stage=baseline_stage,
                name=baseline_name,
            )
            prediction_flow = self._load_flow_from_artifact(
                self.prediction_artifacts.prediction_flow_path,
                stage="predict",
                name="prediction_flow.pkl",
            )

            aligned_baseline = self._align_flow_to_reference(
                source_flow=baseline_flow,
                reference_flow=dns_flow,
            )
            aligned_prediction = self._align_flow_to_reference(
                source_flow=prediction_flow,
                reference_flow=dns_flow,
            )

            fields = self._select_evaluation_fields(
                dns_flow=dns_flow,
                baseline_flow=aligned_baseline,
                prediction_flow=aligned_prediction,
            )
            if not fields:
                raise ValueError(
                    "No common non-empty fields found for evaluation among DNS, baseline, and prediction flows."
                )

            rows: List[Dict[str, Any]] = []
            summary: Dict[str, Dict[str, Any]] = {}
            for field_name in fields:
                dns_flat = self._flatten_field_for_metrics(getattr(dns_flow, field_name))
                baseline_flat = self._flatten_field_for_metrics(
                    getattr(aligned_baseline, field_name)
                )
                prediction_flat = self._flatten_field_for_metrics(
                    getattr(aligned_prediction, field_name)
                )

                baseline_metrics = self._regression_metrics(
                    y_true=dns_flat, y_pred=baseline_flat
                )
                prediction_metrics = self._regression_metrics(
                    y_true=dns_flat, y_pred=prediction_flat
                )

                baseline_rmse = baseline_metrics["rmse"]
                prediction_rmse = prediction_metrics["rmse"]
                rmse_ratio = (
                    prediction_rmse / baseline_rmse if baseline_rmse > 0 else float("inf")
                )

                summary[field_name] = {
                    "baseline_vs_dns": baseline_metrics,
                    "sparta_vs_dns": prediction_metrics,
                    "rmse_ratio_sparta_over_baseline": float(rmse_ratio),
                }

                rows.append(
                    {
                        "field": field_name,
                        "comparison": "baseline_vs_dns",
                        **baseline_metrics,
                    }
                )
                rows.append(
                    {
                        "field": field_name,
                        "comparison": "sparta_vs_dns",
                        **prediction_metrics,
                    }
                )

            metrics_path = eval_dir / "evaluation_metrics.json"
            with metrics_path.open("w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "fields": fields,
                        "summary": summary,
                    },
                    handle,
                    indent=2,
                )

            metrics_csv_path = eval_dir / "evaluation_metrics.csv"
            with metrics_csv_path.open("w", encoding="utf-8", newline="") as handle:
                writer = DictWriter(
                    handle,
                    fieldnames=["field", "comparison", "mae", "rmse", "max_abs_error"],
                )
                writer.writeheader()
                writer.writerows(rows)

            plots_dir = context.plots_dir
            if self.config.evaluation.save_plots:
                self._save_evaluation_plots(
                    dns_flow=dns_flow,
                    baseline_flow=aligned_baseline,
                    prediction_flow=aligned_prediction,
                    fields=fields,
                    output_dir=plots_dir,
                )

            artifacts = SpartaEvaluationArtifacts(
                status="done",
                metrics_path=metrics_path,
                metrics_csv_path=metrics_csv_path,
                plots_dir=plots_dir,
                notes=[
                    "Phase 3 complete: baseline and prediction aligned to DNS reference coordinates.",
                    "Phase 3 complete: metrics exported to JSON and CSV.",
                    "Phase 3 complete: parity/profile plots generated.",
                ],
            )
        except Exception:
            self._update_stage_summary(stage="evaluate", status="failed")
            self.logger.exception("Stage evaluate failed")
            raise

        self.evaluation_artifacts = artifacts
        self._update_stage_summary(stage="evaluate", status=artifacts.status)
        self.logger.info("Stage evaluate finished: metrics=%s", artifacts.metrics_path)
        return artifacts

    def run(self) -> Dict[str, Any]:
        """
        Run initialize -> train -> predict -> evaluate.
        """
        context = self.initialize()
        train_artifacts = self.train()
        prediction_artifacts = self.predict()
        evaluation_artifacts = self.evaluate()

        return {
            "context": context,
            "train": train_artifacts,
            "predict": prediction_artifacts,
            "evaluate": evaluation_artifacts,
        }

    def _ensure_initialized(self) -> SpartaRunContext:
        if self.context is None:
            return self.initialize()
        return self.context

    def _load_dns_flow(self) -> FlowData:
        self.logger.info("Loading DNS flow from %s", self.config.data.dns_case_path)
        return KTHLoader(
            data_path=self.config.data.dns_case_path,
            flow_data=FlowData(),
        ).load()

    def _load_baseline_flow(self) -> FlowData:
        self._ensure_u_is_requested(
            fields=self.config.data.baseline_loader_fields,
            label="baseline_loader_fields",
        )
        self.logger.info(
            "Loading baseline flow from %s (time=%s)",
            self.config.data.baseline_case_path,
            self.config.data.baseline_time,
        )
        return FOAMLoader(
            data_path=self.config.data.baseline_case_path,
            flow_data=FlowData(),
            time=self.config.data.baseline_time,
            fields=self.config.data.baseline_loader_fields,
            sample_location=self.config.data.sample_location,
            streamwise_average=self.config.data.streamwise_average,
            streamwise_fields=None,
            streamwise_yz_atol=self.config.data.streamwise_yz_atol,
        ).load()

    def _load_frozen_flow(
        self,
        *,
        sample_location: Optional[str] = None,
        streamwise_average: Optional[bool] = None,
        fields: Optional[List[str]] = None,
        field_map: Optional[Dict[str, str]] = None,
    ) -> FlowData:
        effective_fields = (
            list(fields) if fields is not None else list(self.config.data.frozen_loader_fields)
        )
        self._ensure_u_is_requested(
            fields=effective_fields,
            label="frozen_loader_fields",
        )

        effective_field_map = (
            dict(field_map) if field_map is not None else dict(self.config.data.frozen_field_map)
        )
        effective_sample_location = (
            sample_location if sample_location is not None else self.config.data.sample_location
        )
        effective_streamwise_average = (
            streamwise_average
            if streamwise_average is not None
            else self.config.data.streamwise_average
        )

        self.logger.info(
            "Loading frozen flow from %s (time=%s, sample_location=%s, streamwise_average=%s)",
            self.config.data.frozen_case_path,
            self.config.data.frozen_time,
            effective_sample_location,
            effective_streamwise_average,
        )
        return FOAMLoader(
            data_path=self.config.data.frozen_case_path,
            flow_data=FlowData(),
            time=self.config.data.frozen_time,
            fields=effective_fields,
            field_map=effective_field_map,
            sample_location=effective_sample_location,
            streamwise_average=effective_streamwise_average,
            streamwise_fields=None,
            streamwise_yz_atol=self.config.data.streamwise_yz_atol,
        ).load()

    def _build_training_flow(
        self,
        dns_flow: FlowData,
        frozen_flow: FlowData,
        train_dir: Path,
    ) -> FlowData:
        required_dns_fields = ("U", "k")
        required_frozen_fields = ("omega", "Rij", "residual")

        for field_name in required_dns_fields:
            values = np.asarray(getattr(dns_flow, field_name))
            if values.size == 0:
                raise ValueError(f"DNS flow missing required field '{field_name}'")

        for field_name in required_frozen_fields:
            values = np.asarray(getattr(frozen_flow, field_name))
            if values.size == 0:
                raise ValueError(f"Frozen flow missing required field '{field_name}'")

        # 1) Build a cell-centred 3D grid from frozen RANS to make OpenFOAM
        #    feature post-processing use the exact same operators as runtime.
        frozen_feature_grid = self._load_frozen_flow(
            sample_location="cell",
            streamwise_average=False,
            fields=["U", "omega", "k", "residual", "Rij"],
            field_map=self.config.data.frozen_field_map,
        )

        mapped_3d = FlowData()
        mapped_3d.coords = np.asarray(frozen_feature_grid.coords, dtype=float).copy()
        mapped_3d.x_vec = np.asarray(frozen_feature_grid.x_vec, dtype=float).copy()
        mapped_3d.y_vec = np.asarray(frozen_feature_grid.y_vec, dtype=float).copy()
        mapped_3d.z_vec = np.asarray(frozen_feature_grid.z_vec, dtype=float).copy()
        mapped_3d.n_points = int(mapped_3d.coords.shape[0])
        mapped_3d.grid_shape = np.asarray(frozen_feature_grid.grid_shape, dtype=int).copy()
        mapped_3d.simulation_config = frozen_feature_grid.simulation_config

        mapped_3d = interpolate_fields_between_flows(
            source_flow=dns_flow,
            target_flow=mapped_3d,
            field_names=["U", "k"],
            source_dims=(1, 2),
            target_dims=(1, 2),
            method=self.config.data.interpolation_method,
            fallback_method="nearest",
        )
        mapped_3d.omega = np.asarray(frozen_feature_grid.omega, dtype=float).copy()

        # 2) Write mapped DNS U into a temporary OpenFOAM case and compute
        #    gradU/Sij/Wij/T1/T2/T3/I1/I2 with OpenFOAM post-process utility.
        feature_case = train_dir / "feature_postprocess_case"
        if feature_case.exists():
            shutil.rmtree(feature_case)
        shutil.copytree(self.config.data.frozen_case_path, feature_case)

        write_flow_data_to_openfoam(
            case_path=str(feature_case),
            flow_data=mapped_3d,
            time=self.config.data.frozen_time,
            fields=["U"],
            field_map={"U": "U"},
            atol=1e-10,
            create_if_missing=False,
        )

        self.logger.info(
            "Running OpenFOAM SpaRTA feature post-process on temporary case: %s",
            feature_case,
        )
        run_sparta_feature_postprocess(
            case_path=str(feature_case),
            time=self.config.data.frozen_time,
            omega_floor=1e-30,
            run_surface_sampling=False,
            check=True,
        )

        # 3) Load OpenFOAM-computed features with the same sampling mode used
        #    by the training dataset (typically 2D streamwise-averaged points).
        feature_fields = [
            "U",
            "gradU",
            "Sij",
            "Wij",
            "T1",
            "T2",
            "T3",
            "I1",
            "I2",
        ]
        feature_flow = FOAMLoader(
            data_path=str(feature_case),
            flow_data=FlowData(),
            time=self.config.data.frozen_time,
            fields=feature_fields,
            sample_location=self.config.data.sample_location,
            streamwise_average=self.config.data.streamwise_average,
            streamwise_fields=None,
            streamwise_yz_atol=self.config.data.streamwise_yz_atol,
        ).load()

        # 4) Assemble final training flow on frozen target coordinates.
        target = FlowData()
        target.coords = np.asarray(frozen_flow.coords, dtype=float).copy()
        target.x_vec = np.asarray(frozen_flow.x_vec, dtype=float).copy()
        target.y_vec = np.asarray(frozen_flow.y_vec, dtype=float).copy()
        target.z_vec = np.asarray(frozen_flow.z_vec, dtype=float).copy()
        target.n_points = int(target.coords.shape[0])
        target.grid_shape = np.asarray(frozen_flow.grid_shape, dtype=int).copy()
        target.simulation_config = frozen_flow.simulation_config

        merged = interpolate_fields_between_flows(
            source_flow=dns_flow,
            target_flow=target,
            field_names=["k"],
            source_dims=(1, 2),
            target_dims=(1, 2),
            method=self.config.data.interpolation_method,
            fallback_method="nearest",
        )

        merged = Preprocessor.map_fields_by_coords(
            source_data=feature_flow,
            target_data=merged,
            field_names=feature_fields,
            atol=self.config.data.streamwise_yz_atol,
        )
        merged.omega = np.asarray(frozen_flow.omega, dtype=float).copy()
        merged.Rij = np.asarray(frozen_flow.Rij, dtype=float).copy()
        merged.residual = np.asarray(frozen_flow.residual, dtype=float).copy()

        merged.is_loaded = True
        merged.is_preprocessed = True
        merged.validate()
        return merged

    @staticmethod
    def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        residual = np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)
        return {
            "mae": float(np.mean(np.abs(residual))),
            "rmse": float(np.sqrt(np.mean(residual**2))),
            "max_abs_error": float(np.max(np.abs(residual))),
        }

    @staticmethod
    def _ensure_u_is_requested(fields: List[str], label: str) -> None:
        if "U" not in fields:
            raise ValueError(
                f"'{label}' must include 'U' because FOAMLoader requires it for metadata and gradU."
            )

    def _prepare_prediction_case(self, source_case: Path, output_case: Path) -> None:
        if not source_case.is_dir():
            raise FileNotFoundError(f"Baseline case path does not exist: {source_case}")

        if source_case.resolve() == output_case.resolve():
            self.logger.info("Prediction output case is baseline case (in-place mode).")
            return

        if output_case.exists():
            self.logger.info("Prediction output case already exists: %s", output_case)
        else:
            output_case.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(source_case, output_case)
            self.logger.info("Copied baseline case: %s -> %s", source_case, output_case)

        self._sanitize_case_for_rerun(case_path=output_case)

    def _write_prediction_expressions(
        self,
        case_path: Path,
        b_expression: str,
        r_expression: str,
    ) -> tuple[Path, Path]:
        constant_dir = case_path / "constant"
        constant_dir.mkdir(parents=True, exist_ok=True)

        model_expression_path = constant_dir / self.config.prediction.model_expression_filename
        residual_expression_path = (
            constant_dir / self.config.prediction.residual_expression_filename
        )

        model_expression_path.write_text(self._quote_expression(b_expression), encoding="utf-8")
        residual_expression_path.write_text(
            self._quote_expression(r_expression), encoding="utf-8"
        )
        return model_expression_path, residual_expression_path

    def _configure_prediction_case(self, case_path: Path) -> None:
        case = self._open_foam_case(case_path=case_path)

        turb_file = case.file("constant/turbulenceProperties")
        turb_file["simulationType"] = "RAS"
        ras = turb_file["RAS"]
        ras["RASModel"] = self.config.prediction.turbulence_model
        ras["turbulence"] = "on"
        ras["printCoeffs"] = "on"
        ras["kOmegaSSTACoeffs"] = {
            "bModelExpression": (
                "#include",
                f'"{self.config.prediction.model_expression_filename}"',
            ),
            "RModelExpression": (
                "#include",
                f'"{self.config.prediction.residual_expression_filename}"',
            ),
        }

        control_file = case.file("system/controlDict")
        libs = list(control_file.get("libs", []))
        lib_name = self.config.prediction.turbulence_library
        normalized = {str(item).strip('"') for item in libs}
        if lib_name not in normalized:
            libs.append(f'"{lib_name}"')
        control_file["libs"] = libs

        # kOmegaSSTA solver path expects this divergence scheme entry.
        fv_schemes = case.file("system/fvSchemes")
        div_schemes = fv_schemes["divSchemes"]
        if "div(nonlinearStress)" not in div_schemes.keys():
            div_schemes["div(nonlinearStress)"] = ("Gauss", "linear")

    def _configure_baseline_reference_case(self, case_path: Path) -> None:
        case = self._open_foam_case(case_path=case_path)

        turb_file = case.file("constant/turbulenceProperties")
        turb_file["simulationType"] = "RAS"
        ras = turb_file["RAS"]
        ras["RASModel"] = self.config.prediction.baseline_turbulence_model
        ras["turbulence"] = "on"
        ras["printCoeffs"] = "on"

        try:
            if "kOmegaSSTACoeffs" in ras.keys():
                del ras["kOmegaSSTACoeffs"]
        except Exception:
            # Optional cleanup only; unknown keys are typically ignored.
            pass

        control_file = case.file("system/controlDict")
        libs = list(control_file.get("libs", []))
        lib_name = self.config.prediction.turbulence_library
        libs = [entry for entry in libs if str(entry).strip('"') != lib_name]
        control_file["libs"] = libs

    def _run_prediction_solver(self, case_path: Path, log_path: Path) -> None:
        command = [self.config.prediction.solver_executable, "-case", str(case_path)]
        self.logger.info("Running solver: %s", " ".join(command))

        timeout = self.config.prediction.solver_timeout_seconds
        with log_path.open("w", encoding="utf-8") as handle:
            try:
                result = subprocess.run(
                    command,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    check=False,
                    timeout=timeout if timeout > 0 else None,
                )
            except FileNotFoundError as exc:
                raise FileNotFoundError(
                    f"Solver executable not found: {self.config.prediction.solver_executable}"
                ) from exc
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError(
                    f"Prediction solver timed out after {timeout}s. Log: {log_path}"
                ) from exc

        if result.returncode != 0:
            raise RuntimeError(
                f"Prediction solver failed with return code {result.returncode}. "
                f"Log: {log_path}"
            )

    def _run_case_and_save_flow(
        self,
        case_path: Path,
        log_path: Path,
        artifact_path: Path,
        skip_message: str,
    ) -> None:
        if self.config.prediction.enabled:
            self._run_prediction_solver(case_path=case_path, log_path=log_path)
        else:
            log_path.write_text(skip_message, encoding="utf-8")

        flow = self._load_case_flow(case_path=case_path)
        flow.save(str(artifact_path))

    def _load_case_flow(self, case_path: Path) -> FlowData:
        requested = list(self.config.data.baseline_loader_fields)
        if "U" not in requested:
            requested.insert(0, "U")

        # Include common fields for downstream comparison if available.
        for optional_field in ("nut", "Rij", "residual"):
            if optional_field not in requested:
                requested.append(optional_field)

        prediction_loader = FOAMLoader(
            data_path=str(case_path),
            flow_data=FlowData(),
            time=self.config.data.baseline_time,
            fields=requested,
            sample_location=self.config.data.sample_location,
            streamwise_average=self.config.data.streamwise_average,
            streamwise_fields=None,
            streamwise_yz_atol=self.config.data.streamwise_yz_atol,
        )
        return prediction_loader.load()

    def _load_flow_from_artifact(self, path: Optional[Path], stage: str, name: str) -> FlowData:
        if path is None:
            raise FileNotFoundError(
                f"Missing artifact path for {name}. Run stage '{stage}' first."
            )
        if not path.exists():
            raise FileNotFoundError(
                f"Artifact file not found: {path}. Run stage '{stage}' first."
            )
        return FlowData().load(str(path))

    def _resolve_baseline_eval_artifact(self) -> tuple[Optional[Path], str, str]:
        if self.prediction_artifacts is None:
            return self.train_artifacts.baseline_flow_path, "train", "baseline_flow.pkl"

        if self.prediction_artifacts.baseline_flow_path is not None:
            return (
                self.prediction_artifacts.baseline_flow_path,
                "predict",
                "baseline_reference_flow.pkl",
            )

        return self.train_artifacts.baseline_flow_path, "train", "baseline_flow.pkl"

    def _sanitize_case_for_rerun(self, case_path: Path) -> None:
        removed: List[str] = []

        for child in case_path.iterdir():
            if child.is_dir():
                if child.name == "postProcessing":
                    shutil.rmtree(child)
                    removed.append(child.name)
                    continue

                if child.name.startswith("processor"):
                    shutil.rmtree(child)
                    removed.append(child.name)
                    continue

                if self._is_numeric_time_dir(child.name):
                    if abs(float(child.name)) > 1e-12:
                        shutil.rmtree(child)
                        removed.append(child.name)
                continue

            if child.name.startswith("log.") or child.name == "surface.csv":
                child.unlink(missing_ok=True)
                removed.append(child.name)

        if removed:
            self.logger.info(
                "Sanitized case for rerun (%s): removed %s",
                case_path,
                sorted(removed),
            )
        else:
            self.logger.info("Case already clean for rerun: %s", case_path)

    @staticmethod
    def _is_numeric_time_dir(name: str) -> bool:
        try:
            float(name)
            return True
        except ValueError:
            return False

    def _align_flow_to_reference(self, source_flow: FlowData, reference_flow: FlowData) -> FlowData:
        target = FlowData()
        target.coords = np.asarray(reference_flow.coords, dtype=float).copy()
        target.x_vec = np.asarray(reference_flow.x_vec, dtype=float).copy()
        target.y_vec = np.asarray(reference_flow.y_vec, dtype=float).copy()
        target.z_vec = np.asarray(reference_flow.z_vec, dtype=float).copy()
        target.n_points = int(target.coords.shape[0])
        target.grid_shape = np.asarray(reference_flow.grid_shape, dtype=int).copy()
        target.simulation_config = source_flow.simulation_config

        return interpolate_fields_between_flows(
            source_flow=source_flow,
            target_flow=target,
            field_names=self._flow_fields_available(source_flow),
            source_dims=(1, 2),
            target_dims=(1, 2),
            method=self.config.data.interpolation_method,
            fallback_method="nearest",
        )

    def _select_evaluation_fields(
        self,
        dns_flow: FlowData,
        baseline_flow: FlowData,
        prediction_flow: FlowData,
    ) -> List[str]:
        priority = ["U", "k", "omega", "Rij", "residual"]
        selected: List[str] = []
        for field_name in priority:
            dns_values = np.asarray(getattr(dns_flow, field_name))
            base_values = np.asarray(getattr(baseline_flow, field_name))
            pred_values = np.asarray(getattr(prediction_flow, field_name))
            if dns_values.size == 0 or base_values.size == 0 or pred_values.size == 0:
                continue
            if (
                dns_values.shape[0] != dns_flow.coords.shape[0]
                or base_values.shape[0] != baseline_flow.coords.shape[0]
                or pred_values.shape[0] != prediction_flow.coords.shape[0]
            ):
                continue
            selected.append(field_name)
        return selected

    @staticmethod
    def _flow_fields_available(flow_data: FlowData) -> List[str]:
        non_point_fields = {
            "simulation_config",
            "n_points",
            "grid_shape",
            "x_vec",
            "y_vec",
            "z_vec",
            "coords",
            "is_loaded",
            "is_preprocessed",
        }
        fields: List[str] = []
        for field_name in flow_data.get_field_names():
            if field_name in non_point_fields:
                continue
            values = np.asarray(getattr(flow_data, field_name))
            if values.size == 0 or values.ndim == 0:
                continue
            if values.shape[0] != flow_data.coords.shape[0]:
                continue
            fields.append(field_name)
        return fields

    @staticmethod
    def _flatten_field_for_metrics(values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        if values.ndim == 1:
            return values.reshape(-1)
        return values.reshape(values.shape[0], -1).reshape(-1)

    def _save_evaluation_plots(
        self,
        dns_flow: FlowData,
        baseline_flow: FlowData,
        prediction_flow: FlowData,
        fields: List[str],
        output_dir: Path,
    ) -> None:
        import matplotlib.pyplot as plt

        output_dir.mkdir(parents=True, exist_ok=True)
        yz_order = np.lexsort((dns_flow.coords[:, 2], dns_flow.coords[:, 1]))
        y = dns_flow.coords[:, 1]
        z = dns_flow.coords[:, 2]

        for field_name in fields:
            dns_components = dict(
                extract_field_components(getattr(dns_flow, field_name), field_name)
            )
            baseline_components = dict(
                extract_field_components(getattr(baseline_flow, field_name), field_name)
            )
            prediction_components = dict(
                extract_field_components(getattr(prediction_flow, field_name), field_name)
            )

            common_labels = [
                label
                for label in dns_components.keys()
                if label in baseline_components and label in prediction_components
            ]
            for label in common_labels:
                dns_vec = np.asarray(dns_components[label], dtype=float)
                base_vec = np.asarray(baseline_components[label], dtype=float)
                pred_vec = np.asarray(prediction_components[label], dtype=float)

                fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

                # Parity plot
                axes[0].scatter(dns_vec, base_vec, s=8, alpha=0.5, label="Baseline")
                axes[0].scatter(dns_vec, pred_vec, s=8, alpha=0.5, label="SpaRTA")
                vmin = float(min(np.min(dns_vec), np.min(base_vec), np.min(pred_vec)))
                vmax = float(max(np.max(dns_vec), np.max(base_vec), np.max(pred_vec)))
                axes[0].plot([vmin, vmax], [vmin, vmax], "k--", linewidth=1)
                axes[0].set_title(f"{label}: Parity")
                axes[0].set_xlabel("DNS")
                axes[0].set_ylabel("Prediction")
                axes[0].legend(loc="best")

                # Ordered profile-like trend
                axes[1].plot(dns_vec[yz_order], label="DNS", linewidth=1.3)
                axes[1].plot(base_vec[yz_order], label="Baseline", linewidth=1.1)
                axes[1].plot(pred_vec[yz_order], label="SpaRTA", linewidth=1.1)
                axes[1].set_title(f"{label}: Ordered YZ Trend")
                axes[1].set_xlabel("Point index (sorted by y,z)")
                axes[1].set_ylabel(label)
                axes[1].legend(loc="best")

                fig.tight_layout()
                fig.savefig(output_dir / f"{label}_comparison.png", dpi=160)
                plt.close(fig)

                plot_2d_field_triplet(
                    y=y,
                    z=z,
                    dns_val=dns_vec,
                    baseline_val=base_vec,
                    sparta_val=pred_vec,
                    field_label=label,
                    output_path=output_dir / f"{label}_2d_fields.png",
                )

                plot_2d_baseline_sparta_difference(
                    y=y,
                    z=z,
                    baseline_val=base_vec,
                    sparta_val=pred_vec,
                    field_label=label,
                    output_path=output_dir / f"{label}_baseline_sparta_difference_2d.png",
                )

    @staticmethod
    def _quote_expression(expression: str) -> str:
        escaped = expression.replace('"', '\\"')
        return f'"{escaped}"\n'

    @staticmethod
    def _open_foam_case(case_path: Path) -> Any:
        try:
            import foamlib  # type: ignore
        except Exception as exc:
            raise ImportError(
                "foamlib is required for prediction case dictionary edits."
            ) from exc

        return foamlib.FoamCase(str(case_path))

    def _update_stage_summary(self, stage: str, status: str) -> None:
        context = self._ensure_initialized()
        summary_path = context.artifacts_dir / "workflow_stage_summary.json"

        if summary_path.exists():
            with summary_path.open("r", encoding="utf-8") as handle:
                summary = json.load(handle)
        else:
            summary = {"run_id": context.run_id}

        summary[stage] = status
        with summary_path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
