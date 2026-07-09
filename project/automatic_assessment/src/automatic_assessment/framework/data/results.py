"""
Typed result objects for the experiment pipeline (artifact schema v3).

The pipeline fills these dataclasses directly; saving and reporting only
consume their helper methods. There is exactly one place where metric
names/structures are defined — here — so schema drift between pipeline,
saving, and visualization is impossible by construction.

Artifact schema v3 (written by reporting/saving.py):
- config.yaml   : nested run configuration incl. schema_version + stage
- metrics.yaml  : nested metrics (val/test x scaled/unscaled, per-target
                  blocks keyed by TARGET NAME, per-fold summaries)
- predictions.csv : TIDY long format — one row per (split, user, target)
                  with y_true / y_pred in scaled AND original units
- metrics_per_target.csv : tidy per-target metric table
- learning_curve.csv     : epoch, train_loss, val_loss (fold 0)
"""

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

ARTIFACT_SCHEMA_VERSION = 3


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------

@dataclass
class ExperimentConfig:
    """Run configuration (replaces the loose config dict)."""
    epochs: int = 30
    hyperparameter_mode: str = "default"   # 'default' | 'optimize'
    n_trials: int = 30
    early_stopping_patience: Optional[int] = 2
    seed: int = 42
    targets: List[str] = field(default_factory=list)
    augmentation_ratio: int = 0
    note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ----------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------

@dataclass
class MetricTable:
    """Per-target regression metrics for one (split, scale) combination."""
    targets: List[str]
    rmse: np.ndarray   # (n_targets,)
    mae: np.ndarray    # (n_targets,)
    r2: np.ndarray     # (n_targets,)

    @classmethod
    def from_predictions(cls, y_true: np.ndarray, y_pred: np.ndarray, targets: List[str]) -> "MetricTable":
        """y_true / y_pred: (n_samples, n_targets). Argument order matters for R^2!"""
        y_true = np.atleast_2d(np.asarray(y_true, dtype=np.float64))
        y_pred = np.atleast_2d(np.asarray(y_pred, dtype=np.float64))
        residuals = y_true - y_pred
        rmse = np.sqrt(np.mean(residuals ** 2, axis=0))
        mae = np.mean(np.abs(residuals), axis=0)
        r2 = np.array([r2_score(y_true[:, i], y_pred[:, i]) for i in range(y_true.shape[1])])
        return cls(targets=list(targets), rmse=rmse, mae=mae, r2=r2)

    # -- aggregates ----------------------------------------------------
    @property
    def rmse_mean(self) -> float:
        return float(np.mean(self.rmse))

    @property
    def mae_mean(self) -> float:
        return float(np.mean(self.mae))

    @property
    def r2_mean(self) -> float:
        return float(np.mean(self.r2))

    # -- accessors -----------------------------------------------------
    def for_target(self, target: str) -> Dict[str, float]:
        i = self.targets.index(target)
        return {"rmse": float(self.rmse[i]), "mae": float(self.mae[i]), "r2": float(self.r2[i])}

    def to_dict(self, baseline: Optional["MetricTable"] = None) -> Dict[str, Any]:
        """Nested dict for metrics.yaml (per-target keyed by target name)."""
        per_target = {}
        for i, name in enumerate(self.targets):
            entry = {
                "rmse": float(self.rmse[i]),
                "mae": float(self.mae[i]),
                "r2": float(self.r2[i]),
            }
            if baseline is not None:
                entry["baseline_rmse"] = float(baseline.rmse[i])
            per_target[name] = entry
        out = {
            "rmse_mean": self.rmse_mean,
            "mae_mean": self.mae_mean,
            "r2_mean": self.r2_mean,
            "per_target": per_target,
        }
        if baseline is not None:
            out["baseline_rmse_mean"] = baseline.rmse_mean
        return out


# ----------------------------------------------------------------------
# Per-fold validation details
# ----------------------------------------------------------------------

@dataclass
class FoldResult:
    """One LOGO-CV fold (= one held-out user)."""
    fold: int
    user_id: int
    val_loss: float
    best_epoch: Optional[int]
    history: Dict[str, List[float]]
    val_preds: np.ndarray               # (n_val, n_targets) scaled
    val_actuals: np.ndarray
    val_preds_unscaled: np.ndarray
    val_actuals_unscaled: np.ndarray
    baseline_preds: np.ndarray          # dummy (train-mean) predictions, scaled
    baseline_preds_unscaled: np.ndarray
    model_parameters: int = 0

    def summary(self) -> Dict[str, Any]:
        return {
            "fold": int(self.fold),
            "user_id": int(self.user_id),
            "val_loss": float(self.val_loss),
            "best_epoch": int(self.best_epoch) if self.best_epoch else None,
        }


# ----------------------------------------------------------------------
# Split-level results
# ----------------------------------------------------------------------

def _predictions_long_rows(split: str, fold, user_ids, targets,
                           preds, actuals, preds_u, actuals_u) -> List[Dict[str, Any]]:
    """Tidy rows: one per (sample, target)."""
    preds = np.atleast_2d(np.asarray(preds))
    actuals = np.atleast_2d(np.asarray(actuals))
    preds_u = np.atleast_2d(np.asarray(preds_u))
    actuals_u = np.atleast_2d(np.asarray(actuals_u))
    rows = []
    for i, uid in enumerate(user_ids):
        for t_idx, target in enumerate(targets):
            rows.append({
                "split": split,
                "fold": fold,
                "user_id": int(uid),
                "target": target,
                "y_true": float(actuals[i, t_idx]),
                "y_pred": float(preds[i, t_idx]),
                "y_true_unscaled": float(actuals_u[i, t_idx]),
                "y_pred_unscaled": float(preds_u[i, t_idx]),
            })
    return rows


@dataclass
class ValidationResult:
    """Aggregated LOGO-CV validation over all folds."""
    targets: List[str]
    loss: float
    baseline_loss: float
    metrics: MetricTable                 # scaled
    metrics_unscaled: MetricTable
    baseline_metrics: MetricTable        # scaled
    baseline_metrics_unscaled: MetricTable
    folds: List[FoldResult]

    @property
    def best_epochs(self) -> List[int]:
        return [f.best_epoch for f in self.folds if f.best_epoch]

    def fold_frame(self) -> pd.DataFrame:
        return pd.DataFrame([f.summary() for f in self.folds])

    def predictions_long(self) -> pd.DataFrame:
        rows = []
        for f in self.folds:
            rows.extend(_predictions_long_rows(
                "val", f.fold, [f.user_id], self.targets,
                f.val_preds, f.val_actuals, f.val_preds_unscaled, f.val_actuals_unscaled,
            ))
        return pd.DataFrame(rows)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "loss": float(self.loss),
            "baseline_loss": float(self.baseline_loss),
            "scaled": self.metrics.to_dict(baseline=self.baseline_metrics),
            "unscaled": self.metrics_unscaled.to_dict(baseline=self.baseline_metrics_unscaled),
        }


@dataclass
class TestResult:
    """Single evaluation on the hold-out test split."""
    targets: List[str]
    user_ids: List[int]
    loss: float
    baseline_loss: float
    metrics: MetricTable
    metrics_unscaled: MetricTable
    baseline_metrics: MetricTable
    baseline_metrics_unscaled: MetricTable
    preds: np.ndarray
    actuals: np.ndarray
    preds_unscaled: np.ndarray
    actuals_unscaled: np.ndarray
    baseline_preds_unscaled: np.ndarray
    model_parameters: int = 0
    duration: str = ""

    def predictions_long(self) -> pd.DataFrame:
        return pd.DataFrame(_predictions_long_rows(
            "test", -1, self.user_ids, self.targets,
            self.preds, self.actuals, self.preds_unscaled, self.actuals_unscaled,
        ))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "loss": float(self.loss),
            "baseline_loss": float(self.baseline_loss),
            "scaled": self.metrics.to_dict(baseline=self.baseline_metrics),
            "unscaled": self.metrics_unscaled.to_dict(baseline=self.baseline_metrics_unscaled),
        }


# ----------------------------------------------------------------------
# Whole experiment
# ----------------------------------------------------------------------

@dataclass
class ExperimentInfo:
    """Bookkeeping about the run itself."""
    start_time: str = ""
    duration: str = ""
    strategy: str = "logo_cv_validation"
    input_shapes: Dict[str, Any] = field(default_factory=dict)
    total_model_parameters: int = 0
    duration_final_test: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class ExperimentResult:
    """
    Everything one experiment produced. `test` is None for tuning-only
    runs; run_final_test returns a NEW ExperimentResult with `test` set
    (the tuning-only result object is never mutated).
    """
    model_name: str
    config: ExperimentConfig
    best_params: Dict[str, Any]
    validation: ValidationResult
    final_epochs: int
    info: ExperimentInfo
    test: Optional[TestResult] = None
    best_trial_val_loss: Optional[float] = None
    tuning_trials: Optional[pd.DataFrame] = None
    param_importances: Optional[Dict[str, float]] = None

    @property
    def stage(self) -> str:
        return "tuning+final_test" if self.test is not None else "tuning"

    @property
    def targets(self) -> List[str]:
        return list(self.config.targets)

    # -- helpers used by saving / reporting -----------------------------

    def predictions_long(self) -> pd.DataFrame:
        """Tidy predictions of all available splits (val first)."""
        frames = [self.validation.predictions_long()]
        if self.test is not None:
            frames.append(self.test.predictions_long())
        return pd.concat(frames, ignore_index=True)

    def metrics_nested(self) -> Dict[str, Any]:
        """Content of metrics.yaml."""
        out = {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "stage": self.stage,
            "val": self.validation.to_dict(),
            "folds": [f.summary() for f in self.validation.folds],
        }
        if self.best_trial_val_loss is not None:
            out["best_trial_val_loss"] = float(self.best_trial_val_loss)
        if self.test is not None:
            out["test"] = self.test.to_dict()
        return out

    def per_target_frame(self) -> pd.DataFrame:
        """Tidy per-target metric table (metrics_per_target.csv)."""
        rows = []
        blocks = [("val", "scaled", self.validation.metrics, self.validation.baseline_metrics),
                  ("val", "unscaled", self.validation.metrics_unscaled, self.validation.baseline_metrics_unscaled)]
        if self.test is not None:
            blocks += [("test", "scaled", self.test.metrics, self.test.baseline_metrics),
                       ("test", "unscaled", self.test.metrics_unscaled, self.test.baseline_metrics_unscaled)]
        for split, scale, table, baseline in blocks:
            for i, target in enumerate(table.targets):
                rows.append({
                    "split": split, "scale": scale, "target": target,
                    "rmse": float(table.rmse[i]), "mae": float(table.mae[i]),
                    "r2": float(table.r2[i]),
                    "baseline_rmse": float(baseline.rmse[i]),
                })
        return pd.DataFrame(rows)

    def learning_curve_frame(self) -> Optional[pd.DataFrame]:
        """Fold-0 learning curve (representative)."""
        if not self.validation.folds:
            return None
        history = self.validation.folds[0].history
        if not history:
            return None
        df = pd.DataFrame(history)
        if df.empty:
            return None
        df.insert(0, "epoch", range(1, len(df) + 1))
        return df

    def config_nested(self) -> Dict[str, Any]:
        """Content of config.yaml."""
        return {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "stage": self.stage,
            "model_name": self.model_name,
            "experiment": self.config.to_dict(),
            "best_params": self.best_params,
            "final_epochs": int(self.final_epochs),
            "best_epochs_per_fold": [int(e) for e in self.validation.best_epochs],
            "best_trial_val_loss": float(self.best_trial_val_loss) if self.best_trial_val_loss is not None else None,
            "info": self.info.to_dict(),
        }
