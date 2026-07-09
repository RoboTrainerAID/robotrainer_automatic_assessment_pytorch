import os
import inspect
import shutil
from datetime import datetime

import pandas as pd
import yaml

from automatic_assessment.framework.data.results import ExperimentResult


class SavingModule:
    """
    Persists an ExperimentResult (artifact schema v3).

    Files written per run folder:
    - config.yaml            : nested run config (schema_version, stage,
                               experiment, best_params, epochs, info)
    - metrics.yaml           : nested metrics — val/test x scaled/unscaled,
                               per-target blocks keyed by TARGET NAME,
                               per-fold summaries
    - predictions.csv        : TIDY long format — one row per
                               (split, user, target) with y_true/y_pred in
                               scaled AND original units
    - metrics_per_target.csv : tidy per-target metric table
    - learning_curve.csv     : fold-0 loss curve
    - tuning_trials.csv / param_importances.csv (optimize mode)
    - model_source.py        : snapshot of the model class source
    """

    DEFAULT_OUTPUT_ROOT = "/workspace/automatic_assessment/experiment_results"

    def __init__(self, model_name: str, output_root: str = None):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        root = output_root or self.DEFAULT_OUTPUT_ROOT
        self.output_dir = os.path.join(root, f"{timestamp}_{model_name}")
        os.makedirs(self.output_dir, exist_ok=True)

    def save_model_source(self, model_class):
        try:
            source_file = inspect.getsourcefile(model_class)
            if source_file:
                dst = os.path.join(self.output_dir, "model_source.py")
                shutil.copy(source_file, dst)
                print(f"Saved model source code to {dst}")
        except Exception as e:
            print(f"Could not save model source: {e}")

    # ------------------------------------------------------------------

    def save_results(self, result: ExperimentResult):
        # 1. config.yaml
        with open(os.path.join(self.output_dir, "config.yaml"), "w") as f:
            yaml.dump(result.config_nested(), f, sort_keys=False)

        # 2. metrics.yaml
        with open(os.path.join(self.output_dir, "metrics.yaml"), "w") as f:
            yaml.dump(result.metrics_nested(), f, sort_keys=False)

        # 3. predictions.csv (tidy long: val folds + optional test)
        preds = result.predictions_long()
        if not preds.empty:
            preds.to_csv(os.path.join(self.output_dir, "predictions.csv"), index=False)

        # 4. metrics_per_target.csv (tidy)
        result.per_target_frame().to_csv(
            os.path.join(self.output_dir, "metrics_per_target.csv"), index=False)

        # 5. learning_curve.csv (fold 0)
        lc = result.learning_curve_frame()
        if lc is not None:
            lc.to_csv(os.path.join(self.output_dir, "learning_curve.csv"), index=False)

        # 6. Tuning artifacts (optimize mode)
        if isinstance(result.tuning_trials, pd.DataFrame) and not result.tuning_trials.empty:
            result.tuning_trials.to_csv(os.path.join(self.output_dir, "tuning_trials.csv"), index=False)

        if isinstance(result.param_importances, dict) and len(result.param_importances) > 0:
            imp_df = pd.DataFrame(
                [{"parameter": k, "importance": v} for k, v in result.param_importances.items()]
            ).sort_values("importance", ascending=False).reset_index(drop=True)
            imp_df.to_csv(os.path.join(self.output_dir, "param_importances.csv"), index=False)

        print(f"Results (stage: {result.stage}) saved to {self.output_dir}")
