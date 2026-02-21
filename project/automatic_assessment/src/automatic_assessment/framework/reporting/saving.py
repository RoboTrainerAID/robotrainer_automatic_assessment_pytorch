import os
import pandas as pd
import yaml
from datetime import datetime
import numpy as np
import inspect
import shutil

class SavingModule:
    def __init__(self, model_name: str):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = f"/workspace/automatic_assessment/experiment_results/{timestamp}_{model_name}"
        os.makedirs(self.output_dir, exist_ok=True)

    def save_model_source(self, model_class):
        try:
            source_file = inspect.getsourcefile(model_class)
            if (source_file):
                dst = os.path.join(self.output_dir, "model_source.py")
                shutil.copy(source_file, dst)
                print(f"Saved model source code to {dst}")
        except Exception as e:
            print(f"Could not save model source: {e}")

    def save_results(self, results: dict):
        fold_data = results.get('fold_data', [])
        test_metrics = results.get('test_metrics', {})
        baseline_metrics = results.get('baseline_metrics', {})
        experiment_info = results.get('experiment_info', {})
        
        # FIX: Ensure we look for 'pipeline_config' if 'config' key is missing or empty
        config = results.get('config', results.get('pipeline_config', {}))
        
        # ---------------------------------------------------------
        # 1. Save Config & Best Params (Meta Info)
        # ---------------------------------------------------------
        # Extract best params from the first fold (since they are shared/optimized before final test)
        best_params = {}
        if fold_data and 'best_params' in fold_data[0]:
            best_params = fold_data[0]['best_params']
            
        full_config = {
            "experiment_info": experiment_info,
            "pipeline_config": config,
            "best_params": best_params
        }
        
        with open(os.path.join(self.output_dir, "config.yaml"), 'w') as f:
            yaml.dump(full_config, f, sort_keys=False)

        # ---------------------------------------------------------
        # 1b. Save Tuning Trials & Param Importances
        # ---------------------------------------------------------
        if fold_data:
            # Tuning trials (Optuna trials dataframe)
            tuning_trials = fold_data[0].get('tuning_trials')
            if tuning_trials is not None:
                if isinstance(tuning_trials, pd.DataFrame) and not tuning_trials.empty:
                    tuning_trials.to_csv(os.path.join(self.output_dir, "tuning_trials.csv"), index=False)
            
            # Param importances (dict from Optuna)
            param_importances = fold_data[0].get('param_importances')
            if param_importances is not None and isinstance(param_importances, dict) and len(param_importances) > 0:
                # Convert to DataFrame with 'parameter' and 'importance' columns
                imp_df = pd.DataFrame([
                    {"parameter": k, "importance": v}
                    for k, v in param_importances.items()
                ])
                imp_df = imp_df.sort_values('importance', ascending=False).reset_index(drop=True)
                imp_df.to_csv(os.path.join(self.output_dir, "param_importances.csv"), index=False)

        # ---------------------------------------------------------
        # 2. Save Learning Curve (Debugging)
        # ---------------------------------------------------------
        # FIX: Pipeline stores this as 'history', not 'learning_curve'
        if fold_data:
            # Check both keys to be safe
            lc_data = fold_data[0].get('history', fold_data[0].get('learning_curve'))
            
            if lc_data:
                # lc_data can be:
                # - a dict of lists: {'train_loss': [...], 'val_loss': [...]}
                # - a list of dicts: [{'epoch': 1, 'train_loss': ...}, ...]
                
                if isinstance(lc_data, dict):
                    # Dict of lists format — convert to DataFrame directly
                    lc_df = pd.DataFrame(lc_data)
                    
                    # Add 'epoch' column if missing
                    if 'epoch' not in lc_df.columns:
                        lc_df.insert(0, 'epoch', range(1, len(lc_df) + 1))
                        
                elif isinstance(lc_data, list) and len(lc_data) > 0:
                    lc_df = pd.DataFrame(lc_data)
                    
                    # Add 'epoch' column if missing
                    if 'epoch' not in lc_df.columns:
                        lc_df.insert(0, 'epoch', range(1, len(lc_df) + 1))
                else:
                    lc_df = pd.DataFrame()
                
                # Normalize common column name variants
                rename_map = {}
                for col in lc_df.columns:
                    col_lower = col.lower()
                    if col_lower in ('training_loss', 'loss', 'train'):
                        rename_map[col] = 'train_loss'
                    elif col_lower in ('validation_loss', 'val', 'valid_loss'):
                        rename_map[col] = 'val_loss'
                if rename_map:
                    lc_df = lc_df.rename(columns=rename_map)
                
                # Check if empty (e.g. sklearn models might return empty history)
                if not lc_df.empty:
                    lc_df.to_csv(os.path.join(self.output_dir, "learning_curve.csv"), index=False)

        # ---------------------------------------------------------
        # 3. Save Predictions to CSV
        # ---------------------------------------------------------
        preds_file = os.path.join(self.output_dir, "predictions.csv")
        all_preds_rows = []
        
        for fold_idx, f in enumerate(fold_data):
            user_ids = f["user_id"]
            
            # Normalize user_ids to list
            if np.isscalar(user_ids) or (isinstance(user_ids, np.ndarray) and user_ids.ndim == 0):
                 user_ids = [user_ids]
            elif isinstance(user_ids, np.ndarray):
                user_ids = user_ids.flatten().tolist()
            elif not isinstance(user_ids, list):
                user_ids = list(user_ids)

            preds = f['test_preds']
            actuals = f['test_actuals']
            
            # Helper to wrap scalars
            if np.isscalar(preds): preds = [preds]
            if np.isscalar(actuals): actuals = [actuals]
            
            count = min(len(user_ids), len(preds))
            
            for i in range(count):
                uid = user_ids[i]
                current_pred = preds[i]
                current_actual = actuals[i]
                
                # Normalize to list (handle multi-target)
                p_list = [float(current_pred)] if np.isscalar(current_pred) else [float(p) for p in current_pred]
                a_list = [float(current_actual)] if np.isscalar(current_actual) else [float(a) for a in current_actual]
                
                row = {
                    "fold": fold_idx,
                    "user_id": int(uid),
                }

                # Flatten targets: pred_target_0, etc.
                for t_idx, val in enumerate(p_list):
                    row[f"pred_target_{t_idx}"] = val
                    
                for t_idx, val in enumerate(a_list):
                    row[f"actual_target_{t_idx}"] = val
                
                all_preds_rows.append(row)
                
        full_df = pd.DataFrame(all_preds_rows)
        full_df.to_csv(preds_file, index=False)

        # ---------------------------------------------------------
        # 4. Save Summary Metrics (metrics.yaml)
        # ---------------------------------------------------------
        summary = test_metrics.copy()
        
        # Merge Final Test Baseline Metrics
        # Prefix them with 'test_' to distinguish from validation baseline
        for k, v in baseline_metrics.items():
            if not k.startswith("test_"):
                summary[f"test_{k}"] = float(v)
            else:
                summary[k] = float(v)

        summary['test_loss'] = float(results.get('test_loss', 0.0))
        summary['test_baseline_loss'] = float(results.get('baseline_loss', 0.0))
        
        # Save Validation Baseline Metrics (from tuning phase)
        val_baseline_metrics = results.get('val_baseline_metrics', {})
        for k, v in val_baseline_metrics.items():
            # Store as val_baseline_rmse_target_X etc.
            summary[f"val_{k}"] = float(v)
        
        # Add per-fold summaries
        summary["fold_metrics"] = []
        
        # Aggregators for Validation Metrics (CV average)
        val_metric_lists = {}
        
        for f in fold_data:
            # Handle UID for summary
            uid_data = f["user_id"]
            if isinstance(uid_data, (np.ndarray, list)):
                if isinstance(uid_data, np.ndarray):
                    uid_save = uid_data.flatten().tolist()
                else:
                    uid_save = uid_data
                if isinstance(uid_save, list) and len(uid_save) == 1:
                    uid_save = int(uid_save[0])
            else:
                uid_save = int(uid_data)

            fold_summary = {
                "fold": f["fold"],
                "user_id": uid_save,
                "test_loss": float(f["test_loss"]),
                "baseline_loss": float(f.get("baseline_loss", 0.0))
            }
            summary["fold_metrics"].append(fold_summary)
            
            # Aggregate validation metrics from this fold
            if "val_metrics" in f:
                for metric_name, val in f["val_metrics"].items():
                    if "rmse" in metric_name:
                        # FIX: Strip the existing 'val_' prefix to avoid double prefix
                        clean_name = metric_name
                        if clean_name.startswith("val_"):
                            clean_name = clean_name[4:]  # Remove 'val_' prefix
                        
                        if clean_name not in val_metric_lists:
                            val_metric_lists[clean_name] = []
                        val_metric_lists[clean_name].append(val)
            
            # Aggregate Validation Baseline metrics stored in fold_data
            if "baseline_metrics" in f:
                for metric_name, val in f["baseline_metrics"].items():
                    if "rmse" in metric_name:
                        # Strip existing prefix and re-add cleanly
                        clean_name = metric_name
                        if clean_name.startswith("baseline_"):
                            clean_name = clean_name[9:]  # Remove 'baseline_'
                        
                        key = f"baseline_{clean_name}"
                        if key not in val_metric_lists:
                            val_metric_lists[key] = []
                        val_metric_lists[key].append(val)

        # Compute Mean for all Validation Metrics
        # FIX: Use 'val_' prefix once (not 'mean_val_val_')
        for metric_name, values in val_metric_lists.items():
            summary[f"val_{metric_name}"] = float(np.mean(values))
        
        with open(os.path.join(self.output_dir, "metrics.yaml"), 'w') as f:
            yaml.dump(summary, f)
            
        print(f"Results saved to {self.output_dir}")