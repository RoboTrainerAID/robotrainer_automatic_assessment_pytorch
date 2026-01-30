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
        pipeline_config = results.get('pipeline_config', {})
        
        # 1. Save Config & Hyperparams
        # Prepare expanded config content
        full_config = {
            "experiment_info": experiment_info,
            "pipeline_config": pipeline_config,
        }

        if fold_data:
            # Create subfolder for fold-specific hyperparameters
            hp_dir = os.path.join(self.output_dir, "fold_hyperparameters")
            os.makedirs(hp_dir, exist_ok=True)
            
            for fold_info in fold_data:
                if 'best_params' in fold_info:
                    fold_idx = fold_info['fold']
                    params = fold_info['best_params']
                    # Save separate param file for each fold
                    with open(os.path.join(hp_dir, f"fold_{fold_idx}_params.yaml"), 'w') as f:
                        yaml.dump(params, f)

            # Add representative best_params (Fold 0) to main config for quick access
            if 'best_params' in fold_data[0]:
                full_config['hyperparameters_fold_0'] = fold_data[0]['best_params']

        # Save main config.yaml
        with open(os.path.join(self.output_dir, "config.yaml"), 'w') as f:
            yaml.dump(full_config, f, sort_keys=False)

        # 1.b Save Learning Curve Data (from first fold)
        if fold_data and 'history' in fold_data[0] and fold_data[0]['history']:
            history = fold_data[0]['history']
            history_df = pd.DataFrame(history)
            history_df['epoch'] = range(1, len(history_df) + 1)
            history_df.to_csv(os.path.join(self.output_dir, "learning_curve.csv"), index=False)
            
            # Save Tuning Trials (Fold 0)
            if 'tuning_trials' in fold_data[0] and fold_data[0]['tuning_trials'] is not None:
                fold_data[0]['tuning_trials'].to_csv(os.path.join(self.output_dir, "tuning_trials.csv"), index=False)
            
            # Save Parameter Importances (Fold 0)
            if 'param_importances' in fold_data[0] and fold_data[0]['param_importances'] is not None:
                importances_df = pd.DataFrame(list(fold_data[0]['param_importances'].items()), columns=['parameter', 'importance'])
                importances_df.to_csv(os.path.join(self.output_dir, "param_importances.csv"), index=False)
                
            # Save Attention Weights (Fold 0)
            if 'attention_weights' in fold_data[0] and fold_data[0]['attention_weights'] is not None:
                attn = fold_data[0]['attention_weights']
                # Save as npy for full structure
                np.save(os.path.join(self.output_dir, "attention_weights.npy"), attn)
                # Save mean per path as csv for quick looking
                # Shape: (Samples, Paths, 1) -> Mean over samples -> (Paths)
                mean_attn = np.mean(attn, axis=0).flatten()
                pd.DataFrame(mean_attn, columns=['mean_weight']).to_csv(os.path.join(self.output_dir, "mean_attention_weights.csv"))

        # 2. Aggregate Fold Predictions
        all_preds = []
        for fold_info in fold_data:
            preds = fold_info['test_preds']
            actuals = fold_info['test_actuals']
            
            df = pd.DataFrame(preds, columns=[f"pred_target_{i}" for i in range(preds.shape[1])])
            df['user_id'] = fold_info['user_id']
            df['fold'] = fold_info['fold']
            
            # Add actuals for parity plotting
            for i in range(actuals.shape[1]):
                df[f"actual_target_{i}"] = actuals[:, i]
            all_preds.append(df)
        
        full_df = pd.concat(all_preds)
        full_df.to_csv(os.path.join(self.output_dir, "predictions.csv"), index=False)

        # 3. Save Summary Metrics
        summary = test_metrics.copy()
        
        # Merge baseline metrics
        summary.update(baseline_metrics)
        summary['test_loss'] = results.get('test_loss', 0.0)
        summary['baseline_loss'] = results.get('baseline_loss', 0.0)
        
        # Add per-fold summaries
        summary["fold_metrics"] = []
        
        # Initialize aggregators for all RMSE metrics found in validation
        val_metric_lists = {}
        
        for f in fold_data:
            fold_summary = {
                "fold": f["fold"],
                "user_id": int(f["user_id"]),
                "test_loss": float(f["test_loss"]),
                "baseline_loss": float(f.get("baseline_loss", 0.0))
            }
            # No longer merging test_fold_metrics since they were removed
                
            summary["fold_metrics"].append(fold_summary)
            
            # Aggregate validation metrics (including specific targets)
            if "val_metrics" in f:
                for metric_name, val in f["val_metrics"].items():
                    if "rmse" in metric_name:
                        if metric_name not in val_metric_lists:
                            val_metric_lists[metric_name] = []
                        val_metric_lists[metric_name].append(val)

        # Compute mean for all validation RMSE metrics
        for metric_name, values in val_metric_lists.items():
            summary[f"mean_{metric_name}"] = float(np.mean(values))
        
        with open(os.path.join(self.output_dir, "metrics.yaml"), 'w') as f:
            yaml.dump(summary, f)
            
        print(f"Results saved to {self.output_dir}")