import os
import pandas as pd
import yaml
from datetime import datetime
import numpy as np

class SavingModule:
    def __init__(self, model_name: str):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = f"experiments/{model_name}_{timestamp}"
        os.makedirs(self.output_dir, exist_ok=True)

    def save_results(self, results: dict):
        fold_data = results['fold_data']
        test_metrics = results['test_metrics']
        
        # 1. Save Hyperparams
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

            # Save representative config (Fold 0) to root for quick access
            if 'best_params' in fold_data[0]:
                best_params = fold_data[0]['best_params']
                with open(os.path.join(self.output_dir, "config.yaml"), 'w') as f:
                    yaml.dump(best_params, f)

        # 1.b Save Learning Curve Data (from first fold)
        if fold_data and 'history' in fold_data[0] and fold_data[0]['history']:
            history = fold_data[0]['history']
            history_df = pd.DataFrame(history)
            history_df['epoch'] = range(1, len(history_df) + 1)
            history_df.to_csv(os.path.join(self.output_dir, "learning_curve.csv"), index=False)

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
        
        # Add per-fold summaries
        summary["fold_metrics"] = []
        val_rmses = []
        
        for f in fold_data:
            fold_summary = {
                "fold": f["fold"],
                "user_id": int(f["user_id"]),
                "test_loss": float(f["test_loss"]),
            }
            # No longer merging test_fold_metrics since they were removed
                
            summary["fold_metrics"].append(fold_summary)
            
            # Check nested dictionary for val metric
            if "val_metrics" in f and "val_rmse_mean" in f["val_metrics"]:
                val_rmses.append(f["val_metrics"]["val_rmse_mean"])

        if val_rmses:
            summary["mean_val_rmse"] = float(np.mean(val_rmses))
        
        with open(os.path.join(self.output_dir, "metrics.yaml"), 'w') as f:
            yaml.dump(summary, f)
            
        print(f"Results saved to {self.output_dir}")