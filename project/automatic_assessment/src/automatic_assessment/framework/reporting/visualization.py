import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

class VisualizationModule:
    def __init__(self, experiment_path):
        self.path = experiment_path
        self.preds_df = pd.read_csv(os.path.join(experiment_path, "predictions.csv"))
        self.figures_path = os.path.join(experiment_path, "plots")
        os.makedirs(self.figures_path, exist_ok=True)
        
        # Load metrics if available
        metrics_path = os.path.join(experiment_path, "metrics.yaml")
        if os.path.exists(metrics_path):
            with open(metrics_path, "r") as f:
                self.metrics = yaml.safe_load(f)
        else:
            self.metrics = {}
        
        # Identify how many targets we have
        self.target_cols = [c for c in self.preds_df.columns if c.startswith("actual_target_")]
        self.n_targets = len(self.target_cols)

    def generate_parity_plots(self):
        """Creates a scatter plot of Actual vs Predicted for each target."""
        fig, axes = plt.subplots(1, self.n_targets, figsize=(6 * self.n_targets, 5), squeeze=False)
        
        for i, target_col in enumerate(self.target_cols):
            pred_col = f"pred_target_{i}"
            ax = axes[0, i]
            
            sns.regplot(x=target_col, y=pred_col, data=self.preds_df, 
                        ax=ax, scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
            
            # Draw diagonal identity line
            min_val = min(self.preds_df[target_col].min(), self.preds_df[pred_col].min())
            max_val = max(self.preds_df[target_col].max(), self.preds_df[pred_col].max())
            ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.7)
            
            # Use pre-calculated metrics if available (using simplified names)
            if f"test_r2_target_{i}" in self.metrics:
                r2 = self.metrics[f"test_r2_target_{i}"]
                mae = self.metrics[f"test_mae_target_{i}"]
            else:
                r2 = r2_score(self.preds_df[target_col], self.preds_df[pred_col])
                mae = mean_absolute_error(self.preds_df[target_col], self.preds_df[pred_col])
            
            ax.set_title(f"Target {i}\n$R^2$: {r2:.2f} | MAE: {mae:.2f}")
            ax.set_xlabel("Ground Truth (Scaled)")
            ax.set_ylabel("Predictions (Scaled)")

        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_path, "parity_plots.png"))
        plt.close()

    def generate_error_distribution(self):
        """Visualizes the residuals to check for heteroscedasticity or bias."""
        fig, axes = plt.subplots(1, self.n_targets, figsize=(6 * self.n_targets, 5), squeeze=False)
        
        for i, target_col in enumerate(self.target_cols):
            pred_col = f"pred_target_{i}"
            residuals = self.preds_df[target_col] - self.preds_df[pred_col]
            ax = axes[0, i]
            
            sns.histplot(residuals, kde=True, ax=ax, color='purple')
            ax.axvline(0, color='black', linestyle='--')
            ax.set_title(f"Residual Distribution: Target {i}")
            ax.set_xlabel("Error (Actual - Predicted)")

        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_path, "residuals.png"))
        plt.close()

    def generate_user_performance_report(self):
        """Shows RMSE per user to identify high-variance outliers."""
        user_metrics = []
        for user_id, group in self.preds_df.groupby('user_id'):
            user_res = {'user_id': user_id}
            for i in range(self.n_targets):
                rmse = np.sqrt(mean_squared_error(group[f"actual_target_{i}"], group[f"pred_target_{i}"]))
                user_res[f"target_{i}_rmse"] = rmse
            user_metrics.append(user_res)
        
        df_user = pd.DataFrame(user_metrics)
        
        plt.figure(figsize=(10, 6))
        df_melt = df_user.melt(id_vars='user_id', value_vars=[f"target_{i}_rmse" for i in range(self.n_targets)])
        sns.barplot(x='user_id', y='value', hue='variable', data=df_melt)
        plt.title("Per-User Error Analysis (LOGO CV Results)")
        plt.ylabel("RMSE (Scaled)")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_path, "user_performance.png"))
        plt.close()
        
        return df_user

    def generate_learning_curve(self):
        """Plots the learning curve (loss per epoch) for the first fold using available data."""
        lc_path = os.path.join(self.path, "learning_curve.csv")
        if not os.path.exists(lc_path):
            print("No learning curve data found.")
            return

        df = pd.read_csv(lc_path)
        
        # Using sklearn's LearningCurveDisplay style, but manually plotting loss history
        # (sklearn's LearningCurveDisplay is for sample_sizes, we want loss/epoch)
        plt.figure(figsize=(10, 6))
        plt.plot(df['epoch'], df['train_loss'], label='Training Loss', marker='o', markersize=3)
        
        if 'val_loss' in df.columns and df['val_loss'].notna().any():
            plt.plot(df['epoch'], df['val_loss'], label='Validation/Test Loss', marker='o', markersize=3)
        
        plt.title("Learning Curve (Fold 0)")
        plt.xlabel("Epoch")
        plt.ylabel("Loss (Huber)")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_path, "learning_curve.png"))
        plt.close()