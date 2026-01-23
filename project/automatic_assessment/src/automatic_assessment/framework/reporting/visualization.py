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

    def generate_attention_plot(self):
        """
        Loads attention weights and generates a bar chart of path importance.
        """
        path_weights_file = os.path.join(self.path, "mean_attention_weights.csv")
        if not os.path.exists(path_weights_file):
            # Fallback if only npy exists
            npy_file = os.path.join(self.path, "attention_weights.npy")
            if not os.path.exists(npy_file):
                return
            # Load npy
            attn = np.load(npy_file) # (Samples, Paths, 1)
            mean_attn = np.mean(attn, axis=0).flatten()
            df = pd.DataFrame(mean_attn, columns=['mean_weight'])
        else:
            df = pd.read_csv(path_weights_file)
            
        plt.figure(figsize=(10, 6))
        df = df.reset_index().rename(columns={'index': 'Path ID'})
        
        sns.barplot(data=df, x='Path ID', y='mean_weight', hue='Path ID', palette='viridis', legend=False)
        plt.title("Learned Path Attention Importance (Fold 0)")
        plt.ylabel("Mean Attention Weight")
        plt.xlabel("Path Index")
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        plt.tight_layout()
        
        plt.savefig(os.path.join(self.figures_path, "attention_importance.png"))
        plt.close()
        print(f"Saved attention plot to {self.figures_path}/attention_importance.png")

    def generate_tuning_plot(self):
        """
        Loads tuning trials and generates convergence plot.
        """
        trials_file = os.path.join(self.path, "tuning_trials.csv")
        if not os.path.exists(trials_file):
            return
            
        df = pd.read_csv(trials_file)
        if 'value' not in df.columns or 'number' not in df.columns:
            return

        # Sort checks
        df = df.sort_values('number')
        
        # Values might be negative MSE if maximizing? or just MSE (minimized).
        # Optuna stores what objective returns. Our objective returns val_loss (Huber).
        scores = df['value'].values
        
        # Calculate best so far
        best_so_far = np.minimum.accumulate(scores)
        
        plt.figure(figsize=(10, 6))
        
        # Plot trials
        plt.scatter(df['number'], scores, c='gray', alpha=0.5, label='Individual Trial')
        
        # Plot best
        plt.plot(df['number'], best_so_far, c='red', lw=2, label='Best So Far')
        
        # Best point line
        best_idx = np.argmin(scores)
        best_val = scores[best_idx]
        plt.axhline(best_val, color='green', linestyle=':', label=f'Best: {best_val:.4f}')
        
        # Vertical phases
        n_trials = len(df)
        # Heuristic phases if enough trials
        if n_trials >= 20:
            # Assuming TPE sampler which starts with random exploration
            # Usually first 10-20% is random/exploration
            exploration_end = int(n_trials * 0.2)
            plt.axvline(exploration_end, color='blue', linestyle='--', alpha=0.5)
            plt.text(exploration_end, max(scores), " Exploration End ", rotation=90, verticalalignment='top')
            
        plt.title("Hyperparameter Tuning Convergence (Fold 0)")
        plt.xlabel("Trial Number")
        plt.ylabel("Validation Loss")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plt.savefig(os.path.join(self.figures_path, "tuning_convergence.png"))
        plt.close()
        print(f"Saved tuning convergence plot to {self.figures_path}/tuning_convergence.png")

    def generate_rmse_comparison(self):
        """
        Generates a grouped bar chart comparing Validation and Test RMSE for each target.
        Includes a baseline line at RMSE=1.0.
        """
        if not self.metrics:
            print("No metrics loaded for RMSE comparison.")
            return

        # Extract target specific RMSEs
        data = []
        
        # Identify target indices from keys like 'test_rmse_target_0' or 'mean_val_rmse_target_0'
        for key, value in self.metrics.items():
            if "rmse_target" not in key:
                continue
                
            parts = key.split('_')
            # Structure often: [test, rmse, target, X] or [mean, val, rmse, target, X]
            try:
                target_idx_loc = parts.index('target') + 1
                target_id = parts[target_idx_loc]
                
                if key.startswith('test_'):
                    split = 'Test'
                elif 'val_' in key:
                    split = 'Validation'
                else:
                    continue
                    
                data.append({
                    "Target": f"Target {target_id}",
                    "RMSE": float(value),
                    "Split": split
                })
            except (ValueError, IndexError):
                continue
                
        if not data:
            print("No target-specific RMSE metrics found for plotting.")
            return
            
        df_plot = pd.DataFrame(data)
        
        # Plotting
        plt.figure(figsize=(10, 6))
        sns.set_theme(style="whitegrid")
        
        sns.barplot(
            data=df_plot, 
            x='Target', 
            y='RMSE', 
            hue='Split', 
            palette="muted"
        )
        
        # Add Random Guessing Baseline
        plt.axhline(y=1.0, color='black', linestyle='--', linewidth=2, label='Random Guessing (RMSE=1.0)')
        
        plt.title("RMSE Comparison: Validation vs Test per Target", fontsize=16)
        plt.ylabel("RMSE (Scaled)", fontsize=12)
        plt.xlabel("")
        plt.legend(title="Split")
        plt.tight_layout()
        
        save_path = os.path.join(self.figures_path, "rmse_comparison.png")
        plt.savefig(save_path)
        plt.close()
        print(f"Saved {save_path}")

    def generate_all_plots(self):
        """Generates all available plots and reports."""
        self.generate_parity_plots()
        self.generate_error_distribution()
        self.generate_learning_curve()
        self.generate_user_performance_report()
        self.generate_attention_plot()
        self.generate_tuning_plot()
        self.generate_rmse_comparison()
