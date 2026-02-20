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
                rmse = self.metrics.get(f"test_rmse_target_{i}", 
                                        np.sqrt(mean_squared_error(self.preds_df[target_col], self.preds_df[pred_col])))
            else:
                r2 = r2_score(self.preds_df[target_col], self.preds_df[pred_col])
                rmse = np.sqrt(mean_squared_error(self.preds_df[target_col], self.preds_df[pred_col]))
            
            ax.set_title(f"Target {i}\n$R^2$: {r2:.2f} | RMSE: {rmse:.2f}")
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
        Includes baseline dots for both validation and test baselines.
        """
        if not self.metrics:
            print("No metrics loaded for RMSE comparison.")
            return

        # Load target names from config.yaml
        config_path = os.path.join(self.path, "config.yaml")
        target_names = {}
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            targets_list = config.get('pipeline_config', {}).get('targets', [])
            for i, name in enumerate(targets_list):
                target_names[i] = name

        # Find all target indices present
        target_indices = set()
        for key in self.metrics.keys():
            if "rmse_target_" in key and "baseline" not in key:
                try:
                    parts = key.split('_')
                    idx_pos = parts.index('target') + 1
                    target_indices.add(int(parts[idx_pos]))
                except (ValueError, IndexError):
                    continue

        target_indices = sorted(target_indices)

        if not target_indices:
            print("No target-specific RMSE metrics found for plotting.")
            return

        # Collect Model RMSE (Val and Test)
        data = []
        for t_idx in target_indices:
            label = target_names.get(t_idx, f"Target {t_idx}")

            val_key = f"val_rmse_target_{t_idx}"
            if val_key in self.metrics:
                data.append({
                    "Target": label,
                    "RMSE": float(self.metrics[val_key]),
                    "Split": "Validation"
                })

            test_key = f"test_rmse_target_{t_idx}"
            if test_key in self.metrics:
                data.append({
                    "Target": label,
                    "RMSE": float(self.metrics[test_key]),
                    "Split": "Test"
                })

        if not data:
            print("No target-specific RMSE metrics found for plotting.")
            return

        df_plot = pd.DataFrame(data)

        # Plotting
        fig, ax = plt.subplots(figsize=(max(10, len(target_indices) * 2.5), 6))
        sns.set_theme(style="whitegrid")

        barplot = sns.barplot(
            data=df_plot,
            x='Target',
            y='RMSE',
            hue='Split',
            hue_order=["Validation", "Test"],
            palette="muted",
            ax=ax
        )

        # Theoretical baseline reference line
        ax.axhline(y=1.0, color='black', linestyle='--', linewidth=1.5, alpha=0.5, label='Scaled Mean Baseline (≈1.0)')

        # --- Overlay baseline dots aligned to actual bar positions ---
        # Extract bar positions from the rendered barplot
        # Bars are grouped: for each target, first bar = Validation, second bar = Test
        val_bar_centers = []
        test_bar_centers = []

        patches = barplot.patches
        n_targets = len(target_indices)
        # sns.barplot renders all bars of one hue first, then all of the next hue
        # So: patches[0..n_targets-1] = Validation bars, patches[n_targets..2*n_targets-1] = Test bars
        for i in range(n_targets):
            val_patch = patches[i]
            val_bar_centers.append(val_patch.get_x() + val_patch.get_width() / 2)

        for i in range(n_targets, 2 * n_targets):
            test_patch = patches[i]
            test_bar_centers.append(test_patch.get_x() + test_patch.get_width() / 2)

        # Plot Validation Baseline dots on top of Validation bars
        val_bl_x = []
        val_bl_y = []
        for i, t_idx in enumerate(target_indices):
            val_bl_key = f"val_baseline_rmse_target_{t_idx}"
            if val_bl_key in self.metrics:
                val_bl_x.append(val_bar_centers[i])
                val_bl_y.append(float(self.metrics[val_bl_key]))

        if val_bl_x:
            ax.scatter(val_bl_x, val_bl_y, color='#1f77b4', marker='D', s=70,
                       zorder=5, edgecolors='black', linewidths=0.8, label='Val Baseline (Dummy)')

        # Plot Test Baseline dots on top of Test bars
        test_bl_x = []
        test_bl_y = []
        for i, t_idx in enumerate(target_indices):
            test_bl_key = f"test_baseline_rmse_target_{t_idx}"
            if test_bl_key in self.metrics:
                test_bl_x.append(test_bar_centers[i])
                test_bl_y.append(float(self.metrics[test_bl_key]))

        if test_bl_x:
            ax.scatter(test_bl_x, test_bl_y, color='#ff7f0e', marker='D', s=70,
                       zorder=5, edgecolors='black', linewidths=0.8, label='Test Baseline (Dummy)')

        ax.set_title("RMSE Comparison: Validation vs Test per Target", fontsize=16)
        ax.set_ylabel("RMSE (Scaled)", fontsize=12)
        ax.set_xlabel("")
        plt.xticks(rotation=30, ha='right')
        ax.legend(title="Legend", loc='upper right')
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

if __name__ == "__main__":
    # Example usage assume running from src root or similar
    # Adjust path as needed during execution
    experiment_path = "/workspace/automatic_assessment/experiment_results/20260220_174324_CNN_Baseline"
    viz = VisualizationModule(experiment_path)
    viz.generate_all_plots()
