import os
import pandas as pd
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
import glob
from typing import List, Dict, Optional, Any
import numpy as np

class ExperimentComparison:
    def __init__(self, experiments_root_path: str):
        """
        Initialize the comparison module.
        
        Args:
            experiments_root_path: Path to the directory containing experiment folders.
        """
        self.root_path = experiments_root_path
        self.data: List[Dict[str, Any]] = []

    def _parse_folder_name(self, folder_name: str) -> str:
        """Extracts model name from folder name (assuming YYYYMMDD_HHMMSS_ModelName format)."""
        parts = folder_name.split('_')
        if len(parts) >= 3:
            # Join everything after the timestamp parts
            return "_".join(parts[2:])
        return "UnknownModel"

    def _determine_config_type(self, targets: List[str]) -> str:
        """Classifies the experiment configuration based on target count."""
        n = len(targets)
        if n == 1:
            return "Single Target"
        elif n == 4:
            return "Reduced (4 Targets)"
        elif n > 10:
            return "All Targets"
        else:
            return f"{n} Targets"

    def load_data(self) -> pd.DataFrame:
        """Scans all experiment folders and aggregates metrics."""
        self.data = []
        
        # Iterate over directories in root path
        for folder_name in sorted(os.listdir(self.root_path)):
            folder_path = os.path.join(self.root_path, folder_name)
            if not os.path.isdir(folder_path):
                continue
            
            # Paths to expected files
            config_path = os.path.join(folder_path, "config.yaml")
            metrics_path = os.path.join(folder_path, "metrics.yaml")
            
            if not (os.path.exists(config_path) and os.path.exists(metrics_path)):
                continue
                
            try:
                # Load Config
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                
                # Load Metrics
                with open(metrics_path, 'r') as f:
                    metrics = yaml.safe_load(f)
                
                # Extract Info
                model_name = self._parse_folder_name(folder_name)
                
                if 'pipeline_config' in config and 'targets' in config['pipeline_config']:
                    target_names = config['pipeline_config']['targets']
                else:
                    print(f"Skipping {folder_name}: No targets found in config.")
                    continue
                
                config_type = self._determine_config_type(target_names)
                
                # Extract augmentation ratio if available
                augmentation_ratio = config.get('pipeline_config', {}).get('augmentation_ratio', None)
                
                # Loop through targets using the definition from THIS specific run
                for local_index, target_name in enumerate(target_names):
                    # Map local index to metric key
                    rmse_key = f"test_rmse_target_{local_index}"
                    r2_key = f"test_r2_target_{local_index}"
                    baseline_key = f"baseline_rmse_target_{local_index}"
                    
                    if rmse_key in metrics:
                        rmse_val = float(metrics[rmse_key])
                        r2_val = float(metrics.get(r2_key, np.nan))
                        baseline_val = float(metrics.get(baseline_key, np.nan))
                        
                        self.data.append({
                            "Folder": folder_name,
                            "Model Class": model_name,
                            "Target": target_name,  # Using name for grouping
                            "Config Type": config_type,
                            "Augmentation Ratio": augmentation_ratio,
                            "RMSE": rmse_val,
                            "R2": r2_val,
                            "Baseline RMSE": baseline_val
                        })
                        
            except Exception as e:
                print(f"Error processing {folder_name}: {e}")

        df = pd.DataFrame(self.data)
        return df

    def plot_multitarget_comparison(self, output_dir: Optional[str] = None) -> None:
        """
        Generates comparison plots. Creates a separate plot/figure for each Model Class,
        containing subplots for each Target. Comparison across Config Types.
        """
        if not self.data:
            df = self.load_data()
        else:
            df = pd.DataFrame(self.data)
            
        if df.empty:
            print("No data found to plot.")
            return

        if output_dir is None:
            output_dir = os.path.join(self.root_path, "comparison_results")
        os.makedirs(output_dir, exist_ok=True)
        
        # Determine order for plotting consistency
        config_order = ["Single Target", "Reduced (4 Targets)", "All Targets"]
        # Define specific colors for each config type so they remain consistent across plots
        palette_colors = sns.color_palette("viridis", n_colors=len(config_order))
        palette_dict = dict(zip(config_order, palette_colors))
        
        # Iterate over Model Classes
        for model_class, model_df in df.groupby("Model Class"):
            print(f"Checking plots for Model: {model_class}")
            
            # Filter targets: Only include targets that have more than 1 Config Type available
            all_targets = model_df['Target'].unique()
            valid_targets = []
            for t in all_targets:
                subset = model_df[model_df['Target'] == t]
                if len(subset['Config Type'].unique()) > 1:
                    valid_targets.append(t)
            
            valid_targets = sorted(valid_targets)
            n_targets = len(valid_targets)
            
            if n_targets == 0:
                print(f"Skipping plot for {model_class}: No targets have comparison data (Config Type > 1).")
                continue

            print(f"Generating multitarget comparison plot for {model_class} with {n_targets} targets.")
            
            relevant_data = model_df[model_df['Target'].isin(valid_targets)]
            max_rmse = relevant_data['RMSE'].max() if not relevant_data.empty else 1.0
            y_limit = max(1.1, max_rmse * 1.1)
            
            n_cols = 3
            n_rows = (n_targets + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
            if isinstance(axes, np.ndarray):
                axes = axes.flatten()
            else:
                axes = [axes]
            
            for i, target in enumerate(valid_targets):
                ax = axes[i]
                subset = model_df[model_df['Target'] == target]
                
                sns.barplot(
                    data=subset, 
                    x="Config Type", 
                    y="RMSE", 
                    hue="Config Type",
                    ax=ax, 
                    order=[c for c in config_order if c in subset["Config Type"].unique()],
                    palette=palette_dict,
                    legend=False
                )
                
                ax.axhline(y=1.0, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
                
                ax.set_ylim(0, y_limit)
                
                ax.set_title(target)
                ax.set_xlabel("")
                ax.set_ylabel("RMSE")
                
                plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
                ax.legend()
                
                for container in ax.containers:
                    ax.bar_label(container, fmt='%.2f')

            for j in range(i + 1, len(axes)):
                axes[j].axis('off')
                
            plt.suptitle(f"RMSE Comparison per Target - {model_class}", fontsize=16)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            
            save_path = os.path.join(output_dir, f"multitarget_comparison_{model_class}.png")
            plt.savefig(save_path)
            plt.close()
            print(f"Saved multitarget comparison plot to {save_path}")

    def plot_augmentation_ratio_comparison(self, output_dir: Optional[str] = None) -> None:
        """
        Generates comparison plots showing the impact of augmentation ratio on RMSE per model.
        Creates a separate plot/figure for each Model Class, containing subplots for each Target.
        Only generates plots if augmentation_ratio is available and differs between runs.
        """
        if not self.data:
            df = self.load_data()
        else:
            df = pd.DataFrame(self.data)
            
        if df.empty:
            print("No data found to plot.")
            return
        
        # Check if augmentation_ratio is available and varies
        if 'Augmentation Ratio' not in df.columns or df['Augmentation Ratio'].isna().all():
            print("Augmentation ratio data not available.")
            return
        
        unique_ratios = df['Augmentation Ratio'].dropna().unique()
        if len(unique_ratios) <= 1:
            print("Augmentation ratio does not vary across experiments.")
            return

        if output_dir is None:
            output_dir = os.path.join(self.root_path, "comparison_results")
        os.makedirs(output_dir, exist_ok=True)
        
        # Sort augmentation ratios for consistent ordering
        sorted_ratios = sorted(unique_ratios)
        palette_colors = sns.color_palette("coolwarm", n_colors=len(sorted_ratios))
        palette_dict = dict(zip(sorted_ratios, palette_colors))
        
        # Iterate over Model Classes
        for model_class, model_df in df.groupby("Model Class"):
            print(f"Checking augmentation ratio plots for Model: {model_class}")
            
            # Filter to only rows with augmentation ratio
            model_df = model_df.dropna(subset=['Augmentation Ratio'])
            
            if model_df.empty:
                print(f"Skipping {model_class}: No augmentation ratio data available.")
                continue
            
            # Check if this model has multiple augmentation ratios
            model_ratios = model_df['Augmentation Ratio'].unique()
            if len(model_ratios) <= 1:
                print(f"Skipping {model_class}: Only one augmentation ratio found.")
                continue
            
            # Filter targets: Only include targets that have more than 1 Augmentation Ratio available
            all_targets = model_df['Target'].unique()
            valid_targets = []
            for t in all_targets:
                subset = model_df[model_df['Target'] == t]
                if len(subset['Augmentation Ratio'].unique()) > 1:
                    valid_targets.append(t)
            
            valid_targets = sorted(valid_targets)
            n_targets = len(valid_targets)
            
            if n_targets == 0:
                print(f"Skipping plot for {model_class}: No targets have comparison data (>1 augmentation ratio).")
                continue

            print(f"Generating augmentation ratio comparison plot for {model_class} with {n_targets} targets.")
            
            relevant_data = model_df[model_df['Target'].isin(valid_targets)]
            max_rmse = relevant_data['RMSE'].max() if not relevant_data.empty else 1.0
            y_limit = max(1.1, max_rmse * 1.1)
            
            n_cols = 3
            n_rows = (n_targets + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
            if isinstance(axes, np.ndarray):
                axes = axes.flatten()
            else:
                axes = [axes]
            
            for i, target in enumerate(valid_targets):
                ax = axes[i]
                subset = model_df[model_df['Target'] == target]
                
                sns.barplot(
                    data=subset, 
                    x="Augmentation Ratio", 
                    y="RMSE", 
                    hue="Augmentation Ratio",
                    ax=ax, 
                    order=[r for r in sorted_ratios if r in subset["Augmentation Ratio"].unique()],
                    palette=palette_dict,
                    legend=False
                )
                
                # Reference line at 1.0
                ax.axhline(y=1.0, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Ref 1.0')
                
                # Baseline line (if available)
                baseline_val = subset['Baseline RMSE'].mean()
                if not np.isnan(baseline_val):
                    ax.axhline(y=baseline_val, color='orange', linestyle=':', linewidth=1.5, alpha=0.7, label=f'Baseline {baseline_val:.2f}')
                
                ax.set_ylim(0, y_limit)
                ax.set_title(target)
                ax.set_xlabel("Augmentation Ratio")
                ax.set_ylabel("RMSE")
                
                plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
                ax.legend()
                
                for container in ax.containers:
                    ax.bar_label(container, fmt='%.2f')

            for j in range(i + 1, len(axes)):
                axes[j].axis('off')
                
            plt.suptitle(f"RMSE vs Augmentation Ratio per Target - {model_class}", fontsize=16)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            
            save_path = os.path.join(output_dir, f"augmentation_ratio_comparison_{model_class}.png")
            plt.savefig(save_path)
            plt.close()
            print(f"Saved augmentation ratio comparison plot to {save_path}")

    def plot_model_comparison(self, output_dir: Optional[str] = None) -> None:
        """
        Generates comparison plots between Models.
        1. Aggregated Boxplot: Mean score across all targets that a model predicts.
        2. Per-Target Comparison: Detailed RMSE comparison per target.
        """
        if not self.data:
            df = self.load_data()
        else:
            df = pd.DataFrame(self.data)
            
        if df.empty:
            print("No data found to plot.")
            return

        if output_dir is None:
            output_dir = os.path.join(self.root_path, "comparison_results")
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Aggregated Boxplot
        plt.figure(figsize=(10, 8))
        sns.boxplot(data=df, x="Model Class", y="RMSE", hue="Model Class", palette="Set2", legend=False)
        sns.stripplot(data=df, x="Model Class", y="RMSE", color='black', alpha=0.3, jitter=0.2)
        plt.axhline(y=1.0, color='black', linestyle='--', linewidth=2, label="Ref 1.0")
        
        plt.title("Model Performance Distribution across All Targets")
        plt.ylabel("RMSE")
        plt.xlabel("Model Class")
        plt.xticks(rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "aggregated_model_comparison.png"))
        plt.close()
        
        # 2. Per-target Subplots
        targets = sorted(df['Target'].unique())
        n_targets = len(targets)
        
        if n_targets == 0:
            return

        n_cols = 3
        n_rows = (n_targets + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 6 * n_rows))
        if isinstance(axes, np.ndarray):
            axes = axes.flatten()
        else:
            axes = [axes]
            
        max_rmse = df['RMSE'].max() if not df.empty else 1.0
        y_limit = max(1.2, max_rmse * 1.1)

        for i, target in enumerate(targets):
            ax = axes[i]
            subset = df[df['Target'] == target]
            
            if subset.empty:
                ax.axis('off')
                continue
            
            sns.barplot(
                data=subset,
                x="Model Class",
                y="RMSE",
                hue="Model Class",
                palette="Set2",
                errorbar=None,
                ax=ax,
                legend=False
            )
            
            # Lines
            ax.axhline(y=1.0, color='black', linestyle='--', linewidth=1.5, label='Ref 1.0')
            
            # Baseline Dummy Score (If Available)
            # Assuming baseline is roughly consistent per target, take mean
            baseline_val = subset['Baseline RMSE'].mean()
            if not np.isnan(baseline_val):
                ax.axhline(y=baseline_val, color='red', linestyle=':', linewidth=2, label=f'Baseline {baseline_val:.2f}')
                
            ax.set_ylim(0, y_limit)
            ax.set_title(target)
            ax.set_xlabel("")
            ax.set_ylabel("RMSE")
            
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
            ax.legend()
            
            for container in ax.containers:
                ax.bar_label(container, fmt='%.2f')
                
        for j in range(i + 1, len(axes)):
            axes[j].axis('off')

        plt.suptitle("Model Comparison per Target", fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(os.path.join(output_dir, "detailed_model_comparison.png"))
        plt.close()
        print("Saved detailed model comparison plots.")

    def generate_aggregated_report(self, output_dir: Optional[str] = None) -> None:
        """Saves the aggregated dataframe to CSV."""
        if not self.data:
            self.load_data()
            
        if output_dir is None:
            output_dir = os.path.join(self.root_path, "comparison_results")
        os.makedirs(output_dir, exist_ok=True)
        
        df = pd.DataFrame(self.data)
        csv_path = os.path.join(output_dir, "aggregated_metrics.csv")
        df.to_csv(csv_path, index=False)
        print(f"Saved aggregated metrics to {csv_path}")

if __name__ == "__main__":
    # Example usage assume running from src root or similar
    # Adjust path as needed during execution
    results_path = "/workspace/automatic_assessment/experiment_results/multitarget_full_comparison"
    if os.path.exists(results_path):
        comp = ExperimentComparison(results_path)
        comp.plot_multitarget_comparison()
        comp.plot_model_comparison()
        comp.plot_augmentation_ratio_comparison()
        comp.generate_aggregated_report()
    else:
        print(f"Path {results_path} does not exist.")
