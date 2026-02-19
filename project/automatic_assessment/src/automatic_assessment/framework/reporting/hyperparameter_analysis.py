import os
import yaml
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Any, Optional
from collections import Counter

class HyperparameterAnalysis:
    def __init__(self, experiments_root_path: str):
        """
        Initialize the analysis module.
        
        Args:
            experiments_root_path: Path to the directory containing experiment folders.
        """
        self.root_path = experiments_root_path
        self.data: List[Dict[str, Any]] = []

    def _parse_folder_name(self, folder_name: str) -> str:
        """Extracts model name from folder name (assuming YYYYMMDD_HHMMSS_ModelName format)."""
        parts = folder_name.split('_')
        if len(parts) >= 3:
            # Join everything after the timestamp parts (Year, Month, Day, Hour, Min, Sec usually imply 2 parts for date/time if combined or separated)
            # Actually standard format often YYYYMMDD_HHMMSS_ModelName -> 2 splits for timestamp
            # But the user example is 20260217_222900_MLP_SharedEncoder
            # Split results: ['20260217', '222900', 'MLP', 'SharedEncoder']
            # We want 'MLP_SharedEncoder'
            return "_".join(parts[2:])
        return "UnknownModel"

    def load_data(self) -> pd.DataFrame:
        """Scans all experiment folders and aggregates hyperparameter data."""
        self.data = []
        
        if not os.path.exists(self.root_path):
            print(f"Path {self.root_path} does not exist.")
            return pd.DataFrame()

        # Iterate over directories in root path
        for folder_name in sorted(os.listdir(self.root_path)):
            folder_path = os.path.join(self.root_path, folder_name)
            if not os.path.isdir(folder_path):
                continue
            
            # Paths to expected files
            config_path = os.path.join(folder_path, "config.yaml")
            
            if not os.path.exists(config_path):
                continue
                
            try:
                # Load Config
                with open(config_path, 'r') as f:
                    config_data = yaml.safe_load(f)
                
                # Check for hyperparameters
                # Support both flat structure or nested under 'hyperparameters_fold_0' (from example)
                hyperparameters = {}
                
                if 'hyperparameters_fold_0' in config_data:
                    hyperparameters = config_data['hyperparameters_fold_0']
                elif 'hyperparameters' in config_data:
                    hyperparameters = config_data['hyperparameters']
                
                if not hyperparameters:
                    continue

                # Identify Model
                # Try to find model name in config, fallback to folder name
                model_name = config_data.get('model_name')
                if not model_name:
                    model_name = self._parse_folder_name(folder_name)

                # Flatten Row
                row = {
                    "Folder": folder_name,
                    "Model": model_name,
                    **hyperparameters
                }
                
                self.data.append(row)
                        
            except Exception as e:
                print(f"Error processing {folder_name}: {e}")

        df = pd.DataFrame(self.data)
        return df

    def analyze_hyperparameters(self, output_dir: Optional[str] = None):
        """
        Analyzes hyperparameters per model.
        Generates distributions plots and calculates the average/best configuration.
        """
        df = self.load_data()
        
        if df.empty:
            print("No data found to analyze.")
            return

        if output_dir is None:
            output_dir = os.path.join(self.root_path, "hyperparameter_analysis")
        os.makedirs(output_dir, exist_ok=True)
        
        # Iterate over each model type found
        for model_name, model_df in df.groupby("Model"):
            print(f"\nAnalyzing Model: {model_name} ({len(model_df)} experiments)")
            
            model_out_dir = os.path.join(output_dir, model_name)
            os.makedirs(model_out_dir, exist_ok=True)
            
            # Identify hyperparameter columns (exclude metadata)
            metadata_cols = ["Folder", "Model"]
            param_cols = [c for c in model_df.columns if c not in metadata_cols]
            
            best_config = {}
            
            # For each hyperparameter
            for param in param_cols:
                values = model_df[param].dropna()
                if values.empty:
                    continue
                
                is_numeric = pd.api.types.is_numeric_dtype(values)
                unique_vals = values.nunique()
                
                plt.figure(figsize=(8, 6))
                
                # Plotting Logic
                if is_numeric and unique_vals > 10:
                    # Continuous / High cardinality numeric -> Hist + KDE
                    sns.histplot(values, kde=True, bins=20)
                    plt.title(f"Distribution of {param} ({model_name})")
                    
                    # Store average (mean) for best config
                    best_config[param] = values.mean()
                    
                else:
                    # Categorical or Low dimensionality numeric -> Count Plot
                    # Convert to string to treat as categorical for plotting
                    sns.countplot(x=values, order=values.value_counts().index)
                    plt.title(f"Frequency of {param} ({model_name})")
                    plt.xticks(rotation=45)
                    
                    # Store mode for best config
                    # value_counts().idxmax() returns the most frequent value
                    if not values.empty:
                        best_config[param] = values.mode()[0]
                
                plt.tight_layout()
                plt.savefig(os.path.join(model_out_dir, f"dist_{param}.png"))
                plt.close()
                
            # Save "Average Best" Config to YAML
            best_config_path = os.path.join(model_out_dir, "average_best_config.yaml")
            
            # Convert numpy types to python types for YAML serialization
            best_config_serializable = {}
            for k, v in best_config.items():
                if isinstance(v, (np.integer, np.int64, int)):
                    best_config_serializable[k] = int(v)
                elif isinstance(v, (np.floating, np.float64, float)):
                    best_config_serializable[k] = float(v)
                else:
                    best_config_serializable[k] = v
                    
            with open(best_config_path, 'w') as f:
                yaml.dump(best_config_serializable, f, default_flow_style=False)
            
            print(f"Saved average best config to {best_config_path}")
            
            # Save summary CSV
            summary_stats = model_df[param_cols].describe(include='all')
            summary_stats.to_csv(os.path.join(model_out_dir, "hyperparameter_statistics.csv"))

if __name__ == "__main__":
    # Adjust path as needed
    results_path = "/workspace/automatic_assessment/experiment_results/multi_objective"
    
    if os.path.exists(results_path):
        analysis = HyperparameterAnalysis(results_path)
        analysis.analyze_hyperparameters()
    else:
        print(f"Path {results_path} does not exist.")
