"""
Statistical feature extraction for timeseries data.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Callable
from pathlib import Path

# To support type hinting without circular imports at runtime if needed, 
# but here we assume the Loader is available or we use Any.
from automatic_assessment.dataset.timeseries_loader import PathData

class TimeseriesFeatures:
    """
    Handles feature extraction from loaded timeseries data.
    """

    def __init__(self, dataset: List[PathData]) -> None:
        self.dataset = dataset
        self.df = pd.DataFrame()
        self.feature_fns = self._default_features()

    @staticmethod
    def _default_features() -> Dict[str, Callable[[np.ndarray], float]]:
        """Returns a dictionary of default statistical functions."""
        return {
            "mean": np.mean,
            "std": np.std,
            "min": np.min,
            "max": np.max,
            "median": np.median,
            "iqr": lambda v: np.percentile(v, 75) - np.percentile(v, 25),
            "rms": lambda v: np.sqrt(np.mean(v ** 2)),
            "peak_to_peak": lambda v: np.max(v) - np.min(v),
        }

    def extract_features(self) -> pd.DataFrame:
        """
        Generates a DataFrame with statistical features and scalar values.
        Structure: user, path, source_timeseries, ...features...
        """
        # Assume standard processing index 2 (Value) for timeseries
        COL_VALUE = 2 
        rows = []

        for pd_data in self.dataset:
            # 1. Process standard Timeseries
            for ts_name, arr in pd_data.timeseries.items():
                if arr.ndim < 2 or arr.shape[1] <= COL_VALUE:
                    continue
                    
                values = arr[:, COL_VALUE]
                if len(values) == 0:
                    continue
                
                row = {
                    "user_id": pd_data.user_id,
                    "path_id": pd_data.path_id,
                    "source_timeseries": ts_name,
                }

                for fname, func in self.feature_fns.items():
                    try:
                        row[fname] = float(func(values))
                    except Exception:
                        row[fname] = np.nan
                
                rows.append(row)

            # 2. Process Scalar Features (load as 'value' or similar)
            for s_name, s_val in pd_data.scalar_features.items():
                row = {
                    "user_id": pd_data.user_id,
                    "path_id": pd_data.path_id,
                    "source_timeseries": s_name,
                    "value": s_val
                    # Statistical columns will be NaN for these rows
                }
                rows.append(row)

        self.df = pd.DataFrame(rows)
        return self.df

    def save_features_to_csv(self, output_path: Path) -> None:
        """Saves the extracted features to a CSV file."""
        if self.df.empty:
            print("Warning: Feature DataFrame is empty. Nothing to save.")
            return

        # Ensure directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.df.to_csv(output_path, index=False)
        print(f"Features saved to {output_path}")
