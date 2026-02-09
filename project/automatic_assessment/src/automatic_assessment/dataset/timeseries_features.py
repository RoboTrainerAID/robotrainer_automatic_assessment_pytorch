"""
Statistical feature extraction for timeseries data.
"""

import numpy as np
from typing import Dict, List, Any, Callable

def default_features() -> Dict[str, Callable[[np.ndarray], float]]:
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

def extract_features(paths_data: List[Any], feature_fns: Dict[str, Callable] = None) -> List[Dict]:
    """
    Apply statistical functions to the value column of each timeseries in each path.
    :param paths_data: List of PathData objects
    :param feature_fns: Dictionary of name -> function(array1d) -> float
    :return: List of dictionaries (one per path) flattened for DataFrame creation
    """
    if feature_fns is None:
        feature_fns = default_features()
    
    # Assume the config COL_VALUE is 2, or pass it? 
    # Since I don't want to couple this tightly to the config module unless necessary,
    # I'll default to 2 or check if the object has it. 
    # But for simplicity, I'll assume standard processing index 2 (Value).
    COL_VALUE = 2 

    rows = []
    for pd in paths_data:
        row = {
            "user_id": pd.user_id,
            "path_id": pd.path_id,
        }
        
        for ts_name, arr in pd.timeseries.items():
            if arr.ndim < 2 or arr.shape[1] <= COL_VALUE:
                continue
                
            values = arr[:, COL_VALUE]
            if len(values) == 0:
                continue
            
            for fname, func in feature_fns.items():
                try:
                    val = func(values)
                    row[f"{ts_name}_{fname}"] = float(val)
                except Exception:
                    row[f"{ts_name}_{fname}"] = np.nan
        
        rows.append(row)
    return rows
