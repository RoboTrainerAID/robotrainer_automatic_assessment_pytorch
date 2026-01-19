import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from typing import Union, List

def calculate_metrics(y_true: Union[np.ndarray, List[np.ndarray]], 
                      y_pred: Union[np.ndarray, List[np.ndarray]], 
                      prefix: str = "test") -> dict:
    """
    Calculates regression metrics (RMSE, MAE, R2) for multi-output data.
    Automatically concatenates list inputs.
    
    Args:
        y_true: Ground truth array or list of arrays.
        y_pred: Predictions array or list of arrays.
        prefix: String prefix for dictionary keys (e.g., 'test', 'val')
        
    Returns:
        Dictionary with keys. Includes per-target metrics and averages.
    """
    if isinstance(y_true, list):
        y_true = np.concatenate(y_true, axis=0)
    if isinstance(y_pred, list):
        y_pred = np.concatenate(y_pred, axis=0)

    metrics = {}
    n_targets = y_true.shape[1]
    
    # Calculate residuals
    residuals = y_true - y_pred
    mse_per_target = np.mean(residuals**2, axis=0)
    rmse_per_target = np.sqrt(mse_per_target)
    mae_per_target = np.mean(np.abs(residuals), axis=0)
    
    # Global Averages (Uniform average across targets)
    metrics[f"{prefix}_rmse_mean"] = float(np.mean(rmse_per_target))
    metrics[f"{prefix}_mae_mean"] = float(np.mean(mae_per_target))
    metrics[f"{prefix}_r2_mean"] = float(r2_score(y_true, y_pred)) 
    
    # Per-target metrics
    for i in range(n_targets):
        metrics[f"{prefix}_rmse_target_{i}"] = float(rmse_per_target[i])
        metrics[f"{prefix}_mae_target_{i}"] = float(mae_per_target[i])
        metrics[f"{prefix}_r2_target_{i}"] = float(r2_score(y_true[:, i], y_pred[:, i]))
        
    return metrics
