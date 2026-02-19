import numpy as np
import pandas as pd
from typing import List
from scipy.stats import pearsonr

def select_features_by_correlation(X: np.ndarray, y: np.ndarray, correlation_threshold: float = 0.1) -> List[int]:
    """
    Selects features that have a mean absolute correlation with the targets above a threshold.
    
    Logic:
    1. Calculate Pearson correlation between each feature and each target.
    2. For each feature, take the maximum absolute correlation across all targets.
        (Alternative: mean absolute correlation. Using MAX ensures a feature is kept if it's 
         highly relevant for at least one target).
    3. Filter features where max_abs_corr >= correlation_threshold.
    
    Args:
        X: (N_samples, N_paths, N_features) or (N_samples, N_features) numpy array.
        y: (N_samples, N_targets) numpy array.
        correlation_threshold: Features with max absolute correlation below this are dropped.
        
    Returns:
        List[int] of indices of selected features.
    """
    
    # Handle X shape: Flatten Paths if 3D
    if X.ndim == 3:
        # X is (N, Paths, Features). 
        # Strategy: Flatten Paths into Samples dimension to treat each path as an instance
        N_samples, N_paths, N_feats = X.shape
        X_flat = X.reshape(N_samples * N_paths, N_feats)
        
        # Prepare y: Repeat for each path to match X rows
        y_flat = np.repeat(y, N_paths, axis=0) # (N*P, T)
    else:
        # Assume (N, F)
        X_flat = X
        y_flat = y

    # Ensure no NaNs - simple imputation for correlation check
    if np.isnan(X_flat).any():
        col_mean = np.nanmean(X_flat, axis=0)
        inds = np.where(np.isnan(X_flat))
        X_flat[inds] = np.take(col_mean, inds[1])
        
    n_features = X_flat.shape[1]
    n_targets = y_flat.shape[1]
    
    # Check for NaNs in y as well
    valid_mask = ~np.isnan(y_flat).any(axis=1)
    if not valid_mask.all():
        X_flat = X_flat[valid_mask]
        y_flat = y_flat[valid_mask]
    
    # If no data left, return all features (fallback) or empty
    if X_flat.shape[0] < 2:
        return list(range(n_features))

    max_correlations = np.zeros(n_features)

    # Vectorized correlation calculation is tricky with numpy for all pairs without big memory.
    # Iterating features is safer for memory.
    
    # Pre-compute y standard deviations and centered y for speed
    y_centered = y_flat - np.mean(y_flat, axis=0)
    y_norm = np.linalg.norm(y_centered, axis=0)
    # Avoid div by zero
    y_norm[y_norm == 0] = 1.0 

    for f_idx in range(n_features):
        x_col = X_flat[:, f_idx]
        
        # Check variance
        if np.std(x_col) == 0:
            max_correlations[f_idx] = 0.0
            continue
            
        x_c = x_col - np.mean(x_col)
        x_n = np.linalg.norm(x_c)
        if x_n == 0:
            x_n = 1.0
            
        # Compute correlation vector (1 feature vs all targets)
        # corr = (x . y) / (|x| * |y|)
        dot_products = np.dot(y_centered.T, x_c) # (N_targets,)
        corrs = dot_products / (y_norm * x_n)
        
        # Take the maximum absolute correlation this feature has with ANY target
        max_abs_corr = np.max(np.abs(corrs))
        max_correlations[f_idx] = max_abs_corr

    # Filter
    selected_indices = [i for i, c in enumerate(max_correlations) if c >= correlation_threshold]
    
    # Ensure at least 1 feature is selected to prevent downstream dimension errors
    if not selected_indices:
        # Fallback: select feature with highest max correlation
        best_idx = int(np.argmax(max_correlations))
        if max_correlations[best_idx] > 0:
            selected_indices = [best_idx]
        else:
            # If absolutely no correlation (all 0), pick 0th feature just to have something non-empty
            selected_indices = [0]
        
    return selected_indices
