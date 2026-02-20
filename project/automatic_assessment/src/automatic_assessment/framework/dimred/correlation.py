import numpy as np
import pandas as pd
from typing import List
from tqdm import tqdm

def select_features_by_correlation(X: np.ndarray, y: np.ndarray, correlation_threshold: float = 0.1) -> List[int]:
    """
    Selects features based on correlation with targets using a balanced strategy.
    
    Logic:
    1. Filter: Identify N features that have at least one target correlation > threshold.
    2. Distribute: Divide N slots among targets (e.g., N=40, Targets=8 -> 5 slots/target).
    3. Select: Pick top 5 features for Target A, top 5 for Target B, etc.
    4. Fill: If duplicates occur (Set < N), fill remaining slots with next best global features.
    
    Args:
        X: (N_samples, N_paths, N_features) or (N_samples, N_features) numpy array.
        y: (N_samples, N_targets) numpy array.
        correlation_threshold: Cutoff.
        
    Returns:
        List[int] of indices of selected features.
    """
    X_in = X
    y_in = y

    # 1. Flatten if 3D (Samples, Paths, Features)
    if X.ndim == 3:
        N_samples, N_paths, N_feats = X.shape
        # Treat each path as a sample for correlation analysis
        X_flat = X.reshape(N_samples * N_paths, N_feats)
        # Repeat Y to match X rows
        y_flat = np.repeat(y, N_paths, axis=0) 
    else:
        X_flat = X
        y_flat = y

    n_features = X_flat.shape[1]
    n_targets = y_flat.shape[1]
    
    # Safety Check
    if X_flat.shape[0] < 2:
        return list(range(n_features))

    # 2. Compute Correlation Matrix (Features x Targets)
    # Standard manual calculation for speed and control
    # Normalize X and Y first
    X_mean = X_flat - X_flat.mean(axis=0)
    y_mean = y_flat - y_flat.mean(axis=0)
    
    # Avoid division by zero
    X_std = X_flat.std(axis=0)
    X_std[X_std == 0] = 1.0
    
    y_std = y_flat.std(axis=0)
    y_std[y_std == 0] = 1.0
    
    X_norm = X_mean / X_std
    y_norm = y_mean / y_std
    
    # Correlation = (X_norm.T @ y_norm) / N
    N = X_flat.shape[0]
    corr_matrix = np.abs(np.dot(X_norm.T, y_norm) / N) # Shape: (n_features, n_targets)

    # 3. Determine 'Desired Count' (Features passing global threshold)
    max_corr_per_feature = np.max(corr_matrix, axis=1) # Shape: (n_features,)
    
    # Indices of features passing threshold
    valid_mask = max_corr_per_feature >= correlation_threshold
    passing_indices = np.where(valid_mask)[0]
    desired_count = len(passing_indices)
    
    # Fallback if too few features pass
    if desired_count < 1:
        print(f"[Correlation] No features met threshold {correlation_threshold}. Max found: {max_corr_per_feature.max():.4f}")
        # Select at least top 1 to prevent crash
        best_idx = np.argmax(max_corr_per_feature)
        return [int(best_idx)]

    # 4. Per-Target Selection
    # How many features per target? 
    # Ceiling division ensuring coverage
    k_per_target = int(np.ceil(desired_count / n_targets))
    
    selected_indices = set()
    
    # Select Top K for each target
    for t_idx in range(n_targets):
        # Get correlations for this target
        t_corrs = corr_matrix[:, t_idx]
        # Get top K indices
        # argsort is ascending, so take last k
        top_k_indices = np.argsort(t_corrs)[-k_per_target:]
        
        # Add to set (deduplicates automatically)
        for idx in top_k_indices:
            # Only add if it actually passed the global threshold? 
            # Often better to respect the 'Best per target' even if slightly below threshold 
            # IF we must reach 'desired_count'. 
            # But strict logic says: only features passing threshold.
            if valid_mask[idx]:
                selected_indices.add(idx)
    
    # 5. Fill Up
    # We might have fewer than 'desired_count' because of duplicates (overlap between targets)
    current_count = len(selected_indices)
    
    if current_count < desired_count:
        fill_needed = desired_count - current_count
        
        # Sort ALL passing features by their global max correlation
        # Filter strictly those passing threshold
        sorted_candidates = np.argsort(max_corr_per_feature)[::-1] # Descending
        
        fill_count = 0
        for idx in sorted_candidates:
            if fill_count >= fill_needed:
                break
            
            if idx not in selected_indices and valid_mask[idx]:
                selected_indices.add(idx)
                fill_count += 1
                
    final_selection = sorted(list(selected_indices))

    tqdm.write(f"[Correlation Selection] Threshold: {correlation_threshold}, Total selected: {len(final_selection)} / {n_features}. per target: {k_per_target}")

    # print(f"[Correlation Selection] Threshold: {correlation_threshold}")
    # print(f"  > Features passing threshold (Desired Count): {desired_count} / {n_features}")
    # print(f"  > Selection logic: Top {k_per_target} per target (Total targets: {n_targets})")
    # print(f"  > Unique features selected: {len(final_selection)}")
    
    return [int(i) for i in final_selection]
