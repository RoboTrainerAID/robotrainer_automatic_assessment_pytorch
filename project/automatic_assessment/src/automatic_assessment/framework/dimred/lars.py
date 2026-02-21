import numpy as np
import pandas as pd
import warnings
from sklearn.linear_model import lars_path
from sklearn.exceptions import ConvergenceWarning
from collections import Counter
from typing import List
from tqdm import tqdm

# Helper to map feature -> source
def get_source_name(feat_name, possible_sources: List[str]) -> str:
    # Sort by length descending so we match "Force_L" before "Force" if both exist
    for src in sorted(possible_sources, key=len, reverse=True):
        if src in feat_name:
            return src
    
    print(f"Warning: Could not map feature '{feat_name}' to any known source.")
    return None


def select_multitarget_top_sources_lars(X: pd.DataFrame, target_df: pd.DataFrame, possible_sources: List[str], top_n_sources: int = 20) -> List[str]:
    """
    Runs LARS on each target independently.
    Ranks sources by how many targets selected them.
    Returns the 'top_n_sources' globally.
    """
    source_votes = Counter()
    
    print(f"--- Running Multi-Target LARS on {target_df.shape[1]} targets ---")
    
    # 1. Loop through each target
    for target_col in target_df.columns:
        if target_col == 'user': continue
            
        y = target_df[target_col].values
        
        # skip if target has NaNs
        if np.isnan(y).any(): continue
            
        # Run LARS Path
        # method='lasso' is standard LARS-LASSO
        _, active_indices, _ = lars_path(X.values, y, method='lasso')
        
        # Find the Top K unique sources for THIS target
        found_sources = set()
        for idx in active_indices:
            feat_name = X.columns[idx]
            src = get_source_name(feat_name, possible_sources)
            
            if src not in found_sources:
                found_sources.add(src)
                # VOTE for this source
                source_votes[src] += 1
            
            # Stop once we have enough for this target
            if len(found_sources) >= top_n_sources:
                break
        
        print(f"  > Target '{target_col}': identified {len(found_sources)} key sources.")

    # 2. Global Ranking
    # Sort by number of votes (descending)
    most_common = source_votes.most_common(top_n_sources)
    
    print(f"\n--- Global Top {top_n_sources} Sources ---")
    selected_sources = []
    for rank, (src, count) in enumerate(most_common):
        print(f"  {rank+1}. {src} (Selected by {count} targets)")
        selected_sources.append(src)
        
    return selected_sources

# ================= USAGE =================
# X_scaled must be standardized!
# final_sources = select_multitarget_top_sources_lars(X_scaled, target_df, top_n_sources=20)


def select_top_sources_lars(X: pd.DataFrame, y: pd.Series, top_k_sources: int = 20) -> List[str]:
    """
    Selects top K unique time-series SOURCES using the LARS regularization path.
    
    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix (columns must follow naming convention like 'Source_Mean')
    y : pd.Series
        Target variable
    top_k_sources : int
        Number of unique sources to select
        
    Returns
    -------
    selected_sources : list
        List of the top K source names
    """
    # 2. Compute LARS Path
    # This returns the coefficients for *every* step of alpha.
    # The 'active' list tells us the order indices were added.
    print(f"Computing LARS path for {X.shape[1]} features...")
    
    # alphas, active, coefs = lars_path(X.values, y.values, method='lasso')
    # Note: sklearn's lars_path return signature is: (alphas, active, coefs)
    # 'active' is a list of indices in the order they entered the model.
    _, active_indices, _ = lars_path(X.values, y.values, method='lasso')
    
    # 3. Iterate through entry order to find unique sources
    selected_sources = set()
    ordered_sources = [] # To keep rank order
    
    print("Scanning LARS entry order...")
    
    feature_names = X.columns
    
    for idx in active_indices:
        # Get feature name from index
        feat_name = feature_names[idx]
        
        # Identify its source
        source = get_source_name(feat_name)
        
        # Check if source is new
        if source not in selected_sources:
            selected_sources.add(source)
            ordered_sources.append(source)
            print(f"  > Source #{len(selected_sources)} found: {source} (triggered by {feat_name})")
        
        # Stop condition
        if len(selected_sources) >= top_k_sources:
            break
            
    print(f"\nSelection Complete. Found {len(ordered_sources)} unique sources.")
    return ordered_sources

def select_multitarget_top_features_lars(X: np.ndarray, y: np.ndarray, top_n_features: int = 20) -> List[int]:
    """
    Runs LARS on each target independently on RAW TENSORS (as numpy).
    
    Selection strategy (two-phase):
      Phase 1 — Per-target guarantee:
        Reserve half the budget (N//2) and distribute equally across targets.
        For each target, take its top (N//2 // n_targets) features by LARS entry order.
        This guarantees that every target's most important features are represented.
      Phase 2 — Global vote fill-up:
        Fill the remaining slots (up to N) from a global ranking of all features
        scored by how many targets selected them via LARS.
    
    X: (N_samples, N_paths, N_features) or (N_samples, N_features)
    y: (N_samples, N_targets)
    
    Returns: List[int] of indices of selected features.
    """
    
    # Handle X shape
    if X.ndim == 3:
        N_samples, N_paths, N_feats = X.shape
        X_flat = X.reshape(N_samples * N_paths, N_feats)
        y_flat = np.repeat(y, N_paths, axis=0)  # (N*P, T)
    else:
        X_flat = X
        y_flat = y

    total_features = X_flat.shape[1]
    n_targets = y_flat.shape[1]
    
    # Budget
    per_target_budget = max(1, round((top_n_features / 2) / n_targets), 0)  # Ensure at least 1 per target
    
    # Storage
    per_target_top = {}      # {target_idx: [ordered list of feature indices]}
    feature_votes = Counter()
    
    # -------------------------------------------------------
    # Run LARS for each target
    # -------------------------------------------------------
    for t_idx in range(n_targets):
        target_vals = y_flat[:, t_idx]
        
        if np.isnan(target_vals).any():
            per_target_top[t_idx] = []
            continue
            
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=ConvergenceWarning)
                _, active_indices, _ = lars_path(X_flat, target_vals, method='lasso')
            
            # Store ordered active indices for this target
            per_target_top[t_idx] = list(active_indices)
            
            # Count votes from a reasonable depth
            vote_depth = min(top_n_features//2, len(active_indices))
            for feat_idx in active_indices[:vote_depth]:
                feature_votes[feat_idx] += 1
                
        except Exception as e:
            per_target_top[t_idx] = []

    # -------------------------------------------------------
    # Phase 1: Per-target guaranteed selection
    # -------------------------------------------------------
    selected = set()
    
    for t_idx in range(n_targets):
        top_for_target = per_target_top.get(t_idx, [])
        added = 0
        for feat_idx in top_for_target:
            if added >= per_target_budget:
                break
            selected.add(feat_idx)
            added += 1
    
    phase1_count = len(selected)
    
    # -------------------------------------------------------
    # Phase 2: Fill up from global vote ranking
    # -------------------------------------------------------
    remaining_slots = top_n_features - len(selected)
    
    if remaining_slots > 0:
        # Sort by votes descending, then by index for stability
        ranked_by_votes = sorted(feature_votes.keys(), 
                                  key=lambda idx: (-feature_votes[idx], idx))
        
        for feat_idx in ranked_by_votes:
            if len(selected) >= top_n_features:
                break
            if feat_idx not in selected:
                selected.add(feat_idx)
    
    # Sort for stable ordering
    selected_indices = sorted(selected)
    
    # -------------------------------------------------------
    # Reporting
    # -------------------------------------------------------
    phase2_count = len(selected_indices) - phase1_count
    
    # tqdm.write(f"[LARS Selection] Total Features: {total_features}, Selected: {len(selected_indices)}.")
    # tqdm.write(f"  Phase 1 (per-target top {per_target_budget}): {phase1_count} unique features")
    # tqdm.write(f"  Phase 2 (global vote fill-up):  {phase2_count} additional features")
    
    # Show per-target contribution
    # for t_idx in range(n_targets):
    #     top_for_target = per_target_top.get(t_idx, [])[:per_target_budget]
    #     tqdm.write(f"  Target {t_idx}: guaranteed features = {top_for_target}")
    
    # Show top features by vote
    # tqdm.write(f"\n  Top features by vote count:")
    # for rank, (idx, count) in enumerate(feature_votes.most_common(min(70, len(selected_indices)))):
    #     marker = "*" if idx in selected_indices else " "
    #     tqdm.write(f"   {marker} Feature {idx}: {count} votes")

    return selected_indices


# ================= USAGE EXAMPLE =================
# assuming X_train (scaled) and y_train exist

# top_sources = select_top_sources_lars(X_train, y_train, top_k_sources=20)
# print(top_sources)