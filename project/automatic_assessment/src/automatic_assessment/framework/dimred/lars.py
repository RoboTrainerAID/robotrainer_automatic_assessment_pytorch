import numpy as np
import pandas as pd
from sklearn.linear_model import lars_path
from collections import Counter
from typing import List

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

# ================= USAGE EXAMPLE =================
# assuming X_train (scaled) and y_train exist

# top_sources = select_top_sources_lars(X_train, y_train, top_k_sources=20)
# print(top_sources)