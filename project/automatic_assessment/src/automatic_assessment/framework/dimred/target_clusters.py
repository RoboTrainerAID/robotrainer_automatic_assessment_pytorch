import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.ensemble import RandomForestRegressor
from typing import List, Dict, Tuple, Any

from automatic_assessment.framework.data.dataset import DatasetConv1s

def analyze_target_structure(target_df: pd.DataFrame, correlation_threshold: float=1.5) -> Dict[int, List[str]]:
    """
    Step 1: Analyze correlations and group similar targets.
    """
    print(f"\n--- Analyzing Target Structure (Clustering) ---")
    
    # 1. Clean Data
    targets = target_df.set_index("user")
    targets = targets.dropna(axis=1, how='all').fillna(targets.mean())
    
    # 2. Correlation Matrix
    corr = targets.corr(method='spearman')
    
    # Hierarchical Clustering
    Z = linkage(corr, 'ward')
    
    # Plot Dendrogram (Optional, helpful for paper/thesis)
    plt.figure(figsize=(10, 5))
    dendrogram(Z, labels=corr.columns, leaf_rotation=90)
    plt.title("Clinical Target Clustering")
    plt.tight_layout()
    
    # Save figure to disk instead of attempting to show it interactively
    plot_path = "target_clustering_dendrogram.png"
    plt.savefig(plot_path)
    print(f"Dendrogram plot saved to '{plot_path}'")
    plt.close()
    
    # Form Clusters
    # fcluster determines cluster labels based on distance threshold
    # Distance = 1 - Correlation approx.
    # t=1 - correlation_threshold
    cluster_labels = fcluster(Z, t=4, criterion='maxclust')
    
    clusters = {}
    for target_name, cluster_id in zip(corr.columns, cluster_labels):
        if cluster_id not in clusters: clusters[cluster_id] = []
        clusters[cluster_id].append(target_name)
        
    print(f"\nFound {len(clusters)} distinct target clusters (max. cluster threshold = 4):")
    for cid, t_list in clusters.items():
        print(f"  Cluster {cid}: {t_list}")
        
    return clusters

def robust_permutation_test(X: np.ndarray, y: np.ndarray, groups: np.ndarray, n_permutations: int=50) -> Tuple[float, float, float]:
    """
    Step 2: Rigorous 'Y-Scrambling' Test.
    Returns a p-value for learnability.
    """
    # Ensure inputs are numpy arrays
    if hasattr(X, 'values'):
        X = X.values
    if hasattr(y, 'values'):
        y = y.values
    if hasattr(groups, 'values'):
        groups = groups.values
    
    logo = LeaveOneGroupOut()
    model = RandomForestRegressor(
        n_estimators=100, 
        max_depth=4, 
        random_state=42, 
        n_jobs=-1, 
        min_samples_leaf=3,
        max_features="sqrt")
    
    # 1. Calculate True Score (sRMSE)
    true_rmses = []
    y_std = y.std() + 1e-8
    
    for tr, te in logo.split(X, y, groups):
        model.fit(X[tr], y[tr])
        preds = model.predict(X[te])
        true_rmses.append(np.sqrt(np.mean((y[te] - preds)**2)))
        
    true_score = np.mean(true_rmses) / y_std
    
    # 2. Permutation Loop
    perm_scores = []
    y_shuffled = y.copy() # standardized numpy array
    
    for i in range(n_permutations):
        np.random.shuffle(y_shuffled) # Shuffle targets globally (break X-Y link)
        
        fold_rmses = []
        for tr, te in logo.split(X, y_shuffled, groups):
            model.fit(X[tr], y_shuffled[tr])
            preds = model.predict(X[te])
            fold_rmses.append(np.sqrt(np.mean((y_shuffled[te] - preds)**2)))
            
        perm_scores.append(np.mean(fold_rmses) / y_std)
    
    # 3. Calculate P-Value
    # P-value = (Number of shuffled scores BETTER than true score) / Total Permutations
    # "Better" means LOWER sRMSE
    n_better_random = sum(np.array(perm_scores) <= true_score)
    p_value = (n_better_random + 1) / (n_permutations + 1)
    
    return true_score, np.mean(perm_scores), p_value

# ================= INTEGRATION EXAMPLE =================

def select_best_targets(X, target_df, groups):
    """
    The Method:
    Train your model on features $X$ and true targets $Y$. 
    Record the Score (e.g., 1$R^2$ or sRMSE)
    Scramble the target vector 3$Y$ randomly (break the relationship between user and score)
    Train the model again on $X$ and $Y_{shuffled}$. Record the Score.
    Repeat 100 times.Selection
    Rule: A target is "Real" only if the True Score is better than 95% of the Shuffled Scores (p-value < 0.05).
    Why it's better: It proves the model found a physical signal, not just a statistical fluke.
    """
    # 1. Structure Analysis (Grouping)
    clusters = analyze_target_structure(target_df)
    
    # 2. For each cluster, find the "Representative" Target
    # The one with the best p-value
    
    final_targets = []
    
    print("\n--- Running Permutation Tests on Clusters ---")
    
    for cid, t_list in clusters.items():
        print(f"\nEvaluating Cluster {cid} {t_list}...")
        
        best_p = 1.0
        best_t = None
        best_srmse = 999.0
        
        for t in t_list:
            y = target_df.set_index("user").reindex(groups)[t]
            if y.isna().any(): continue
            
            score, perm_mu, p_val = robust_permutation_test(X, y, groups)
            
            print(f"  > {t:<20} | sRMSE: {score:.3f} | Random: {perm_mu:.3f} | p-val: {p_val:.3f}")
            
            # We pick the target with the lowest p-value (most statistically significant)
            if p_val < best_p:
                best_p = p_val
                best_t = t
                best_srmse = score
            elif p_val == best_p and score < best_srmse:
                # Tie-breaker: Lower error
                best_t = t
                best_srmse = score
        
        if best_t and best_p < 0.10: # Only keep if reasonably significant
            print(f"  *** Selected Leader: {best_t} (p={best_p:.3f}) ***")
            final_targets.append(best_t)
        else:
            print(f"  --- Cluster Discarded (No target met significance threshold) ---")

    print(f"\nFinal Selected Targets: {final_targets}")
    return final_targets

# Example usage:
if __name__ == "__main__":
    IN_DIR = "/data/dataset_conv@2hz"

    dataset = DatasetConv1s(path=IN_DIR, recreate=False)
    X, y, users, feature_names = dataset.get_path_level_dataset()
    users = np.array(users)
    target_df = dataset.target_df

    select_best_targets(X, target_df, users)

