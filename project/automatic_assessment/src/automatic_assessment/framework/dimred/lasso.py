import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV, MultiTaskLassoCV
from typing import List, Tuple

# from automatic_assessment.framework.data.dataset import DatasetConv1s

def select_features(X_train_mean: np.ndarray, y_train_scaled: np.ndarray, n_features: int = 20) -> tuple[np.ndarray, np.ndarray]:
        """LASSO Feature Selection: Returns top n_features."""
        # Increase max_iter to prevent ConvergenceWarning on complex datasets
        # Default is usually 1000, 10000+ is safer for convergence.
        max_iter = 1000
        
        if y_train_scaled.shape[1] > 1:
            # Multi-output: use MultiTaskLasso to find shared features
            lasso = MultiTaskLassoCV(
                cv=5, 
                selection='random', 
                max_iter=max_iter
            ).fit(X_train_mean, y_train_scaled)
            coef_importance = np.sum(np.abs(lasso.coef_), axis=0)
        else:
            # Single-target
            lasso = LassoCV(
                cv=5, 
                max_iter=max_iter
            ).fit(X_train_mean, y_train_scaled.ravel())
            coef_importance = np.abs(lasso.coef_)
        
        # Sort by importance (descending) and take top n_features
        importances = np.argsort(coef_importance)[::-1]
        selected_idx = importances[:n_features]
            
        return selected_idx, coef_importance

def select_top_n_features_lasso(
    X: np.ndarray, 
    y: np.ndarray, 
    feature_names: List[str], 
    n: int, 
    alpha: float = 0.3
) -> Tuple[np.ndarray, List[str]]:
    """
    Selects the top n features based on Lasso regression importance scores.
    Importance is defined as the mean absolute coefficient across targets.
    
    Args:
        X: Feature matrix (n_samples, n_features)
        y: Target matrix (n_samples, n_targets)
        feature_names: List of feature names
        n: Number of features to select
        alpha: Lasso regularization strength
        
    Returns:
        X_filtered: Feature matrix with only the top n features (n_samples, n)
        selected_features: List of names of the top n features
    """
    # 2. SCALE THE DATA (Crucial for LASSO)
    # Scale Features
    scaler_x = StandardScaler()
    X_scaled = scaler_x.fit_transform(X)

    # Scale Targets
    # YES, target scaling influences Lasso significantly, especially in multi-output settings.
    # If targets have different scales (e.g., 0-1 vs 50-300), the coefficients for the larger target
    # will be much larger. Since Lasso penalizes the sum of absolute coefficients (L1 norm) with a 
    # shared alpha, the penalty will disproportionately affect the smaller target's coefficients 
    # (potentially zeroing them out) while having little effect on the larger target's coefficients.
    # Scaling ensures the penalty applies fairly and coefficients are comparable for ranking.
    scaler_y = StandardScaler()
    y_scaled = scaler_y.fit_transform(y)

    # 3. Initialize and fit LASSO
    # alpha is the penalty strength (lambda). 
    lasso = Lasso(alpha=alpha) 
    lasso.fit(X_scaled, y_scaled)

    # 4. Get the list of selected features and calculate importance scores
    # LASSO drives coefficients of useless features to exactly 0.
    # The "Score" is the magnitude of the coefficient.
    # For multi-output y, lasso.coef_ is (n_targets, n_features). 
    # We use the mean absolute coefficient across targets as the ranking score.
    coefs = lasso.coef_

    if coefs.ndim == 1:
        # Single target case
        feature_scores = np.abs(coefs)
    else:
        # Multi-target case: select feature if it's non-zero for at least one target
        # Score: Mean absolute coefficient across all targets
        feature_scores = np.mean(np.abs(coefs), axis=0)

    # Create a DataFrame for ranking
    df_ranking = pd.DataFrame({
        'Feature': feature_names,
        'Importance_Score': feature_scores,
        'Original_Index': range(len(feature_names))
    })

    # Sort by importance
    df_ranking = df_ranking.sort_values(by='Importance_Score', ascending=False)

    # Report amount after alpha lasso
    n_nonzero = (df_ranking['Importance_Score'] > 0).sum()
    print(f"Features with non-zero coefficients after Lasso (alpha={alpha}): {n_nonzero} / {len(feature_names)}")

    # Mandatory features to always include
    # mandatory_feats = ["sex_value", "age"]
    mandatory_feats = []
    
    # Split into mandatory and candidates
    # We check which mandatory features are actually present in the feature list
    is_mandatory = df_ranking['Feature'].isin(mandatory_feats)
    df_mandatory = df_ranking[is_mandatory]
    df_candidates = df_ranking[~is_mandatory]
    
    # Calculate how many slots are left for candidates
    n_mandatory_found = len(df_mandatory)
    n_candidates_to_select = max(0, n - n_mandatory_found)
    
    # Select top candidates
    df_selected_candidates = df_candidates.head(n_candidates_to_select)
    
    # Combine: Mandatory first, then top candidates
    top_n_df = pd.concat([df_mandatory, df_selected_candidates])
    
    # Get indices and names
    selected_indices = top_n_df['Original_Index'].values
    selected_features = top_n_df['Feature'].tolist()
    
    # Filter X (return original unscaled values)
    X_filtered = X[:, selected_indices]

    print("\n--- Feature Importance Report ---")
    print("Score definition: Mean Absolute Coefficient magnitude across targets.")
    print(f"Mandatory features included: {df_mandatory['Feature'].tolist()}")
    print(f"Showing top {min(10, len(top_n_df))} out of {len(top_n_df)} selected features:\n")
    print(top_n_df[['Feature', 'Importance_Score']].head(10).to_string(index=False))
    
    return X_filtered, selected_features

if __name__ == "__main__":
    dataset = DatasetConv1s(recreate=False)
    X, y, users, feature_names = dataset.get_user_level_dataset()
    
    # Example usage
    print(f"Original X shape: {X.shape}")
    X_new, selected_feats = select_top_n_features_lasso(X, y, feature_names, n=20, alpha=0.3)
    print(f"Filtered X shape: {X_new.shape}")
