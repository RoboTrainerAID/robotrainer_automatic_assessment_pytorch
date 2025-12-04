import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class SmartImputer(BaseEstimator, TransformerMixin):
    """
    Imputer that can use global mean, user-specific mean, or path-specific mean.
    Also drops features with missing values exceeding a threshold.
    """
    # Class-level set to track warnings across instances (e.g. during CV tuning)
    _warned_configs = set()

    def __init__(self, strategy='global_mean', max_nan_threshold=0.15, user_col='user', path_col='path'):
        self.strategy = strategy
        self.max_nan_threshold = max_nan_threshold
        self.user_col = user_col
        self.path_col = path_col
        self.fill_values_ = {}
        self.global_means_ = None
        self.cols_to_drop_ = []
        self.feature_names_in_ = []
        self.feature_names_out_ = []

    def fit(self, X: pd.DataFrame, y=None):
        if not isinstance(X, pd.DataFrame):
             raise ValueError("SmartImputer requires input X to be a pandas DataFrame.")

        self.feature_names_in_ = X.columns.tolist()
        n_samples = X.shape[0]
        
        # Identify numeric columns
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        
        # 1. Identify columns to drop
        self.cols_to_drop_ = []
        for col in numeric_cols:
            # Skip user/path columns from dropping logic, we handle them explicitly
            if col == self.user_col or col == self.path_col:
                continue
                
            nan_count = X[col].isna().sum()
            if nan_count / n_samples > self.max_nan_threshold:
                self.cols_to_drop_.append(col)
                
                # Check if we already warned about this column with this threshold
                config_key = col
                if config_key not in SmartImputer._warned_configs:
                    print(f"Warning: Dropping feature '{col}' due to > {self.max_nan_threshold:.1%} missing values.")
                    SmartImputer._warned_configs.add(config_key)

        # 2. Calculate Global Means (for fallback and global strategy)
        # Only for columns we keep
        cols_to_keep = [c for c in numeric_cols if c not in self.cols_to_drop_]
        self.global_means_ = X[cols_to_keep].mean()

        # 3. Calculate Specific Means
        if self.strategy == 'user_mean' and self.user_col in X.columns:
            self.fill_values_ = X.groupby(self.user_col)[cols_to_keep].mean()
        elif self.strategy == 'path_mean' and self.path_col in X.columns:
            self.fill_values_ = X.groupby(self.path_col)[cols_to_keep].mean()
            
        # Determine output features
        self.feature_names_out_ = [c for c in self.feature_names_in_ 
                                   if c not in self.cols_to_drop_]
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        if not isinstance(X, pd.DataFrame):
             raise ValueError("SmartImputer requires input X to be a pandas DataFrame.")
             
        X_copy = X.copy()
        
        # Drop columns
        if self.cols_to_drop_:
            X_copy.drop(columns=self.cols_to_drop_, inplace=True, errors='ignore')
            
        # Columns to impute
        numeric_cols = X_copy.select_dtypes(include=[np.number]).columns
        cols_to_impute = [c for c in numeric_cols if c not in [self.user_col, self.path_col]]

        # Optimization: Vectorized filling instead of column-loop
        if self.strategy == 'user_mean' and self.user_col in X_copy.columns:
            # 1. Align means to the rows based on user ID
            # self.fill_values_ is indexed by user_id
            means_aligned = self.fill_values_.reindex(X_copy[self.user_col])
            
            # 2. Reset index to match X_copy for correct alignment during fillna
            means_aligned.index = X_copy.index
            
            # 3. Fill only the relevant columns
            # Intersect columns to avoid KeyErrors if fill_values_ has different cols
            common_cols = list(set(cols_to_impute) & set(means_aligned.columns))
            if common_cols:
                X_copy[common_cols] = X_copy[common_cols].fillna(means_aligned[common_cols])
            
        elif self.strategy == 'path_mean' and self.path_col in X_copy.columns:
            means_aligned = self.fill_values_.reindex(X_copy[self.path_col])
            means_aligned.index = X_copy.index
            common_cols = list(set(cols_to_impute) & set(means_aligned.columns))
            if common_cols:
                X_copy[common_cols] = X_copy[common_cols].fillna(means_aligned[common_cols])

        # Fallback to global
        if self.global_means_ is not None:
            # Ensure we only use columns that exist in both
            common_cols = list(set(cols_to_impute) & set(self.global_means_.index))
            if common_cols:
                X_copy[common_cols] = X_copy[common_cols].fillna(self.global_means_[common_cols])
        
        return X_copy.values

    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names_out_)