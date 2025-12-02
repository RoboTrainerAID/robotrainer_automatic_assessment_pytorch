import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler



def data_normalization(df):
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    df[numeric_columns] = (df[numeric_columns] - df[numeric_columns].min()) / (df[numeric_columns].max() - df[numeric_columns].min())
    return df


def apply_pca(X, n_components=0.95):
    """
    Apply PCA to the data.
    """

    X = np.array(X)
    print(f"PCA Input shape: {X.shape}")
    
    # Überprüfe Matrix-Größe für LAPACK-Kompatibilität
    total_elements = X.shape[0] * X.shape[1]
    print(f"Total matrix elements: {total_elements}")
    
    if total_elements > 2e9:  # Grenze für LAPACK
        print("Matrix too large for standard PCA. Using TruncatedSVD...")
        
        
        imputer = SimpleImputer(strategy="mean")
        X_imputed = imputer.fit_transform(X)
        
        # Standardization before truncated SVD
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_imputed)
        
        # Specify number of components
        if isinstance(n_components, float):
            n_comp = min(min(X.shape) - 1, 100)  # Maximum 100 components
        else:
            n_comp = min(n_components, min(X.shape) - 1)
            
        svd = TruncatedSVD(n_components=n_comp, random_state=42)
        X_pca = svd.fit_transform(X_scaled)

        # Compute cumulative explained variance
        explained_variance_ratio = svd.explained_variance_ratio_

        print(f"TruncatedSVD: {n_comp} components, Explained Variance: {explained_variance_ratio.sum():.3f}")

    else:
        # Standard PCA
        imputer = SimpleImputer(strategy="mean")
        X_imputed = imputer.fit_transform(X)

        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X_imputed)
        explained_variance_ratio = pca.explained_variance_ratio_

    X_pca_df = pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(X_pca.shape[1])])

    # Scree Plot to visualize explained variance by each principal component
    plt.plot(range(1, len(explained_variance_ratio)+1), explained_variance_ratio.cumsum(), marker='o')
    plt.xlabel("Number of Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.title("Scree Plot")
    plt.show()

    return X_pca_df, explained_variance_ratio


def prepare_input_blocks(df, pipeline_config, mode="row"):
    """
    Prepares X and y for the model.
    
    mode:
        "row"        -> each row = 1 sample
        "user"       -> all rows per user = 1 sample
        "user_path"  -> all rows per user+path = 1 sample
    """
    topics = pipeline_config.topics

    if mode == "row":
        X = df.drop(columns=topics + ["user", "path"], errors="ignore")
        y = df[topics].values
        return X.values, y

    elif mode == "user":
        groups = df.groupby("user")
    elif mode == "user_path":
        groups = df.groupby(["user", "path"])
    else:
        raise ValueError(f"Unknown mode: {mode}")

    X_blocks = []
    y_blocks = []

    for _, group in groups:
        feature_cols = [c for c in group.columns if c not in topics + ["user", "path"]]
        features = group[feature_cols].values.flatten() 
        X_blocks.append(features)

        y_block = group[topics].iloc[0].values
        y_blocks.append(y_block)

    max_len = max(len(x) for x in X_blocks)
    X_padded = np.array([np.pad(x, (0, max_len - len(x)), 'constant') for x in X_blocks])

    return X_padded, np.array(y_blocks)


def ensure_numeric(X):
    if not np.issubdtype(X.dtype, np.number):
        print("X contains non-numeric values. Converting to numeric values...")
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        X = pd.get_dummies(X, drop_first=True)
        X = X.values  

    X = X.astype(float)
    return X
    