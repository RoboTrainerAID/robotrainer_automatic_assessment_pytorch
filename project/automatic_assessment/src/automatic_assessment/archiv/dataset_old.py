import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset as TorchDataset
from sklearn.preprocessing import StandardScaler

from automatic_assessment.framework.data.preprocessing import Preprocessor
from automatic_assessment.framework.data.augmentation import user_augmentation
from automatic_assessment.framework.dimred.lars import select_multitarget_top_sources_lars, get_source_name
# from automatic_assessment.lasso import select_top_n_features_lasso

class Dataset(TorchDataset):
    def __init__(self, sampling_frequency, folder_path, recreate):
        """
        Original Dataset Structure:
        1. timeseries_df
            - Shape: (n_samples * n_paths * variable_timesteps, n_timeseries)
            - Values: (28 * 20 * max. 79, 50) = (44240, 50)
        2. path_related_df
            - Shape: (n_samples * n_paths, n_path_features)
            - Values: (28 * 20, 914) = (560, 914)
        3. user_related_df
            - Shape: (n_samples, n_user_features)
            - Values: (28, 3)
        4. target_df
            - Shape: (n_samples, n_targets)
            - Values: (28, 14)
        """
        self.folder_path = folder_path
        if recreate:
            print(f"Creating dataset at {folder_path} (Sampling: {sampling_frequency} Hz)...")
            prep = Preprocessor(sampling_frequency, folder_path)
            self.timeseries_df, self.path_related_df, self.user_related_df, self.target_df = prep.get_data()
        else:
            print(f"Loading existing dataset from {folder_path}...")
            self.timeseries_df = pd.read_csv(f"{folder_path}/timeseries.csv")
            self.path_related_df = pd.read_csv(f"{folder_path}/path.csv")
            self.user_related_df = pd.read_csv(f"{folder_path}/user.csv")
            self.target_df = pd.read_csv(f"{folder_path}/target.csv")

    def create_augmented_and_filtered_dataset(self, selected_targets, augmentation_ratio: int, apply_lasso: bool, subfolder: str, apply_lars: bool = False, max_n_timeseries: int = 20, max_n_path_features: int = 20, max_n_extracted_ts_features: int = 500):
        """
        Filters targets and optionally augments the dataset.
        Modifies internal dataframes in-place.
        """
        # 1. Filter Targets
        # TODO: Later use target_cluster.py to select best targets
        # We ensure 'user' is kept for ID matching
        self.target_df = Preprocessor.filter_dataframe(self.target_df, selected_targets + ['user'])

        # 2. Multi-Target LARS Feature Selection
        if apply_lars:
            # --- 2a. Time-Series Source Selection ---
            # Identify columns derived from time-series features (candidates for TS filtering)
            actual_path_cols = Preprocessor.PATH_RELATED_COLS + Preprocessor.TASK_DIFFICULTY_COLS
            ts_feature_cols = [c for c in self.path_related_df.columns 
                               if c not in actual_path_cols and c not in ['user', 'path']]
            
            print(f"Applying LARS for Time-Series selection (Top {max_n_timeseries} sources out of {len(ts_feature_cols)} TS-derived feats)...")

            # Flatten to (User x Features) and align with Targets
            X_ts, y_aligned = self._flatten_and_align(self.path_related_df, ts_feature_cols, self.target_df)
            
            # Scale
            scaler_X, scaler_y = StandardScaler(), StandardScaler()
            X_scaled = pd.DataFrame(scaler_X.fit_transform(X_ts), columns=X_ts.columns)
            y_scaled = pd.DataFrame(scaler_y.fit_transform(y_aligned), columns=y_aligned.columns)
            
            # Select Sources matching timeseries columns
            selected_sources = select_multitarget_top_sources_lars(
                X_scaled, y_scaled, 
                possible_sources=list(self.timeseries_df.columns), 
                top_n_sources=max_n_timeseries
            )
            self.timeseries_df = Preprocessor.filter_dataframe(self.timeseries_df, selected_sources)

            # --- 2b. Path-Related Feature Selection ---
            # Now we look at the path_related_df itself to reduce its width.
            # Candidates: All numeric columns except indices
            print(f"Applying LARS for Path Feature selection (Top {max_n_path_features} features)...")
            
            # Flatten to (User x Features)
            X_path, _ = self._flatten_and_align(self.path_related_df, actual_path_cols, self.target_df)
            
            # Scale Reuse y_scaled
            X_path_scaled = pd.DataFrame(scaler_X.fit_transform(X_path), columns=X_path.columns)
            
            selected_path_feats = select_multitarget_top_sources_lars(
                X_path_scaled, y_scaled, 
                possible_sources=actual_path_cols, 
                top_n_sources=max_n_path_features
            )

            selected_path_extracted_ts_feats = select_multitarget_top_sources_lars(
                X_scaled, y_scaled, 
                possible_sources=ts_feature_cols, 
                top_n_sources=max_n_extracted_ts_features
            )

            self.path_related_df = Preprocessor.filter_dataframe(self.path_related_df, selected_path_feats + selected_path_extracted_ts_feats)


        # 3. Data Augmentation
        # ratio > 0 implies we want to add synthetic data
        if augmentation_ratio > 0:
            original_path_related_cols = Preprocessor.PATH_RELATED_COLS
            
            # Call the modular function
            # It returns the COMBINED (Original + Augmented) dataframes
            self.target_df, self.user_related_df, self.path_related_df, self.timeseries_df = user_augmentation(
                self.target_df,
                self.user_related_df,
                self.path_related_df,
                self.timeseries_df,
                augmentation_ratio,
                original_path_related_cols
            )
            
        # 4. Ensure int columns
        # re-casting types if jittering introduced floats to integer columns
        # self.user_related_df['user'] = self.user_related_df['user'].astype(int)
        # self.path_related_df['user'] = self.path_related_df['user'].astype(int)

        # 5. LASSO Feature Selection
        # TODO: Later use LASSO on the path-related + user_level features
        # if apply_lasso:
        #     select_top_n_features_lasso()

        # 6. Save Processed Dataset
        self._save_csv_dataset(subfolder=subfolder)

    def _flatten_and_align(self, feature_df: pd.DataFrame, feature_cols: list[str], target_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Helper to flatten path-level DataFrame to User-level (wide format) and align with targets.
        Used for LARS feature selection where X and y must match on dimension 0 (User).
        """
        # Filter specific feature columns into sub DF (keeps 'user', 'path' indices)
        # We need 'path' to pivot.
        cols = list(set(feature_cols + ['path']))
        subset = Preprocessor.filter_dataframe(feature_df, cols, "Feature DF")
        
        # Pivot: index='user', columns='path', values=features
        pivoted = subset.pivot(index='user', columns='path', values=feature_cols)
        
        # Flatten MultiIndex columns: "feature_name" + "_path_" + "path_id"
        pivoted.columns = [f"{col[0]}_path_{col[1]}" for col in pivoted.columns]
        pivoted = pivoted.reset_index()
        
        # Merge with targets to ensure alignment
        aligned = pd.merge(pivoted, target_df, on='user', how='inner')
        
        # Separate X and y
        y_cols = [c for c in target_df.columns if c != 'user']
        X = aligned.drop(columns=['user'] + y_cols)
        y = aligned[y_cols]
        
        return X, y
            

    def _save_csv_dataset(self, subfolder: str):
        """
        Saves the 4 datasets to CSV files in the specified folder.
        """
        full_path = os.path.join(self.folder_path, subfolder)
        if not os.path.exists(full_path):
            os.makedirs(full_path)

        self.timeseries_df.to_csv(os.path.join(full_path, "timeseries.csv"), index=False)
        self.path_related_df.to_csv(os.path.join(full_path, "path.csv"), index=False)
        self.user_related_df.to_csv(os.path.join(full_path, "user.csv"), index=False)
        self.target_df.to_csv(os.path.join(full_path, "target.csv"), index=False)
        print(f"\nDatasets saved to: {full_path}")

    def get_all_level_dataset(self, padding: bool = True) -> tuple[tuple[np.ndarray | list, np.ndarray, np.ndarray], np.ndarray, np.ndarray, dict]:
        """
        Returns datasets with all levels.

        Handling Variable Timesteps:
        If padding=True:
            Time-series data (x_ts) is zero-padded to the maximum number of timesteps found across all user/path combinations.
            Since x_ts is initialized with zeros, any timestep index beyond the actual length of a specific time-series remains 0.
        If padding=False:
            Returns a nested list structure for x_ts where dimensions vary.

        X tuple: (x_ts, x_path, x_user)
            1. x_ts: Time-Series Dataset
                - Shape (padded): (n_samples, n_paths, n_timeseries, max_timesteps)
                - Shape (unpadded): (n_timeseries, variable_timesteps)
                - Values: (28, 20, 50, variable_timesteps)
            2. x_path: Path-Level Dataset
                - Shape: (n_samples, n_paths, n_path_features)
                - Values: (28, 20, )
            3. x_user: User-Level Dataset
                - Shape: (n_samples, n_user_features)
                - Values: (28, 3)

        y: Targets
            - Shape: (n_samples, n_targets)
            - Values: (28, 14)
        """
        print(f"\nConstructing All-Level Dataset (padding={padding})...")

        # 1. Identify Indexing
        users = sorted(self.user_related_df['user'].unique())
        n_users = len(users)
        
        unique_paths = sorted(self.path_related_df['path'].unique())
        n_paths = len(unique_paths)

        # 2. X User
        user_df = self.user_related_df.set_index('user').reindex(users)
        x_user = user_df.values.astype(np.float32)
        user_feat_names = list(user_df.columns)

        # 3. X Path
        path_feat_cols = [c for c in self.path_related_df.columns if c not in ['user', 'path']]
        n_path_feats = len(path_feat_cols)
        
        # Sort/Index to ensure shape (n_users, n_paths, n_features)
        # Using multi-index reindexing to align strictly by (user, path)
        path_df = self.path_related_df.set_index(['user', 'path'])
        mi = pd.MultiIndex.from_product([users, unique_paths], names=['user', 'path'])
        path_df_aligned = path_df.reindex(mi).fillna(0) # Fill missing paths with 0 if necessary
        
        # Reshape directly: (n_users * n_paths, n_features) -> (n_users, n_paths, n_features)
        x_path = path_df_aligned[path_feat_cols].values.astype(np.float32)
        x_path = x_path.reshape(n_users, n_paths, n_path_feats)
        
        # 4. X Time-Series
        ts_feat_cols = [c for c in self.timeseries_df.columns if c not in ['user', 'path', 'time']]
        n_ts_feats = len(ts_feat_cols)
        
        # Determine max dimension for static array allocation
        # This defines the fixed size for the time dimension. Shorter series will be padded.
        max_timesteps = self.timeseries_df.groupby(['user', 'path']).size().max()
        
        if padding:
            # Shape: (n_users, n_paths, n_features, timesteps)
            # Initialize with zeros. This automatically handles zero-padding for sequences shorter than max_timesteps.
            x_ts = np.zeros((n_users, n_paths, n_ts_feats, max_timesteps), dtype=np.float32)
        else:
            # List of lists to hold variable length arrays
            # Default to empty array (features, 0) if path is missing
            x_ts = [[np.zeros((n_ts_feats, 0), dtype=np.float32) for _ in range(n_paths)] for _ in range(n_users)]
        
        user_map = {u: i for i, u in enumerate(users)}
        path_map = {p: i for i, p in enumerate(unique_paths)}
        
        # Sort to ensure data chunks are contiguous if iterating
        ts_df_sorted = self.timeseries_df.sort_values(by=['user', 'path', 'time'])
        
        for (u, p), group in ts_df_sorted.groupby(['user', 'path']):
            if u in user_map and p in path_map:
                u_idx = user_map[u]
                p_idx = path_map[p]
                
                vals = group[ts_feat_cols].values.T # (features, timesteps)
                
                if padding:
                    length = vals.shape[1]
                    # Assign to correct slice. Elements from length to max_timesteps remain 0 (padding).
                    x_ts[u_idx, p_idx, :, :length] = vals
                else:
                    x_ts[u_idx][p_idx] = vals

        # 5. Y Targets
        target_df = self.target_df.set_index('user').reindex(users)
        y = target_df.values.astype(np.float32)
        
        feature_names = {
            'ts': ts_feat_cols,
            'path': path_feat_cols,
            'user': user_feat_names
        }

        print(f"  x_ts shape: {x_ts.shape if padding else 'List (Variable Length)'} (Zero-padded to max length: {max_timesteps} if padding=True)")
        print(f"  x_path shape: {x_path.shape}")
        print(f"  x_user shape: {x_user.shape}")
        print(f"  y shape: {y.shape}")

        return (x_ts, x_path, x_user), y, np.array(users), feature_names

    def get_user_level_time_series_dataset(self) -> tuple[np.ndarray, np.ndarray, list, list]:
        """
        Total Samples: 140 (with Augmentation x4), 28 (original Users)
        Features (Paths * Timeseries) per Sample: 20 * 50 = 1000
        Timesteps per Timeseries: Max Timesteps with 0-Padding (all 79s)
        Targets per Sample: 4

        X shape: (n_samples, n_features, max_timesteps)
        Y shape: (n_samples, n_targets)
        """
        print("\nConstructing User-Level Time-Series Dataset (Stacked Paths)...")
        
        # 1. Prepare Metadata
        ts_df = self.timeseries_df.sort_values(by=['user', 'path', 'time'])
        
        users = sorted(ts_df['user'].unique())
        unique_paths = sorted(ts_df['path'].unique())
        
        # Identify feature columns
        exclude_cols = {'user', 'path', 'time'}
        raw_feature_cols = [c for c in ts_df.columns if c not in exclude_cols]
        
        n_samples = len(users)
        n_paths = len(unique_paths)
        n_raw_features = len(raw_feature_cols)
        n_features = n_paths * n_raw_features
        
        # Determine Max Timesteps
        path_group_sizes = ts_df.groupby(['user', 'path']).size()
        max_timesteps = path_group_sizes.max()
        
        print(f"  Samples (Users): {n_samples}")
        print(f"  Paths per User: {n_paths}")
        print(f"  Raw Features per Path: {n_raw_features}")
        print(f"  Total Features (Channels): {n_features}")
        print(f"  Max Timesteps: {max_timesteps}")
        
        # 2. Initialize X: (N, C, L) -> (n_samples, n_features, max_timesteps)
        X = np.zeros((n_samples, n_features, max_timesteps), dtype=np.float32)
        
        # Map helpers
        user_map = {u: i for i, u in enumerate(users)}
        path_map = {p: i for i, p in enumerate(unique_paths)}
        
        # 3. Fill Data
        # Ensure we iterate in a way that aligns with maps
        for (u, p), group in ts_df.groupby(['user', 'path']):
            if u not in user_map or p not in path_map:
                continue
                
            u_idx = user_map[u]
            p_idx = path_map[p]
            
            # Start index for this path's features in the channel dimension
            feat_start = p_idx * n_raw_features
            feat_end = feat_start + n_raw_features
            
            # Get values (T, F), transpose to (F, T) for assigning to (C, L)
            vals = group[raw_feature_cols].values.T 
            
            current_time = vals.shape[1]
            
            # Fill slice: X[u_idx, feature_slice, time_slice]
            X[u_idx, feat_start:feat_end, :current_time] = vals
            
        # 4. Prepare Y
        if 'user' in self.target_df.columns:
            target_df_indexed = self.target_df.set_index('user')
        else:
            target_df_indexed = self.target_df
            
        y_aligned = target_df_indexed.reindex(users)
        y = y_aligned.values.astype(np.float32)
        
        # 5. Generate Feature Names
        feature_names = []
        for p in unique_paths:
            for f in raw_feature_cols:
                feature_names.append(f"{p}_{f}")
                
        print(f"  X Shape: {X.shape}")
        print(f"  y Shape: {y.shape}")
        
        return X, y, users, feature_names

    def get_path_level_time_series_dataset(self, padding: bool = True) -> tuple[np.ndarray | list[np.ndarray], np.ndarray, list, list]:
        """
        Creates a dataset using time-series data.
        Each sample X represents all the time-series data for one path. There are multiple paths per user.

        Total Samples: 560 (Users * Paths)
        Features (Timeseries) per Sample: 50
        Duration per Timeseries: Varibale (2-79s) or with 0-Padding (all 79s)
        Targets per Sample: 14

        X shape: If padding=True, (n_samples, max_timesteps, n_features)
                 If padding=False (n_samples, variable_timesteps, n_features)
        Y shape: (n_samples, n_targets)
        
        Args:
            padding (bool): If True, pads sequences to maximum length and returns a 3D tensor.
                            If False, returns a list of variable-length 2D arrays.

        Returns:
            X: If padding=True, shape is (n_samples, max_timesteps, n_features).
               If padding=False, list of n_samples arrays, where each is (timesteps, n_features).
            y_vals: (n_samples, n_targets)
            users: list of user IDs corresponding to samples
            feature_names: list of feature names
        """
        print(f"\nConstructing Path-Level Time-Series Dataset (Padding={padding})...")
        
        # 1. Prepare Data
        # Sort by user, path, time to ensure data contiguity and deterministic order
        ts_df = self.timeseries_df.sort_values(by=['user', 'path', 'time'])
        
        # Identify non-feature columns
        exclude_cols = {'user', 'path', 'time'}
        feature_cols = [c for c in ts_df.columns if c not in exclude_cols]
        n_features = len(feature_cols)
        
        # 2. Determine Shapes
        # Group by sample identifier (user, path)
        grouped = ts_df.groupby(['user', 'path'])
        
        # Get count
        n_samples = len(grouped)
        
        # Stats on sequence lengths
        lengths = grouped.size()
        min_len = lengths.min()
        max_len = lengths.max()
        
        print(f"  Samples (Paths): {n_samples}")
        print(f"  Features: {n_features}")
        print(f"  Sequence Lengths: Min={min_len}, Max={max_len}")
        
        # 3. Initialize Containers
        if padding:
            # X: (samples, max_timesteps, features)
            # Initialize with zeros for padding
            X_data = np.zeros((n_samples, max_len, n_features), dtype=np.float32)
        else:
            # X: List of arrays (variable length)
            X_data = []
        
        # y: (samples, targets)
        if 'user' in self.target_df.columns:
            target_df_indexed = self.target_df.set_index('user')
        else:
            target_df_indexed = self.target_df
        
        # Ensure we only check columns that are actual targets (if target_df_indexed still has index set correctly)
        n_targets = target_df_indexed.shape[1]
        y_vals = np.zeros((n_samples, n_targets), dtype=np.float32)
        
        users_list = []
        
        # 4. Fill Data
        # Iterating groupby to fill data
        idx = 0
        for (u_id, _), group in grouped:
            # Extract features
            vals = group[feature_cols].values.astype(np.float32)
            
            if padding:
                seq_len = len(vals)
                # Fill tensor with data, leaving rest as zeros (padding)
                X_data[idx, :seq_len, :] = vals
            else:
                # No padding, strictly variable length
                X_data.append(vals)
            
            # Fill y (replicate user target for this path)
            if u_id in target_df_indexed.index:
                # Use .values ensures we get the array
                y_vals[idx] = target_df_indexed.loc[u_id].values
            
            users_list.append(u_id)
            idx += 1
            
        print(f"  X Tensor Shape: {X_data.shape}")
        print(f"  y Shape: {y_vals.shape}")
        
        return X_data, y_vals, users_list, feature_cols

    def get_user_level_dataset(self) -> tuple[np.ndarray, np.ndarray, list, list]:
        """
        Creates a dataset using path-level features merged with user info.

        Total Samples: 28 (Users)
        All Paths flattened and concated into single feature vector per user.
        Paths per Sample: 20
        Features per Path: ~900
        Total Features per Sample: ~18000
        Targets per Sample: 14

        X shape: (n_samples, n_features * n_paths)
        Y shape: (n_samples, n_targets)

        Returns:
            X_flat: (n_users, n_features_total)
            y_vals: (n_users, n_targets)
            users: list of user IDs
            feature_names: list of feature names
        """
        print("\nConstructing User-Level Dataset...")
        
        # 1. Pivot Path Data to flatten paths per user
        # Exclude 'user' from values, use it as index
        # 'path' is the column to pivot on
        path_df = self.path_related_df.copy()
        
        # Pivot: index=user, columns=path, values=all other columns
        pivoted = path_df.pivot(index='user', columns='path')
        
        # Flatten MultiIndex columns: feature_name + _path_ + path_id
        pivoted.columns = [f"{col[0]}_path_{col[1]}" for col in pivoted.columns]
        
        # Reset index to make 'user' a column again for merging
        pivoted = pivoted.reset_index()
        
        # 2. Merge User Info
        # user_related_df has 'user', 'sex_value', 'age'
        user_df = self.user_related_df.copy()
        X_df = pd.merge(pivoted, user_df, on='user', how='inner')
        
        # Sort by user to ensure alignment
        X_df = X_df.sort_values('user')
        users = X_df['user'].tolist()

        X_df.to_csv("debug_path_level_X.csv", index=False)
        
        # 3. Construct X (Features)
        # Drop 'user' column
        feature_cols = [c for c in X_df.columns if c != 'user']
        X_vals = X_df[feature_cols].values.astype(np.float32)
        
        print(f"  Users: {len(users)}")
        print(f"  Total Features per User: {X_vals.shape[1]}")
        
        # 4. Construct Y (Targets)
        target_df = self.target_df.copy()
        if 'user' in target_df.columns:
            target_df = target_df.set_index('user')
            
        # Reindex to match user order in X
        y_aligned = target_df.reindex(users)
        y_vals = y_aligned.values.astype(np.float32)
        
        return X_vals, y_vals, users, feature_cols

    def get_path_level_dataset(self) -> tuple[np.ndarray, np.ndarray, list, list]:
        """
        Creates a dataset using path-level features merged with user info.
        Expands the dataset so each (user, path) combination is a sample.
        Targets are repeated for each path of the same user.

        Total Samples: 560 (Users * Paths)
        Features per Sample: 914
        Targets per Sample: 14
        X shape: (n_samples, n_features)
        Y shape: (n_samples, n_targets)
        
        Returns:
            X_vals: (n_samples, n_features)
            y_vals: (n_samples, n_targets)
            users: list of user IDs (length n_samples)
            feature_names: list of feature names
        """
        print("\nConstructing Path-Level Dataset (Expanded per Path)...")
        
        # 1. Start with Path Data
        path_df = self.path_related_df.copy()
        
        # 2. Merge User Info
        user_df = self.user_related_df.copy()
        # Merge on 'user'
        merged_df = pd.merge(path_df, user_df, on='user', how='inner')
        
        # 3. Merge Targets
        target_df = self.target_df.copy()
        target_cols = [c for c in target_df.columns if c != 'user']
        
        # Merge targets (inner join ensures we only keep users with targets)
        full_df = pd.merge(merged_df, target_df, on='user', how='inner')
        
        # Sort for determinism
        full_df = full_df.sort_values(['user', 'path'])
        
        # 4. Extract Data
        users = full_df['user'].tolist()
        
        # Features: All columns except metadata (user, path) and targets
        exclude_cols = ['user', 'path'] + target_cols
        feature_cols = [c for c in full_df.columns if c not in exclude_cols]
        
        X_vals = full_df[feature_cols].values.astype(np.float32)
        y_vals = full_df[target_cols].values.astype(np.float32)
        
        print(f"  Total Samples: {len(users)} (Users * Paths)")
        print(f"  Features per Sample: {X_vals.shape[1]}")
        print(f"  Targets per Sample: {y_vals.shape[1]}")
        
        return X_vals, y_vals, users, feature_cols

    def get_flattened_dataset(self):
        """
        Creates a flattened dataset using time-series data.
        Each sample X represents all the time-series data for one user. There are multiple paths per user.

        Total Samples: 28 (Users)
        Features (Max Timesteps * Paths *  Features) per Sample: 79 * 20 * 50 = 79000
        Duration per Timeseries: Max Timesteps with 0-Padding (all 79s)
        Targets per Sample: 14

        X shape: (n_samples, max_timesteps, n_features)
        Y shape: (n_samples, n_targets)
        Returns:
            X_flat: (n_users, n_features_total)
            y_vals: (n_users, n_targets)
            users: list of user IDs
            flat_feature_names: list of feature names for flattened vector
        """
        print("\nConstructing Flattened Time-Series Dataset...")
        
        # 1. Define Dimensions
        # Users
        users = sorted(self.timeseries_df['user'].unique())
        n_users = len(users)
        
        # Paths
        # Assuming 'path' column contains identifiers.
        unique_paths = sorted(self.timeseries_df['path'].unique())
        n_paths = len(unique_paths)
        
        # Features
        # Exclude indices
        feature_cols = [c for c in self.timeseries_df.columns if c not in ['user', 'path', 'time']]
        n_features = len(feature_cols)
        
        # Time
        # Calculate max timesteps per path
        path_stats = self.timeseries_df.groupby(['user', 'path']).size()
        max_timesteps = path_stats.max()
        
        print(f"  Users: {n_users}")
        print(f"  Fixed Number of Paths: {n_paths} (IDs: {unique_paths})")
        print(f"  Max Timesteps per Path: {max_timesteps} (Padding applied to this length)")
        print(f"  Features per Timestep: {n_features}")
        
        total_vector_len = n_paths * max_timesteps * n_features
        print(f"  -> Total Feature Vector Length per User: {total_vector_len}")

        # 2. Construct X (Features)
        # Shape: (n_users, n_paths, max_timesteps, n_features)
        # Initialize with zeros for padding
        X_tensor = np.zeros((n_users, n_paths, max_timesteps, n_features), dtype=np.float32)
        
        user_to_idx = {u: i for i, u in enumerate(users)}
        path_to_idx = {p: i for i, p in enumerate(unique_paths)}
        
        # Iterate over groups to fill data
        # Sort by time to ensure order
        timeseries_df = self.timeseries_df.sort_values(['user', 'path', 'time'])
        
        # Using groupby is safer than iterating rows
        for (u, p), group in timeseries_df.groupby(['user', 'path']):
            if u in user_to_idx and p in path_to_idx:
                u_idx = user_to_idx[u]
                p_idx = path_to_idx[p]
                
                vals = group[feature_cols].values
                length = min(len(vals), max_timesteps)
                
                # Fill data (rest remains 0)
                X_tensor[u_idx, p_idx, :length, :] = vals[:length]
                
        # Flatten: (n_users, n_paths * max_timesteps * n_features)
        X_flat = X_tensor.reshape(n_users, -1)
        
        # 3. Construct Y (Targets)
        if 'user' in self.target_df.columns:
            target_df = self.target_df.set_index('user')
            
        # Reindex to match user order in X
        y_aligned = target_df.reindex(users)
        y_vals = y_aligned.values.astype(np.float32)
        
        # Generate feature names for flattened vector
        flat_feature_names = []
        for p in unique_paths:
            for t in range(max_timesteps):
                for f in feature_cols:
                    flat_feature_names.append(f"path_{p}_t{t}_{f}")

        return X_flat, y_vals, users, flat_feature_names

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx])
    
    
class DatasetConv1s(Dataset):
    def __init__(self, path: str = "/data/dataset_conv@1s", recreate: bool = False):
        super().__init__(sampling_frequency=1, folder_path=path, recreate=recreate)

class DatasetFreq1hzAugmentedx4(Dataset):
    def __init__(self, path: str = "/data/dataset_conv@1s/augmentedx4", recreate: bool = False):
        super().__init__(sampling_frequency=1, folder_path=path, recreate=recreate)

class DatasetFreq2hz(Dataset):
    def __init__(self, path: str = "/data/dataset_conv@2hz", recreate: bool = False):
        super().__init__(sampling_frequency=2, folder_path=path, recreate=recreate)

class DatasetFreq2hzAugmentedx4(Dataset):
    def __init__(self, path: str = "/data/dataset_conv@2hz/augmentedx4", recreate: bool = False):
        super().__init__(sampling_frequency=2, folder_path=path, recreate=recreate)

if __name__ == "__main__":
    dataset = DatasetFreq2hz(recreate=False)
    dataset.create_augmented_and_filtered_dataset(
        selected_targets=['Balance Test', 'Single Leg Stance', 'Hand Grip Right', 'Throwing Beanbag at Target'],
        augmentation_ratio=4,
        apply_lasso=False,
        subfolder="augmentedx4",
        apply_lars=True,
        max_n_timeseries=40,
        max_n_path_features=10,
        max_n_extracted_ts_features=100
    )
    dataset.get_all_level_dataset(padding=True)
