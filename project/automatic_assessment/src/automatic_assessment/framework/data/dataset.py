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

    def create_augmented_dataset(self, augmentation_ratio: int, subfolder: str):
        """
        Augments the dataset by the specified ratio.
        Modifies internal dataframes in-place.
        """
        print(f"\nCreating augmented dataset with ratio {augmentation_ratio}...")

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

            # Save Processed Dataset
            self._save_csv_dataset(subfolder=subfolder)

    def target_and_feature_selection(self, selected_targets, apply_lars: bool = False, max_n_timeseries: int = 20, max_n_path_features: int = 20, max_n_extracted_ts_features: int = 500):
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

        # 4. Ensure int columns
        # re-casting types if jittering introduced floats to integer columns
        # self.user_related_df['user'] = self.user_related_df['user'].astype(int)
        # self.path_related_df['user'] = self.path_related_df['user'].astype(int)

        # 5. LASSO Feature Selection
        # TODO: Later use LASSO on the path-related + user_level features
        # if apply_lasso:
        #     select_top_n_features_lasso()


    def _flatten_and_align(self, feature_df: pd.DataFrame, feature_cols: list[str], target_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Helper to flatten path-level DataFrame to User-level (wide format) and align with targets.
        Used for LARS feature selection where X and y must match on dimension 0 (User).
        """
        # Check for missing columns explicitly
        missing_cols = [c for c in feature_cols if c not in feature_df.columns]
        if missing_cols:
            raise ValueError(f"The following required columns are missing from the dataframe: {missing_cols}")

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
    # dataset = Dataset(sampling_frequency=2, folder_path="/data/test", recreate=True)
    dataset = DatasetFreq2hz(recreate=False)
    dataset.create_augmented_dataset(augmentation_ratio=4, subfolder="augmentedx4")
    # --> DatasetFreq2hzAugmentedx4
    
    # dataset.target_and_feature_selection(
    #     selected_targets=['Balance Test', 'Single Leg Stance', 'Hand Grip Right', 'Throwing Beanbag at Target'],
    #     apply_lars=True,
    #     max_n_timeseries=40,
    #     max_n_path_features=10,
    #     max_n_extracted_ts_features=100
    # )
    # dataset.get_all_level_dataset(padding=True)
