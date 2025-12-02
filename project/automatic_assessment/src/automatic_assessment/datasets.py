import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split, LeaveOneOut, LeaveOneGroupOut

class Dataset:
    """
    Unified Dataset class handling creation, loading, preprocessing, and splitting.
    Supports 'user', 'path', and time-based aggregation types.
    """
    
    # Raw Data Paths (Constants)
    RAW_TIMESERIES = "/data/KATE_AA_dataset.csv"
    RAW_DEMOGRAPHICS = "/data/demographics.csv"
    RAW_MOTOR_TESTS = "/data/motoric_test.csv"
    RAW_TASK_DIFFICULTY = "/data/task_difficulty.csv"
    
    # Columns to drop during preprocessing
    DROP_COLS = [
        "hrv", "virtual_force_resulting_force_y", 
        "virtual_force_resulting_force_x", 
        "virtual_force_position_x", "virtual_force_position_y"
    ]
    
    # Target columns
    TARGET_COLS = [
        'Balance Test', 'Single Leg Stance', 'Robotrainer Front', 'Robotrainer Left', 
        'Robotrainer Right', 'Hand Grip Left', 'Hand Grip Right', 'Jump & Reach', 
        'Tandem Walk', 'Figure 8 Walk', 'Jumping Sideways', 'Throwing Beanbag at Target',
        'Tapping Test', 'Ruler Drop Test'
    ]

    def __init__(self, dataset_type: str, recreate: bool = False):
        """
        Args:
            dataset_type: 'user', 'path', or a time string like '1s', '100ms'.
            recreate: Whether to force recreation of the CSV file.
        """
        self.type = dataset_type
        self.recreate = recreate
        self.output_path = f"/data/merged_dataset@{self.type}.csv"
        self.df = None
        
        # Load or Create
        self._load_or_create()

    def _load_or_create(self):
        """Checks if dataset exists. If not (or if recreate is True), creates it."""
        if self.recreate or not os.path.exists(self.output_path):
            print(f"Creating dataset at {self.output_path} (Type: {self.type})...")
            self._create_dataset()
        else:
            print(f"Loading existing dataset from {self.output_path}...")
        
        self.df = pd.read_csv(self.output_path)
        self._validate_data()

    def _create_dataset(self):
        """Orchestrates the loading of raw files, merging, cleaning, and saving."""
        # 1. Load Raw
        timeseries_df = pd.read_csv(self.RAW_TIMESERIES)
        demographics_df = pd.read_csv(self.RAW_DEMOGRAPHICS)
        motor_tests_df = pd.read_csv(self.RAW_MOTOR_TESTS)
        task_difficulty_df = pd.read_csv(self.RAW_TASK_DIFFICULTY)

        # 2. Process Time Series (Downsample/Aggregate)
        processed_ts = self._process_timeseries(timeseries_df)

        # 3. Merge
        merged_df = self._merge_datasets(processed_ts, demographics_df, motor_tests_df, task_difficulty_df)

        # 4. Preprocess (Cleaning, Mapping, Dropping)
        # Gender Mapping
        gender_mapping = {'Männlich': 0, 'Weiblich': 1, 'Divers': 2}
        if 'gender' in merged_df.columns:
            merged_df['gender'] = merged_df['gender'].map(gender_mapping).fillna(-1)

        # Drop Columns
        merged_df = merged_df.drop(columns=[c for c in self.DROP_COLS if c in merged_df.columns])

        # 5. Filter Users (Consistency Check)
        user_counts = merged_df['user'].value_counts()
        if not user_counts.empty:
            expected_count = user_counts.mode()[0]
            users_to_drop = user_counts[user_counts != expected_count].index.tolist()
            if users_to_drop:
                print(f"Warning: Dropping users without exactly {expected_count} entries: {users_to_drop}")
                merged_df = merged_df[~merged_df['user'].isin(users_to_drop)]

        # 6. Save
        merged_df.to_csv(self.output_path, index=False)

    def _process_timeseries(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handles the aggregation logic based on self.type."""
        if self.type == 'user':
            return df.groupby('user').mean(numeric_only=True).reset_index()
        elif self.type == 'path':
            return df.groupby(['user', 'path']).mean(numeric_only=True).reset_index()
        else:
            # Time-based
            if 'time' not in df.columns: raise KeyError("Missing 'time' column")
            df['time'] = pd.to_datetime(df['time'], unit='ns')
            df = df.set_index('time')
            return df.groupby(['user', 'path']).resample(self.type).mean(numeric_only=True).reset_index()

    def _merge_datasets(self, timeseries_df, demographics_df, motor_test_df, task_df):
        """Merges static and task difficulty data."""
        merged = pd.merge(timeseries_df, demographics_df, on='user', how='left')
        merged = pd.merge(merged, motor_test_df, on='user', how='left')

        # Task Difficulty Merge Logic
        if self.type != 'user' and 'path' in merged.columns:
            merged['path'] = merged['path'].astype(int)
            if 'path' in task_df.columns:
                task_df['path'] = task_df['path'].astype(int)
                merged = pd.merge(merged, task_df, on='path', how='left')
        
        return merged

    def _validate_data(self):
        """Basic validation after loading."""
        print(f"Dataset Loaded. Shape: {self.df.shape}. Users: {self.df['user'].nunique()}")

    def get_features_targets(self):
        """Separates X, y, and user groups."""
        y = self.df[self.TARGET_COLS].values
        feature_cols = [c for c in self.df.columns if c not in self.TARGET_COLS and c != 'user']
        X = self.df[feature_cols].values
        users = self.df['user'].values
        return X, y, users, self.TARGET_COLS

    def get_train_test_split(self, test_size=4, random_state=0):
        """
        Splits data ensuring all records of a specific user stay in the same set.
        Returns (X_train, X_test, y_train, y_test, users_train, users_test)
        """
        X, y, users, _ = self.get_features_targets()
        unique_users = np.unique(users)
        
        train_users, test_users = train_test_split(unique_users, test_size=test_size, random_state=random_state)
        
        train_mask = np.isin(users, train_users)
        test_mask = np.isin(users, test_users)
        
        return (
            X[train_mask], X[test_mask],
            y[train_mask], y[test_mask],
            users[train_mask], users[test_mask]
        )

    @property
    def has_multiple_rows_per_user(self):
        """Returns True if the dataset contains multiple rows per user (requires aggregation)."""
        return self.type != 'user'

    def get_cv_strategy(self):
        """Returns the Cross-Validation object (e.g., LOOCV or LOGO)."""
        if self.has_multiple_rows_per_user:
            return LeaveOneGroupOut()
        else:
            return LeaveOneOut()

    def get_fit_params(self, X, y, groups):
        """Returns a dictionary of parameters to pass to .fit() (e.g., groups)."""
        if self.has_multiple_rows_per_user:
            return {'groups': groups}
        else:
            return {}

    def get_cv_splitter(self, X, y, groups):
        """Returns the generator for cross-validation splitting."""
        cv = self.get_cv_strategy()
        if self.has_multiple_rows_per_user:
            return cv.split(X, y, groups=groups)
        else:
            return cv.split(X)