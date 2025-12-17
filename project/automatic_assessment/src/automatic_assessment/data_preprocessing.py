import pandas as pd
import numpy as np
import os
# from sklearn.model_selection import train_test_split, LeaveOneOut, LeaveOneGroupOut
# from sklearn.utils import shuffle

class Preprocessor:
    """
    Unified Dataset class handling creation, loading, preprocessing, and splitting.
    Supports 'user', 'path', and time-based aggregation types.
    """
    
    # Raw Data Paths (Constants)
    RAW_TIMESERIES = "/data/KATE_AA_dataset.csv"
    RAW_DEMOGRAPHICS = "/data/demographics.csv"
    RAW_MOTOR_TESTS = "/data/motoric_test.csv"
    RAW_TASK_DIFFICULTY = "/data/task_difficulty.csv"

    INDICES_COLS = ['user', 'path', 'time']

    DEMOGRAPHICS_COLS = ['age', 'sex_value']
    
    # Columns to separate from timeseries data because they are aggregated per path
    # Those topics are averages of other columns '_avg' or counts '_num_'.
    PATH_RELATED_COLS = [
        'right_num_steps',
        'right_num_strides',
        'right_step_length_avg',
        'right_step_duration_avg',
        'right_stride_duration_avg',
        'right_stride_length_avg',
        'right_stride_swing_time_avg',
        'right_stride_stance_time_avg',
        'left_num_steps',
        'left_num_strides',
        'left_step_length_avg',
        'left_step_duration_avg',
        'left_stride_length_avg',
        'left_stride_duration_avg',
        'left_stride_swing_time_avg',
        'left_stride_stance_time_avg',
        'speed_avg',
        'cadence_avg',
        'total_duration'
    ]
    
    # For topics that are only present when forces were in the path
    TIMESERIES_IMPUTE_WITH_ZERO_COLS = [
        "virtual_force_resulting_force_y",
        "virtual_force_resulting_force_x",
        "virtual_force_position_x",
        "virtual_force_position_y",
        "virtual_force_velocity_out_y",
        "virtual_force_velocity_out_x",
        "virtual_force_resulting_velocity_x",
        "virtual_force_resulting_velocity_y",
        "virtual_force_velocity_in_y",
        "virtual_force_velocity_in_x",
    ]

    # Topics that need backfill/forwardfill in time-series datasets
    TIMESERIES_IMPUTE_WITH_CLOSEST_COLS = [
        'force_input_raw_y',
        'force_input_raw_x',
        'ppi',
        'robot_pose_z',
        'robotrainer_deviation_left',
        'robot_pose_y',
        'output_data_linear_z',
        'velocity_output_angular_z',
        'output_data_linear_x',
        'output_data_angular_y',
        'output_data_angular_z',
        'hrv',
        'heart_rate',
        'force_input_scaled_limited_y',
        'force_input_scaled_limited_x',
        'path_index_left',
        'path_index_front',
        'ppg_ch1',
        'velocity_output_linear_x',
        'velocity_output_linear_y',
        'ppg_ch2',
        'torque_input_raw_z',
        'ppg_ch0',
        'robotrainer_deviation_right',
        'path_index_right',
        'ppg_ch3',
        'robotrainer_deviation_front',
        'robot_pose_x',
        'left_step_duration',
        'right_stride_duration',
        'right_step_length',
        'left_stride_swing_time',
        'right_step_duration',
        'left_stride_duration',
        'right_stride_length',
        'right_stride_swing_time',
        'left_step_length',
        'left_stride_stance_time',
        'left_stride_length',
        'right_stride_stance_time'
    ]

    TASK_DIFFICULTY_COLS = [
        'path_type', 'area_num', 'force_strength', 'force_direction', 'inverted'
    ]
    
    # Target columns
    TARGET_COLS = [
        'Balance Test', 'Single Leg Stance', 'Robotrainer Front', 'Robotrainer Left', 
        'Robotrainer Right', 'Hand Grip Left', 'Hand Grip Right', 'Jump & Reach', 
        'Tandem Walk', 'Figure 8 Walk', 'Jumping Sideways', 'Throwing Beanbag at Target',
        'Tapping Test', 'Ruler Drop Test'
    ]

    def __init__(self, sampling_frequency: int, path: str):
        """
        Args:
            dataset_type: 'user', 'path', or a time string like '1s', '100ms'.
            recreate: Whether to force recreation of the CSV file.
        """
        self.sampling_frequency = sampling_frequency
        self.output_path = path
        
        self.timeseries_df, self.path_related_df, self.user_related_df, self.target_df = self._create_dataset()
        # self._validate_data()


    def _create_dataset(self):
        """Orchestrates the loading of raw files, merging, cleaning, and saving."""
        # Load raw csv files
        raw_timeseries_df = pd.read_csv(self.RAW_TIMESERIES)
        demographics_df = pd.read_csv(self.RAW_DEMOGRAPHICS)
        motor_tests_df = pd.read_csv(self.RAW_MOTOR_TESTS)
        raw_task_difficulty_df = pd.read_csv(self.RAW_TASK_DIFFICULTY)

        # Filter loaded dataframes
        user_related_df = self._filter_dataframe(demographics_df, self.DEMOGRAPHICS_COLS, self.RAW_DEMOGRAPHICS)
        target_df = self._filter_dataframe(motor_tests_df, self.TARGET_COLS, self.RAW_MOTOR_TESTS)
        timeseries_df = self._filter_dataframe(raw_timeseries_df, self.TIMESERIES_IMPUTE_WITH_ZERO_COLS + self.TIMESERIES_IMPUTE_WITH_CLOSEST_COLS, self.RAW_TIMESERIES)
        path_related_df = self._filter_dataframe(raw_timeseries_df, self.PATH_RELATED_COLS, self.RAW_TIMESERIES)
        task_difficulty_df = self._filter_dataframe(raw_task_difficulty_df, self.TASK_DIFFICULTY_COLS, self.RAW_TASK_DIFFICULTY)

        # Impute path_related_df (0 or NaN) with the average of the user's other paths and merge task difficulty
        path_related_df = self._impute_path_related_data(path_related_df)

        # Merge path related data with task difficulty
        path_related_df = pd.merge(path_related_df, task_difficulty_df, on=['path'], how='left')
        path_related_df.to_csv("path_related_data.csv", index=False)
        
        # Impute time-series data with closest values from the same user/path
        resampled_df, resampled_before_imputing_df = self._downsample_and_impute_timeseries(timeseries_df)

        # Validation
        self._generate_quality_report(resampled_df, resampled_before_imputing_df)
        
        resampled_df.to_csv(self.output_path, index=False)

        return resampled_df, path_related_df, user_related_df, target_df

    def _filter_dataframe(self, df: pd.DataFrame, cols_to_keep: list[str], df_name: str = "DataFrame") -> pd.DataFrame:
        """
        Filters the dataframe to keep only the specified columns and any indices columns found.
        Prints warnings for missing columns.
        """
        # Always include indices if they exist in the df
        indices_in_df = [c for c in self.INDICES_COLS if c in df.columns]
        
        # Identify missing columns from the requested list
        missing_cols = [c for c in cols_to_keep if c not in df.columns]
        if missing_cols:
            print(f"WARNING: {df_name} is missing expected columns: {missing_cols}")
        
        # Determine final columns to keep (indices + requested)
        # Using set to avoid duplicates, then list comprehension to preserve original order
        target_cols = set(indices_in_df + cols_to_keep)
        final_cols = [c for c in df.columns if c in target_cols]
        
        return df[final_cols]

    def _impute_path_related_data(self, path_related_df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregates path-related data (mean per path), imputes missing values using user averages,
        and merges with task difficulty data.
        """
        # Remove 'time' from path_related_df as it shouldn't be aggregated (it would become average timestamp)
        if 'time' in path_related_df.columns:
            path_related_df = path_related_df.drop(columns=['time'])
            
        # Validation: Check that PATH_RELATED_COLS have a single unique value per user/path
        grouped = path_related_df.groupby(['user', 'path'])
        unique_counts = grouped.nunique()

        # Check for multiple unique values
        if (unique_counts > 1).any().any():
            print("PATH_RELATED_DF: WARNING: Multiple unique values found for a path+user group. It is expected to have only 1 unique value per group. The following will be averaged.")
            for col in self.PATH_RELATED_COLS:
                mask = unique_counts[col] > 1
                if mask.any():
                    for u, p in unique_counts.index[mask]:
                        values = grouped.get_group((u, p))[col].unique()
                        print(f"  Column: '{col}' in Cell: User {u}, Path {p}: {values}")

        # Create path_related_df:
        # Logic: If unique count is 1, take that value. If > 1, take mean. If 0, it remains NaN.
        path_related_df = grouped.mean().reset_index()

        # Check for missing values: only list if ALL path related columns are zero or nan for a group
        is_zero_or_nan = (path_related_df == 0) | (path_related_df.isna())
        all_missing_mask = is_zero_or_nan.all(axis=1)

        if all_missing_mask.any():
            print("PATH_RELATED_DF: WARNING: The following groups have ONLY 0 or NaN values across ALL path-related columns (These will be imputed by averaged user values.):")
            print(path_related_df.loc[all_missing_mask, ['user', 'path']])

        # Validation: Missing value percentage per column before imputing
        print("\nPATH_RELATED_DF: Top 5 percentage of NaN values per path-related column (before imputing):")
        self._report_missing_values(path_related_df, path_related_df.columns)
        
        # Impute missing values (0 or NaN) with the average of the user's other paths
        print("\nPATH_RELATED_DF: Imputing missing path-related values (0 or NaN) with user average of the other paths...")
        for col in path_related_df.columns:
            # Identify missing values (0 or NaN)
            is_missing = (path_related_df[col] == 0) | (path_related_df[col].isna())
            
            if is_missing.any():
                # Calculate user-level mean for this column, ignoring 0s and NaNs
                user_means = path_related_df[col].replace(0, np.nan).groupby(path_related_df['user']).transform('mean')
                
                # Fill the identified missing rows with the calculated user mean
                path_related_df.loc[is_missing, col] = user_means[is_missing]
        
        return path_related_df
    

    def _downsample_and_impute_timeseries(self, df: pd.DataFrame) -> pd.DataFrame:

        # Convert float seconds to DateTime (using a dummy epoch) to ensure robust resampling alignment
        # Timedelta resampling with origin='start_day' can sometimes be inconsistent across pandas versions
        dummy_epoch = pd.Timestamp("2000-01-01")
        df = df.copy()
        df['dt'] = dummy_epoch + pd.to_timedelta(df['time'], unit='s')
        
        # Set index for resampling
        df = df.set_index('dt')

        # Group by identifiers, then resample the time index
        # Calculate bin size from frequency (Hz).
        interval = pd.Timedelta(seconds=1) / self.sampling_frequency
        print(f"\nTIMESERIES_DF: Resampling time-series data to interval: {interval} (Frequency: {self.sampling_frequency} Hz)")
        
        # include_groups=False prevents duplication of grouping keys into columns and silences warning
        # origin='epoch' ensures alignment to 1970-01-01 00:00:00. Since our data is 2000-01-01 + time, 
        # and 2000-01-01 is aligned with minute/second boundaries, this aligns bins to X.000s
        # label='right' sets the bin label to the right edge
        resampled_raw = df.groupby(['user', 'path']).resample(interval, include_groups=False, origin='epoch', label='right').mean(numeric_only=True)
        
        # Cleanup to return to flat format with float time
        resampled_raw = resampled_raw.reset_index()
        # Convert back to seconds relative to the dummy epoch
        resampled_raw['time'] = (resampled_raw['dt'] - dummy_epoch).dt.total_seconds()
        resampled_raw = resampled_raw.drop(columns=['dt'])
        resampled = resampled_raw.copy()

        # Validation: Missing value percentage per column before imputing
        print("\nTIMESERIES_DF: Top 5 percentage of NaN values per timeseries column (before imputing with closest value within same path):")
        self._report_missing_values(resampled, self.TIMESERIES_IMPUTE_WITH_CLOSEST_COLS)


        # Impute time-series data with closest next/previous values from the same user/path
        for col in self.TIMESERIES_IMPUTE_WITH_CLOSEST_COLS:
            if col in resampled.columns:
                # bfill: fills NaNs with next valid observation (closest next value)
                # ffill: fills remaining NaNs at the end with last valid observation
                resampled[col] = resampled.groupby(['user', 'path'])[col].transform(lambda x: x.bfill().ffill())

        # Validation: Missing value percentage per column before imputing
        print("\nTIMESERIES_DF: Top 5 percentage of NaN values per timeseries column (before imputing with average of all paths of the same user):")
        self._report_missing_values(resampled, self.TIMESERIES_IMPUTE_WITH_CLOSEST_COLS)
            
        # Check for groups where ffill/bfill won't work (all NaNs in a group)
        for col in self.TIMESERIES_IMPUTE_WITH_CLOSEST_COLS:
            if col in resampled.columns:
                grouped_count = resampled.groupby(['user', 'path'])[col].count()
                missing_groups = grouped_count[grouped_count == 0]
                
                if not missing_groups.empty:
                     print(f"TIMESERIES_DF: WARNING: Column '{col}' has NO valid values for the following User+Path groups (Will be imputed with average of all paths of the same user):")
                     print(missing_groups.index.tolist())

                # Impute remaining NaNs with the average of all paths of the same user
                if resampled[col].isna().any():
                    user_means = resampled.groupby('user')[col].transform('mean')
                    resampled[col] = resampled[col].fillna(user_means)

        # Impute specific columns that should be filled with zero
        for col in self.TIMESERIES_IMPUTE_WITH_ZERO_COLS:
            if col in resampled.columns:
                # Fill with 0
                resampled[col] = resampled[col].fillna(0)

        return resampled, resampled_raw

    def _report_missing_values(self, df, columns, top_n=5):
        missing_stats = []
        for col in columns:
            if col in df.columns:
                missing_count = (df[col].isna()).sum()
                percentage = (missing_count / len(df)) * 100
                if percentage > 0:
                    missing_stats.append((col, percentage))

        if not missing_stats:
            print("  No columns with missing values.")
            return

        missing_stats.sort(key=lambda x: x[1], reverse=True)

        for col, percentage in missing_stats[:top_n]:
            print(f"  {col}: {percentage:.2f}%")

        remaining = len([x for x in missing_stats[top_n:] if x[1] > 0])
        if remaining > 0:
            print(f"  ... and {remaining} other columns with missing values")


    def _generate_quality_report(self, final_df: pd.DataFrame, raw_df: pd.DataFrame):
        """
        Generates and prints a comprehensive data quality report.
        """
        print(f"\n{'='*30} DATA QUALITY REPORT {'='*30}")
        
        # 1. Dimensions & Users
        n_users = final_df['user'].nunique()
        n_rows, n_cols = final_df.shape
        
        # Calculate per-user stats
        user_counts = final_df['user'].value_counts()
        avg_rows_per_user = user_counts.mean()
        min_rows_per_user = user_counts.min()
        max_rows_per_user = user_counts.max()

        print(f"\n[Dimensions]")
        print(f"  Unique Users: {n_users}")
        print(f"  Total Rows:   {n_rows}")
        print(f"  Avg Rows/User: {avg_rows_per_user:.2f}")
        print(f"  Min Rows/User: {min_rows_per_user}")
        print(f"  Max Rows/User: {max_rows_per_user}")
        print(f"  Total Cols:   {n_cols} (including indices)")
        
        # 2. Data Volume & Raw Stats
        # Calculate total datapoints in raw dataframe
        raw_total_points = raw_df.size
        raw_nans = raw_df.isna().sum().sum()
        
        # Calculate percentage
        if raw_total_points > 0:
            raw_nan_pct = (raw_nans / raw_total_points) * 100
        else:
            raw_nan_pct = 0.0
            
        final_total_points = final_df.size
        
        print(f"\n[Data Volume & Imputation]")
        print(f"  Total datapoints in raw dataset:   {raw_total_points:,}")
        print(f"  Total missing values in raw data:  {raw_nans:,}")
        print(f"  Percentage of imputed datapoints:  {raw_nan_pct:.2f}% (based on raw NaNs)")
        print(f"  Total datapoints in final dataset: {final_total_points:,}")

        # 3. Column Validation
        expected_cols = self.TIMESERIES_IMPUTE_WITH_ZERO_COLS + self.TIMESERIES_IMPUTE_WITH_CLOSEST_COLS
        print(f"\n[Column Validation]")
        missing_expected = [c for c in expected_cols if c not in final_df.columns]
        if missing_expected:
            print(f"  WARNING: The following expected columns are MISSING in the final dataset:")
            for c in missing_expected:
                print(f"    - {c}")
        else:
            print(f"  OK: All {len(expected_cols)} expected columns are present.")

        # 4. Final Missing Values
        print(f"\n[Final Missing Values]")
        self._report_missing_values(final_df, final_df.columns.tolist())
        
        # 5. Time Series Specifics
        if 'time' in final_df.columns:
            print(f"\n[Time Series Properties]")
            print(f"  Sampling Frequency: {self.sampling_frequency} Hz")

            # Calculate duration per path
            path_stats = final_df.groupby(['user', 'path'])['time'].agg(['min', 'max'])
            path_durations = path_stats['max'] - path_stats['min']

            print(f"  Longest path duration:  {path_durations.max():.2f}s")
            print(f"  Shortest path duration: {path_durations.min():.2f}s")
            print(f"  Average path duration:  {path_durations.mean():.2f}s")
        print(f"{'='*80}\n")

    def get_features_targets(self):
        """Separates X, y, and user groups."""
        y = self.df[self.TARGET_COLS].values
        # Keep user and path in X for SmartImputer
        feature_cols = [c for c in self.df.columns if c not in self.TARGET_COLS]
        X = self.df[feature_cols] # Return DataFrame
        users = self.df['user'].values
        return X, y, users, self.TARGET_COLS

    @property
    def feature_names(self) -> list:
        """Returns the list of feature column names."""
        return [c for c in self.df.columns if c not in self.TARGET_COLS and c != 'user']

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
        
        X_train, X_test = X[train_mask], X[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]
        users_train, users_test = users[train_mask], users[test_mask]

        # Shuffle the training set to mix users (Best practice for SGD/Adam)
        X_train, y_train, users_train = shuffle(X_train, y_train, users_train, random_state=random_state)

        # Note: We do NOT shuffle the test set. 
        # Keeping it ordered by User->Path->Time is better for visualization and analysis.

        return X_train, X_test, y_train, y_test, users_train, users_test

    @property
    def has_multiple_rows_per_user(self):
        """Returns True if the dataset contains multiple rows per user (requires aggregation)."""
        return self.sampling_frequency != 'user'

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
        

if __name__ == "__main__":

    dataset = Preprocessor(sampling_frequency=1, path="/data/dataset_conv@1s.csv")

    # --- 2. Prepare Data ---
    # X_train_val, X_test, y_train_val, y_test, users_train_val, users_test = dataset.get_train_test_split()
    
    # print(f"Train/Val Samples: {len(X_train_val)} (Users: {len(np.unique(users_train_val))})")
    # print(f"Test Samples: {len(X_test)} (Users: {len(np.unique(users_test))})")
    # print(f"Number of Features: {X_train_val.shape[1]}")