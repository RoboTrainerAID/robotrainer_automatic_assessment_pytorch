"""
Imputation logic for missing or empty timeseries data.
"""

import numpy as np
from typing import Dict, List, Any, Optional
from automatic_assessment.dataset.timeseries_loader import PathData

class TimeseriesImputer:
    """
    Handles imputation of missing or broken timeseries/scalar data based on validation reports.
    """

    def __init__(self, dataset: Dict[int, Dict[int, PathData]], config: Any) -> None:
        self.dataset = dataset
        self.cfg = config

    def impute_from_report(self, report: Dict[int, List[Dict[str, Any]]]) -> None:
        """
        Iterates through the validation report and imputes missing data.
        """
        print("\nStarting Imputation Process...")
        
        for user_id, issues in report.items():
            for issue in issues:
                path_id = issue['path_id']
                issue_type = issue['type']
                target_names = issue.get('target_names', []) # List of missing keys

                if not target_names:
                    continue

                if user_id not in self.dataset or path_id not in self.dataset[user_id]:
                    continue
                
                pd_obj = self.dataset[user_id][path_id]

                if issue_type in ["missing_timeseries", "empty_timeseries"]:
                    for ts_name in target_names:
                        self._impute_timeseries(pd_obj, ts_name)
                
                elif issue_type == "missing_scalar":
                    for scalar_name in target_names:
                        self._impute_scalar(pd_obj, scalar_name)

    def impute_zero_disturbance(self) -> None:
        """
        Proactively adds zero-valued disturbance force timeseries to paths 
        that are known to have no disturbance (defined in config).
        Fills with 0.0 every second.
        """
        print("\nImputing zero-disturbance forces for safe paths...")
        
        dist_cols = ["disturbance_force_x", "disturbance_force_y"]
        count = 0

        for user_id, paths in self.dataset.items():
            for path_id, pd_data in paths.items():
                
                if path_id in self.cfg.PATHS_WITHOUT_DISTURBANCE:
                    
                    # Determine duration and start time
                    duration = 0.0
                    start_time = 0.0

                    # Check scalar features first for duration
                    if "duration" in pd_data.scalar_features:
                        duration = pd_data.scalar_features["duration"]
                    # Fallback to meta (legacy or raw data)
                    elif pd_data.meta and "duration" in pd_data.meta:
                        try:
                            duration = float(pd_data.meta["duration"])
                        except (ValueError, TypeError):
                            pass

                    if pd_data.meta and "trimmed_start_timestamp" in pd_data.meta:
                         start_time = float(pd_data.meta["trimmed_start_timestamp"])
                    
                    # Fallback for timing if meta/scalars fails
                    if duration == 0.0 and pd_data.timeseries:
                        try:
                            # Use robot_vel_x as reliable reference if present
                            if "robot_vel_x" in pd_data.timeseries:
                                ref_ts = pd_data.timeseries["robot_vel_x"]
                            else:
                                ref_ts = next(iter(pd_data.timeseries.values()))
                                
                            if ref_ts.size > 0:
                                start_time = ref_ts[0, self.cfg.COL_RAW_TS]
                                duration = ref_ts[-1, self.cfg.COL_REL_TS]
                        except Exception:
                            pass
                    
                    # Default if completely unknown
                    if duration == 0.0:
                        duration = 10.0

                    # Generate time points (1Hz)
                    # 0, 1, 2, ..., duration
                    rel_times = np.arange(0, duration + 0.1, 1.0)
                    raw_times = start_time + rel_times
                    values = np.zeros_like(rel_times)
                    
                    # Stack into (N, 3) matrix: [raw, rel, val]
                    # shape (N, 3)
                    new_data = np.column_stack((raw_times, rel_times, values))

                    for dist_name in dist_cols:
                        if dist_name not in pd_data.timeseries:
                            pd_data.timeseries[dist_name] = new_data
                            count += 1
        
        print(f"  Added zero-disturbance data to {count} missing entries.")

    def _calculate_robust_mean(self, values: List[float]) -> float:
        """
        Calculates a robust mean by removing outliers using the IQR method.
        """
        if not values:
            return 0.0
        
        data = np.array(values)
        if data.size < 4:
            return float(np.mean(data))
            
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        iqr = q3 - q1
        
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        # Filter values within bounds
        filtered_data = data[(data >= lower_bound) & (data <= upper_bound)]
        
        if filtered_data.size == 0:
            return float(np.mean(data))
            
        return float(np.mean(filtered_data))

    def _impute_timeseries(self, target_pd: PathData, ts_name: str) -> None:
        """
        Imputes a missing timeseries by calculating a user-specific robust average 
        from all other valid paths of the SAME user.
        Creates a single entry at the middle of the path duration.
        """
        # 1. Collect valid values for this timeseries from only the SAME USER
        valid_values = []
        user_paths = self.dataset.get(target_pd.user_id, {})

        for p_id, pd_data in user_paths.items():
            # Skip the target itself
            if p_id == target_pd.path_id:
                continue
            
            if ts_name in pd_data.timeseries:
                arr = pd_data.timeseries[ts_name]
                if arr.size > 0 and arr.shape[1] > self.cfg.COL_VALUE:
                    # Append all values to the list for robust statistics
                    valid_values.extend(arr[:, self.cfg.COL_VALUE].tolist())

        if not valid_values:
            print(f"  Warning: No valid data found for User {target_pd.user_id} for '{ts_name}'. Cannot impute.")
            return

        # 2. Calculate Robust Mean
        imputed_value = self._calculate_robust_mean(valid_values)

        # 3. Determine Timestamp (Middle of the session)
        # Try to get duration from scalars, then meta, otherwise estimate from other ts
        duration = 0.0
        start_time = 0.0
        
        if "duration" in target_pd.scalar_features:
            duration = target_pd.scalar_features["duration"]
        elif target_pd.meta and "duration" in target_pd.meta:
            # Some meta files might have duration in string or float
            try:
                duration = float(target_pd.meta["duration"])
            except (ValueError, TypeError):
                print(f"  Warning: Invalid duration format in meta for User {target_pd.user_id} Path {target_pd.path_id}.")

        # Construct the single datapoint
        # [timestamp, rel_time, value]
        mid_rel = duration / 2.0
        mid_raw = start_time + mid_rel if start_time > 0 else 0.0 # Raw might be invalid if 0
        
        # Create (1, 3) array
        new_entry = np.array([[mid_raw, mid_rel, imputed_value]])
        
        target_pd.timeseries[ts_name] = new_entry
        print(f"  [Imputed] User {target_pd.user_id} Path {target_pd.path_id}: '{ts_name}' set to {imputed_value:.4f}")

    def _impute_scalar(self, target_pd: PathData, scalar_name: str) -> None:
        """
        Imputes a missing scalar by calculating a user-specific robust average.
        """
        valid_values = []
        user_paths = self.dataset.get(target_pd.user_id, {})

        for p_id, pd_data in user_paths.items():
            if p_id == target_pd.path_id:
                continue
            
            if scalar_name in pd_data.scalar_features:
                valid_values.append(pd_data.scalar_features[scalar_name])

        if not valid_values:
            print(f"  Warning: No valid data found for User {target_pd.user_id} scalar '{scalar_name}'. Cannot impute.")
            return

        imputed_value = self._calculate_robust_mean(valid_values)
        target_pd.scalar_features[scalar_name] = imputed_value
        print(f"  [Imputed] User {target_pd.user_id} Path {target_pd.path_id}: Scalar '{scalar_name}' set to {imputed_value:.4f}")
