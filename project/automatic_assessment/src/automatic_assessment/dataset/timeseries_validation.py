"""
Validation logic for the loaded timeseries dataset.
"""

from typing import Dict, List, Any

import numpy as np

from automatic_assessment.dataset.timeseries_loader import PathData


class TimeseriesValidator:
    """
    Validates the structure and content of the loaded timeseries dataset.
    """

    @staticmethod
    def validate_dataset(data: Dict[int, Dict[int, PathData]], config: Any, check_all_disturbance: bool = True) -> Dict[int, List[Dict[str, Any]]]:
        """
        Validates the dataset and returns a report of checks/issues.
        
        :param check_all_disturbance: If False, ignore missing disturbance forces on paths defined in config.
                                           Set to True after imputation to verify they were added.
        Returns:
            Dict[user_id, List[IssueDict]]
            IssueDict contains keys: path_id, type, details, target_names
        """
        report = {}

        for user_id, paths in data.items():
            user_issues = []
            for path_id, pd in paths.items():
                
                # Determine expected timeseries logic
                expected_ts = config.TIMESERIES_TO_LOAD.copy()
                
                # Conditionally filter out disturbance forces for safe paths
                if not check_all_disturbance and path_id in config.PATHS_WITHOUT_DISTURBANCE:
                    expected_ts = [ts for ts in expected_ts if not ts.startswith("disturbance_force")]

                # 1. Check for missing timeseries files
                missing_files = [ts for ts in expected_ts if ts not in pd.timeseries]
                if missing_files:
                    user_issues.append({
                        "path_id": path_id,
                        "type": "missing_timeseries",
                        "details": f"Missing expected files: {missing_files}",
                        "target_names": missing_files
                    })

                # 2. Check for missing expected scalars
                missing_scalars = [s for s in config.DATA_WITH_A_SINGLE_VALUE_PER_PATH if s not in pd.scalar_features]
                if missing_scalars:
                    user_issues.append({
                        "path_id": path_id,
                        "type": "missing_scalar",
                        "details": f"Missing expected scalars: {missing_scalars}",
                        "target_names": missing_scalars
                    })

                # 3. Check for empty timeseries
                short_keys = []
                for ts_name, arr in pd.timeseries.items():
                    if len(arr) < 1:
                        short_keys.append(ts_name)

                if short_keys:
                    user_issues.append({
                        "path_id": path_id,
                        "type": "empty_timeseries",
                        "details": f"Timeseries is empty: {short_keys}",
                        "target_names": short_keys
                    })

                # 4. Check for NaN/Inf VALUES inside the timeseries
                invalid_value_keys = []
                invalid_details = []
                for ts_name, arr in pd.timeseries.items():
                    if arr.size == 0:
                        continue
                    n_bad = int(np.size(arr) - np.isfinite(arr).sum())
                    if n_bad > 0:
                        invalid_value_keys.append(ts_name)
                        invalid_details.append(f"{ts_name}: {n_bad} non-finite entries")

                if invalid_value_keys:
                    user_issues.append({
                        "path_id": path_id,
                        "type": "invalid_values",
                        "details": f"Non-finite values found: {invalid_details}",
                        "target_names": invalid_value_keys
                    })

                # 5. Check for NaN/Inf scalar features
                bad_scalars = [
                    name for name, value in pd.scalar_features.items()
                    if not np.isfinite(value)
                ]
                if bad_scalars:
                    user_issues.append({
                        "path_id": path_id,
                        "type": "invalid_scalar_values",
                        "details": f"Non-finite scalar features: {bad_scalars}",
                        "target_names": bad_scalars
                    })

            if user_issues:
                report[user_id] = user_issues

        return report

    @staticmethod
    def print_validation_report(report: Dict[int, List[Dict[str, Any]]]) -> None:
        """
        Prints the validation report to the console.
        """
        if report:
            print("\n" + "="*40)
            print("DATASET VALIDATION REPORT")
            print("="*40)
            for user_id in sorted(report.keys()):
                print(f"\nUser {user_id}:")
                for issue in report[user_id]:
                    print(f"  - [{issue['type']}] Path {issue['path_id']}: {issue['details']}")
            print("\n" + "="*40 + "\n")
        else:
            print("Dataset validation passed with no issues.\n")

    @staticmethod
    def print_dataset_summary(data: Dict[int, Dict[int, PathData]]) -> None:
        """
        Prints a summary of the loaded dataset statistics.
        Includes number of users, range of paths per user, range of timeseries per path,
        and range of path durations.
        """
        num_users = len(data)
        if num_users == 0:
            print("Dataset Summary: Empty")
            return

        # Calculate path statistics per user
        path_counts = [len(paths) for paths in data.values()]
        min_paths = min(path_counts)
        max_paths = max(path_counts)

        # Calculate timeseries statistics per path and durations
        ts_counts = []
        durations = []
        for user_data in data.values():
            for pd in user_data.values():
                ts_counts.append(len(pd.timeseries))
                
                # Check scalar features first
                if "duration" in pd.scalar_features:
                    durations.append(pd.scalar_features["duration"])
                # Fallback to meta
                elif pd.meta and "duration" in pd.meta:
                    try:
                        durations.append(float(pd.meta["duration"]))
                    except (ValueError, TypeError):
                        pass
        
        min_ts = min(ts_counts) if ts_counts else 0
        max_ts = max(ts_counts) if ts_counts else 0
        
        min_dur = min(durations) if durations else 0.0
        max_dur = max(durations) if durations else 0.0

        print("-" * 60)
        print(f"Dataset Summary")
        print(f"  Users: {num_users}")
        print(f"  Paths per User: {min_paths} - {max_paths}")
        print(f"  Timeseries per Path: {min_ts} - {max_ts}")
        print(f"  Path Duration (s): {min_dur:.2f} - {max_dur:.2f}")
        print("-" * 60)