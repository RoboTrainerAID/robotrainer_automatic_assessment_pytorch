"""
Advanced statistical feature extraction for timeseries data.
"""

import numpy as np
import pandas as pd
from typing import Dict, Callable
from scipy.signal import periodogram
from automatic_assessment.dataset.timeseries_loader import PathData
import automatic_assessment.dataset.config as config


class TimeseriesFeatures:

    def __init__(self, dataset: Dict[int, Dict[int, PathData]]) -> None:
        self.dataset = dataset
        self.features_df = pd.DataFrame()
        self.feature_fns = self._feature_functions()

    # ============================================================
    # Robust Feature Functions
    # ============================================================

    def _robust_clip(self, v):
        q1 = np.percentile(v, 25)
        q3 = np.percentile(v, 75)
        iqr = q3 - q1
        lower = q1 - 3 * iqr
        upper = q3 + 3 * iqr
        return np.clip(v, lower, upper)

    def _feature_functions(self) -> Dict[str, Callable]:

        return {

            "trimmed_mean": lambda v, t=None: float(np.mean(self._robust_clip(v))),
            "std": lambda v, t=None: float(np.std(v)),
            "p95": lambda v, t=None: float(np.percentile(v, 95)),
            "p05": lambda v, t=None: float(np.percentile(v, 5)),
            "rms": lambda v, t=None: float(np.sqrt(np.mean(self._robust_clip(v) ** 2))),
            "abs_mean": lambda v, t=None: float(np.mean(np.abs(v))),
            "energy": lambda v, t=None: float(np.sum(self._robust_clip(v) ** 2)),

            "rms_diff": lambda v, t=None: float(
                np.sqrt(np.mean(np.diff(self._robust_clip(v)) ** 2))
            ) if len(v) > 1 else 0.0,

            "impulse": lambda v, t: float(
                np.sum(np.abs(self._robust_clip(v[:-1])) * np.diff(t))
            ) if t is not None and len(t) > 1 else 0.0,
            
            # --- New Physical Integrals ---
            "integral": lambda v, t: float(
                np.sum(v[:-1] * np.diff(t))
            ) if t is not None and len(t) > 1 else 0.0,

            "abs_integral": lambda v, t: float(
                np.sum(np.abs(v[:-1]) * np.diff(t))
            ) if t is not None and len(t) > 1 else 0.0,
            # ------------------------------

            "band_power_voluntary": lambda v, t=None: self._band_power(
                v, t, config.VOLUNTARY_BAND
            ),

            "band_power_tremor": lambda v, t=None: self._band_power(
                v, t, config.TREMOR_BAND
            ),
        }

    def _band_power(self, v: np.ndarray, t: np.ndarray, band: tuple) -> float:
        if len(v) < 4:
            return 0.0
        v = self._robust_clip(v)
        fs = self._estimate_fs(t)
        f, Pxx = periodogram(v, fs=fs)
        mask = (f >= band[0]) & (f <= band[1])
        return float(np.sum(Pxx[mask]))

    def _estimate_fs(self, t: np.ndarray) -> float:
        """
        Estimates sampling frequency from the time array.
        Uses median time difference to be robust against dropped packets or gaps.
        """
        if t is None or len(t) < 2:
             return 10.0
        
        # Calculate time differences
        dt = np.diff(t)
        
        # Filter strictly positive intervals to avoid division by zero or negative time
        dt = dt[dt > 1e-6]
        
        if len(dt) == 0:
            return 10.0
            
        # Use median to ignore potential large gaps in data
        median_dt = np.median(dt)
        
        return 1.0 / median_dt

    # ============================================================
    # Derived Timeseries
    # ============================================================

    def _add_force_magnitudes(self, ts_dict: Dict[str, np.ndarray]) -> None:
        """
        Derives physical quantities including equivalent magnitudes, Power, and Work.
        Lever arm (r) = 0.35m.
        """
        r_handle = 0.35  # meters
        
        # Helper to get column vectors if they exist
        def get_vec(name_x, name_y, name_z=None):
            if name_x in ts_dict and name_y in ts_dict:
                vx = ts_dict[name_x][:, config.COL_VALUE]
                vy = ts_dict[name_y][:, config.COL_VALUE]
                vz = ts_dict[name_z][:, config.COL_VALUE] if (name_z and name_z in ts_dict) else np.zeros_like(vx)
                # Return time reference from x
                return ts_dict[name_x][:, :2], vx, vy, vz
            return None

        # 1. Total User Effort (Force + Torque)
        # F_equiv = sqrt(Fx^2 + Fy^2 + (Tz / r)^2)
        user_f_data = get_vec("user_force_x", "user_force_y", "user_torque_z")
        if user_f_data:
            time_cols, fx, fy, tz = user_f_data
            
            # Linear Magnitude
            lin_mag = np.sqrt(fx**2 + fy**2)
            ts_dict["user_force_lin_mag"] = np.column_stack((time_cols, lin_mag))
            
            # Combined Equivalent Magnitude (at handle)
            f_tan = tz / r_handle # Convert torque to tangential force
            total_mag = np.sqrt(fx**2 + fy**2 + f_tan**2)
            ts_dict["user_force_total_mag"] = np.column_stack((time_cols, total_mag))

        # 2. Total Handle Velocity (Linear + Rotational)
        # V_equiv = sqrt(Vx^2 + Vy^2 + (Wz * r)^2)
        robot_v_data = get_vec("robot_vel_x", "robot_vel_y", "robot_vel_rot_z")
        if robot_v_data:
            time_cols, vx, vy, wz = robot_v_data
            
            # Linear Magnitude
            lin_mag = np.sqrt(vx**2 + vy**2)
            ts_dict["robot_vel_lin_mag"] = np.column_stack((time_cols, lin_mag))
            
            # Combined Equivalent Magnitude (velocity of handle)
            v_tan = wz * r_handle
            total_mag = np.sqrt(vx**2 + vy**2 + v_tan**2)
            ts_dict["robot_vel_total_mag"] = np.column_stack((time_cols, total_mag))

        # 3. User Power Interaction
        # P = F.v + T.w
        if user_f_data and robot_v_data:
            # Align lengths if necessary (usually robust if loaded together)
            n = min(len(user_f_data[1]), len(robot_v_data[1]))
            
            fx, fy, tz = user_f_data[1][:n], user_f_data[2][:n], user_f_data[3][:n]
            vx, vy, wz = robot_v_data[1][:n], robot_v_data[2][:n], robot_v_data[3][:n]
            time_cols = user_f_data[0][:n]

            # Power calculation (Watts)
            # Dot product of force and velocity vectors
            p_trans = fx * vx + fy * vy
            p_rot   = tz * wz
            p_total = p_trans + p_rot

            ts_dict["user_power"] = np.column_stack((time_cols, p_total))

            # 4. Work (Accumulated Energy)
            # We calculate this as a timeseries of cumulative sum for visualization,
            # but the 'integral' feature in feature_fns will calculate single-value Total Work later.
            # Work = Integral(P dt)
            if n > 1:
                times = time_cols[:, 0] # Use Raw timestamps for diff
                dt = np.diff(times)
                # Compute work increments (Joules)
                dW = p_total[:-1] * dt
                # Prepend 0 for initial state
                work_cum = np.pad(np.cumsum(dW), (1, 0), 'constant')
                ts_dict["user_work_cum"] = np.column_stack((time_cols, work_cum))

        # Disturbance Magnitude (Linear only usually)
        if "disturbance_force_x" in ts_dict and "disturbance_force_y" in ts_dict:
            fx = ts_dict["disturbance_force_x"]
            fy = ts_dict["disturbance_force_y"]
            mag = np.sqrt(fx[:, config.COL_VALUE]**2 + fy[:, config.COL_VALUE]**2)
            ts_dict["disturbance_force_mag"] = np.column_stack(
                (fx[:, :2], mag)
            )

    # ============================================================
    # Correlation
    # ============================================================

    def _compute_correlation(self, ts1, ts2):

        t1, v1 = ts1[:, 1], ts1[:, 2]
        t2, v2 = ts2[:, 1], ts2[:, 2]

        common_t = np.linspace(
            max(t1.min(), t2.min()),
            min(t1.max(), t2.max()),
            200,
        )

        v1_interp = np.interp(common_t, t1, v1)
        v2_interp = np.interp(common_t, t2, v2)

        if len(v1_interp) < 3:
            return 0.0

        # Check for constant arrays to avoid RuntimeWarning in corrcoef
        if np.std(v1_interp) < 1e-9 or np.std(v2_interp) < 1e-9:
            return 0.0

        corr = np.corrcoef(v1_interp, v2_interp)[0, 1]
        
        if np.isnan(corr):
            return 0.0
            
        return float(corr)

    # ============================================================
    # Main Extraction
    # ============================================================

    def extract_features(self) -> pd.DataFrame:

        rows = []

        for user_id, paths in self.dataset.items():
            for path_id, pd_data in paths.items():
                
                # Base row with identifiers (Wide Format: 1 row per path)
                row_data = {
                    "user_id": user_id,
                    "path_id": path_id,
                }
                
                # Add existing scalar features (e.g. from meta or single value files)
                if pd_data.scalar_features:
                    row_data.update(pd_data.scalar_features)

                # Add duration from meta if available
                if pd_data.meta and "duration" in pd_data.meta:
                    try:
                        row_data["duration"] = float(pd_data.meta["duration"])
                    except (ValueError, TypeError):
                        pass

                ts_dict = dict(pd_data.timeseries)

                # Add derived magnitudes and physical quantities
                self._add_force_magnitudes(ts_dict)

                # ---- Per timeseries features ----
                for ts_name, feature_list in config.TS_FEATURES.items():

                    if ts_name not in ts_dict:
                        continue

                    arr = ts_dict[ts_name]
                    # Data validation
                    if arr.ndim < 2 or arr.shape[1] <= config.COL_VALUE:
                        continue
                        
                    values = arr[:, config.COL_VALUE]
                    times = arr[:, config.COL_REL_TS]

                    for fname in feature_list:
                        func = self.feature_fns.get(fname)
                        if func is None:
                            continue
                        
                        # Create unique column name: <ts_name>_<feature>
                        col_name = f"{ts_name}_{fname}"

                        try:
                            row_data[col_name] = func(values, times)
                        except Exception:
                            row_data[col_name] = np.nan

                # ---- Correlation features ----
                for ts1_name, ts2_name in config.CORRELATION_FEATURES:

                    if ts1_name in ts_dict and ts2_name in ts_dict:

                        corr = self._compute_correlation(
                            ts_dict[ts1_name],
                            ts_dict[ts2_name],
                        )

                        col_name = f"{ts1_name}_corr_{ts2_name}"
                        row_data[col_name] = corr

                rows.append(row_data)

        self.features_df = pd.DataFrame(rows)
        return self.features_df

    # ============================================================
    # Save
    # ============================================================

    def save_features_to_csv(self):
        if self.features_df.empty:
            print("Warning: Feature DataFrame is empty.")
            return

        config.CSV_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        self.features_df.to_csv(config.CSV_OUTPUT_PATH, index=False)
        print(f"Features saved to {config.CSV_OUTPUT_PATH}")
