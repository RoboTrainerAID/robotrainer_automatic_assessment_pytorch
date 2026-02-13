"""
Preprocessor for deriving new timeseries from existing raw data.
Calculates physical quantities like equivalent forces, power, and work.
"""

import numpy as np
from typing import Dict, Tuple, Any
from automatic_assessment.dataset.timeseries_loader import PathData
import automatic_assessment.dataset.config as config


class TimeseriesDerivedTS:
    """
    Augments the dataset with derived timeseries (magnitudes, power).
    Assumes:
      - X, Y, Rot components of the SAME source are already synchronized (same length/timestamps).
      - Force and Velocity components differ in frequency and require synchronization.
    """

    def __init__(self, dataset: Dict[int, Dict[int, PathData]]) -> None:
        self.dataset = dataset

    def process(self) -> None:
        """
        Iterates over all paths and adds derived timeseries to the dataset.
        """
        print("Preprocessing derived timeseries (Magnitudes, Power, Work)...")
        count = 0
        for user_id, paths in self.dataset.items():
            for path_id, pd_data in paths.items():
                self._calculate_magnitudes(pd_data.timeseries)
                self._calculate_power_work(pd_data.timeseries)
                count += 1
        print(f"  Processed {count} paths.")

    def _calculate_magnitudes(self, ts_dict: Dict[str, np.ndarray]) -> None:
        """
        Calculates linear and total magnitudes.
        """
        r_handle = config.HANDLE_RADIUS

        for out_name, defs in config.DERIVED_MAGNITUDES.items():
            name_x = defs.get("x")
            name_y = defs.get("y")
            name_rot = defs.get("rot")
            is_velocity = defs.get("is_velocity", False)

            if not name_x or not name_y:
                continue
            
            # Check availability
            if name_x not in ts_dict or name_y not in ts_dict:
                continue

            # Load Data (Assumed Synchronized)
            data_x = ts_dict[name_x]
            data_y = ts_dict[name_y]
            
            # Use X timestamps as reference
            time_cols = data_x[:, :2] # Raw, Rel
            vals_x = data_x[:, config.COL_VALUE]
            vals_y = data_y[:, config.COL_VALUE]

            # Ensure same length (robustness check)
            n = min(len(vals_x), len(vals_y))
            if len(vals_x) != len(vals_y):
                 vals_x = vals_x[:n]
                 vals_y = vals_y[:n]
                 time_cols = time_cols[:n]

            # 1. Linear Component Term: x^2 + y^2
            sq_sum = vals_x**2 + vals_y**2

            # 2. Rotational Component Term (if requested and present)
            if name_rot and name_rot in ts_dict:
                data_rot = ts_dict[name_rot]
                vals_rot = data_rot[:n, config.COL_VALUE] # Assume sync
                
                # Convert to Tangential Equivalent
                # F_tan = T / r,  V_tan = w * r
                if is_velocity:
                    vals_tan = vals_rot * r_handle
                else:
                    vals_tan = vals_rot / r_handle
                
                sq_sum += vals_tan**2
            
            # Final Magnitude
            magnitude = np.sqrt(sq_sum)
            
            # Save Result
            ts_dict[out_name] = np.column_stack((time_cols, magnitude))


    def _calculate_power_work(self, ts_dict: Dict[str, np.ndarray]) -> None:
        """
        Calculates Power (P = F*v) and Work (CumSum P*dt).
        Handles synchronization between Force and Velocity.
        """
        for base_name, defs in config.DERIVED_POWER_WORK.items():
            fx_n = defs.get("force_x")
            fy_n = defs.get("force_y")
            tz_n = defs.get("torque_z")
            
            vx_n = defs.get("vel_x")
            vy_n = defs.get("vel_y")
            wz_n = defs.get("vel_rot_z")

            # Must have at least translational components
            if not (fx_n in ts_dict and fy_n in ts_dict and vx_n in ts_dict and vy_n in ts_dict):
                continue

            # 1. Prepare Force Data (Synchronized Group)
            # We stack them into one array for easier Time sync: [Raw, Rel, Fx, Fy, Tz]
            f_times = ts_dict[fx_n][:, :2]
            fx = ts_dict[fx_n][:, config.COL_VALUE]
            fy = ts_dict[fy_n][:, config.COL_VALUE]
            
            # Only include Torque if present
            has_rot = (tz_n in ts_dict and wz_n in ts_dict)
            if has_rot:
                tz = ts_dict[tz_n][:, config.COL_VALUE]
                # Ensure length match within Force group
                n_f = min(len(fx), len(tz))
                force_block = np.column_stack((f_times[:n_f], fx[:n_f], fy[:n_f], tz[:n_f]))
            else:
                force_block = np.column_stack((f_times, fx, fy))

            # 2. Prepare Velocity Data (Synchronized Group)
            # [Raw, Rel, Vx, Vy, Wz]
            v_times = ts_dict[vx_n][:, :2]
            vx = ts_dict[vx_n][:, config.COL_VALUE]
            vy = ts_dict[vy_n][:, config.COL_VALUE]
            
            if has_rot:
                wz = ts_dict[wz_n][:, config.COL_VALUE]
                n_v = min(len(vx), len(wz))
                vel_block = np.column_stack((v_times[:n_v], vx[:n_v], vy[:n_v], wz[:n_v]))
            else:
                vel_block = np.column_stack((v_times, vx, vy))

            # 3. Synchronize Force vs Velocity
            # _synchronize_groups will interpolate 'secondary' columns to 'reference' timestamps
            time_cols, aligned_f, aligned_v = self._synchronize_groups(force_block, vel_block)
            
            if len(time_cols) == 0:
                continue

            # 4. Calculate Power
            # aligned_f cols: [Fx, Fy, (Tz)]
            # aligned_v cols: [Vx, Vy, (Wz)]
            
            p_trans = aligned_f[:, 0] * aligned_v[:, 0] + aligned_f[:, 1] * aligned_v[:, 1]
            p_rot = np.zeros_like(p_trans)
            
            if has_rot:
                p_rot = aligned_f[:, 2] * aligned_v[:, 2]
                
            p_total = p_trans + p_rot

            # Save Power
            ts_dict[f"{base_name}_power"] = np.column_stack((time_cols, p_total))

            # 5. Calculate Work
            times = time_cols[:, 0]
            if len(times) > 1:
                dt = np.diff(times)
                # Simple rectangular integration: P[i] * dt[i]
                dW = p_total[:-1] * dt
                work_cum = np.pad(np.cumsum(dW), (1, 0), 'constant')
                ts_dict[f"{base_name}_work_cum"] = np.column_stack((time_cols, work_cum))


    def _synchronize_groups(self, group_a: np.ndarray, group_b: np.ndarray, margin: float = 0.05) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Synchronizes two groups of signals based on timestamps.
        Inputs assumed format: [Raw_Time, Rel_Time, Val1, Val2, ...]
        
        Strategy:
        1. Pick the group with FEWER samples (lower frequency) as the Reference Time Basis.
        2. Interpolate the OTHER group's values to match these timestamps.
           (Interpolation is generally better than nearest neighbor for continuous physical signals).
        
        Returns:
            (Time_Cols, Aligned_Vals_A, Aligned_Vals_B)
        """
        if group_a.shape[0] == 0 or group_b.shape[0] == 0:
             return np.array([]), np.array([]), np.array([])

        # Identify Reference (Shorter)
        if group_a.shape[0] <= group_b.shape[0]:
            ref_group, sec_group = group_a, group_b
            swapped = False
        else:
            ref_group, sec_group = group_b, group_a
            swapped = True
            
        ref_times = ref_group[:, 0]
        sec_times = sec_group[:, 0]
        
        # Prepare Output Arrays
        # Reference values are taken as-is
        aligned_ref_vals = ref_group[:, 2:]
        
        # Secondary values need interpolation
        sec_vals = sec_group[:, 2:]
        num_cols = sec_vals.shape[1]
        aligned_sec_vals = np.zeros((len(ref_times), num_cols))
        
        for i in range(num_cols):
            # linear interpolation for each component
            aligned_sec_vals[:, i] = np.interp(ref_times, sec_times, sec_vals[:, i])
            
        # Return properly ordered
        time_cols = ref_group[:, :2]
        
        if not swapped:
            return time_cols, aligned_ref_vals, aligned_sec_vals
        else:
            return time_cols, aligned_sec_vals, aligned_ref_vals