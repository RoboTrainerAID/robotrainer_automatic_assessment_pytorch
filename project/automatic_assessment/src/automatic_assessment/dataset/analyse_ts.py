import pandas as pd
import os
import matplotlib.pyplot as plt
import yaml
import glob
import math
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator
from typing import List, Optional, Union, Dict, Any
from collections import Counter

from automatic_assessment.dataset.timeseries_loader import TimeseriesLoader, PathData, TimeseriesDataset
import automatic_assessment.dataset.config as config

def get_timeseries_as_df(path_data: PathData, ts_names: List[str]) -> pd.DataFrame:
    """
    Helper function to align multiple timeseries from PathData into a single DataFrame based on timestamp.
    Assumes index 0 is timestamp and index 2 (COL_VALUE) is the value.
    """
    dfs = []
    
    for name in ts_names:
        if name not in path_data.timeseries:
            print(f"Warning: Timeseries {name} not found in path data.")
            continue
        
        arr = path_data.timeseries[name]
        # Create temporary DF for alignment
        # arr[:, 0] is Timestamp, arr[:, 2] is Value
        temp_df = pd.DataFrame(arr, columns=["time", "rel_time", "value", *[f"extra_{i}" for i in range(arr.shape[1]-3)]])
        # We only care about time and value for now
        temp_df = temp_df[["time", "value"]].rename(columns={"value": name})
        # Sort by time just in case
        temp_df = temp_df.sort_values("time")
        dfs.append(temp_df)
    
    if not dfs:
        return pd.DataFrame()

    # Use the first DF as base and merge others using merge_asof
    base_df = dfs[0]
    for i in range(1, len(dfs)):
        base_df = pd.merge_asof(base_df, dfs[i], on="time", direction="nearest")
        
    return base_df

def analyze_dataset_summary(path_data: PathData) -> None:
    """
    Prints a summary of the loaded PathData including available timeseries and stats.

    Args:
        path_data (PathData): The loaded path data object.
    """
    print("\n--- Path Data Summary Analysis ---")
    print(f"User ID: {path_data.user_id}, Path ID: {path_data.path_id}")
    
    if not path_data.timeseries and not path_data.scalar_features:
        print("Dataset is empty.")
        return

    # 3. Full list of columns with stats
    print("\nTimeseries Statistics:")
    print(f"{'Name':<30} | {'Freq (Hz)':<10} | {'Min':<12} | {'Max':<12} | {'Mean':<12}")
    print("-" * 88)
    
    for name, arr in path_data.timeseries.items():
        try:
            # Calculate Frequency
            if arr.shape[0] > 1:
                times = arr[:, config.COL_REL_TS]
                duration = times[-1] - times[0]
                if duration > 0:
                    freq = (arr.shape[0] - 1) / duration
                    freq_str = f"{freq:.1f}" 
                else:
                    freq_str = "inf"
            else:
                freq_str = "-"

            # Assuming value is at index 2 (COL_VALUE)
            vals = arr[:, config.COL_VALUE]
            min_val = np.min(vals)
            max_val = np.max(vals)
            mean_val = np.mean(vals)
            
            min_str = f"{min_val:.4g}"
            max_str = f"{max_val:.4g}"
            mean_str = f"{mean_val:.4g}"
            
        except Exception:
            freq_str, min_str, max_str, mean_str = "Err", "Err", "Err", "Err"

        print(f"{name:<30} | {freq_str:<10} | {min_str:<12} | {max_str:<12} | {mean_str:<12}")

    if path_data.scalar_features:
        print("\nScalar Features:")
        for k, v in path_data.scalar_features.items():
            print(f"  {k}: {v}")


def plot_timeseries_over_time(path_data: PathData, ts_names: List[str]) -> None:
    """
    Plots one or multiple timeseries over time as line graphs.
    """
    if not ts_names:
        return

    df = get_timeseries_as_df(path_data, ts_names)
    if df.empty or "time" not in df.columns:
        print("No valid data found to plot time series.")
        return

    # Convert timestamp to relative time starting at 0 for better readability
    start_time = df["time"].iloc[0]
    rel_time = df["time"] - start_time

    plt.figure(figsize=(12, 6))
    for name in ts_names:
        if name in df.columns:
            plt.plot(rel_time, df[name], label=name)

    plt.title(f"Timeseries for {path_data.path_id}")
    plt.xlabel("Time (s)")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    
    filename = "_".join(ts_names)[:50] + "_timeline.png"
    output_path = os.path.join("/workspace/automatic_assessment/figures/dataset", filename)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    print(f"Timeseries plot saved to {output_path}")
    plt.close()


def plot_histogram(path_data: PathData, ts_name: str, bins: int = 50) -> None:
    """
    Plots a histogram for a given timeseries in the PathData.
    """
    if ts_name not in path_data.timeseries:
        print(f"Error: Timeseries '{ts_name}' not found.")
        return

    try:
        arr = path_data.timeseries[ts_name]
        vals = arr[:, config.COL_VALUE]

        plt.figure(figsize=(10, 6))
        plt.hist(vals, bins=bins, edgecolor='black', alpha=0.7)
        plt.title(f"Histogram of {ts_name}")
        plt.xlabel(ts_name)
        plt.ylabel("Frequency")
        plt.grid(axis='y', alpha=0.5)
        plt.savefig(f"/workspace/automatic_assessment/figures/dataset/{ts_name}_histogram.png")
        print(f"Histogram for '{ts_name}' plotted.")
    except Exception as e:
        print(f"Error plotting histogram for '{ts_name}': {e}")


def plot_robot_path(path_data: PathData) -> None:
    """
    Plots the 2D path of the robot using robot_pos_x and robot_pos_y columns.
    """
    x_col = "robot_pos_x"
    y_col = "robot_pos_y"

    df = get_timeseries_as_df(path_data, [x_col, y_col])
    
    if x_col not in df.columns or y_col not in df.columns:
        print(f"Error: Required columns '{x_col}' or '{y_col}' for path plotting not found.")
        return

    try:
        plt.figure(figsize=(10, 8))
        # Plot path
        plt.plot(df[x_col], df[y_col], linestyle='-', linewidth=1, label='Path')
        
        # Mark start and end points
        if not df.empty:
            plt.scatter(df[x_col].iloc[0], df[y_col].iloc[0], c='green', marker='o', label='Start')
            plt.scatter(df[x_col].iloc[-1], df[y_col].iloc[-1], c='red', marker='x', label='End')

        plt.title(f"Robot Path (2D Position) - {path_data.path_id}")
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.legend()
        plt.grid(True)
        plt.axis('equal')  # Ensure aspect ratio is equal for correct spatial representation

        # Ensure directory exists
        output_path = "/workspace/automatic_assessment/figures/dataset/robot_path.png"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        plt.savefig(output_path)
        print(f"Robot path plot saved to {output_path}")
        plt.close()
    except Exception as e:
        print(f"Error plotting robot path: {e}")


def load_scenario_by_path_number(path_number: int, folder_path: str) -> dict:
    """
    Loads a scenario YAML file corresponding to a path number from a folder.
    Matches files starting with '{path_number}_'.
    """
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Scenario folder not found: {folder_path}")

    pattern = os.path.join(folder_path, f"{path_number}_*.yaml")
    files = glob.glob(pattern)
    
    if not files:
        raise FileNotFoundError(f"No scenario file found searching for pattern: {pattern}")
    
    # Load the first match
    file_path = files[0]
    # print(f"Loading scenario file: {file_path}")
    with open(file_path, 'r') as f:
        return yaml.safe_load(f)


def parse_scenario(scenario_dict: dict) -> dict:
    """
    Parses a loaded scenario dictionary into a structured format for plotting.
    Extracts path points, forces (scaled to Newtons and Meters), and areas.
    """
    parsed = {
        'path_points': [],
        'forces': [],
        'areas': []
    }

    # 1. Parse Path Points
    path_data = scenario_dict.get('path', {})
    point_names = path_data.get('points', [])
    
    # Iterate through the list of point keys (point0, point1...)
    for key in point_names:
        if key in path_data:
            pt = path_data[key]
            parsed['path_points'].append((pt.get('x', 0.0), pt.get('y', 0.0)))

    # 2. Parse Forces
    force_section = scenario_dict.get('force', {})
    config = force_section.get('config', {})
    newton_per_meter = config.get('newton_per_meter', 1.0)
    force_names = config.get('force_names', [])
    data_section = force_section.get('data', {})

    for fname in force_names:
        if fname not in data_section:
            continue
        
        fdata = data_section[fname]
        area = fdata.get('area', {})
        arrow = fdata.get('arrow', {})
        margin = fdata.get('margin', {})

        # Center position
        start_x = area.get('x', 0.0)
        start_y = area.get('y', 0.0)

        # Margin position for radius calculation
        margin_x = margin.get('x', 0.0)
        margin_y = margin.get('y', 0.0)
        
        # Radius is distance from center to margin
        radius = math.sqrt((start_x - margin_x)**2 + (start_y - margin_y)**2)

        # Arrow vector logic:
        arrow_val_x = arrow.get('x', 0.0)
        arrow_val_y = arrow.get('y', 0.0)

        strength = math.sqrt(arrow_val_x**2 + arrow_val_y**2)
        
        # Scale for plotting (meters)
        vec_x = arrow_val_x / newton_per_meter
        vec_y = arrow_val_y / newton_per_meter

        parsed['forces'].append({
            'name': fname,
            'center': (start_x, start_y),
            'radius': radius,
            'vector': (vec_x, vec_y),
            'strength': strength
        })

    # 3. Parse Areas (Regions without force vectors, e.g., inverted rotation)
    area_section = scenario_dict.get('area', {})
    if area_section:
        config_area = area_section.get('config', {})
        area_names = config_area.get('area_names', [])
        data_section = area_section.get('data', {})

        for aname in area_names:
            if aname not in data_section:
                continue
            
            adata = data_section[aname]
            center_info = adata.get('area', {})
            margin_info = adata.get('margin', {})
            funcs = adata.get('area_functions', [])
            
            cx = center_info.get('x', 0.0)
            cy = center_info.get('y', 0.0)
            
            mx = margin_info.get('x', 0.0)
            my = margin_info.get('y', 0.0)
            
            radius = math.sqrt((cx - mx)**2 + (cy - my)**2)
            
            label = ", ".join(funcs) if funcs else aname

            parsed['areas'].append({
                'name': aname,
                'center': (cx, cy),
                'radius': radius,
                'label': label
            })

    return parsed


def find_scenario_pois(parsed_scenario: dict, path_data: PathData) -> Dict[str, List[tuple]]:
    """
    Finds points of interest where the robot enters or leaves force or effect areas.
    Returns a dictionary with keys 'forces' and 'areas' containing list of (x,y) tuples.
    """
    pois = {'forces': [], 'areas': []}
    
    # Load required timeseries
    df = get_timeseries_as_df(path_data, ["robot_pos_x", "robot_pos_y"])
    
    if df.empty or "robot_pos_x" not in df.columns or "robot_pos_y" not in df.columns:
        return pois

    # Create numpy arrays for faster computation
    path_x = df["robot_pos_x"].values
    path_y = df["robot_pos_y"].values
    
    # Helper to find crossings
    def _get_crossings(regions):
        crossings = []
        for region in regions:
            cx, cy = region['center']
            r = region['radius']
            
            # Calculate distance squared from center for all points
            dist_sq = (path_x - cx)**2 + (path_y - cy)**2
            r_sq = r**2
            
            # Boolean array: True if inside, False if outside
            inside = dist_sq < r_sq
            
            # Find transitions: where value changes from previous
            if len(inside) > 1:
                # XOR with shifted version to find indices where status changes
                transitions = np.where(inside[:-1] != inside[1:])[0]
                
                for idx in transitions:
                    # idx is the point before change, take idx + 1 as the crossing point
                    crossings.append((path_x[idx + 1], path_y[idx + 1]))
        return crossings

    pois['forces'] = _get_crossings(parsed_scenario.get('forces', []))
    pois['areas'] = _get_crossings(parsed_scenario.get('areas', []))
            
    return pois


def find_detailed_scenario_pois(parsed_scenario: dict, path_data: PathData) -> List[Dict[str, Any]]:
    """
    Finds POIs with detailed info: timestamp, index, coordinate, region name, type (entry/exit).
    Returns a list of dicts sorted by index (time).
    Groups POIs by area and filters jitter (entry/exit pairs < 0.2m) if count > 2.
    """
    detailed_pois = []
    
    df = get_timeseries_as_df(path_data, ["robot_pos_x", "robot_pos_y"])
    if df.empty or "robot_pos_x" not in df.columns: 
        return []

    path_x = df["robot_pos_x"].values
    path_y = df["robot_pos_y"].values
    time_vals = df["time"].values
    
    # Determine offset to relative coordinates if needed (assuming raw coords in file match scenario coords)
    # Usually scenario coords are global.

    def _process_regions(regions, region_cat):
        for region in regions:
            name = region.get('name', 'unknown')
            cx, cy = region['center']
            r = region['radius']
            
            dist_sq = (path_x - cx)**2 + (path_y - cy)**2
            r_sq = r**2
            inside = dist_sq < r_sq # boolean array
            
            local_pois = []
            
            if len(inside) > 1:
                # Find indices where state changes
                # inside[:-1] != inside[1:] gives indices i where status at i is different from i+1
                transitions = np.where(inside[:-1] != inside[1:])[0]
                
                for idx in transitions:
                    # transition is happening between idx and idx+1
                    # check direction
                    is_entering = not inside[idx] and inside[idx+1]
                    transition_type = "entry" if is_entering else "exit"
                    
                    local_pois.append({
                        'index': idx + 1,
                        'time': time_vals[idx + 1],
                        'x': path_x[idx + 1],
                        'y': path_y[idx + 1],
                        'region_name': name,
                        'region_category': region_cat, # 'force' or 'area'
                        'transition': transition_type
                    })
            
            # Group filtering logic: If > 2 points, filter close consecutive pairs (jitter)
            if len(local_pois) > 2:
                changed = True
                # Iterative pass to handle potential chain removals or complex jitter
                while changed and len(local_pois) > 2:
                    changed = False
                    indices_to_remove = set()
                    
                    i = 0
                    while i < len(local_pois) - 1:
                        p1 = local_pois[i]
                        p2 = local_pois[i+1]
                        
                        # Check distance between consecutive events
                        dist = math.sqrt((p1['x'] - p2['x'])**2 + (p1['y'] - p2['y'])**2)
                        
                        if dist < 0.2:
                            # Mark pair for removal
                            indices_to_remove.add(i)
                            indices_to_remove.add(i+1)
                            i += 2 # Skip pair
                            changed = True
                        else:
                            i += 1
                    
                    if indices_to_remove:
                        local_pois = [p for k, p in enumerate(local_pois) if k not in indices_to_remove]

            detailed_pois.extend(local_pois)

    _process_regions(parsed_scenario.get('forces', []), 'force')
    _process_regions(parsed_scenario.get('areas', []), 'area')
    
    # Sort by appearance in path
    detailed_pois.sort(key=lambda x: x['index'])
    return detailed_pois


def analyze_scenario_pois_distribution(dataset: TimeseriesDataset, scenario_folder: str) -> None:
    """
    Iterates through every user and path, calculates scenario POIs, and summarizes the distribution
    relative to the expected number of interactions based on the scenario file.
    """
    print("\n--- Analyzing Scenario POI Distribution ---")
    
    delta_counts = [] # Difference: Actual - Expected
    path_details = []
    
    total_paths = 0
    skipped = 0
    perfect_matches = 0

    for user_id, paths in dataset.items():
        for path_id, path_data in paths.items():
            total_paths += 1
            try:
                # Load and parse scenario
                raw_scenario = load_scenario_by_path_number(path_id, scenario_folder)
                parsed_scen = parse_scenario(raw_scenario)
                
                # Calculate Expected POIs (Entry + Exit for each region)
                if path_id == 11:
                    # Special Case: Path 11 is designed to stay completely inside an area.
                    # Ideally 0 crossings if performed correctly.
                    expected_count = 0
                else:
                    num_forces = len(parsed_scen.get('forces', []))
                    num_areas = len(parsed_scen.get('areas', []))
                    expected_count = 2 * (num_forces + num_areas)
                
                # Find detailed POIs
                pois = find_detailed_scenario_pois(parsed_scen, path_data)
                actual_count = len(pois)
                
                delta = actual_count - expected_count
                delta_counts.append(delta)
                
                if delta == 0:
                    perfect_matches += 1
                
                # Store summary for this path
                regions_touched = set([p['region_name'] for p in pois])
                
                path_details.append({
                    'u_id': user_id,
                    'p_id': path_id,
                    'actual': actual_count,
                    'expected': expected_count,
                    'delta': delta,
                    'regions': list(regions_touched)
                })
                
            except (FileNotFoundError, Exception) as e:
                # print(f"Skipping U{user_id} P{path_id}: {e}")
                skipped += 1
                continue

    # Summary Statistics
    print(f"\nTotal Paths Processed: {total_paths}")
    print(f"Skipped (No Scenario/Error): {skipped}")
    print(f"Perfect Scenario Matches (Actual == Expected): {perfect_matches}")
    
    if not delta_counts:
        print("No POIs calculated.")
        return
        
    freq_dist = Counter(delta_counts)
    
    print("\nDeviation from Expected POI Count (Actual - Expected):")
    print(f"{'Deviation':<10} | {'Frequency':<10} | {'Percentage':<10}")
    print("-" * 36)
    
    # Sort by deviation value
    sorted_deltas = sorted(freq_dist.keys())
    n_successful = len(delta_counts)
    
    for d in sorted_deltas:
        freq = freq_dist[d]
        perc = (freq / n_successful) * 100
        sign = "+" if d > 0 else ""
        label = f"{sign}{d}"
        print(f"{label:<10} | {freq:<10} | {perc:<9.1f}%")
        
    print("-" * 36)
    
    # Analyze outliers
    print("\nSignificant Deviation Examples (> +/- 2):")
    for d in path_details:
        if abs(d['delta']) >= 2:
            status = "Extra interactions" if d['delta'] > 0 else "Missed interactions"
            print(f"  U{d['u_id']} P{d['p_id']}: {d['actual']} vs {d['expected']} expected ({status})")


def find_sensor_pois(path_data: PathData, sensor_col: str) -> list:
    """
    Finds points of interest derived from sensor data extremes.
    """
    pois = []
    
    df = get_timeseries_as_df(path_data, ["robot_pos_x", "robot_pos_y", sensor_col])
    
    if df.empty or sensor_col not in df.columns:
        return pois
    
    # Ensure standard index for referencing
    df = df.reset_index(drop=True)
    sensor_series = df[sensor_col]
    
    if sensor_series.dropna().empty:
        return pois

    # Indices of interest
    indices = set()
    
    # Max and Min values
    indices.add(sensor_series.idxmax())
    indices.add(sensor_series.idxmin())
    
    # Max and Min Derivative
    derivative = sensor_series.diff()
    if not derivative.dropna().empty:
        indices.add(derivative.idxmax())
        indices.add(derivative.idxmin())
    
    # Extract Coordinates
    for idx in indices:
        if pd.notna(idx): 
            row = df.iloc[idx]
            pois.append((row["robot_pos_x"], row["robot_pos_y"]))
            
    return pois


def plot_scenario_with_robot(parsed_scenario: dict, path_data: PathData = None, 
                             sensor_col: str = None, 
                             scenario_pois: Dict[str, List[tuple]] = None, sensor_pois: list = None) -> None:
    """
    Plots the scenario path, forces, areas, and overlays the robot path if provided.
    Offsets all coordinates so the robot start position is (0,0).
    Also mirrors the X-axis (multiplies X by -1) so movements in negative X appear as positive distance.
    """
    try:
        # Determine Offset based on Robot Start Position
        offset_x, offset_y = 0.0, 0.0
        
        plot_df = pd.DataFrame()
        if path_data:
            cols_to_load = ["robot_pos_x", "robot_pos_y"]
            if sensor_col:
                cols_to_load.append(sensor_col)
            plot_df = get_timeseries_as_df(path_data, cols_to_load)
            
            if not plot_df.empty:
                offset_x = plot_df["robot_pos_x"].iloc[0]
                offset_y = plot_df["robot_pos_y"].iloc[0]
                # Shift robot path
                plot_df["robot_pos_x"] -= offset_x
                plot_df["robot_pos_y"] -= offset_y
                
                # MIRROR X AXIS (Transform data to positive coordinates)
                plot_df["robot_pos_x"] *= -1

        plt.figure(figsize=(10, 4))
        ax = plt.gca()

        # Plot Scenario Path (Shifted & Mirrored)
        path_points = parsed_scenario.get('path_points', [])
        if path_points:
            xs, ys = zip(*path_points)
            xs = (np.array(xs) - offset_x) * -1 # Mirror
            ys = np.array(ys) - offset_y
            plt.plot(xs, ys, color='black', linewidth=2, label='Scenario Path') #alpha=0.5)

        # Plot Forces (Shifted & Mirrored)
        has_forces = False
        for force in parsed_scenario.get('forces', []):
            has_forces = True
            cx = (force['center'][0] - offset_x) * -1 # Mirror center
            cy = force['center'][1] - offset_y
            radius = force['radius']
            vx, vy = force['vector']
            vx = vx * -1 # Mirror vector direction
            strength = force['strength']

            # Force Area (Circle)
            circle = plt.Circle((cx, cy), radius, color='red', fill=False, linestyle='--', linewidth=1.5)
            ax.add_patch(circle)

            # Force Vector (Arrow)
            plt.arrow(cx, cy, vx, vy, color='red', head_width=0.15, length_includes_head=True)
            
            # Label
            label_x = cx + vx
            label_y = cy + vy
            plt.text(label_x - 0.7, label_y, f"{strength:.0f} N", color='red', fontsize=10, fontweight='bold')

        # Plot Areas (Shifted & Mirrored)
        has_areas = False
        for area in parsed_scenario.get('areas', []):
            has_areas = True
            cx = (area['center'][0] - offset_x) * -1 # Mirror center
            cy = area['center'][1] - offset_y
            radius = area['radius']
            label = area['label']
            
            # Area Circle (Orange dashed/dot)
            circle = plt.Circle((cx, cy), radius, color='orange', fill=False, linestyle='-.', linewidth=1.5)
            ax.add_patch(circle)
            
            # Label (Function name)
            plt.text(cx, cy + radius -0.4, label, color='orange', fontsize=9, fontweight='bold', ha='center')

        # Overlay Robot Path
        if not plot_df.empty:
            x_vals = plot_df["robot_pos_x"].values
            y_vals = plot_df["robot_pos_y"].values
            
            # Mark Start (0,0) and End points
            plt.scatter(0, 0, c='black', marker='o', s=60, zorder=5)
            plt.text(0, -0.2, "Start", fontsize=10, color='black', fontweight='bold', zorder=5, ha='center', va='top')
            
            plt.scatter(x_vals[-1], y_vals[-1], c='black', marker='o', s=60, zorder=5)
            plt.text(x_vals[-1], y_vals[-1] - 0.2, "End", fontsize=10, color='black', fontweight='bold', zorder=5, ha='center', va='top')

            if sensor_col and sensor_col in plot_df.columns:
                sensor_vals = plot_df[sensor_col].values
                
                # Continuous LineCollection for smooth gradient
                points = np.array([x_vals, y_vals]).T.reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)

                # Cap styles projecting/round help fill gaps between segments
                norm = plt.Normalize(sensor_vals.min(), sensor_vals.max())
                lc = LineCollection(segments, cmap='viridis', norm=norm, zorder=4)
                
                # Use value at start of segment for color
                lc.set_array(sensor_vals[:-1])
                lc.set_linewidth(10) 
                
                # 'round' capstyle smooths joints visually
                # lc.set_capstyle('round') 
                
                line = ax.add_collection(lc)
                cbar = plt.colorbar(line, ax=ax, label=sensor_col)
            else:
                plt.plot(x_vals, y_vals, color='blue', linestyle='-', alpha=0.6, linewidth=5)
        else:
            print("Warning: No valid pose data available for overlay.")

        # Plot Scenario POIs (Shifted & Mirrored)
        if scenario_pois:
            force_pts = scenario_pois.get('forces', [])
            area_pts = scenario_pois.get('areas', [])
            
            if force_pts:
                fx = (np.array([p[0] for p in force_pts]) - offset_x) * -1
                fy = np.array([p[1] for p in force_pts]) - offset_y
                plt.scatter(fx, fy, c='red', marker='x', s=100, linewidths=2.5, zorder=10, label='Area Event')
            
            if area_pts:
                ax_pts = (np.array([p[0] for p in area_pts]) - offset_x) * -1
                ay_pts = np.array([p[1] for p in area_pts]) - offset_y
                plt.scatter(ax_pts, ay_pts, c='orange', marker='x', s=100, linewidths=2.5, zorder=10, label='Area Event')

        # Plot Sensor POIs (Shifted & Mirrored)
        if sensor_pois:
            sens_x = (np.array([p[0] for p in sensor_pois]) - offset_x) * -1
            sens_y = np.array([p[1] for p in sensor_pois]) - offset_y
            plt.scatter(sens_x, sens_y, facecolors='none', edgecolors='magenta', s=300, linewidths=2.5, zorder=10, label='Sensor Event')
                
        # Construct Custom Legend
        legend_elements = [Line2D([0], [0], color='black', lw=2, label='Scenario Path')] #alpha=0.5
        if has_forces: legend_elements.append(Line2D([0], [0], color='red', linestyle='--', lw=1.5, label='Force Area'))
        if has_areas: legend_elements.append(Line2D([0], [0], color='orange', linestyle='-.', lw=1.5, label='Effect Area'))
        legend_elements.extend([
            Line2D([0], [0], color='teal' if sensor_col else 'blue', lw=3, label='Robot Path'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='black', markersize=8, label='Start, End')
        ])
        
        if scenario_pois:
             if isinstance(scenario_pois, dict):
                 if scenario_pois.get('forces'):
                     legend_elements.append(Line2D([0], [0], marker='x', color='w', markeredgecolor='red', markersize=10, markeredgewidth=2, label='Area Event'))
                 if scenario_pois.get('areas'):
                     legend_elements.append(Line2D([0], [0], marker='x', color='w', markeredgecolor='orange', markersize=10, markeredgewidth=2, label='Area Event'))
             else:
                 legend_elements.append(Line2D([0], [0], marker='x', color='w', markeredgecolor='red', markersize=10, markeredgewidth=2, label='Scenario Event'))
                 
        if sensor_pois:
            legend_elements.append(Line2D([0], [0], marker='o', color='w', markeredgecolor='magenta', markerfacecolor='none', markersize=10, markeredgewidth=2, label='Sensor Event'))


        plt.legend(handles=legend_elements, loc='upper left')

        title_str = "Scenario Visualization"
        if path_data: title_str += f" ({path_data.path_id})"
        plt.title(title_str)
        plt.xlabel("X [m]")
        plt.ylabel("Y [m]")
        
        # Grid settings
        plt.axis('equal')
        ax.xaxis.set_major_locator(MultipleLocator(1))
        ax.yaxis.set_major_locator(MultipleLocator(1))
        plt.grid(True, which='major', linestyle='-', alpha=0.5)

        # Removed invert_xaxis() as data is now manually mirrored
        # ax.invert_xaxis()

        filename = f"scenario_overlay_{sensor_col}.png" if sensor_col else "scenario_overlay.png"
        output_path = os.path.join("/workspace/automatic_assessment/figures/dataset", filename)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path)
        print(f"Scenario overlay plot saved to {output_path}")
        plt.close()

    except Exception as e:
        print(f"Error plotting scenario: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    
    # 1. Initialize Loader and Load Data
    scenario_folder = "/data/raw/scenarios"
    dataset_path = "/data/raw/timeseries_numpy_processed"
    print(f"Loading processed dataset from: {dataset_path}")
    
    loader = TimeseriesLoader(config)
    dataset = loader.load(dataset_path)
    
    # 2. Select User and Path for Analysis using IDs (Integers)
    target_user_id = 12  
    target_path_id = 9   
    
    path_data = dataset[target_user_id][target_path_id]

    # --- New Distribution Analysis ---
    # analyze_scenario_pois_distribution(dataset, scenario_folder)
    
    # 3. Analyze
    analyze_dataset_summary(path_data)
    
    # 4. Plots
    # plot_histogram(path_data, "hrv")
    # plot_robot_path(path_data)
    # plot_timeseries_over_time(path_data, ["disturbance_force_x", "disturbance_force_y", "disturbance_force_lin_mag"])
    # plot_timeseries_over_time(path_data, ["robot_vel_x", "robot_vel_y", "robot_vel_lin_mag", "robot_vel_total_mag"])
    # plot_timeseries_over_time(path_data, ["user_force_x", "user_force_y", "user_force_lin_mag", "user_force_total_mag"])
    # plot_timeseries_over_time(path_data, ["user_power", "user_force_lin_mag", "user_force_total_mag", "user_work_cum"])
    # plot_timeseries_over_time(path_data, ["ppi", "hrv", "heart_rate"])

    # 5. Scenario Overlay
    raw_scenario = load_scenario_by_path_number(path_number=target_path_id, folder_path=scenario_folder)
    parsed_data = parse_scenario(raw_scenario)
    
    # Calculate POIs
    sensor_column = "user_power"
    scen_pois = find_scenario_pois(parsed_data, path_data)
    sens_pois = find_sensor_pois(path_data, sensor_column)
    
    plot_scenario_with_robot(parsed_data, path_data, sensor_col=sensor_column, 
                                scenario_pois=scen_pois, sensor_pois=sens_pois)
