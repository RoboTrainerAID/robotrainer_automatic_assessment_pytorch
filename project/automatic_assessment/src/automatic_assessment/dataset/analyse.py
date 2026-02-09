import pandas as pd
import os
import matplotlib.pyplot as plt
import yaml
import glob
import math
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

def load_dataset(file_path: str) -> pd.DataFrame:
    """
    Loads a CSV file into a pandas DataFrame.

    Args:
        file_path (str): The path to the .csv file.

    Returns:
        pd.DataFrame: The loaded data.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Error: File not found at {file_path}")
    
    # low_memory=False is often helpful for large datasets with mixed types
    return pd.read_csv(file_path, low_memory=False)

def save_dataset(df: pd.DataFrame, file_path: str) -> None:
    """
    Saves a pandas DataFrame to a CSV file.

    Args:
        df (pd.DataFrame): The DataFrame to save.
        file_path (str): The destination path.
    """
    try:
        df.to_csv(file_path, index=False)
        print(f"Dataset successfully saved to {file_path}")
    except Exception as e:
        print(f"Error saving dataset: {e}")


def extract_user_dataset(source_path: str, dest_path: str, user_id: int) -> None:
    """
    Loads the full dataset, filters for a specific user, and saves the reduced dataset.

    Args:
        source_path (str): The file path of the full dataset.
        dest_path (str): The file path to save the filtered dataset.
        user_id (int): The user ID to filter by.
    """
    print(f"Extracting data for user {user_id} from sources...")
    try:
        df = load_dataset(source_path)
        if "user" in df.columns:
            user_df = df[df["user"] == user_id]
            if not user_df.empty:
                save_dataset(user_df, dest_path)
            else:
                print(f"No data found for user {user_id}.")
        else:
            print("Column 'user' not found in dataset.")
    except Exception as e:
        print(f"Error extracting user dataset: {e}")


def analyze_dataset_summary(df: pd.DataFrame) -> None:
    """
    Prints a summary of the dataset including path info, durations, and column frequencies.
    Also reports Min, Max, and Mean for each column.

    Args:
        df (pd.DataFrame): The pandas DataFrame to analyze.
    """
    print("\n--- Dataset Summary Analysis ---")

    if df.empty:
        print("Dataset is empty, cannot generate summary.")
        return

    if "time" not in df.columns:
        print("Error: 'time' column required for analysis.")
        return

    # Ensure time is proper type (numeric or datetime)
    if not pd.api.types.is_numeric_dtype(df["time"]):
        try:
            df["time"] = pd.to_datetime(df["time"])
        except Exception:
            pass  # Attempt to continue

    # 1. Report number of paths
    # Assuming 'path' column exists based on requirement, otherwise infer 1 path
    path_col = "path" if "path" in df.columns else None
    
    # Store time series for easy access
    time_data = df["time"]

    if path_col:
        num_paths = df[path_col].nunique()
        print(f"Number of paths: {num_paths}")
        
        # 2. Duration of each path sorted by min to max
        # Group by path, calculate range
        groups = df.groupby(path_col)["time"]
        durations = groups.max() - groups.min()
        sorted_durations = durations.sort_values()
        
        print("\nDuration of each path (sorted min to max):")
        print(sorted_durations)
        
    else:
        print("Number of paths: 1 (inferred)")
        min_t = df["time"].min()
        max_t = df["time"].max()
        total_dur_val = max_t - min_t
        print(f"Duration: {total_dur_val}")
    
    # 3. Full list of columns with stats
    print("\nColumn Statistics:")
    print(f"{'Column Name':<30} | {'Freq (Hz)':<10} | {'Points':<8} | {'Min':<12} | {'Max':<12} | {'Mean':<12}")
    print("-" * 95)
    
    for col in df.columns:
        # Determine valid data: No NaNs, and for numeric columns (except time) No 0s
        # This prevents leading/trailing zeros from artificially extending duration or count
        is_numeric = pd.api.types.is_numeric_dtype(df[col])
        
        if is_numeric and col != "time":
            mask = (df[col].notna()) & (df[col] != 0)
        else:
            mask = df[col].notna()

        # Extract valid data 
        valid_data = df.loc[mask, col]
        
        count = len(valid_data)
        freq = 0.0
        
        # Calculate duration based on the timestamps of first and last valid data points
        if count > 1:
            # Get the index of the first and last valid measurement
            first_idx = valid_data.index[0]
            last_idx = valid_data.index[-1]
            
            t_start = df.loc[first_idx, "time"]
            t_end = df.loc[last_idx, "time"]
            
            col_duration = t_end - t_start
            
            # Convert duration to seconds float
            if hasattr(col_duration, 'total_seconds'):
                dur_seconds = col_duration.total_seconds()
            else:
                dur_seconds = float(col_duration)
            
            if dur_seconds > 0:
                freq = count / dur_seconds
        
        try:
            min_str, max_str, mean_str = "N/A", "N/A", "N/A"
            
            if not valid_data.empty:
                min_val = valid_data.min()
                max_val = valid_data.max()
                
                if is_numeric:
                    mean_val = valid_data.mean()
                    mean_str = f"{mean_val:.4g}"
                    min_str = f"{min_val:.4g}"
                    max_str = f"{max_val:.4g}"
                else:
                    min_str = str(min_val)
                    max_str = str(max_val)
            
            # Truncate strings if too long
            if len(min_str) > 12: min_str = min_str[:10] + ".."
            if len(max_str) > 12: max_str = max_str[:10] + ".."
            
        except Exception:
            min_str, max_str, mean_str = "Err", "Err", "Err"

        print(f"{col:<30} | {freq:<10.3f} | {count:<8} | {min_str:<12} | {max_str:<12} | {mean_str:<12}")


def plot_histogram(df: pd.DataFrame, column: str, bins: int = 50) -> None:
    """
    Plots a histogram for a given column in the DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        column (str): The name of the column to plot.
        bins (int): Number of bins for the histogram.
    """
    if column not in df.columns:
        print(f"Error: Column '{column}' not found in dataset.")
        return

    try:
        if not pd.api.types.is_numeric_dtype(df[column]):
            print(f"Warning: Column '{column}' is not numeric. Plotting may fail or look unexpected.")

        plt.figure(figsize=(10, 6))
        df[column].hist(bins=bins, edgecolor='black', alpha=0.7)
        plt.title(f"Histogram of {column}")
        plt.xlabel(column)
        plt.ylabel("Frequency")
        plt.grid(axis='y', alpha=0.5)
        plt.savefig(f"/workspace/automatic_assessment/figures/dataset/{column}_histogram.png")
        print(f"Histogram for '{column}' plotted.")
    except Exception as e:
        print(f"Error plotting histogram for '{column}': {e}")


def plot_robot_path(df: pd.DataFrame) -> None:
    """
    Plots the 2D path of the robot using robot_pose_x and robot_pose_y columns.

    Args:
        df (pd.DataFrame): The DataFrame containing the robot position data.
    """
    x_col = "robot_pose_x"
    y_col = "robot_pose_y"

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

        plt.title("Robot Path (2D Position)")
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

    Args:
        path_number (int): The path identifier.
        folder_path (str): Directory containing scenario YAML files.

    Returns:
        dict: The loaded YAML content.
    """
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Scenario folder not found: {folder_path}")

    pattern = os.path.join(folder_path, f"{path_number}_*.yaml")
    files = glob.glob(pattern)
    
    if not files:
        raise FileNotFoundError(f"No scenario file found searching for pattern: {pattern}")
    
    # Load the first match
    file_path = files[0]
    print(f"Loading scenario file: {file_path}")
    with open(file_path, 'r') as f:
        return yaml.safe_load(f)


def parse_scenario(scenario_dict: dict) -> dict:
    """
    Parses a loaded scenario dictionary into a structured format for plotting.
    Extracts path points and forces (scaled to Newtons and Meters).

    Args:
        scenario_dict (dict): The raw YAML dictionary.

    Returns:
        dict: A dictionary containing 'path_points' (list of tuples) and 
              'forces' (list of dicts with center, radius, vector, strength).
    """
    parsed = {
        'path_points': [],
        'forces': []
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
        # The YAML 'arrow' x/y seems to represent the Force Vector scaled by NPM?
        # Based on example: 
        #   vector_in_meters = arrow_val / newton_per_meter
        #   strength_in_newtons = length_in_meters * newton_per_meter = length_of_arrow_val
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

    return parsed


def synchronize_sensor_to_pose(df: pd.DataFrame, sensor_col: str) -> pd.DataFrame:
    """
    Matches sensor data to robot location based on closest timestamp.

    Args:
        df (pd.DataFrame): Input dataframe with time, pose, and sensor columns.
        sensor_col (str): The sensor column name to sync.

    Returns:
        pd.DataFrame: DataFrame with matched robot_pose_x, robot_pose_y, and <sensor_col>.
    """
    required_cols = ["time", "robot_pose_x", "robot_pose_y", sensor_col]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"Missing columns for synchronization: {missing}")
        return pd.DataFrame()

    # Ensure sorted by time
    pose_df = df[["time", "robot_pose_x", "robot_pose_y"]].dropna().sort_values("time")
    sensor_df = df[["time", sensor_col]].dropna().sort_values("time")

    if pose_df.empty or sensor_df.empty:
        print("Pose or sensor data empty after filtering.")
        return pd.DataFrame()
    
    try:
        # Use merge_asof to find nearest sensor value for each pose.
        # This aligns the sensor readings to the pose timestamps.
        merged = pd.merge_asof(pose_df, sensor_df, on="time", direction="nearest")
        return merged
    except Exception as e:
        print(f"Synchronization error: {e}")
        return pd.DataFrame()


def find_scenario_pois(parsed_scenario: dict, robot_df: pd.DataFrame) -> list:
    """
    Finds points of interest where the robot enters or leaves force areas.

    Args:
        parsed_scenario (dict): The parsed scenario containing forces.
        robot_df (pd.DataFrame): The robot path dataframe.

    Returns:
        list: A list of (x, y) tuples representing intersections.
    """
    pois = []
    if robot_df is None or robot_df.empty:
        return pois
    if "robot_pose_x" not in robot_df.columns or "robot_pose_y" not in robot_df.columns:
        return pois

    # Create numpy arrays for faster computation
    path_x = robot_df["robot_pose_x"].values
    path_y = robot_df["robot_pose_y"].values
    
    # Iterate through each force to check for boundary crossings
    for force in parsed_scenario.get('forces', []):
        cx, cy = force['center']
        r = force['radius']
        
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
                pois.append((path_x[idx + 1], path_y[idx + 1]))
            
    return pois


def find_sensor_pois(robot_df: pd.DataFrame, sensor_col: str) -> list:
    """
    Finds points of interest derived from sensor data extremes and derivative extremes.
    Matches these points to robot locations using time synchronization.

    Args:
        robot_df (pd.DataFrame): Dataframe containing raw data.
        sensor_col (str): The sensor column name.

    Returns:
        list: A list of (x, y) tuples.
    """
    pois = []
    
    # helper function created before
    synced_df = synchronize_sensor_to_pose(robot_df, sensor_col)
    
    if synced_df.empty:
        return pois
    
    # Ensure standard index for referencing
    synced_df = synced_df.reset_index(drop=True)
    sensor_series = synced_df[sensor_col]
    
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
            row = synced_df.iloc[idx]
            pois.append((row["robot_pose_x"], row["robot_pose_y"]))
            
    return pois


def plot_scenario_with_robot(parsed_scenario: dict, robot_df: pd.DataFrame = None, 
                             sensor_col: str = None, path_id: int = None,
                             scenario_pois: list = None, sensor_pois: list = None) -> None:
    """
    Plots the scenario path and forces, and overlays the robot path if provided.
    Can also overlay points of interest.

    Args:
        parsed_scenario (dict): The output from parse_scenario.
        robot_df (pd.DataFrame, optional): DataFrame containing robot_pose_x/y.
        sensor_col (str, optional): Column name to use for coloring the robot path gradient.
        path_id (int, optional): The path ID for the title.
        scenario_pois (list, optional): List of (x,y) locations for scenario matches (Red Crosses).
        sensor_pois (list, optional): List of (x,y) locations for sensor matches (Circles).
    """
    try:
        plt.figure(figsize=(10, 4))
        ax = plt.gca()

        # Plot Scenario Path
        path_points = parsed_scenario.get('path_points', [])
        if path_points:
            xs, ys = zip(*path_points)
            plt.plot(xs, ys, color='black', linewidth=2, label='Scenario Path')

        # Plot Forces
        for force in parsed_scenario.get('forces', []):
            cx, cy = force['center']
            radius = force['radius']
            vx, vy = force['vector']
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

        # Overlay Robot Path
        if robot_df is not None and not robot_df.empty:
            
            # Prepare data: Synced if sensor_col provided, else raw pose
            plot_df = pd.DataFrame()
            if sensor_col:
                plot_df = synchronize_sensor_to_pose(robot_df, sensor_col)
            elif "robot_pose_x" in robot_df.columns and "robot_pose_y" in robot_df.columns:
                plot_df = robot_df

            if not plot_df.empty and "robot_pose_x" in plot_df.columns and "robot_pose_y" in plot_df.columns:
                x_vals = plot_df["robot_pose_x"].values
                y_vals = plot_df["robot_pose_y"].values
                
                # Mark Start and End points with distinct symbols and text
                # Start: Black dot, text under
                plt.scatter(x_vals[0], y_vals[0], c='black', marker='o', s=60, zorder=5)
                plt.text(x_vals[0], y_vals[0] - 0.4, "Start", fontsize=10, color='black', fontweight='bold', zorder=5, ha='center', va='top')
                
                # End: Black dot (was Red X), text under
                plt.scatter(x_vals[-1], y_vals[-1], c='black', marker='o', s=60, zorder=5)
                plt.text(x_vals[-1], y_vals[-1] - 0.4, "End", fontsize=10, color='black', fontweight='bold', zorder=5, ha='center', va='top')

                if sensor_col and sensor_col in plot_df.columns:
                    # Use LineCollection for a continuous, non-obstructing gradient line
                    sensor_vals = plot_df[sensor_col].values
                    
                    # Create segments (x, y) -> (next_x, next_y)
                    points = np.array([x_vals, y_vals]).T.reshape(-1, 1, 2)
                    segments = np.concatenate([points[:-1], points[1:]], axis=1)

                    # Create and add collection
                    norm = plt.Normalize(sensor_vals.min(), sensor_vals.max())
                    lc = LineCollection(segments, cmap='viridis', norm=norm, zorder=4)
                    
                    # Set array correlates color to the value at the starting point of the segment
                    lc.set_array(sensor_vals[:-1])
                    lc.set_linewidth(10)
                    line = ax.add_collection(lc)
                    
                    # Add colorbar for the gradient legend
                    cbar = plt.colorbar(line, ax=ax, label=sensor_col)
                    
                else:
                    # Fallback to single color line
                    plt.plot(x_vals, y_vals, 
                             color='blue', linestyle='-', alpha=0.6, linewidth=5, label='Robot Path')
                    if sensor_col:
                        print(f"Warning: Sensor column '{sensor_col}' not found in synchronized data.")
            else:
                print("Warning: No valid pose data available for overlay.")

        # Plot Scenario POIs
        if scenario_pois:
            sx, sy = zip(*scenario_pois)
            plt.scatter(sx, sy, c='red', marker='x', s=100, linewidths=2.5, zorder=10, label='Scenario Event')

        # Plot Sensor POIs
        if sensor_pois:
            sens_x, sens_y = zip(*sensor_pois)
            # Circles around the plotted sensor path
            plt.scatter(sens_x, sens_y, facecolors='none', edgecolors='magenta', s=300, linewidths=2.5, zorder=10, label='Sensor Event')
                
        # Constuct Custom Legend
        legend_elements = [
            Line2D([0], [0], color='black', lw=2, label='Scenario Path'),
            Line2D([0], [0], color='red', linestyle='--', lw=1.5, label='Force Area'),
            Line2D([0], [0], color='teal' if sensor_col else 'blue', lw=3, label='Robot Path'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='black', markersize=8, label='Start, End'),
        ]
        
        if scenario_pois:
            legend_elements.append(Line2D([0], [0], marker='x', color='w', markeredgecolor='red', markersize=10, markeredgewidth=2, label='Scenario Event'))
        if sensor_pois:
            legend_elements.append(Line2D([0], [0], marker='o', color='w', markeredgecolor='magenta', markerfacecolor='none', markersize=10, markeredgewidth=2, label='Sensor Event'))

        plt.legend(handles=legend_elements, loc='best')

        title_str = "Scenario Visualization"
        if path_id is not None:
            title_str += f" (Path {path_id})"
        if sensor_col:
            title_str += f" - Sensor: {sensor_col}"
        
        plt.title(title_str)
        plt.xlabel("X [m]")
        plt.ylabel("Y [m]")
        plt.axis('equal')
        plt.grid(True)

        filename = f"scenario_overlay_{sensor_col}.png" if sensor_col else "scenario_overlay.png"
        output_path = os.path.join("/workspace/automatic_assessment/figures/dataset", filename)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path)
        print(f"Scenario overlay plot saved to {output_path}")
        plt.close()

    except Exception as e:
        print(f"Error plotting scenario: {e}")


if __name__ == "__main__":
    path_to_csv = "/data/raw/KATE_AA_dataset.csv"
    output_single_user_csv = "/data/raw/KATE_AA_dataset_user_1.csv"
    # Assuming scenarios are stored here; adjust as needed for environment
    scenario_folder = "/data/raw/scenarios" 

    # extract_user_dataset(path_to_csv, output_single_user_csv, user_id=12)
    print(f"Loading user dataset: {output_single_user_csv}")
    user_1_df = load_dataset(output_single_user_csv)
    
    # Filter for path = 10 to isolate one timeline
    path_id_to_filter = 14
    user_1_df = user_1_df[user_1_df["path"] == path_id_to_filter]
    # save_dataset(user_1_df, f"/data/user_1_path_{path_id_to_filter}.csv")
        
    analyze_dataset_summary(user_1_df)

    # plot_histogram(user_1_df, "robot_pose_y")
    # plot_robot_path(user_1_df)

    
    raw_scenario = load_scenario_by_path_number(path_number=path_id_to_filter, folder_path=scenario_folder)
    parsed_data = parse_scenario(raw_scenario)
    
    sensor_column = "robotrainer_deviation_front"
    
    # Calculate POIs
    scen_pois = find_scenario_pois(parsed_data, user_1_df)
    sens_pois = find_sensor_pois(user_1_df, sensor_column)
    
    plot_scenario_with_robot(parsed_data, user_1_df, sensor_col=sensor_column, path_id=path_id_to_filter,
                             scenario_pois=scen_pois, sensor_pois=sens_pois)
