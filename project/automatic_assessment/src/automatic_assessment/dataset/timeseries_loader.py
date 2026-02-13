"""
Data loading and container logic.
"""

from __future__ import annotations

import json
import numpy as np
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from collections.abc import MutableMapping

@dataclass
class PathData:
    """All loaded (and preprocessed) timeseries for a single path."""
    user_id: int
    path_id: int
    meta: dict = field(default_factory=dict)
    timeseries: Dict[str, np.ndarray] = field(default_factory=dict)
    scalar_features: Dict[str, float] = field(default_factory=dict)
    motion_start_timestamp: Optional[float] = None


class TimeseriesDataset(MutableMapping):
    """
    A wrapper around the dataset dictionary that adds utility methods like saving.
    Behaves like a dictionary: dataset[user_id][path_id] -> PathData
    """
    def __init__(self, data: Dict[int, Dict[int, PathData]] = None):
        self.data = data if data is not None else {}

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, value):
        self.data[key] = value

    def __delitem__(self, key):
        del self.data[key]

    def __iter__(self):
        return iter(self.data)

    def __len__(self):
        return len(self.data)
    
    def save(self, output_root: Union[str, Path]) -> None:
        """
        Saves the current state of the dataset to a folder structure matching the input.
        """
        root = Path(output_root).resolve()
        print(f"Saving dataset to: {root}")
        
        root.mkdir(parents=True, exist_ok=True)
        
        count = 0
        for user_id, paths in self.data.items():
            user_dir = root / f"U{user_id}"
            user_dir.mkdir(exist_ok=True)
            
            for path_id, pd_obj in paths.items():
                path_dir = user_dir / f"path_{path_id}"
                path_dir.mkdir(exist_ok=True)
                
                # Save Meta
                if pd_obj.meta:
                    with open(path_dir / "meta.json", 'w') as f:
                        json.dump(pd_obj.meta, f, indent=4)
                
                # Save Scalars
                for name, value in pd_obj.scalar_features.items():
                    # Save as 0-d array (scalar)
                    np.save(str(path_dir / f"{name}.npy"), np.array(value))
                
                # Save Timeseries
                for name, arr in pd_obj.timeseries.items():
                    np.save(str(path_dir / f"{name}.npy"), arr)
                    
                count += 1
                
        print(f"Datasets saved. ({count} paths)")


class TimeseriesLoader:
    """
    Handles file system traversal and loading of raw numpy data.
    """

    def __init__(self, config: Any) -> None:
        self.cfg = config

    def load(self, folder: str = None) -> TimeseriesDataset:
        """Discover and load all users and paths into a TimeseriesDataset."""
        if folder is None:
            folder = self.cfg.DATASET_FOLDER

        print(f"Loading dataset from: {folder}")
        root = Path(folder).resolve()
        data_map = {}
        
        if not root.exists():
            print(f"Error: Dataset root not found at {root}")
            return TimeseriesDataset(data_map)

        user_dirs = sorted([d for d in root.iterdir() if d.is_dir() and d.name.startswith("U")])
        
        for user_dir in user_dirs:
            try:
                user_id = int(user_dir.name[1:])
            except ValueError:
                print(f"Skipping invalid user directory: {user_dir.name}")
                continue
                
            data_map[user_id] = {}
            
            # Sort paths numerically (path_1, path_2, ...)
            path_dirs = sorted(
                [d for d in user_dir.iterdir() if d.is_dir() and d.name.startswith("path_")],
                key=lambda p: int(p.name.split("_")[1]) if "_" in p.name and p.name.split("_")[1].isdigit() else 0
            )

            for path_dir in path_dirs:
                try:
                    path_id = int(path_dir.name.split("_")[1])
                except (IndexError, ValueError):
                    continue

                pd = self._load_path(user_id, path_id, path_dir)
                if pd:
                    data_map[user_id][path_id] = pd

        return TimeseriesDataset(data_map)

    def _load_path(self, user_id: int, path_id: int, path_dir: Path) -> Optional[PathData]:
        pd = PathData(user_id=user_id, path_id=path_id)
        
        # Load Meta
        meta_path = path_dir / "meta.json"
        if meta_path.exists():
            with open(meta_path, 'r') as f:
                pd.meta = json.load(f)

        # Iterate over all .npy files in the directory
        npy_files = list(path_dir.glob("*.npy"))
        
        # Get set of all valid timeseries names from config (Raw + Derived)
        valid_ts_names = set(self.cfg.ALL_VALID_TIMESERIES)

        for fpath in npy_files:
            name = fpath.stem # filename without extension
            
            # Case 1: Scalar Features
            if name in self.cfg.DATA_WITH_A_SINGLE_VALUE_PER_PATH:
                try:
                    # Load as array first
                    arr = np.load(str(fpath))
                    
                    # If this file has timestamps (columns), extract only the value column.
                    if arr.ndim == 2 and arr.shape[1] > self.cfg.COL_VALUE:
                         arr = arr[:, self.cfg.COL_VALUE]

                    # Normalize scalar values immediately
                    if arr.size == 1:
                        pd.scalar_features[name] = float(arr.item())
                    elif arr.size > 1:
                        pd.scalar_features[name] = float(arr.item(0))
                        
                except Exception as e:
                    print(f"Error loading scalar {fpath}: {e}")
            
            # Case 2: Timeseries Data (Raw or Derived)
            elif name in valid_ts_names:
                try:
                    arr = np.load(str(fpath))
                    # Basic check for timeseries shape (N, >=3cols usually [raw, rel, val...])
                    if arr.ndim == 2 and arr.shape[1] >= 3:
                         pd.timeseries[name] = arr
                except Exception as e:
                    print(f"Error loading {fpath}: {e}")
            
            # Otherwise we ignore unwanted files

        return pd
