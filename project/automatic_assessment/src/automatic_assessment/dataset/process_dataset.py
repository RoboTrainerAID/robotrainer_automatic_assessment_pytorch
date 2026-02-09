"""
Main entry point for processing the timeseries dataset.
Imports configuration from dataset_settings.py
"""

import sys
from pathlib import Path
import automatic_assessment.dataset.dataset_settings as config
from automatic_assessment.dataset.timeseries_loader import TimeseriesLoader
from automatic_assessment.dataset.timeseries_features import extract_features

def main():
    print(f"Loading dataset from: {config.DATASET_ROOT}")
    
    # Initialize loader with the config module
    loader = TimeseriesLoader(config)
    loader.load_all()
    
    # Get loaded data
    all_paths = loader.get_all_paths()
    print(f"\nProcessing complete. Found {len(all_paths)} valid paths.")
    
    # Extract features
    print("Extracting statistical features...")
    features = extract_features(all_paths)
    
    if features:
        print(f"Extracted features for {len(features)} paths.")
        print("Example keys:", list(features[0].keys())[:5])
    else:
        print("No features extracted.")

if __name__ == "__main__":
    main()
