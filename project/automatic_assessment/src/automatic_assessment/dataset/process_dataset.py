"""
Main entry point for processing the timeseries dataset.
Imports configuration from dataset_settings.py
"""

import sys
from pathlib import Path
import automatic_assessment.dataset.config as config
from automatic_assessment.dataset.timeseries_loader import TimeseriesLoader
from automatic_assessment.dataset.timeseries_features import TimeseriesFeatures

def main():
    print(f"Loading dataset from: {config.DATASET_ROOT}")
    
    # Initialize loader with the config module
    loader = TimeseriesLoader(config)
    loader.load_all()
    
    # Get loaded data
    all_paths = loader.get_all_paths()
    print(f"\nProcessing complete. Found {len(all_paths)} valid paths.")
    
    # Extract features using the new TimeseriesFeatures class
    print("Extracting statistical features...")
    extractor = TimeseriesFeatures(all_paths)
    
    # Store features in loader as per requirement and get dictionary/df
    loader.features = extractor.extract_features()
    
    if loader.features is not None and not loader.features.empty:
        # Save to CSV using the config path
        print(f"Saving features to {config.CSV_OUTPUT_PATH}")
        extractor.save_features_to_csv(config.CSV_OUTPUT_PATH)

if __name__ == "__main__":
    main()
