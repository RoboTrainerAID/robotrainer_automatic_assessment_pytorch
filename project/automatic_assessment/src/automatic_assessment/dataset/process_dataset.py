"""
Main entry point for processing the timeseries dataset.
Imports configuration from dataset_settings.py
"""

import sys
from pathlib import Path
import automatic_assessment.dataset.config as config
from automatic_assessment.dataset.timeseries_loader import TimeseriesLoader
from automatic_assessment.dataset.timeseries_preprocessor import TimeseriesPreprocessor
from automatic_assessment.dataset.timeseries_derived_ts import TimeseriesDerivedTS
from automatic_assessment.dataset.timeseries_features import TimeseriesFeatureExtractor
from automatic_assessment.dataset.timeseries_validation import TimeseriesValidator
from automatic_assessment.dataset.timeseries_imputer import TimeseriesImputer

def create_features_csv():
    # Load processed dataset
    loader = TimeseriesLoader(config)
    dataset = loader.load("/data/raw/timeseries_numpy_processed")
    
    # Extract features
    print("Extracting statistical features...")
    extractor = TimeseriesFeatureExtractor(dataset)
    features = extractor.extract_features()
    
    # Save to CSV using the config path
    extractor.save_features_to_csv()


def main():
    # 0. Load raw data
    loader = TimeseriesLoader(config)
    dataset = loader.load(config.DATASET_FOLDER)
    
    # 1. Preprocessing (Trimming & Cleaning)
    preprocessor = TimeseriesPreprocessor(dataset, config)
    preprocessor.process()

    # 2. First Validation
    # print("\nValidating dataset (Pre-Imputation)...")
    validation_report = TimeseriesValidator.validate_dataset(dataset, config, check_all_disturbance=False)
    # TimeseriesValidator.print_validation_report(validation_report)
    TimeseriesValidator.print_dataset_summary(dataset)

    # 3. Imputation if needed
    imputer = TimeseriesImputer(dataset, config)
    imputer.impute_from_report(validation_report)
    imputer.impute_zero_disturbance()
    
    # 4. Second Validation
    print("\nValidating dataset (Post-Imputation)...")
    validation_report_post = TimeseriesValidator.validate_dataset(dataset, config)
    TimeseriesValidator.print_validation_report(validation_report_post)
    TimeseriesValidator.print_dataset_summary(dataset)

    # 5. Preprocessing (Derived Timeseries)
    processor = TimeseriesDerivedTS(dataset)
    processor.process()

    dataset.save("/data/raw/timeseries_numpy_processed")
    
if __name__ == "__main__":
    # main()
    create_features_csv()