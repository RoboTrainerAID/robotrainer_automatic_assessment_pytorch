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
    
    # Extract features (validates the feature config and reports
    # missing series / NaN features loudly)
    print("Extracting statistical features...")
    extractor = TimeseriesFeatureExtractor(dataset)
    features = extractor.extract_features()

    # Fail loudly if the final feature table contains NaN values:
    # the training framework asserts finiteness and would crash later
    # with a less specific error (see analysis 3.5).
    feature_cols = [c for c in features.columns if c not in ("user", "path")]
    nan_counts = features[feature_cols].isna().sum()
    nan_cols = nan_counts[nan_counts > 0]
    if len(nan_cols) > 0:
        raise RuntimeError(
            "timeseries_features.csv would contain NaN values in columns: "
            f"{list(nan_cols.index)}. Fix the source data or the imputation "
            "before generating the dataset."
        )
    
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
    
    # 4. Second Validation — after imputation the dataset MUST be complete
    # and finite. Any remaining issue is a hard error (analysis 3.5).
    print("\nValidating dataset (Post-Imputation)...")
    validation_report_post = TimeseriesValidator.validate_dataset(dataset, config)
    TimeseriesValidator.print_validation_report(validation_report_post)
    TimeseriesValidator.print_dataset_summary(dataset)

    if validation_report_post:
        n_issues = sum(len(v) for v in validation_report_post.values())
        raise RuntimeError(
            f"Post-imputation validation failed with {n_issues} issue(s) across "
            f"{len(validation_report_post)} user(s) — see the report above. "
            "The dataset must be complete and finite after imputation."
        )

    # 5. Preprocessing (Derived Timeseries)
    processor = TimeseriesDerivedTS(dataset)
    processor.process()

    dataset.save("/data/raw/timeseries_numpy_processed")
    
if __name__ == "__main__":
    main()
    create_features_csv()