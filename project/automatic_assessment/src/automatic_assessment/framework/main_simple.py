import os

# Force matplotlib to not use any Xwindows backend before other imports
import matplotlib
matplotlib.use('Agg')

# Set memory management configuration to avoid fragmentation
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:512"

from automatic_assessment.framework.core.pipeline_simple import SimplePipeline
from automatic_assessment.framework.reporting.saving import SavingModule
from automatic_assessment.framework.reporting.visualization import VisualizationModule
from automatic_assessment.framework.data.dataset import AssessmentDataset
from automatic_assessment.framework.models.simple_models import SimpleMLPRegressor, ElasticNetModel, LinearRegressionModel, RandomForestLikeMLP
from automatic_assessment.framework.models.mlp_path_shared import MLPSharedEncoder
from automatic_assessment.framework.models.mlp_path_specific import MLPPathSpecific
from automatic_assessment.framework.models.mlp_baseline import MLPBaseline
from automatic_assessment.framework.models.sklearn.sklearn_models import LinearReg, ElasticNetReg, SVRReg, RandomForestReg, SGDReg
from automatic_assessment.framework.models.timeseries.LSTM import HierarchicalTimeseriesLSTM
from automatic_assessment.framework.models.timeseries.LSTM_baseline import LSTMBaseline
from automatic_assessment.framework.models.timeseries.CNN_baseline import CNNBaseline


def main():

    dataset_train = AssessmentDataset("/data/train")

    # X tuple: (x_ts, x_path, x_user)
    #     1. x_ts: Time-Series Dataset
    #         - Shape (variable): (n_samples, n_paths, n_timeseries, variable_length)
    #         - Values: (25, 20, 32, variable_timesteps)
    #     2. x_path: Path-Level Dataset
    #         - Shape: (n_samples, n_paths, n_path_features)
    #         - Values: (25, 20, 73)
    #     3. x_user: User-Level Dataset
    #         - Shape: (n_samples, n_user_features)
    #         - Values: (25, 2)

    # y: Targets
    #     - Shape: (n_samples, n_targets)
    #     - Values: (25, 14)

    all = [
        'Balance Test', 'Single Leg Stance', 'Robotrainer Front', 'Robotrainer Left', 
        'Robotrainer Right', 'Hand Grip Left', 'Hand Grip Right', 'Jump & Reach', 
        'Tandem Walk', 'Figure 8 Walk', 'Jumping Sideways', 'Throwing Beanbag at Target',
        'Tapping Test', 'Ruler Drop Test'
    ]
    clusters = ['Balance Test', 'Single Leg Stance', 'Hand Grip Right', 'Throwing Beanbag at Target']

    singles_list = [[label] for label in all]

    best_performing_targets = [
        'Robotrainer Front', 'Robotrainer Left', 'Robotrainer Right', 
        'Hand Grip Left', 'Hand Grip Right', 'Jump & Reach', 
        'Figure 8 Walk', 'Ruler Drop Test'
    ]

    # List of different target combinations
    # targets_to_test = [clusters] + singles_list
    # targets_to_test = [all, clusters] + singles_list
    # targets_to_test = [clusters]  # For quick testing
    targets_to_test = [best_performing_targets]

    # augmentation_range = [0, 1, 2, 3, 4, 5]
    augmentation_range = [0]

    for ratio in augmentation_range:
        # print(f"Creating augmented dataset with ratio: {ratio}")
        
        # Note: Augmentation logic is currently a placeholder in dataset.py
        # dataset.create_augmented_dataset(ratio)

        for target_set in targets_to_test:
            print(f"\n\n=== Selecting Targets: {target_set} ===\n")

            # Select targets 
            dataset_train.perform_target_selection(target_set)

            X_train, y_train, users_train, feature_names = dataset_train.get_all()
            
            config = {
                "epochs": 50,
                "hyperparameter_mode": 'default', # 'default', 'optimize'
                "n_trials": 75,  # Number of Optuna trials
                "targets": target_set,
                "augmentation_ratio": ratio,
                "note": "Testing",
            }
            
            # List of models to test
            # models_to_test = [SimpleMLPRegressor, ElasticNetModel, LinearRegressionModel, RandomForestLikeMLP]
            # models_to_test = [MLPSharedEncoder, MLPPathSpecific, MLPBaseline]
            # models_to_test = [HierarchicalTimeseriesLSTM]
            # models_to_test = [LinearReg, ElasticNetReg, SVRReg, RandomForestReg, SGDReg, MLPBaseline]
            models_to_test = [CNNBaseline, LSTMBaseline]

            
            for model_class in models_to_test:
                model_name = model_class.model_name
                print(f"\n{'='*60}")
                print(f"STARTING EXPERIMENT FOR: {model_name}")
                print(f"{'='*60}\n")
                
                pipeline = SimplePipeline(model_class, config)
                saver = SavingModule(model_name=f"{model_name}")
                saver.save_model_source(model_class)

                # 1. Hyperparameter Tuning & CV on Train Set
                results = pipeline.run_simple_tuning(X_train, y_train, users_train)
                
                # Extract best params found on training set
                best_params = results['fold_data'][0]['best_params']

                # 2. Final Evaluation on Test Set
                # print("\nLoading Test Set...")
                # dataset_test = AssessmentDataset("/data/test")
                # X_test, y_test, _, _ = dataset_test.get_all()

                # final_test_results = pipeline.run_final_test(X_train, y_train, X_test, y_test, best_params)

                # Merge final results into main results dictionary for saving
                # results.update(final_test_results)
                
                # Update main metrics to reflect final test performance instead of CV estimates 
                # for the plots that use 'test_metrics' key
                # Mapping final_test_ -> test_ for compatibility with existing visualization/reporting
                # renamed_metrics = {k.replace('final_test_', 'test_'): v for k,v in final_test_results['final_test_metrics'].items()}
                # results['test_metrics'] = renamed_metrics
                # results['test_loss'] = final_test_results['final_test_loss']
                
                # Adding final baseline similarly
                # renamed_baseline = {k.replace('final_baseline_', 'baseline_'): v for k,v in final_test_results['final_baseline_metrics'].items()}
                # results['baseline_metrics'] = renamed_baseline
                # results['baseline_loss'] = final_test_results['final_baseline_loss']

                # Overwrite the 'test_preds' in fold_data for visualization (parity plots)
                # Since simple tuning aggregates results in fold 0, we put final test preds there
                # results['fold_data'][0]['test_preds'] = final_test_results['final_test_preds']
                # results['fold_data'][0]['test_actuals'] = final_test_results['final_test_actuals']
                # User ID mapping for test set might be needed if creating user reports, 
                # but dataset.get_all doesn't return users for test easily mapped without ID.
                # Just using placeholder or extracting if available.
                # Since get_all returns users array, we can use it.
                # _, _, users_test, _ = dataset_test.get_all()
                # results['fold_data'][0]['user_id'] = users_test[:, 0] if users_test.ndim > 1 else users_test # Or use full array

                saver.save_results(results)

                visualizer = VisualizationModule(saver.output_dir)
                visualizer.generate_all_plots()

                print(f"Reporting complete for {model_name}. Check the 'plots' folder in the experiment directory.")


if __name__ == "__main__":
    main()