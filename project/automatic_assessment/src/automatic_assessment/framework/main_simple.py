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
from automatic_assessment.framework.models.timeseries.BASE_baseline import BASEBaseline
from automatic_assessment.framework.models.mlp_path_shared_flat import MLPSharedEncoderFLAT
from automatic_assessment.framework.models.timeseries.BASE_baseline_flat import BASEBaselineFLAT
from automatic_assessment.framework.models.timeseries.CNN_baseline_no_embedding import CNNBaselineNOEMBED
from automatic_assessment.framework.models.timeseries.CNN_baseline_no_embedding_flat import CNNBaselineNOEMBEDFLAT


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
    # targets_to_test = [clusters]
    # targets_to_test = [all]
    # targets_to_test = [all] + [best_performing_targets] + singles_list
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
                "epochs": 30,
                "hyperparameter_mode": 'optimize', # 'default', 'optimize'
                "n_trials": 50,  # Number of Optuna trials
                "targets": target_set,
                "augmentation_ratio": ratio,
                "note": "Single targets comparison",
            }
            
            # List of models to test
            # models_to_test = [SimpleMLPRegressor, ElasticNetModel, LinearRegressionModel, RandomForestLikeMLP]
            # models_to_test = [MLPSharedEncoder, MLPPathSpecific, MLPBaseline]
            # models_to_test = [HierarchicalTimeseriesLSTM]
            # models_to_test = [LinearReg, ElasticNetReg, SVRReg, RandomForestReg, SGDReg, MLPBaseline]
            # models_to_test = [CNNBaseline, LSTMBaseline]
            # models_to_test = [LSTMBaseline]
            # models_to_test = [CNNBaseline, MLPSharedEncoder, MLPPathSpecific, MLPBaseline, LinearReg, ElasticNetReg, SVRReg, RandomForestReg]
            models_to_test = [BASEBaseline, MLPSharedEncoderFLAT, BASEBaselineFLAT, CNNBaselineNOEMBED, CNNBaselineNOEMBEDFLAT]

            
            for model_class in models_to_test:
                model_name = model_class.model_name
                print(f"\n{'='*60}")
                print(f"STARTING EXPERIMENT FOR: {model_name}")
                print(f"{'='*60}\n")
                
                pipeline = SimplePipeline(model_class, config)
                saver = SavingModule(model_name=f"{model_name}")
                saver.save_model_source(model_class)

                # 1. Hyperparameter Tuning & CV on Train Set
                # This returns the CV results (optimistic) and the best params found
                tuning_results = pipeline.run_simple_tuning(X_train, y_train, users_train)
                
                # 2. Final Evaluation on Test Set
                print("\nLoading Test Set...")
                dataset_test = AssessmentDataset("/data/test")
                
                # Need target selection on test set too to match dimensions!
                dataset_test.perform_target_selection(target_set)
                
                X_test, y_test, users_test, _ = dataset_test.get_all()

                # Run Final Test
                # This consumes the tuning results, trains the final model on full train,
                # evaluates on test, and returns a consolidated results dictionary.
                final_results = pipeline.run_final_test(
                    X_train, y_train, 
                    X_test, y_test, users_test,
                    tuning_results
                )

                # 3. Save & Report
                saver.save_results(final_results)

                visualizer = VisualizationModule(saver.output_dir)
                visualizer.generate_all_plots()

                print(f"Reporting complete for {model_name}. Check the 'plots' folder in the experiment directory.")


if __name__ == "__main__":
    main()