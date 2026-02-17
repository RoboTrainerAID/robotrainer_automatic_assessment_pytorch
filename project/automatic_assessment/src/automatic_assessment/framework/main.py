import os

# Set memory management configuration to avoid fragmentation
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:512"

from automatic_assessment.framework.core.pipeline import Pipeline
from automatic_assessment.framework.models.timeseries.cnn1d import CNN1D
from automatic_assessment.framework.models.timeseries.hierarchical_cnn import HierarchicalCNN
from automatic_assessment.framework.models.timeseries.mlp import SimpleMLP
from automatic_assessment.framework.models.timeseries.hierarchical_attention import HierarchicalAttentionNetwork
from automatic_assessment.framework.models.timeseries.hierarchical_attention_chatgpt import HierarchicalTimeseriesChatGPT
from automatic_assessment.framework.models.timeseries.hierarchical_attention_chatgpt_hyper import HierarchicalTimeseriesChatGPT as HierarchicalTimeseriesChatGPTHyper
from automatic_assessment.framework.models.timeseries.hierarchical_attention_gemini import HierarchicalTimeseriesGemini
from automatic_assessment.framework.models.timeseries.hierarchical_attention_gemini_1 import HierarchicalTimeseriesGemini1
from automatic_assessment.framework.models.timeseries.hierarchical_attention_gemini_2 import HierarchicalTimeseriesGemini2
from automatic_assessment.framework.models.timeseries.hierarchical_attention_gemini_3 import HierarchicalTimeseriesGemini3
from automatic_assessment.framework.models.timeseries.hierarchical_attention_gemini_21 import HierarchicalTimeseriesGemini21
from automatic_assessment.framework.models.timeseries.hierarchical_attention_gemini_22 import HierarchicalTimeseriesGemini22
from automatic_assessment.framework.models.timeseries.hierarchical_attention_gemini_22_LSTM import HierarchicalTimeseriesLSTM
import automatic_assessment.framework.models.simple_models as simple_models
from automatic_assessment.framework.reporting.saving import SavingModule
from automatic_assessment.framework.reporting.visualization import VisualizationModule
from automatic_assessment.archiv.dataset_old import DatasetFreq2hz


def main():
    dataset = DatasetFreq2hz(recreate=False)
    # dataset.create_augmented_dataset(augmentation_ratio=4, subfolder="augmentedx4")
    # dataset = DatasetFreq2hzAugmentedx4(recreate=False)

    # dataset.target_and_feature_selection(
    #     selected_targets=['Balance Test', 'Single Leg Stance', 'Hand Grip Right', 'Throwing Beanbag at Target'],
    #     apply_lars=True,
    #     max_n_timeseries=40,
    #     max_n_path_features=10,
    #     max_n_extracted_ts_features=100
    # )
    # X, y, users, _ = dataset.get_all_level_dataset(padding=True)

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

    # List of different target combinations
    # targets_to_test = [clusters] + singles_list
    # targets_to_test = [all, clusters] + singles_list
    targets_to_test = [clusters]  # For quick testing

    # augmentation_range = [0, 1, 2, 3, 4, 5]
    augmentation_range = [3]

    for ratio in augmentation_range:
        print(f"Creating augmented dataset with ratio: {ratio}")
        
        dataset.create_augmented_dataset(ratio)

        for target_set in targets_to_test:
            print(f"\n\n=== Selecting Targets: {target_set} ===\n")

            X, y, users, feature_names = dataset.get_all_level_dataset(padding=True)
            
            config = {
                "epochs": 50,
                "hyperparameter_mode": 'default', # 'default', 'optimize_once', 'optimize_every_fold'
                "only_first_fold": True,  # For quick testing
                "n_trials": 50,  # Number of Optuna trials
                "use_lasso": False, 
                "max_n_features": 20,
                "targets": target_set,
                "augmentation_ratio": ratio,
                "note": "Multitarget to all single target comparison",
            }
            
            # List of models to test
            # models_to_test = [SimpleMLP, CNN1D, HierarchicalCNN, HierarchicalAttentionNetwork
            # models_to_test = [HierarchicalTimeseriesChatGPT] 
            # models_to_test = [HierarchicalTimeseriesGemini]
            # models_to_test = [HierarchicalTimeseriesGemini2, HierarchicalTimeseriesGemini1, HierarchicalTimeseriesChatGPTHyper]
            # models_to_test = [HierarchicalTimeseriesChatGPTHyper]
            # models_to_test = [HierarchicalTimeseriesGemini21, HierarchicalTimeseriesLSTM]
            # models_to_test = [HierarchicalTimeseriesLSTM]
            # models_to_test = [simple_models.ElasticNetModel, simple_models.LinearRegressionModel, simple_models.RandomForestLikeMLP, simple_models.SimpleMLPRegressor]
            models_to_test = [HierarchicalTimeseriesGemini21, HierarchicalTimeseriesLSTM, simple_models.ElasticNetModel, simple_models.LinearRegressionModel, simple_models.RandomForestLikeMLP, simple_models.SimpleMLPRegressor]
            
            for model_class in models_to_test:
                model_name = model_class.model_name
                print(f"\n{'='*60}")
                print(f"STARTING EXPERIMENT FOR: {model_name}")
                print(f"{'='*60}\n")
                
                pipeline = Pipeline(model_class, config)
                saver = SavingModule(model_name=f"{model_name}")
                saver.save_model_source(model_class)

                results = pipeline.run_nested_cv(X, y, users)
                
                saver.save_results(results)

                visualizer = VisualizationModule(saver.output_dir)
                visualizer.generate_all_plots()

                print(f"Reporting complete for {model_name}. Check the 'plots' folder in the experiment directory.")


if __name__ == "__main__":
    main()