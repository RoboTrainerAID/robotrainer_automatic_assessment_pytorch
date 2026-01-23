import numpy as np

from automatic_assessment.framework.core.pipeline import Pipeline
from automatic_assessment.framework.models.cnn1d import CNN1D
from automatic_assessment.framework.models.hierarchical_cnn import HierarchicalCNN
from automatic_assessment.framework.models.mlp import SimpleMLP
from automatic_assessment.framework.models.hierarchical_attention import HierarchicalAttentionNetwork
from automatic_assessment.framework.models.hierarchical_attention_chatgpt import HierarchicalTimeseriesChatGPT
from automatic_assessment.framework.models.hierarchical_attention_chatgpt_hyper import HierarchicalTimeseriesChatGPT as HierarchicalTimeseriesChatGPTHyper
from automatic_assessment.framework.models.hierarchical_attention_gemini import HierarchicalTimeseriesGemini
from automatic_assessment.framework.models.hierarchical_attention_gemini_1 import HierarchicalTimeseriesGemini1
from automatic_assessment.framework.models.hierarchical_attention_gemini_2 import HierarchicalTimeseriesGemini2
from automatic_assessment.framework.models.hierarchical_attention_gemini_3 import HierarchicalTimeseriesGemini3
from automatic_assessment.framework.reporting.saving import SavingModule
from automatic_assessment.framework.reporting.visualization import VisualizationModule
from automatic_assessment.framework.data.dataset import DatasetConv1s, DatasetFreq1hzAugmentedx4, DatasetFreq2hzAugmentedx4


def main():
    dataset = DatasetFreq2hzAugmentedx4(recreate=False)
    # X, y, users, _ = dataset.get_path_level_time_series_dataset(padding=True)
    # X, y, users, _ = dataset.get_user_level_time_series_dataset()
    X, y, users, _ = dataset.get_all_level_dataset(padding=True)

    # X tuple: (x_ts, x_path, x_user)
    #     1. x_ts: Time-Series Dataset
    #         - Shape (padded): (n_samples, n_paths, n_timeseries, max_timesteps)
    #         - Shape (unpadded): (n_samples, n_paths, n_timeseries, variable_timesteps)
    #         - Values: (135, 20, 35, variable_timesteps)
    #     2. x_path: Path-Level Dataset
    #         - Shape: (n_samples, n_paths, n_path_features)
    #         - Values: (135, 20, 105)
    #     3. x_user: User-Level Dataset
    #         - Shape: (n_samples, n_user_features)
    #         - Values: (135, 2)

    # y: Targets
    #     - Shape: (n_samples, n_targets)
    #     - Values: (135, 4)
    
    config = {
        "epochs": 50,
        "hyperparameter_mode": 'optimize_once', # 'default', 'optimize_once', 'optimize_every_fold'
        "only_first_fold": False,  # For quick testing
        "n_trials": 120,  # Number of Optuna trials
        "use_lasso": False, 
        "max_n_features": 20,
        "note": "Hperparameter search for Gemini1+2 model",
    }
    
    # List of models to test
    # models_to_test = [SimpleMLP, CNN1D, HierarchicalCNN, HierarchicalAttentionNetwork
    # models_to_test = [HierarchicalTimeseriesChatGPT] 
    # models_to_test = [HierarchicalTimeseriesGemini]
    models_to_test = [HierarchicalTimeseriesGemini1, HierarchicalTimeseriesGemini2]
    # models_to_test = [HierarchicalTimeseriesChatGPTHyper]
    
    for model_class in models_to_test:
        model_name = model_class.model_name
        print(f"\n{'='*60}")
        print(f"STARTING EXPERIMENT FOR: {model_name}")
        print(f"{'='*60}\n")
        
        pipeline = Pipeline(model_class, config)
        saver = SavingModule(model_name=f"{model_name}_NestedCV_MultiOutput")

        results = pipeline.run_nested_cv(X, y, users)
        
        saver.save_results(results)

        visualizer = VisualizationModule(saver.output_dir)
        visualizer.generate_all_plots()

        print(f"Reporting complete for {model_name}. Check the 'plots' folder in the experiment directory.")

if __name__ == "__main__":
    main()