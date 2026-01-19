import numpy as np

from automatic_assessment.framework.core.pipeline import Pipeline
from automatic_assessment.framework.models.cnn1d import CNN1D
from automatic_assessment.framework.models.hierarchical_cnn import HierarchicalCNN
from automatic_assessment.framework.models.mlp import SimpleMLP
from automatic_assessment.framework.reporting.saving import SavingModule
from automatic_assessment.framework.reporting.visualization import VisualizationModule
from automatic_assessment.framework.data.dataset import DatasetConv1s, DatasetFreq1hzAugmentedx4


def main():
    dataset = DatasetFreq1hzAugmentedx4(recreate=False)
    # X, y, users, _ = dataset.get_path_level_time_series_dataset(padding=True)
    X, y, users, _ = dataset.get_user_level_time_series_dataset()

    # Dataset Summary:
    # Total Samples: 140 (with Augmentation x4), 28 (original Users)
    # Features (Paths * Timeseries) per Sample: 20 * 20 = 400
    # Timesteps per Timeseries: Max Timesteps with 0-Padding (all 79s)
    # Targets per Sample: 4

    # X shape: (n_samples, n_features, max_timesteps)
    # Y shape: (n_samples, n_targets)
    
    # Ensure users is numpy array
    users = np.array(users)

    config = {
        "epochs": 50,
        "hyperparameter_mode": 'optimize_once', # 'default', 'optimize_once', 'optimize_every_fold'
        "n_trials": 20,  # Number of Optuna trials
        "use_lasso": False, 
        "max_n_features": 20
    }
    
    # List of models to test
    # models_to_test = [SimpleMLP, CNN1D, HierarchicalCNN]
    models_to_test = [CNN1D]
    
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
        visualizer.generate_parity_plots()
        visualizer.generate_error_distribution()
        visualizer.generate_learning_curve()
        user_report = visualizer.generate_user_performance_report()

        print(f"Reporting complete for {model_name}. Check the 'plots' folder in the experiment directory.")

if __name__ == "__main__":
    main()