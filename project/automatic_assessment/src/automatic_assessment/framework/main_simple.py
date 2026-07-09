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
from automatic_assessment.framework.data.results import ExperimentConfig
from automatic_assessment.framework.utils.seed import set_global_seed
from automatic_assessment.framework.models.simple_models import SimpleMLPRegressor, ElasticNetModel, LinearRegressionModel, RandomForestLikeMLP
from automatic_assessment.framework.models.mlp_path_shared import MLPSharedEncoder
from automatic_assessment.framework.models.mlp_path_specific import MLPPathSpecific
from automatic_assessment.framework.models.mlp_baseline import MLPBaseline
from automatic_assessment.framework.models.sklearn.sklearn_models import LinearReg, ElasticNetReg, SVRReg, RandomForestReg, SGDReg, AutoSklearnReg, TabPFNReg
from automatic_assessment.framework.models.timeseries.LSTM import HierarchicalTimeseriesLSTM
from automatic_assessment.framework.models.timeseries.LSTM_baseline import LSTMBaseline
from automatic_assessment.framework.models.timeseries.CNN_baseline import CNNBaseline
from automatic_assessment.framework.models.timeseries.BASE_baseline import BASEBaseline
from automatic_assessment.framework.models.mlp_path_shared_flat import MLPSharedEncoderFLAT
from automatic_assessment.framework.models.timeseries.BASE_baseline_flat import BASEBaselineFLAT
from automatic_assessment.framework.models.timeseries.CNN_baseline_no_embedding import CNNBaselineNOEMBED
from automatic_assessment.framework.models.timeseries.CNN_baseline_no_embedding_flat import CNNBaselineNOEMBEDFLAT
from automatic_assessment.framework.models.timeseries.BASE_baseline_norm import BASEBaselineNORM
from automatic_assessment.framework.models.timeseries.TCN_multiscale import MultiScaleTCN
from automatic_assessment.framework.models.timeseries.CNN_hybrid_fusion import HybridCNNGRUFusion


def main():

    dataset_train = AssessmentDataset("/data/train")

    # X tuple: (x_path, x_user, g0_x, g0_mask, g1_x, g1_mask, ...)
    #     1. x_path: Path-Level Dataset
    #         - Shape: (n_samples, n_paths, n_path_features), e.g. (25, 20, 88)
    #     2. x_user: User-Level Dataset
    #         - Shape: (n_samples, n_user_features), e.g. (25, 2)
    #     3.+ Timeseries channel groups (config.TS_MODEL_GROUPS order:
    #         mechanical @ 50 Hz, physiological @ 2 Hz, gait @ 2 Hz),
    #         each as a (values, bool-mask) tensor pair of shape
    #         (n_samples, n_paths, n_group_channels, T_group).
    #         Groups keep their native resolution and are NOT aligned
    #         to each other; padding is 0 with mask=False.

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

    # Augmentation ratio sweep: clones are precomputed in the train split
    # (up to config.AUGMENTATION['max_ratio']); the view below selects how
    # many are used — no data regeneration between ratios.
    # augmentation_range = [0, 1, 2, 3, 4, 5]
    augmentation_range = [0]

    for ratio in augmentation_range:
        for target_set in targets_to_test:
            print(f"\n\n=== Targets: {target_set} | augmentation ratio: {ratio} ===\n")

            # Immutable selection of targets + augmentation ratio (R11)
            view_train = dataset_train.view(targets=target_set, augmentation_ratio=ratio)
            X_train, y_train, users_train, feature_names = view_train.get_all()
            
            config = ExperimentConfig(
                epochs=30,
                hyperparameter_mode='default',  # 'default', 'optimize'
                n_trials=30,  # Number of Optuna trials
                early_stopping_patience=2,  # Stop if val loss doesn't improve for N epochs (None to disable)
                seed=42,  # Global seed (torch/numpy/random + Optuna sampler)
                targets=target_set,
                augmentation_ratio=ratio,
                note="augmentation with default hyperparameters",
            )
            
            # List of models to test
            # models_to_test = [SimpleMLPRegressor, ElasticNetModel, LinearRegressionModel, RandomForestLikeMLP]
            # models_to_test = [MLPSharedEncoder, MLPPathSpecific, MLPBaseline]
            # models_to_test = [HierarchicalTimeseriesLSTM]
            # models_to_test = [LinearReg, ElasticNetReg, SVRReg, RandomForestReg, SGDReg, MLPBaseline]
            # models_to_test = [CNNBaseline, LSTMBaseline]
            # models_to_test = [LSTMBaseline]
            # models_to_test = [CNNBaseline, MLPSharedEncoder, MLPPathSpecific, MLPBaseline, LinearReg, ElasticNetReg, SVRReg, RandomForestReg]
            # models_to_test = [BASEBaselineNORM, BASEBaseline, MLPSharedEncoderFLAT, BASEBaselineFLAT, CNNBaselineNOEMBED, CNNBaselineNOEMBEDFLAT]
            # models_to_test = [LSTMBaseline, BASEBaseline, BASEBaselineFLAT, CNNBaselineNOEMBED, CNNBaseline, MLPBaseline, LinearReg, ElasticNetReg, SVRReg]
            # models_to_test = [TabPFNReg] #AutoSklearnReg
            models_to_test = [MultiScaleTCN, HybridCNNGRUFusion]  # new architectures, v3 2026-07-09
            # models_to_test = [BASEBaselineFLAT]

            
            for model_class in models_to_test:
                model_name = model_class.model_name
                if model_name == "LSTM_Baseline":
                    config.epochs = 50
                else:
                    config.epochs = 30

                print(f"\n{'='*60}")
                print(f"STARTING EXPERIMENT FOR: {model_name}")
                print(f"{'='*60}\n")

                # Reproducibility: identical seed state at the start of every experiment
                set_global_seed(config.seed)

                pipeline = SimplePipeline(model_class, config)
                saver = SavingModule(model_name=f"{model_name}")
                saver.save_model_source(model_class)

                # 1. Hyperparameter Tuning & LOGO-CV Validation on Train Set
                # Returns an ExperimentResult (test=None) with per-user CV
                # predictions. NOTE: in 'optimize' mode the val score is a
                # model-selection score (optimistically biased) — see
                # open_improvements.md.
                tuning_results = pipeline.run_simple_tuning(X_train, y_train, users_train)

                # 2. Final Evaluation on Test Set
                print("\nLoading Test Set...")
                dataset_test = AssessmentDataset("/data/test")

                # Same target selection; the test split never contains clones
                X_test, y_test, users_test, _ = dataset_test.view(
                    targets=target_set, augmentation_ratio=0).get_all()

                # Run Final Test
                # Trains the final model on the full training set (mean best
                # epoch from CV) and returns a NEW ExperimentResult with the
                # test results attached; tuning_results stays untouched.
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