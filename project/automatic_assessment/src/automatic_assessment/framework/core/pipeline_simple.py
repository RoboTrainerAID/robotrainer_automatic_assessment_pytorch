import numpy as np
import torch
import optuna
import time
import gc
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from automatic_assessment.framework.core.trainer import Trainer
from automatic_assessment.framework.core.trainer_sklearn import SklearnTrainer
from automatic_assessment.framework.models.sklearn.sklearn_base import SklearnBaseModel
from automatic_assessment.framework.data.data_utils import AugmentedLOGO, prepare_fold_data, slice_data, apply_feature_selection
from automatic_assessment.framework.reporting.metrics import calculate_metrics
from automatic_assessment.framework.utils.time_utils import start_timer, stop_timer
from automatic_assessment.framework.models.dummy import DummyBaseline

class SimplePipeline:
    def __init__(self, model_class, config):
        self.model_class= model_class
        self.config = config

    def _get_trainer(self, model_class, input_dims, output_dim, params):
        if issubclass(model_class, SklearnBaseModel):
            return SklearnTrainer(model_class, input_dims, output_dim, params)
        else:
            return Trainer(model_class, input_dims, output_dim, params)

    def run_simple_tuning(self, X: tuple, y: np.ndarray, users: np.ndarray) -> dict:
        """
        Runs Simple Tuning (Strategy B).
        Performs grid search / optimization on the full dataset using LOGO CV.
        Reports the best validation scores found during tuning as the final 'test' results.
        Note: This estimate is optimistic/biased because hyperparameters were selected 
        to maximize performance on this exact data.
        """
        start_dt, start_perf = start_timer()
        hyperparameter_mode = self.config.get('hyperparameter_mode', 'default')
        
        # Capture Input Shapes
        input_info = {
            "shape_X_ts": list(X[0].shape),
            "shape_X_path": list(X[1].shape),
            "shape_X_user": list(X[2].shape),
            "shape_y": list(y.shape)
        }

        tqdm.write("\n>>> RUNNING SIMPLE PIPELINE (Strategy B) <<<")
        self.model_class.print_summary(X, y)

        best_params = None
        best_val_loss = 0.0
        best_val_metrics = {}
        tuning_trials = None
        param_importances = None

        best_val_metrics_unscaled = {}

        if hyperparameter_mode == 'default':
            tqdm.write("Using Default Hyperparameters...")
            best_params = self.model_class.get_default_parameters()
            # Simple tuning passes raw data, so we must scale inside CV
            best_val_loss, best_val_metrics, best_val_metrics_unscaled = self._evaluate_params_cv(X, y, users, best_params, scale_data=True)
            
        else:
            tqdm.write("Performing Hyperparameter Tuning on full dataset using LOGO CV.")
            # Simple tuning passes raw data, so we must scale inside optimization loop
            best_params, best_val_loss, best_val_metrics, best_val_metrics_unscaled, tuning_trials, param_importances = self._optimize_hyperparameters(X, y, users, scale_data=True)

        # --- Calculate Dummy Baseline for Comparison (CV) ---
        tqdm.write("Calculating Baseline Performance (CV)...")
        baseline_loss, baseline_metrics = self._evaluate_baseline_cv(X, y, users)

        # --- Generate Learning Curve (Training vs Validation History) ---
        tqdm.write("Generating representative learning curve...")
        history = {}
        logo = AugmentedLOGO(include_augmented_in_test=False)
        splits = list(logo.split(groups=users))
        
        # Take the first split as representative
        if len(splits) > 0:
            train_idx, val_idx = splits[0]
            X_t = slice_data(X, train_idx)
            X_v = slice_data(X, val_idx)
            y_t, y_v = y[train_idx], y[val_idx]

            # Scale/Prepare
            Xt_s, yt_s, Xv_s, yv_s, _, _ = prepare_fold_data(X_t, y_t, X_v, y_v)
            
            # Apply Feature Selection if in best_params
            Xt_s, Xv_s = apply_feature_selection(Xt_s, yt_s, Xv_s, 
                                                n_features=best_params.get('n_path_features'), 
                                                correlation_threshold=best_params.get('correlation_threshold'))

            input_dims = self.model_class.get_input_dims(Xt_s)
            trainer = self._get_trainer(self.model_class, input_dims, yt_s.shape[1], best_params)
            
            history = trainer.train_model_and_evaluate_every_epoch(
                Xt_s, yt_s, 
                epochs=self.config.get('epochs', 50), 
                X_val=Xv_s, 
                y_val=yv_s,
                early_stopping_patience=self.config.get('early_stopping_patience')
            )
            trainer.cleanup()

        start_time_str, duration_str = stop_timer(start_dt, start_perf)

        experiment_meta = {
            "start_time": start_time_str,
            "duration": duration_str,
            "input_shapes": input_info,
            "total_model_parameters": 0, # Not training a final model object to return here
            "strategy": "simple_tuning"
        }

        tqdm.write("\n" + "="*40)
        tqdm.write(" SIMPLE RESULTS (Optimistic Estimate) ")
        tqdm.write(f" CV Loss: {best_val_loss:.3f}")
        tqdm.write(f" CV RMSE: {best_val_metrics.get('val_rmse_mean', 0.0):.3f}")
        tqdm.write("-" * 20)
        tqdm.write(f" Baseline Loss: {baseline_loss:.3f}")
        tqdm.write(f" Baseline RMSE: {baseline_metrics.get('baseline_rmse_mean', 0.0):.3f}")
        tqdm.write("="*40 + "\n")

        fold_info = {
            "fold": 0,
            "user_id": -1, # aggregate
            "test_loss": best_val_loss, 
            "baseline_loss": baseline_loss, 
            "val_loss": best_val_loss,
            "test_preds": np.zeros((len(y), y.shape[1])), # Placeholder for saving compatibility
            "test_actuals": np.zeros((len(y), y.shape[1])), # Placeholder
            "best_params": best_params,
            "val_metrics": best_val_metrics,
            "val_metrics_unscaled": best_val_metrics_unscaled,
            "baseline_metrics": baseline_metrics,  # <-- ADD: store val baseline in fold_data
            "tuning_trials": tuning_trials,
            "param_importances": param_importances,
            "history": history
        }

        final_results = {
            "test_metrics": {k.replace('val_', 'test_'): v for k,v in best_val_metrics.items()}, # Map val -> test for reporting
            "baseline_metrics": baseline_metrics,
            "test_loss": best_val_loss,
            "baseline_loss": baseline_loss,
            "fold_data": [fold_info],
            "experiment_info": experiment_meta,
            "pipeline_config": self.config
        }
        return final_results

    def run_final_test(self, X_train: tuple, y_train: np.ndarray, 
                       X_test: tuple, y_test: np.ndarray, users_test: np.ndarray,
                       tuning_results: dict) -> dict:
        """
        Trains the model on the full training set using the best parameters found 
        during tuning, and evaluates detailed metrics on the hold-out test set.
        
        Merges final test results into the tuning_results structure for unified reporting.
        """
        start_dt, start_perf = start_timer()
        tqdm.write("\n>>> RUNNING FINAL TEST ON HOLD-OUT SET <<<")
        
        # Extract best params from tuning results
        best_params = tuning_results['fold_data'][0]['best_params']

        # Prepare Data (Fit scaler on Train, Apply to Train & Test)
        # Using prepare_fold_data logic: X_train -> X_t, X_test -> X_v
        Xt_s, yt_s, Xtest_s, ytest_s, scaler_y, _ = prepare_fold_data(X_train, y_train, X_test, y_test)

        # Apply Feature Selection if in best_params
        Xt_s, Xtest_s = apply_feature_selection(Xt_s, yt_s, Xtest_s, 
                                                n_features=best_params.get('n_path_features'), 
                                                correlation_threshold=best_params.get('correlation_threshold'))

        # Train on full X_train
        input_dims = self.model_class.get_input_dims(Xt_s)
        trainer = self._get_trainer(self.model_class, input_dims, yt_s.shape[1], best_params)
        
        tqdm.write("Training final model on full training set...")
        trainer.train_model(Xt_s, yt_s, epochs=self.config.get('epochs', 50))
        
        # Capture parameter count from the first trained model
        try:
            model_total_params = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
        except (AttributeError, TypeError):
            model_total_params = 0  # Sklearn / non-PyTorch models
        tqdm.write(f"Total trainable parameters: {model_total_params}")
        
        # Evaluate on X_test
        tqdm.write("Evaluating on test set...")
        test_loss, test_preds, test_actuals = trainer.evaluate_model(Xtest_s, ytest_s)
        
        # Run Baseline on Test Set
        dummy_baseline = DummyBaseline()
        dummy_preds, dummy_loss = dummy_baseline.run(yt_s, ytest_s)

        # Calculate Metrics (scaled)
        # Naming them 'test_' directly to match reporting expectations
        final_test_metrics = calculate_metrics([test_actuals], [test_preds], prefix="test")
        final_baseline_metrics = calculate_metrics([test_actuals], [dummy_preds], prefix="baseline")

        # --- Inverse Transform to Original Scale ---
        test_preds_unscaled = scaler_y.inverse_transform(test_preds)
        test_actuals_unscaled = scaler_y.inverse_transform(test_actuals)
        dummy_preds_unscaled = scaler_y.inverse_transform(dummy_preds)

        # Calculate Metrics on Unscaled Data
        unscaled_test_metrics = calculate_metrics([test_actuals_unscaled], [test_preds_unscaled], prefix="test_unscaled")
        unscaled_baseline_metrics = calculate_metrics([test_actuals_unscaled], [dummy_preds_unscaled], prefix="baseline_unscaled")

        start_time_str, duration_str = stop_timer(start_dt, start_perf)

        tqdm.write("\n" + "="*40)
        tqdm.write(" FINAL TEST RESULTS (Hold-Out) ")
        tqdm.write(f" Test Loss: {test_loss:.3f}")
        tqdm.write(f" Test RMSE (scaled): {final_test_metrics.get('test_rmse_mean', 0.0):.3f}")
        tqdm.write(f" Test RMSE (unscaled): {unscaled_test_metrics.get('test_unscaled_rmse_mean', 0.0):.3f}")
        tqdm.write("-" * 20)
        tqdm.write(f" Baseline Loss: {dummy_loss:.3f}")
        tqdm.write(f" Baseline RMSE (scaled): {final_baseline_metrics.get('baseline_rmse_mean', 0.0):.3f}")
        tqdm.write(f" Baseline RMSE (unscaled): {unscaled_baseline_metrics.get('baseline_unscaled_rmse_mean', 0.0):.3f}")
        tqdm.write("="*40 + "\n")

        trainer.cleanup()

        # --- UPDATE RESULTS DICTIONARY ---
        # Clone tuning results to preserve CV info if needed, but overwrite top-level metrics
        final_results = tuning_results.copy()
        
        # Preserve validation baseline before overwriting
        val_baseline_metrics = tuning_results.get('baseline_metrics', {})
        
        # Update Top-Level Metrics
        final_results['test_metrics'] = final_test_metrics
        final_results['baseline_metrics'] = final_baseline_metrics
        final_results['val_baseline_metrics'] = val_baseline_metrics  # <-- ADD: preserve val baseline
        final_results['test_loss'] = test_loss
        final_results['baseline_loss'] = dummy_loss
        
        # Unscaled metrics
        final_results['unscaled_test_metrics'] = unscaled_test_metrics
        final_results['unscaled_baseline_metrics'] = unscaled_baseline_metrics
        
        # Update Experiment Info
        final_results['experiment_info']['duration_final_test'] = duration_str
        final_results['experiment_info']['total_model_parameters'] = model_total_params
        
        # Update Fold Data for Visualization
        # We replace the CV predictions with the Final Test predictions
        # so that parity plots and error distributions reflect the Test Set performance.
        final_results['fold_data'][0]['test_preds'] = test_preds
        final_results['fold_data'][0]['test_actuals'] = test_actuals
        final_results['fold_data'][0]['test_preds_unscaled'] = test_preds_unscaled
        final_results['fold_data'][0]['test_actuals_unscaled'] = test_actuals_unscaled
        final_results['fold_data'][0]['dummy_preds_unscaled'] = dummy_preds_unscaled
        final_results['fold_data'][0]['test_loss'] = test_loss
        final_results['fold_data'][0]['baseline_loss'] = dummy_loss
        
        # Update User IDs for per-user analysis on Test Set
        # Ensure users_test is flat
        u_test_flat = users_test[:, 0] if (users_test.ndim > 1 and users_test.shape[1] == 1) else users_test
        if u_test_flat.ndim > 1: u_test_flat = u_test_flat.flatten()
        final_results['fold_data'][0]['user_id'] = u_test_flat

        return final_results

    def _optimize_hyperparameters(self, X: tuple, y: torch.Tensor, users: np.ndarray, scale_data: bool = False) -> tuple[dict, float, dict, dict, object, dict]:
        """Runs Optuna optimization using Inner LOGO."""
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        n_trials = self.config.get('n_trials', 30)
        
        study = optuna.create_study(direction="minimize")

        with tqdm(total=n_trials, desc="Hyperparam Tuning", leave=False) as pbar:
            def objective(trial):
                params = self.model_class.get_hyperparameter_space(trial)
                val_loss, val_metrics, val_metrics_unscaled = self._evaluate_params_cv(X, y, users, params, scale_data=scale_data)
                trial.set_user_attr("val_metrics", val_metrics)
                trial.set_user_attr("val_metrics_unscaled", val_metrics_unscaled)
                pbar.update(1)
                return val_loss

            study.optimize(objective, n_trials=n_trials, gc_after_trial=True)

        best_val_metrics = study.best_trial.user_attrs["val_metrics"]
        best_val_metrics_unscaled = study.best_trial.user_attrs["val_metrics_unscaled"]

        try:
            importances = optuna.importance.get_param_importances(study)
        except Exception:
            importances = None

        return study.best_params, study.best_trial.value, best_val_metrics, best_val_metrics_unscaled, study.trials_dataframe(), importances

    def _evaluate_params_cv(self, X: tuple, y: torch.Tensor, users, params, scale_data: bool = False) -> tuple[float, dict, dict]:
        """Runs LOGO CV on the provided data with given params.
        Returns: (loss, val_metrics, val_metrics_unscaled)
        """
        logo = AugmentedLOGO(include_augmented_in_test=False)
        all_val_preds = []
        all_val_actuals = []
        all_val_losses = []
        all_val_preds_unscaled = []
        all_val_actuals_unscaled = []
        
        splits = list(logo.split(groups=users))
        
        for fold_idx, (train_idx, val_idx) in enumerate(tqdm(splits, desc="LOGO CV", leave=False)):
            X_t = slice_data(X, train_idx)
            X_v = slice_data(X, val_idx)
            y_t, y_v = y[train_idx], y[val_idx]
            
            scaler_y_fold = None
            if scale_data:
                # If data is raw (Strategy B), we must scale per fold to avoid leakage
                Xt_scaled, yt_scaled, Xv_scaled, yv_scaled, scaler_y_fold, _ = prepare_fold_data(X_t, y_t, X_v, y_v)
            else:
                # If data is already scaled, use as is
                Xt_scaled, yt_scaled, Xv_scaled, yv_scaled = X_t, y_t, X_v, y_v

            # Feature Selection Step
            Xt_scaled, Xv_scaled = apply_feature_selection(Xt_scaled, yt_scaled, Xv_scaled, 
                                                           n_features=params.get('n_path_features'), 
                                                           correlation_threshold=params.get('correlation_threshold'))

            input_dims = self.model_class.get_input_dims(Xt_scaled)

            trainer = self._get_trainer(self.model_class, input_dims, yt_scaled.shape[1], params)
            trainer.train_model_and_evaluate_every_epoch(
                Xt_scaled, yt_scaled, 
                epochs=self.config.get('epochs', 50),
                X_val=Xv_scaled, y_val=yv_scaled,
                early_stopping_patience=self.config.get('early_stopping_patience')
            )

            
            val_loss, val_preds, val_actuals = trainer.evaluate_model(Xv_scaled, yv_scaled)
            
            all_val_preds.append(val_preds)
            all_val_actuals.append(val_actuals)
            all_val_losses.append(val_loss)

            # Inverse transform to original scale if scaler available
            if scaler_y_fold is not None:
                all_val_preds_unscaled.append(scaler_y_fold.inverse_transform(val_preds))
                all_val_actuals_unscaled.append(scaler_y_fold.inverse_transform(val_actuals))

            trainer.cleanup()
            del trainer
            
        val_metrics = calculate_metrics(all_val_preds, all_val_actuals, prefix="val")
        loss = float(np.mean(all_val_losses))

        val_metrics_unscaled = {}
        if all_val_preds_unscaled:
            val_metrics_unscaled = calculate_metrics(all_val_actuals_unscaled, all_val_preds_unscaled, prefix="val_unscaled")
            
        return loss, val_metrics, val_metrics_unscaled

    def _evaluate_baseline_cv(self, X: tuple, y: torch.Tensor, users) -> tuple[float, dict]:
        """
        Runs LOGO CV for the Dummy Baseline to get comparable metrics.
        """
        logo = AugmentedLOGO(include_augmented_in_test=False)
        splits = list(logo.split(groups=users))
        
        all_dummy_preds = []
        all_dummy_actuals = []
        all_dummy_losses = []
        
        dummy_baseline = DummyBaseline()

        for train_idx, val_idx in splits:
            y_t, y_v = y[train_idx], y[val_idx]
            
            # Replicate scaling for consistency
            scaler_y = StandardScaler()
            # Ensure proper type for fit_transform
            y_t_np = y_t.cpu().numpy() if isinstance(y_t, torch.Tensor) else y_t
            y_v_np = y_v.cpu().numpy() if isinstance(y_v, torch.Tensor) else y_v
            
            yt_s = torch.FloatTensor(scaler_y.fit_transform(y_t_np))
            yv_s = torch.FloatTensor(scaler_y.transform(y_v_np))
            
            d_preds, d_loss = dummy_baseline.run(yt_s, yv_s)
            
            all_dummy_preds.append(d_preds)
            all_dummy_actuals.append(yv_s.numpy())
            all_dummy_losses.append(d_loss)
            
        baseline_metrics = calculate_metrics(all_dummy_actuals, all_dummy_preds, prefix="baseline")
        mean_loss = float(np.mean(all_dummy_losses))
        
        return mean_loss, baseline_metrics
