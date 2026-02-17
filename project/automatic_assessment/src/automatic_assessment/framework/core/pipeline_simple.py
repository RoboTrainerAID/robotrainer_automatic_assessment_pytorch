import numpy as np
import torch
import optuna
import time
import gc
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from automatic_assessment.framework.core.trainer import Trainer
from automatic_assessment.framework.data.data_utils import AugmentedLOGO, prepare_fold_data, slice_data
from automatic_assessment.framework.reporting.metrics import calculate_metrics
from automatic_assessment.framework.utils.time_utils import start_timer, stop_timer
from automatic_assessment.framework.models.dummy import DummyBaseline

class SimplePipeline:
    def __init__(self, model_class, config):
        self.model_class= model_class
        self.config = config

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

        if hyperparameter_mode == 'default':
            tqdm.write("Using Default Hyperparameters...")
            best_params = self.model_class.get_default_parameters()
            # Simple tuning passes raw data, so we must scale inside CV
            best_val_loss, best_val_metrics = self._evaluate_params_cv(X, y, users, best_params, scale_data=True)
            
        else:
            tqdm.write("Performing Hyperparameter Tuning on full dataset using LOGO CV.")
            # Simple tuning passes raw data, so we must scale inside optimization loop
            best_params, best_val_loss, best_val_metrics, tuning_trials, param_importances = self._optimize_hyperparameters(X, y, users, scale_data=True)

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
            
            input_dims = self.model_class.get_input_dims(Xt_s)
            trainer = Trainer(self.model_class, input_dims, yt_s.shape[1], best_params)
            
            history = trainer.train_model_and_evaluate_every_epoch(
                Xt_s, yt_s, 
                epochs=self.config.get('epochs', 50), 
                X_val=Xv_s, 
                y_val=yv_s
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

    def run_final_test(self, X_train: tuple, y_train: np.ndarray, X_test: tuple, y_test: np.ndarray, best_params: dict) -> dict:
        """
        Trains the model on the full training set using the best parameters found 
        during tuning, and evaluates detailed metrics on the hold-out test set.
        """
        start_dt, start_perf = start_timer()
        tqdm.write("\n>>> RUNNING FINAL TEST ON HOLD-OUT SET <<<")

        # Prepare Data (Fit scaler on Train, Apply to Train & Test)
        # Using prepare_fold_data logic: X_train -> X_t, X_test -> X_v
        Xt_s, yt_s, Xtest_s, ytest_s, _, _ = prepare_fold_data(X_train, y_train, X_test, y_test)

        # Train on full X_train
        input_dims = self.model_class.get_input_dims(Xt_s)
        trainer = Trainer(self.model_class, input_dims, yt_s.shape[1], best_params)
        
        tqdm.write("Training final model on full training set...")
        trainer.train_model(Xt_s, yt_s, epochs=self.config.get('epochs', 50))
        
        # Evaluate on X_test
        tqdm.write("Evaluating on test set...")
        test_loss, test_preds, test_actuals = trainer.evaluate_model(Xtest_s, ytest_s)
        
        # Run Baseline on Test Set
        dummy_baseline = DummyBaseline()
        dummy_preds, dummy_loss = dummy_baseline.run(yt_s, ytest_s)

        # Calculate Metrics
        final_test_metrics = calculate_metrics([test_actuals], [test_preds], prefix="final_test")
        final_baseline_metrics = calculate_metrics([test_actuals], [dummy_preds], prefix="final_baseline")

        start_time_str, duration_str = stop_timer(start_dt, start_perf)

        tqdm.write("\n" + "="*40)
        tqdm.write(" FINAL TEST RESULTS (Hold-Out) ")
        tqdm.write(f" Test Loss: {test_loss:.3f}")
        tqdm.write(f" Test RMSE: {final_test_metrics.get('final_test_rmse_mean', 0.0):.3f}")
        tqdm.write("-" * 20)
        tqdm.write(f" Baseline Loss: {dummy_loss:.3f}")
        tqdm.write(f" Baseline RMSE: {final_baseline_metrics.get('final_baseline_rmse_mean', 0.0):.3f}")
        tqdm.write("="*40 + "\n")

        trainer.cleanup()

        return {
            "final_test_metrics": final_test_metrics,
            "final_baseline_metrics": final_baseline_metrics,
            "final_test_loss": test_loss,
            "final_baseline_loss": dummy_loss,
            "final_test_preds": test_preds,
            "final_test_actuals": test_actuals,
            "duration": duration_str
        }

    def _optimize_hyperparameters(self, X: tuple, y: torch.Tensor, users: np.ndarray, scale_data: bool = False) -> tuple[dict, float, dict, object, dict]:
        """Runs Optuna optimization using Inner LOGO."""
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        n_trials = self.config.get('n_trials', 30)
        
        study = optuna.create_study(direction="minimize")

        with tqdm(total=n_trials, desc="Hyperparam Tuning", leave=False) as pbar:
            def objective(trial):
                params = self.model_class.get_hyperparameter_space(trial)
                val_loss, val_metrics = self._evaluate_params_cv(X, y, users, params, scale_data=scale_data)
                trial.set_user_attr("val_metrics", val_metrics)
                pbar.update(1)
                return val_loss

            study.optimize(objective, n_trials=n_trials, gc_after_trial=True)

        best_val_metrics = study.best_trial.user_attrs["val_metrics"]

        try:
            importances = optuna.importance.get_param_importances(study)
        except Exception:
            importances = None

        return study.best_params, study.best_trial.value, best_val_metrics, study.trials_dataframe(), importances

    def _evaluate_params_cv(self, X: tuple, y: torch.Tensor, users, params, scale_data: bool = False) -> tuple[float, dict]:
        """Runs LOGO CV on the provided data with given params."""
        logo = AugmentedLOGO(include_augmented_in_test=False)
        all_val_preds = []
        all_val_actuals = []
        all_val_losses = []
        
        splits = list(logo.split(groups=users))
        
        for fold_idx, (train_idx, val_idx) in enumerate(tqdm(splits, desc="Inner CV", leave=False)):
            X_t = slice_data(X, train_idx)
            X_v = slice_data(X, val_idx)
            y_t, y_v = y[train_idx], y[val_idx]
            
            if scale_data:
                # If data is raw (Strategy B), we must scale per fold to avoid leakage
                Xt_scaled, yt_scaled, Xv_scaled, yv_scaled, _, _ = prepare_fold_data(X_t, y_t, X_v, y_v)
            else:
                # If data is already scaled, use as is
                Xt_scaled, yt_scaled, Xv_scaled, yv_scaled = X_t, y_t, X_v, y_v

            input_dims = self.model_class.get_input_dims(Xt_scaled)

            trainer = Trainer(self.model_class, input_dims, yt_scaled.shape[1], params)
            trainer.train_model(Xt_scaled, yt_scaled, epochs=self.config.get('epochs', 50))
            
            val_loss, val_preds, val_actuals = trainer.evaluate_model(Xv_scaled, yv_scaled)
            
            all_val_preds.append(val_preds)
            all_val_actuals.append(val_actuals)
            all_val_losses.append(val_loss)

            trainer.cleanup()
            del trainer
            gc.collect()
            torch.cuda.empty_cache()
            
        val_metrics = calculate_metrics(all_val_preds, all_val_actuals, prefix="val")
        loss = float(np.mean(all_val_losses))
            
        return loss, val_metrics

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
