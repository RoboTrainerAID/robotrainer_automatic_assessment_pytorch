import numpy as np
import torch
import optuna
import time
import gc
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from automatic_assessment.framework.core.trainer import Trainer
from automatic_assessment.framework.data.data_utils import AugmentedLOGO
from automatic_assessment.framework.dimred.lasso import select_features
from automatic_assessment.framework.reporting.metrics import calculate_metrics
from automatic_assessment.framework.utils.time_utils import start_timer, stop_timer
from automatic_assessment.framework.models.dummy import DummyBaseline

class Pipeline:
    def __init__(self, model_class, config):
        self.model_class= model_class
        self.config = config

    def run_nested_cv(self, X: tuple, y: np.ndarray, users: np.ndarray) -> dict:
        start_dt, start_perf = start_timer()

        # Capture Input Shapes
        input_info = {
            "shape_X_ts": list(X[0].shape),
            "shape_X_path": list(X[1].shape),
            "shape_X_user": list(X[2].shape),
            "shape_y": list(y.shape)
        }
        
        # X is (x_ts, x_path, x_user)
        # Use custom outer LOGO splitter to handle augmented data (Test on Real, remove augmented clones from Train)
        outer_splitter = AugmentedLOGO(include_augmented_in_test=False)
        splits = list(outer_splitter.split(groups=users))
        hyperparameter_mode = self.config.get('hyperparameter_mode', 'default')
        
        cached_best_params = None
        fold_data = [] # List storing data for each fold
        model_total_params = 0
        
        all_test_preds = []
        all_test_actuals = []
        all_test_losses = []

        all_dummy_preds = []
        all_dummy_losses = []

        # --- Print Model Summary Once ---
        self.model_class.print_summary(X, y)

        for fold_idx, (train_idx, test_idx) in enumerate(tqdm(splits, desc="Outer Nested CV")):
            X_t = self._slice_data(X, train_idx)
            X_test = self._slice_data(X, test_idx)
            y_t, y_test = y[train_idx], y[test_idx]
            users_t = users[train_idx]
            
            if fold_idx == 0:
                test_users_unique = np.unique(users[test_idx])
                train_users_unique = np.unique(users_t)
                tqdm.write(f"[Outer Fold 0] Test Users ({len(test_users_unique)}): {test_users_unique}")
                tqdm.write(f"[Outer Fold 0] Train Users ({len(train_users_unique)}): {train_users_unique}")

            # --- 1. Scale and Select Features (On Outer Train) ---
            Xt_s, yt_s, Xtest_s, ytest_s, scaler_y, feat_idx = self._prepare_fold_data(X_t, y_t, X_test, y_test)

            # --- 2. Hyperparameter Determination (Inner Loop) ---
            current_params = None
            val_loss = 0.0
            val_metrics = {}
            tuning_trials = None
            param_importances = None
            
            if hyperparameter_mode == 'default':
                current_params = self.model_class.get_default_parameters()
                # Run inner CV to report inner validation score
                val_loss, val_metrics = self._evaluate_params_cv(Xt_s, yt_s, users_t, current_params)
                
            elif hyperparameter_mode == 'optimize_once':
                if cached_best_params is None:
                    tqdm.write(f"[Outer Fold {fold_idx}] Optimizing Hyperparameters (Once)...")
                    current_params, val_loss, val_metrics, tuning_trials, param_importances = self._optimize_hyperparameters(Xt_s, yt_s, users_t)
                    cached_best_params = current_params
                else:
                    tqdm.write(f"[Outer Fold {fold_idx}] Reusing Cached Hyperparameters...")
                    current_params = cached_best_params
                    val_loss, val_metrics = self._evaluate_params_cv(Xt_s, yt_s, users_t, current_params)
                
            elif hyperparameter_mode == 'optimize_every_fold':
                tqdm.write(f"[Outer Fold {fold_idx}] Optimizing Hyperparameters...")
                current_params, val_loss, val_metrics, tuning_trials, param_importances = self._optimize_hyperparameters(Xt_s, yt_s, users_t)

            # --- 3. Final Model Training (Outer Loop) ---
            # Determine Input Dims Object
            input_dims = self.model_class.get_input_dims(Xt_s)
            
            # Train on full Outer Train set with determined params
            trainer = Trainer(self.model_class, input_dims, yt_s.shape[1], current_params)
            
            history = {}
            attention_weights = None
            if fold_idx == 0:
                # Capture parameter count from the first trained model
                model_total_params = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
                # Capture history for the first fold for learning curve visualization
                history = trainer.train_model_and_evaluate_every_epoch(Xt_s, yt_s, epochs=self.config.get('epochs', 50), X_val=Xtest_s, y_val=ytest_s)
                
                # Capture Attention Weights for first fold (last batch of validation)
                if hasattr(trainer.model, 'last_attn_weights') and trainer.model.last_attn_weights is not None:
                    # Detach from graph, move to cpu, convert to numpy
                    attention_weights = trainer.model.last_attn_weights.detach().cpu().numpy()
            else:
                trainer.train_model(Xt_s, yt_s, epochs=self.config.get('epochs', 50))

            
            # --- 4. Evaluate on Outer Test Set ---
            # Note: test_loss is Huber Loss. Metrics are computed explicitly below.
            test_loss, test_preds, test_actuals = trainer.evaluate_model(Xtest_s, ytest_s)

            # --- Run Dummy Baseline ---
            dummy_baseline = DummyBaseline()
            dummy_preds, dummy_loss = dummy_baseline.run(yt_s, ytest_s)

            val_rmse_str = f"{val_metrics.get('val_rmse_mean', 0.0):.3f}"
            tqdm.write(f"[Outer Fold {fold_idx}] | Test User {users[test_idx][0]}: sLoss {test_loss:.3f} | Inner Val: sLoss {val_loss:.3f}, sRMSE {val_rmse_str} | Baseline Loss: {dummy_loss:.3f}")

            # 2. Accumulate for Global Metrics
            all_test_preds.append(test_preds)
            all_test_actuals.append(test_actuals)
            all_test_losses.append(test_loss)
            all_dummy_preds.append(dummy_preds)
            all_dummy_losses.append(dummy_loss)

            fold_info = {
                "fold": fold_idx,
                "user_id": users[test_idx][0],
                "test_loss": test_loss,
                "baseline_loss": dummy_loss,
                "val_loss": val_loss,
                "test_preds": test_preds,
                "test_actuals": test_actuals,
                "feature_indices": feat_idx,
                "best_params": current_params,
                "val_metrics": val_metrics,
                "history": history,
                "tuning_trials": tuning_trials,
                "attention_weights": attention_weights,
                "param_importances": param_importances
            }
            fold_data.append(fold_info)

            if self.config.get('only_first_fold', False) and fold_idx == 0:
                tqdm.write("Only first fold requested; ending after fold 0.")
                break
            
        start_time_str, duration_str = stop_timer(start_dt, start_perf)

        experiment_meta = {
            "start_time": start_time_str,
            "duration": duration_str,
            "input_shapes": input_info,
            "total_model_parameters": model_total_params
        }

        # --- 5. Test Metric Calculation ---
        test_metrics = calculate_metrics(all_test_actuals, all_test_preds, prefix="test")
        test_loss = float(np.mean(all_test_losses))

        baseline_metrics = calculate_metrics(all_test_actuals, all_dummy_preds, prefix="baseline")
        baseline_loss = float(np.mean(all_dummy_losses))

        tqdm.write("\n" + "="*40)
        tqdm.write(" MEAN VAL RESULTS ACROSS FOLDS ")
        tqdm.write(f" Mean Val Loss (Scaled Mean Huber): {np.mean([f['val_loss'] for f in fold_data]):.3f}")
        tqdm.write(f" Mean Val RMSE (Scaled Mean): {np.mean([f['val_metrics'].get('val_rmse_mean', 0.0) for f in fold_data]):.3f}")
        tqdm.write("="*40)
        
        
        tqdm.write("\n" + "="*40)
        tqdm.write(" FINAL TEST RESULTS ")
        tqdm.write(f" Test Loss (Scaled Mean Huber): {test_loss:.3f}")
        tqdm.write(f" Test RMSE (Scaled Mean): {test_metrics['test_rmse_mean']:.3f}")
        tqdm.write(f" Test R2 (Scaled Mean):   {test_metrics['test_r2_mean']:.3f}")
        tqdm.write("-" * 20)
        tqdm.write(f" Baseline Loss: {baseline_loss:.3f}")
        tqdm.write(f" Baseline RMSE: {baseline_metrics['baseline_rmse_mean']:.3f}")
        tqdm.write("="*40 + "\n")

        final_results = {
            "test_metrics": test_metrics,
            "baseline_metrics": baseline_metrics,
            "test_loss": test_loss,
            "baseline_loss": baseline_loss,
            "fold_data": fold_data,
            "experiment_info": experiment_meta,
            "pipeline_config": self.config
        }
        
        return final_results

    def _slice_data(self, X: tuple, indices: np.ndarray) -> tuple:
        """Helper to slice tuple of arrays."""
        return tuple(x[indices] for x in X)

    def _optimize_hyperparameters(self, X: tuple, y: torch.Tensor, users: np.ndarray) -> tuple[dict, float, dict, object, dict]:
        """Runs Optuna optimization using Inner LOGO."""
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        n_trials = self.config.get('n_trials', 30)
        
        study = optuna.create_study(direction="minimize")

        with tqdm(total=n_trials, desc="Hyperparam Tuning", leave=False) as pbar:
            def objective(trial):
                # Suggest params
                params = self.model_class.get_hyperparameter_space(trial)
                
                # Evaluate using Inner CV
                val_loss, val_metrics = self._evaluate_params_cv(X, y, users, params)
                
                # Store additional metrics in the trial for later retrieval
                trial.set_user_attr("val_metrics", val_metrics)
                
                pbar.update(1)

                return val_loss

            study.optimize(objective, n_trials=n_trials, gc_after_trial=True)

        best_val_metrics = study.best_trial.user_attrs["val_metrics"]

        # Calculate parameter importance
        try:
            importances = optuna.importance.get_param_importances(study)
        except Exception:
            importances = None

        return study.best_params, study.best_trial.value, best_val_metrics, study.trials_dataframe(), importances

    def _evaluate_params_cv(self, X: tuple, y: torch.Tensor, users, params) -> tuple[float, dict]:
        """Runs LOGO CV on the provided data with given params."""
        logo = AugmentedLOGO(include_augmented_in_test=False)
        all_val_preds = []
        all_val_actuals = []
        all_val_losses = []
        
        # Use AugmentedLOGO to ensure validation is on real users only and training excludes their augmented clones.
        splits = list(logo.split(groups=users))
        
        for fold_idx, (train_idx, val_idx) in enumerate(tqdm(splits, desc="Inner Nested CV", leave=False)):
            X_t = self._slice_data(X, train_idx)
            X_v = self._slice_data(X, val_idx)
            y_t, y_v = y[train_idx], y[val_idx]
            
            # Determine Input Dims
            input_dims = self.model_class.get_input_dims(X_t)

            trainer = Trainer(self.model_class, input_dims, y_t.shape[1], params)
            trainer.train_model(X_t, y_t, epochs=self.config.get('epochs', 50))
            
            # Validate
            val_loss, val_preds, val_actuals = trainer.evaluate_model(X_v, y_v)
            
            # Accumulate
            all_val_preds.append(val_preds)
            all_val_actuals.append(val_actuals)
            all_val_losses.append(val_loss)

            # Cleanup trainer to free VRAM for next fold
            trainer.cleanup()
            del trainer
            gc.collect()
            torch.cuda.empty_cache()
            
        # Use shared metric calculation (list handling moved to metrics file)
        val_metrics = calculate_metrics(all_val_preds, all_val_actuals, prefix="val")
        loss = float(np.mean(all_val_losses))
            
        return loss, val_metrics

    def _prepare_fold_data(self, X_t: tuple, y_t, X_v: tuple, y_v) -> tuple:
        """Helper to handle scaling and LASSO within the CV loop."""        
        scaler_y = StandardScaler()
        yt_s = torch.FloatTensor(scaler_y.fit_transform(y_t))
        yv_s = torch.FloatTensor(scaler_y.transform(y_v))
        feat_idx = None

        # Structure: (x_ts, x_path, x_user)
        # 1. X_TS (N, P, F, T) -> Scale Per feature F across N, P, T
        xt_ts, xv_ts = X_t[0], X_v[0]
        N, P, F, T = xt_ts.shape
        
        # Reshape to flatten N, P, T -> (N*P*T, F) assuming scaling per feature
        # Transpose to put F last
        xt_ts_flat = xt_ts.transpose(0,1,3,2).reshape(-1, F)
        xv_ts_flat = xv_ts.transpose(0,1,3,2).reshape(-1, F)
        
        scaler_ts = StandardScaler()
        # Scaling
        xt_ts_s = scaler_ts.fit_transform(xt_ts_flat).reshape(N, P, T, F).transpose(0,1,3,2)
        xv_ts_s = scaler_ts.transform(xv_ts_flat).reshape(xv_ts.shape[0], P, T, F).transpose(0,1,3,2)
        
        # 2. X_PATH (N, P, Fp) -> Scale Per feature Fp across N, P
        xt_path, xv_path = X_t[1], X_v[1]
        N, P, Fp = xt_path.shape
        xt_path_flat = xt_path.reshape(-1, Fp)
        xv_path_flat = xv_path.reshape(-1, Fp)
        
        scaler_path = StandardScaler()
        xt_path_s = scaler_path.fit_transform(xt_path_flat).reshape(N, P, Fp)
        xv_path_s = scaler_path.transform(xv_path_flat).reshape(xv_path.shape[0], P, Fp)
        
        # 3. X_USER (N, Fu) -> Scale Per feature Fu across N
        xt_user, xv_user = X_t[2], X_v[2]
        scaler_user = StandardScaler()
        xt_user_s = scaler_user.fit_transform(xt_user)
        xv_user_s = scaler_user.transform(xv_user)

        Xt_final = (torch.FloatTensor(xt_ts_s), torch.FloatTensor(xt_path_s), torch.FloatTensor(xt_user_s))
        Xv_final = (torch.FloatTensor(xv_ts_s), torch.FloatTensor(xv_path_s), torch.FloatTensor(xv_user_s))
        
        return Xt_final, yt_s, Xv_final, yv_s, scaler_y, feat_idx