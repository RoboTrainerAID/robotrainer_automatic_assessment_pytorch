import numpy as np
import torch
import optuna
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from automatic_assessment.framework.core.trainer import Trainer
from automatic_assessment.framework.data.data_utils import AugmentedLOGO
from automatic_assessment.framework.dimred.lasso import select_features
from automatic_assessment.framework.reporting.metrics import calculate_metrics

class Pipeline:
    def __init__(self, model_class, config):
        self.model_class = model_class
        self.config = config

    def run_nested_cv(self, X: np.ndarray, y: np.ndarray, users: np.ndarray) -> dict:
        # Use custom outer LOGO splitter to handle augmented data (Test on Real, remove augmented clones from Train)
        outer_splitter = AugmentedLOGO(include_augmented_in_test=False)
        splits = list(outer_splitter.split(groups=users))
        hyperparameter_mode = self.config.get('hyperparameter_mode', 'default')
        
        cached_best_params = None
        fold_data = [] # List storing data for each fold
        
        all_test_preds = []
        all_test_actuals = []
        all_test_losses = []

        # --- Print Model Summary Once ---
        self.model_class.print_summary(X, y)

        for fold_idx, (train_idx, test_idx) in enumerate(tqdm(splits, desc="Outer Nested CV")):
            X_t, X_test = X[train_idx], X[test_idx]
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
            
            if hyperparameter_mode == 'default':
                if fold_idx == 2:
                    # TODO: remove this
                    break

                current_params = self.model_class.get_default_parameters()
                # Run inner CV to report inner validation score
                val_loss, val_metrics = self._evaluate_params_cv(Xt_s, yt_s, users_t, current_params)
                
            elif hyperparameter_mode == 'optimize_once':
                if cached_best_params is None:
                    tqdm.write(f"[Outer Fold {fold_idx}] Optimizing Hyperparameters (Once)...")
                    current_params, val_loss, val_metrics = self._optimize_hyperparameters(Xt_s, yt_s, users_t)
                    cached_best_params = current_params
                else:
                    tqdm.write(f"[Outer Fold {fold_idx}] Reusing Cached Hyperparameters...")
                    current_params = cached_best_params
                    val_loss, val_metrics = self._evaluate_params_cv(Xt_s, yt_s, users_t, current_params)
                
            elif hyperparameter_mode == 'optimize_every_fold':
                tqdm.write(f"[Outer Fold {fold_idx}] Optimizing Hyperparameters...")
                current_params, val_loss, val_metrics = self._optimize_hyperparameters(Xt_s, yt_s, users_t)

            # --- 3. Final Model Training (Outer Loop) ---
            # Train on full Outer Train set with determined params
            trainer = Trainer(self.model_class, Xt_s.shape[1], yt_s.shape[1], current_params)
            
            # Capture history for the first fold for learning curve visualization
            history = {}
            if fold_idx == 0:
                history = trainer.train_model_and_evaluate_every_epoch(Xt_s, yt_s, epochs=self.config.get('epochs', 50), X_val=Xtest_s, y_val=ytest_s)
            else:
                trainer.train_model(Xt_s, yt_s, epochs=self.config.get('epochs', 50))
            
            # --- 4. Evaluate on Outer Test Set ---
            # Note: test_loss is Huber Loss. Metrics are computed explicitly below.
            test_loss, test_preds, test_actuals = trainer.evaluate_model(Xtest_s, ytest_s)

            val_rmse_str = f"{val_metrics.get('val_rmse_mean', 0.0):.3f}"
            tqdm.write(f"[Outer Fold {fold_idx}] | Test User {users[test_idx][0]}: sLoss {test_loss:.3f} | Inner Val: sLoss {val_loss:.3f}, sRMSE {val_rmse_str}")

            # 2. Accumulate for Global Metrics
            all_test_preds.append(test_preds)
            all_test_actuals.append(test_actuals)
            all_test_losses.append(test_loss)

            fold_info = {
                "fold": fold_idx,
                "user_id": users[test_idx][0],
                "test_loss": test_loss,
                "val_loss": val_loss,
                "test_preds": test_preds,
                "test_actuals": test_actuals,
                "feature_indices": feat_idx,
                "best_params": current_params,
                "val_metrics": val_metrics,
                "history": history
            }
            fold_data.append(fold_info)
            
        # --- 5. Test Metric Calculation ---
        test_metrics = calculate_metrics(all_test_actuals, all_test_preds, prefix="test")
        test_loss = float(np.mean(all_test_losses))

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
        tqdm.write("="*40 + "\n")

        final_results = {
            "test_metrics": test_metrics,
            "test_loss": test_loss,
            "fold_data": fold_data
        }
        
        return final_results

    def _optimize_hyperparameters(self, X: torch.Tensor, y: torch.Tensor, users: np.ndarray) -> tuple[dict, float, dict]:
        """Runs Optuna optimization using Inner LOGO."""
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        
        def objective(trial):
            # Suggest params
            params = self.model_class.get_hyperparameter_space(trial)
            
            # Evaluate using Inner CV
            val_loss, val_metrics = self._evaluate_params_cv(X, y, users, params)
            
            # Store additional metrics in the trial for later retrieval
            trial.set_user_attr("val_metrics", val_metrics)
            
            return val_loss

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=self.config.get('n_trials', 30))

        best_val_metrics = study.best_trial.user_attrs["val_metrics"]

        return study.best_params, study.best_trial.value, best_val_metrics

    def _evaluate_params_cv(self, X: torch.Tensor, y: torch.Tensor, users, params) -> tuple[float, dict]:
        """Runs LOGO CV on the provided data with given params."""
        logo = AugmentedLOGO(include_augmented_in_test=False)
        all_val_preds = []
        all_val_actuals = []
        all_val_losses = []
        
        # Use AugmentedLOGO to ensure validation is on real users only and training excludes their augmented clones.
        splits = logo.split(groups=users)
        
        for fold_idx, (train_idx, val_idx) in enumerate(tqdm(splits, desc="Inner Nested CV", leave=False)):
            X_t, X_v = X[train_idx], X[val_idx] # split tensors directly
            y_t, y_v = y[train_idx], y[val_idx]
            users_t = users[train_idx]

            trainer = Trainer(self.model_class, X_t.shape[1], y_t.shape[1], params)
            trainer.train_model(X_t, y_t, epochs=self.config.get('epochs', 50))
            
            # Validate
            val_loss, val_preds, val_actuals = trainer.evaluate_model(X_v, y_v)
            
            # Accumulate
            all_val_preds.append(val_preds)
            all_val_actuals.append(val_actuals)
            all_val_losses.append(val_loss)
            
        # Use shared metric calculation (list handling moved to metrics file)
        val_metrics = calculate_metrics(all_val_preds, all_val_actuals, prefix="val")
        loss = float(np.mean(all_val_losses))
            
        return loss, val_metrics

    def _prepare_fold_data(self, X_t: np.ndarray, y_t: np.ndarray, X_v: np.ndarray, y_v: np.ndarray) -> tuple:
        """Helper to handle scaling and LASSO within the CV loop."""        
        # Input Assumed: (N, F, T) where F=Features, T=Time
        N_t, F, T = X_t.shape
        N_v = X_v.shape[0]
        
        # Scale X (per feature, across time and samples)
        # 1. Transpose to (N, T, F) so features are last for StandardScaler
        Xt_trans = X_t.transpose(0, 2, 1)
        Xv_trans = X_v.transpose(0, 2, 1)
        
        # 2. Reshape to (N*T, F)
        Xt_flat = Xt_trans.reshape(-1, F)
        Xv_flat = Xv_trans.reshape(-1, F)

        # 3. Scale
        scaler_x = StandardScaler()
        Xt_s_flat = scaler_x.fit_transform(Xt_flat)
        Xv_s_flat = scaler_x.transform(Xv_flat)
        
        # 4. Reshape back to (N, T, F)
        Xt_s_trans = Xt_s_flat.reshape(N_t, T, F)
        Xv_s_trans = Xv_s_flat.reshape(N_v, T, F)

        # 5. Transpose back to (N, F, T)
        Xt_final = Xt_s_trans.transpose(0, 2, 1)
        Xv_final = Xv_s_trans.transpose(0, 2, 1)
        
        # Scale Y
        scaler_y = StandardScaler()
        yt_s = scaler_y.fit_transform(y_t)
        yv_s = scaler_y.transform(y_v)
        
        # LASSO
        feat_idx = np.arange(F)
        if self.config.get('use_lasso', False):
            # Aggregate time dim for feature selection: (N, F, T) -> mean(axis=2) -> (N, F)
            X_mean = Xt_final.mean(axis=2)
            n_features = min(self.config.get("max_n_features"), F)
            feat_idx, _ = select_features(X_mean, yt_s, n_features=n_features)
            
            Xt_final = Xt_final[:, feat_idx, :]
            Xv_final = Xv_final[:, feat_idx, :]
            
        return (torch.FloatTensor(Xt_final), torch.FloatTensor(yt_s), 
                torch.FloatTensor(Xv_final), torch.FloatTensor(yv_s), 
                scaler_y, feat_idx)