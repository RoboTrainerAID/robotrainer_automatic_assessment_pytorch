import numpy as np
import torch
import optuna
from tqdm import tqdm

from automatic_assessment.framework.core.trainer import Trainer
from automatic_assessment.framework.core.trainer_sklearn import SklearnTrainer
from automatic_assessment.framework.models.sklearn.sklearn_base import SklearnBaseModel
from automatic_assessment.framework.data import schema
from automatic_assessment.framework.data.data_utils import AugmentedLOGO, prepare_fold_data, slice_data, apply_feature_selection
from automatic_assessment.framework.data.results import (
    ExperimentConfig, MetricTable, FoldResult, ValidationResult, TestResult,
    ExperimentInfo, ExperimentResult,
)
from automatic_assessment.framework.utils.time_utils import start_timer, stop_timer
from automatic_assessment.framework.models.dummy import DummyBaseline


class SimplePipeline:
    """
    Two-stage pipeline built on the typed result objects
    (framework/data/results.py):

    1. run_simple_tuning -> ExperimentResult (test=None):
       LOGO-CV over the training users, optionally preceded by an Optuna
       hyperparameter search. Validation results contain per-user CV
       predictions (scaled + original units) and the dummy-baseline
       reference evaluated on the SAME folds with the SAME scalers.

       NOTE on interpretation: when hyperparameter_mode='optimize', the
       validation score belongs to the configuration selected by the
       search on these same folds — a model-SELECTION score, optimistic
       as a generalization estimate (see open_improvements.md).

    2. run_final_test -> NEW ExperimentResult (test=TestResult):
       trains one model on all training users using the MEAN best-epoch
       from the CV folds as the epoch budget (early stopping on the test
       evaluation itself would let the test set pick the model) and
       evaluates once on the hold-out split. The tuning-only result
       object is never mutated.
    """

    def __init__(self, model_class, config: ExperimentConfig):
        self.model_class = model_class
        self.config = config

    def _get_trainer(self, model_class, input_dims, output_dim, params):
        if issubclass(model_class, SklearnBaseModel):
            return SklearnTrainer(model_class, input_dims, output_dim, params)
        else:
            return Trainer(model_class, input_dims, output_dim, params)

    # ------------------------------------------------------------------
    # Stage 1: LOGO-CV validation (+ optional hyperparameter search)
    # ------------------------------------------------------------------

    def run_simple_tuning(self, X: tuple, y: np.ndarray, users: np.ndarray) -> ExperimentResult:
        start_dt, start_perf = start_timer()

        tqdm.write("\n>>> RUNNING SIMPLE PIPELINE (LOGO-CV Validation) <<<")
        self.model_class.print_summary(X, y)

        tuning_trials = None
        param_importances = None
        best_trial_val_loss = None

        if self.config.hyperparameter_mode == 'default':
            tqdm.write("Using Default Hyperparameters...")
            best_params = self.model_class.get_default_parameters()
        else:
            tqdm.write("Performing Hyperparameter Tuning on the training users using LOGO CV.")
            best_params, best_trial_val_loss, tuning_trials, param_importances = \
                self._optimize_hyperparameters(X, y, users)

        # --- Definitive CV evaluation with the chosen configuration ---
        # (For 'optimize' this re-runs the best configuration once so that
        #  per-user predictions and histories are available. Due to fresh
        #  weight initialization the score can differ slightly from the
        #  best trial's value; both are reported.)
        tqdm.write("Running definitive LOGO CV with the selected configuration...")
        validation = self._evaluate_params_cv(X, y, users, best_params, collect_details=True)

        # --- Epoch budget for the final model ---
        # Mean of the per-fold best epochs (training-side information only;
        # early stopping on the test evaluation would bias the test score).
        best_epochs = validation.best_epochs
        if self.config.early_stopping_patience is not None and best_epochs:
            final_epochs = int(round(float(np.mean(best_epochs))))
        else:
            final_epochs = int(self.config.epochs)
        final_epochs = max(1, final_epochs)

        start_time_str, duration_str = stop_timer(start_dt, start_perf)

        info = ExperimentInfo(
            start_time=start_time_str,
            duration=duration_str,
            strategy="logo_cv_validation",
            input_shapes=schema.input_shapes_info(X, y),
            total_model_parameters=validation.folds[0].model_parameters if validation.folds else 0,
        )

        tqdm.write("\n" + "=" * 40)
        tqdm.write(" VALIDATION RESULTS (LOGO CV) ")
        if best_trial_val_loss is not None:
            tqdm.write(f" Best Trial CV Loss (selection score): {best_trial_val_loss:.3f}")
        tqdm.write(f" CV Loss: {validation.loss:.3f}")
        tqdm.write(f" CV RMSE: {validation.metrics.rmse_mean:.3f}")
        tqdm.write("-" * 20)
        tqdm.write(f" Baseline Loss: {validation.baseline_loss:.3f}")
        tqdm.write(f" Baseline RMSE: {validation.baseline_metrics.rmse_mean:.3f}")
        tqdm.write(f" Final epoch budget (mean best epoch): {final_epochs}")
        tqdm.write("=" * 40 + "\n")

        return ExperimentResult(
            model_name=self.model_class.model_name,
            config=self.config,
            best_params=best_params,
            validation=validation,
            final_epochs=final_epochs,
            info=info,
            test=None,
            best_trial_val_loss=best_trial_val_loss,
            tuning_trials=tuning_trials,
            param_importances=param_importances,
        )

    # ------------------------------------------------------------------
    # Stage 2: optional final test on the hold-out split
    # ------------------------------------------------------------------

    def run_final_test(self, X_train: tuple, y_train: np.ndarray,
                       X_test: tuple, y_test: np.ndarray, users_test: np.ndarray,
                       tuning_result: ExperimentResult) -> ExperimentResult:
        start_dt, start_perf = start_timer()
        tqdm.write("\n>>> RUNNING FINAL TEST ON HOLD-OUT SET <<<")

        best_params = tuning_result.best_params
        final_epochs = tuning_result.final_epochs
        targets = tuning_result.targets

        # Prepare Data (Fit scaler on Train, Apply to Train & Test)
        Xt_s, yt_s, Xtest_s, ytest_s, scaler_y, _ = prepare_fold_data(X_train, y_train, X_test, y_test)

        # Apply Feature Selection (fit on train only)
        Xt_s, Xtest_s = apply_feature_selection(Xt_s, yt_s, Xtest_s,
                                                n_features=best_params.get('n_path_features'),
                                                correlation_threshold=best_params.get('correlation_threshold'))

        input_dims = self.model_class.get_input_dims(Xt_s)
        trainer = self._get_trainer(self.model_class, input_dims, yt_s.shape[1], best_params)

        tqdm.write(f"Training final model on full training set for {final_epochs} epochs "
                   f"(mean best epoch from CV)...")
        trainer.train_model(Xt_s, yt_s, epochs=final_epochs)

        try:
            model_total_params = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
        except (AttributeError, TypeError):
            model_total_params = 0  # Sklearn / non-PyTorch models
        tqdm.write(f"Total trainable parameters: {model_total_params}")

        tqdm.write("Evaluating on test set...")
        test_loss, test_preds, test_actuals = trainer.evaluate_model(Xtest_s, ytest_s)

        # Baseline on Test Set
        dummy_baseline = DummyBaseline()
        dummy_preds, dummy_loss = dummy_baseline.run(yt_s, ytest_s)

        # Inverse transforms to original units
        test_preds_u = scaler_y.inverse_transform(test_preds)
        test_actuals_u = scaler_y.inverse_transform(test_actuals)
        dummy_preds_u = scaler_y.inverse_transform(dummy_preds)

        test = TestResult(
            targets=targets,
            user_ids=[int(u) for u in np.asarray(users_test).flatten()],
            loss=float(test_loss),
            baseline_loss=float(dummy_loss),
            metrics=MetricTable.from_predictions(test_actuals, test_preds, targets),
            metrics_unscaled=MetricTable.from_predictions(test_actuals_u, test_preds_u, targets),
            baseline_metrics=MetricTable.from_predictions(test_actuals, dummy_preds, targets),
            baseline_metrics_unscaled=MetricTable.from_predictions(test_actuals_u, dummy_preds_u, targets),
            preds=test_preds,
            actuals=test_actuals,
            preds_unscaled=test_preds_u,
            actuals_unscaled=test_actuals_u,
            baseline_preds_unscaled=dummy_preds_u,
            model_parameters=model_total_params,
        )

        start_time_str, duration_str = stop_timer(start_dt, start_perf)
        test.duration = duration_str

        tqdm.write("\n" + "=" * 40)
        tqdm.write(" FINAL TEST RESULTS (Hold-Out) ")
        tqdm.write(f" Test Loss: {test.loss:.3f}")
        tqdm.write(f" Test RMSE (scaled): {test.metrics.rmse_mean:.3f}")
        tqdm.write(f" Test RMSE (unscaled): {test.metrics_unscaled.rmse_mean:.3f}")
        tqdm.write("-" * 20)
        tqdm.write(f" Baseline Loss: {test.baseline_loss:.3f}")
        tqdm.write(f" Baseline RMSE (scaled): {test.baseline_metrics.rmse_mean:.3f}")
        tqdm.write("=" * 40 + "\n")

        trainer.cleanup()

        # New result object; the tuning-only result stays untouched.
        info = ExperimentInfo(
            start_time=tuning_result.info.start_time,
            duration=tuning_result.info.duration,
            strategy=tuning_result.info.strategy,
            input_shapes=tuning_result.info.input_shapes,
            total_model_parameters=model_total_params,
            duration_final_test=duration_str,
        )
        return ExperimentResult(
            model_name=tuning_result.model_name,
            config=tuning_result.config,
            best_params=best_params,
            validation=tuning_result.validation,
            final_epochs=final_epochs,
            info=info,
            test=test,
            best_trial_val_loss=tuning_result.best_trial_val_loss,
            tuning_trials=tuning_result.tuning_trials,
            param_importances=tuning_result.param_importances,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _optimize_hyperparameters(self, X: tuple, y: torch.Tensor, users: np.ndarray):
        """Runs Optuna optimization using LOGO CV as the trial objective."""
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        n_trials = self.config.n_trials

        sampler = optuna.samplers.TPESampler(seed=self.config.seed)
        study = optuna.create_study(direction="minimize", sampler=sampler)

        with tqdm(total=n_trials, desc="Hyperparam Tuning", leave=False) as pbar:
            def objective(trial):
                params = self.model_class.get_hyperparameter_space(trial)
                validation = self._evaluate_params_cv(X, y, users, params, collect_details=False)
                trial.set_user_attr("val_rmse_mean", validation.metrics.rmse_mean)
                pbar.update(1)
                return validation.loss

            study.optimize(objective, n_trials=n_trials, gc_after_trial=True)

        try:
            importances = optuna.importance.get_param_importances(study)
        except Exception:
            importances = None

        return study.best_params, study.best_trial.value, study.trials_dataframe(), importances

    def _evaluate_params_cv(self, X: tuple, y: torch.Tensor, users, params,
                            collect_details: bool = False) -> ValidationResult:
        """
        LOGO CV on the raw data with the given params. Scaling and feature
        selection are fitted inside each fold; the dummy (train-mean)
        baseline is evaluated on the SAME folds with the SAME scalers.

        Returns a ValidationResult. FoldResult details (histories and
        predictions) are populated only when collect_details=True —
        metric tables and losses are always complete.
        """
        targets = list(self.config.targets)
        logo = AugmentedLOGO(include_augmented_in_test=False)
        splits = list(logo.split(groups=users))

        dummy_baseline = DummyBaseline()

        all_preds, all_actuals = [], []
        all_preds_u, all_actuals_u = [], []
        all_base_preds, all_base_preds_u = [], []
        all_losses, all_base_losses = [], []
        folds = []

        for fold_idx, (train_idx, val_idx) in enumerate(tqdm(splits, desc="LOGO CV", leave=False)):
            X_t = slice_data(X, train_idx)
            X_v = slice_data(X, val_idx)
            y_t, y_v = y[train_idx], y[val_idx]

            # Scale per fold (fit on fold-train only) to avoid leakage
            Xt_s, yt_s, Xv_s, yv_s, scaler_y_fold, _ = prepare_fold_data(X_t, y_t, X_v, y_v)

            # Feature Selection Step (fit on fold-train only)
            Xt_s, Xv_s = apply_feature_selection(Xt_s, yt_s, Xv_s,
                                                 n_features=params.get('n_path_features'),
                                                 correlation_threshold=params.get('correlation_threshold'))

            input_dims = self.model_class.get_input_dims(Xt_s)

            trainer = self._get_trainer(self.model_class, input_dims, yt_s.shape[1], params)
            history = trainer.train_model_and_evaluate_every_epoch(
                Xt_s, yt_s,
                epochs=self.config.epochs,
                X_val=Xv_s, y_val=yv_s,
                early_stopping_patience=self.config.early_stopping_patience
            )

            val_loss, val_preds, val_actuals = trainer.evaluate_model(Xv_s, yv_s)

            # Dummy baseline on the same fold/scaler
            base_preds, base_loss = dummy_baseline.run(yt_s, yv_s)

            # Inverse transform to original units (per-fold scaler)
            val_preds_u = scaler_y_fold.inverse_transform(val_preds)
            val_actuals_u = scaler_y_fold.inverse_transform(val_actuals)
            base_preds_u = scaler_y_fold.inverse_transform(base_preds)

            all_preds.append(val_preds)
            all_actuals.append(val_actuals)
            all_preds_u.append(val_preds_u)
            all_actuals_u.append(val_actuals_u)
            all_base_preds.append(base_preds)
            all_base_preds_u.append(base_preds_u)
            all_losses.append(val_loss)
            all_base_losses.append(base_loss)

            if collect_details:
                try:
                    n_params = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
                except (AttributeError, TypeError):
                    n_params = 0
                folds.append(FoldResult(
                    fold=fold_idx,
                    user_id=int(np.asarray(users)[val_idx][0]),
                    val_loss=float(val_loss),
                    best_epoch=getattr(trainer, 'last_best_epoch', None),
                    history=history,
                    val_preds=val_preds,
                    val_actuals=val_actuals,
                    val_preds_unscaled=val_preds_u,
                    val_actuals_unscaled=val_actuals_u,
                    baseline_preds=base_preds,
                    baseline_preds_unscaled=base_preds_u,
                    model_parameters=n_params,
                ))

            trainer.cleanup()
            del trainer

        cat = np.concatenate
        return ValidationResult(
            targets=targets,
            loss=float(np.mean(all_losses)),
            baseline_loss=float(np.mean(all_base_losses)),
            metrics=MetricTable.from_predictions(cat(all_actuals), cat(all_preds), targets),
            metrics_unscaled=MetricTable.from_predictions(cat(all_actuals_u), cat(all_preds_u), targets),
            baseline_metrics=MetricTable.from_predictions(cat(all_actuals), cat(all_base_preds), targets),
            baseline_metrics_unscaled=MetricTable.from_predictions(cat(all_actuals_u), cat(all_base_preds_u), targets),
            folds=folds,
        )
