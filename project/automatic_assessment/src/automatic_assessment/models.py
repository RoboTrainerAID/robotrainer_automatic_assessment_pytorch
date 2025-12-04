from abc import ABC, abstractmethod
import numpy as np
import joblib
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from skopt import BayesSearchCV
import ast

from automatic_assessment.hyperparameters import HyperparameterManager
from automatic_assessment.datasets import Dataset
from automatic_assessment.imputers import SmartImputer
from automatic_assessment.visualization import visualize_tuning_convergence


class Model(ABC):
    """
    Base class for Machine Learning models.
    Handles pipeline management, tuning, training, prediction, and persistence.
    """
    
    def __init__(self, model_type: str, dataset: Dataset):
        self.model_type = model_type
        self.dataset = dataset
        
        # Pass feature count to allow dynamic search space constraints
        n_features = len(dataset.feature_names)-1
        self.hp_manager = HyperparameterManager(model_type, dataset.type, n_features=n_features)
        
        self.pipeline = self._create_pipeline()
        self.best_params = {}

    @abstractmethod
    def _create_pipeline(self) -> Pipeline:
        """Creates the full pipeline including preprocessing and regressor."""
        pass

    def apply_default_params(self):
        """Applies default parameters from HyperparameterManager."""
        print("Loading default parameters from HyperparameterManager...")
        params = self.hp_manager.get_best_params()
        self.best_params = params
        self.pipeline.set_params(**params)
        print(f"Pipeline configured with params: {params}")

    def tune(self, X: np.ndarray, y: np.ndarray, users: np.ndarray, n_iter: int = 50):
        """
        Performs Bayesian Optimization to find best hyperparameters.
        Generates a convergence plot to visualize Chaos -> Improvement -> Elbow.
        
        Args:
            X: Training features.
            y: Training targets.
            users: User IDs corresponding to X (for grouping).
            n_iter: Number of optimization iterations.
        """
        print(f"\n--- Starting Hyperparameter Tuning ({self.model_type.upper()}) ---")
        
        # Retrieve strategy and params directly from the stored dataset
        cv_splitter = self.dataset.get_cv_strategy()
        fit_params = self.dataset.get_fit_params(X, y, users)

        search_spaces = self.hp_manager.get_search_space()
        
        # WRAPPER IMPLEMENTATION:
        # Wrap the pipeline to handle Y-scaling internally per fold
        wrapped_estimator = AutoScalingRegressor(self.pipeline)
        
        # Adjust search space keys to match the wrapper structure (estimator__...)
        wrapped_search_spaces = {f"estimator__{k}": v for k, v in search_spaces.items()}
        
        opt = BayesSearchCV(
            estimator=wrapped_estimator,
            search_spaces=wrapped_search_spaces,
            n_iter=n_iter,
            cv=cv_splitter,
            n_jobs=-1,
            verbose=1,
            random_state=0,
            scoring=None, # Use the estimator's score method (which returns negative MSE)
            return_train_score=False # Save memory
        )
        
        # 1. Run the Optimization
        # Note: We pass RAW y here. The wrapper handles scaling.
        opt.fit(X, y, **fit_params)
        
        print(f"Best CV Score (MSE): {-opt.best_score_:.4f}")
        
        # Strip the 'estimator__' prefix from best_params to apply back to self.pipeline
        best_params_clean = {k.replace("estimator__", ""): v for k, v in opt.best_params_.items()}
        print("Best Parameters found:", best_params_clean)
        
        self.best_params = best_params_clean
        
        # Configure the final pipeline with the best parameters found
        self.pipeline.set_params(**self.best_params)

        # --- VISUALIZATION: OBJECTIVE VS ITERATION ---
        visualize_tuning_convergence(opt.cv_results_, n_iter, self.model_type)

    def train(self, X: np.ndarray, y: np.ndarray):
        """Trains the model on the provided data."""
        self.pipeline.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicts targets for X."""
        return self.pipeline.predict(X)

    def save(self, path: str):
        """Saves the trained pipeline to disk."""
        joblib.dump(self.pipeline, path)
        print(f"Model saved to {path}")

    def load(self, path: str):
        """Loads a trained pipeline from disk."""
        self.pipeline = joblib.load(path)
        print(f"Model loaded from {path}")

    @staticmethod
    def create(model_type: str, dataset: Dataset) -> "Model":
        """Factory method to create Model instances."""
        if model_type == 'svr':
            return SVRModel(dataset)
        elif model_type == 'svr_single':
            return IndependentSVRModel(dataset)
        elif model_type == 'rf':
            return RandomForestModel(dataset)
        elif model_type == 'mlp':
            return MLPModel(dataset)
        else:
            raise ValueError(f"Unknown model type: {model_type}")


class SVRModel(Model):
    def __init__(self, dataset: Dataset):
        super().__init__('svr', dataset)

    def _create_pipeline(self) -> Pipeline:
        # n_jobs=1 ensures BayesSearchCV manages the parallelism (Outer Loop Parallelism)
        regressor = MultiOutputRegressor(SVR(), n_jobs=1)
        steps = [
            ('imputer', SmartImputer(strategy='path_mean')),
            # ('imputer', SmartImputer(strategy='user_mean')),
            # ('imputer', SmartImputer(strategy='path_mean')),
            # ('imputer', SmartImputer(strategy='global_mean')),
            # ('imputer', SimpleImputer(strategy='global_mean')),
            ('scaler', StandardScaler()),
            ('pca', PCA()),
            ('regressor', regressor)
        ]
        return Pipeline(steps)


class RandomForestModel(Model):
    def __init__(self, dataset: Dataset):
        super().__init__('rf', dataset)

    def _create_pipeline(self) -> Pipeline:
        # n_jobs=1 prevents contention with BayesSearchCV
        regressor = RandomForestRegressor(random_state=0, n_jobs=1)
        steps = [
            ('imputer', SmartImputer(strategy='path_mean')),
            ('regressor', regressor)
        ]
        return Pipeline(steps)


class MLPModel(Model):
    def __init__(self, dataset: Dataset):
        super().__init__('mlp', dataset)

    def _create_pipeline(self) -> Pipeline:
        # Use the wrapper class instead of standard MLPRegressor
        regressor = TunableMLPRegressor(random_state=0)
        steps = [
            ('imputer', SmartImputer(strategy='path_mean')),
            ('scaler', StandardScaler()),
            ('pca', PCA()),
            ('regressor', regressor)
        ]
        return Pipeline(steps)
    
class TunableMLPRegressor(MLPRegressor):
    """
    Wrapper for MLPRegressor that accepts hidden_layer_sizes as a string.
    This is a workaround for a known issue in skopt/BayesSearchCV where passing 
    tuples as parameters causes a ValueError.
    """
    def fit(self, X, y):
        # Convert string representation back to tuple if necessary
        if isinstance(self.hidden_layer_sizes, str):
            self.hidden_layer_sizes = ast.literal_eval(self.hidden_layer_sizes)
        return super().fit(X, y)
    

class AutoScalingRegressor(BaseEstimator, RegressorMixin):
    """
    A wrapper that scales targets (Y) internally before fitting, 
    but keeps evaluation in SCALED units to ensure equal importance 
    for multi-output tuning.
    """
    def __init__(self, estimator):
        self.estimator = estimator
        self.y_scaler = StandardScaler()
        self.estimators_ = None # For multi-output compatibility check

    def fit(self, X, y, **kwargs):
        # 1. Scale Y using ONLY the training data provided in this fold
        # This prevents the data leakage.
        self.y_scaler = StandardScaler()
        y_scaled = self.y_scaler.fit_transform(y)
        
        # 2. Clone the base estimator (clean slate)
        self.estimator_ = clone(self.estimator)
        
        # 3. Fit the inner model on Scaled Y
        self.estimator_.fit(X, y_scaled, **kwargs)
        
        # Save nested attributes for debugging (optional)
        if hasattr(self.estimator_, 'estimators_'):
            self.estimators_ = self.estimator_.estimators_
            
        return self

    def predict(self, X):
        # Return SCALED predictions. 
        # We do NOT inverse_transform here because we want the Tuner
        # to calculate error based on scaled units (Equal Importance).
        return self.estimator_.predict(X)

    def score(self, X, y):
        # This is what BayesSearchCV uses to decide who wins.
        
        # 1. Get Scaled Predictions
        y_pred_scaled = self.predict(X)
        
        # 2. Scale the Validation Y using the TRAINING scaler
        # (Simulating real world: we don't know Val stats yet)
        y_val_scaled = self.y_scaler.transform(y)
        
        # 3. Calculate MSE on SCALED data
        return -mean_squared_error(y_val_scaled, y_pred_scaled)


class IndependentSVRModel(Model):
    """
    SVR model that trains a separate regressor for each target column.
    Optimizes hyperparameters independently for each target.
    """
    def __init__(self, dataset: Dataset):
        super().__init__('svr', dataset) # Use 'svr' to get SVR hyperparameters
        self.pipelines = []
        self.best_params_per_target = []

    def _create_pipeline(self) -> Pipeline:
        # Single target pipeline (No MultiOutputRegressor)
        steps = [
            ('imputer', SmartImputer(strategy='user_mean')),
            ('scaler', StandardScaler()),
            ('pca', PCA()),
            ('regressor', SVR())
        ]
        return Pipeline(steps)

    def tune(self, X: np.ndarray, y: np.ndarray, users: np.ndarray, n_iter: int = 50):
        print(f"\n--- Starting Independent Hyperparameter Tuning (SVR per Target) ---")
        
        n_targets = y.shape[1]
        self.best_params_per_target = []
        
        cv_splitter = self.dataset.get_cv_strategy()
        fit_params = self.dataset.get_fit_params(X, y, users)
        
        # Get generic SVR search space
        search_spaces = self.hp_manager.get_search_space()
        
        # Fix keys: remove 'estimator__' which was for MultiOutputRegressor
        # The HP manager returns keys like 'regressor__estimator__C' -> we want 'regressor__C'
        single_target_spaces = {}
        for k, v in search_spaces.items():
            new_k = k.replace('regressor__estimator__', 'regressor__')
            single_target_spaces[new_k] = v
            
        # Wrap for AutoScalingRegressor (adds 'estimator__' prefix back, but for the wrapper)
        wrapped_search_spaces = {f"estimator__{k}": v for k, v in single_target_spaces.items()}

        for i in range(n_targets):
            target_name = self.dataset.TARGET_COLS[i]
            print(f"\n--- Tuning Target {i+1}/{n_targets}: {target_name} ---")
            
            y_single = y[:, i]
            
            pipeline = self._create_pipeline()
            wrapped_estimator = AutoScalingRegressor(pipeline)
            
            opt = BayesSearchCV(
                estimator=wrapped_estimator,
                search_spaces=wrapped_search_spaces,
                n_iter=n_iter,
                cv=cv_splitter,
                n_jobs=-1,
                verbose=0,
                random_state=0,
                scoring=None, 
                return_train_score=False
            )
            
            opt.fit(X, y_single, **fit_params)
            
            print(f"Best Score (MSE): {-opt.best_score_:.4f}")
            
            # Clean params: remove 'estimator__' prefix from wrapper
            best_params_clean = {k.replace("estimator__", ""): v for k, v in opt.best_params_.items()}
            print(f"Best Params: {best_params_clean}")
            self.best_params_per_target.append(best_params_clean)

    def train(self, X: np.ndarray, y: np.ndarray):
        self.pipelines = []
        n_targets = y.shape[1]
        
        use_defaults = len(self.best_params_per_target) == 0
        
        fixed_defaults = {}
        if use_defaults:
             # Get default params and fix keys
            defaults = self.hp_manager.get_best_params()
            for k, v in defaults.items():
                fixed_defaults[k.replace('regressor__estimator__', 'regressor__')] = v
            print("Using default parameters for all targets.")
        
        for i in range(n_targets):
            pipeline = self._create_pipeline()
            
            if not use_defaults:
                pipeline.set_params(**self.best_params_per_target[i])
            else:
                pipeline.set_params(**fixed_defaults)
            
            pipeline.fit(X, y[:, i])
            self.pipelines.append(pipeline)

    def predict(self, X: np.ndarray) -> np.ndarray:
        preds = []
        for pipeline in self.pipelines:
            preds.append(pipeline.predict(X))
        return np.column_stack(preds)
    
    def apply_default_params(self):
        # Defaults are applied during training if no tuning occurred
        print("IndependentSVRModel: Default parameters will be applied during training.")

    def save(self, path: str):
        joblib.dump(self.pipelines, path)
        print(f"Model saved to {path}")

    def load(self, path: str):
        self.pipelines = joblib.load(path)
        print(f"Model loaded from {path}")
