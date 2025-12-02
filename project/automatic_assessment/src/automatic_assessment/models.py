from abc import ABC
import numpy as np
import joblib
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from skopt import BayesSearchCV

from automatic_assessment.hyperparameters import HyperparameterManager
from automatic_assessment.datasets import Dataset

class Model(ABC):
    """
    Base class for Machine Learning models.
    Handles pipeline management, tuning, training, prediction, and persistence.
    """
    
    def __init__(self, model_type: str, dataset: Dataset):
        self.model_type = model_type
        self.dataset = dataset
        self.hp_manager = HyperparameterManager(model_type, dataset.type)
        self.pipeline = None
        self.best_params = {}

    def set_pipeline(self, pipeline: Pipeline, params: dict = None):
        """
        Sets the pipeline and applies parameters.
        Args:
            pipeline: The sklearn Pipeline instance to use.
            params: Dictionary of parameters to apply. 
                    If None, uses defaults from HyperparameterManager.
        """
        self.pipeline = pipeline
        
        if params is None:
            print("Loading default parameters from HyperparameterManager...")
            params = self.hp_manager.get_best_params()
            self.best_params = params
        
        self.pipeline.set_params(**params)
        print(f"Pipeline configured with params: {params}")

    def tune(self, X: np.ndarray, y: np.ndarray, users: np.ndarray, pipeline_template: Pipeline, n_iter: int = 50):
        """
        Performs Bayesian Optimization to find best hyperparameters.
        Uses the stored dataset object to determine CV strategy and fit parameters.
        
        Args:
            X: Training features.
            y: Training targets.
            users: User IDs corresponding to X (for grouping).
            pipeline_template: The pipeline structure to tune.
            n_iter: Number of optimization iterations.
        """
        print(f"\n--- Starting Hyperparameter Tuning ({self.model_type.upper()}) ---")
        
        # Retrieve strategy and params directly from the stored dataset
        cv_splitter = self.dataset.get_cv_strategy()
        fit_params = self.dataset.get_fit_params(X, y, users)

        search_spaces = self.hp_manager.get_search_space()
        
        opt = BayesSearchCV(
            estimator=pipeline_template,
            search_spaces=search_spaces,
            n_iter=n_iter,
            cv=cv_splitter,
            n_jobs=-1,
            verbose=1,
            random_state=0,
            scoring='neg_mean_squared_error'
        )
        
        opt.fit(X, y, **fit_params)
        
        print(f"Best CV Score (MSE): {-opt.best_score_:.4f}")
        print("Best Parameters found:", opt.best_params_)
        
        self.best_params = opt.best_params_
        
        # Configure the final pipeline with the best parameters found
        self.set_pipeline(pipeline_template, self.best_params)

    def train(self, X: np.ndarray, y: np.ndarray):
        """Trains the model on the provided data."""
        if self.pipeline is None:
            raise RuntimeError("Pipeline not set. Call set_pipeline() or tune() first.")
        self.pipeline.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicts targets for X."""
        if self.pipeline is None:
            raise RuntimeError("Model has not been trained yet.")
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
        elif model_type == 'rf':
            return RandomForestModel(dataset)
        else:
            raise ValueError(f"Unknown model type: {model_type}")


class SVRModel(Model):
    def __init__(self, dataset: Dataset):
        super().__init__('svr', dataset)


class RandomForestModel(Model):
    def __init__(self, dataset: Dataset):
        super().__init__('rf', dataset)