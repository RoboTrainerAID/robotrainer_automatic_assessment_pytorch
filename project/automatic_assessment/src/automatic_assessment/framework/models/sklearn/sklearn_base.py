from sklearn.base import BaseEstimator
import numpy as np
import torch
from abc import abstractmethod
from typing import Dict, Any, List, Union

class SklearnBaseModel(BaseEstimator):
    """
    Abstract base class for Scikit-Learn based models ensuring compatibility with 
    the existing pipeline (e.g. print_summary, get_input_dims).
    """
    def __init__(self, input_dims, output_dim, hyperparams):
        """
        Args:
            input_dims: Shape info (ignored by sklearn models usually, but kept for interface limit)
            output_dim: Number of targets (needed for multi-output wrapping)
            hyperparams: Dictionary of hyperparameters
        """
        self.input_dims = input_dims
        self.output_dim = output_dim
        self.hyperparams = hyperparams
        self.model = None # To be initialized in child classes

    @property
    def model_name(self) -> str:
        """String identifier for the model."""
        return self.__class__.__name__
    
    @staticmethod
    def get_input_dims(X: Any) -> List[Any]:
        """Same helper as BaseModel to keep pipeline happy."""
        if isinstance(X, (list, tuple)):
            return [x.shape for x in X]
        return [X.shape]

    @staticmethod
    def get_hyperparameter_space(trial) -> Dict[str, Any]:
        """Returns Optuna search space."""
        return {}

    @staticmethod
    def get_default_parameters() -> Dict[str, Any]:
        """Returns default parameters."""
        return {}

    def _flatten_inputs(self, x: Union[List[torch.Tensor], tuple]) -> np.ndarray:
        """
        Converts the standardized input tuple (x_path, x_user, *ts_groups) into a single 
        flattened numpy array [N, Features].
        Ignores the timeseries groups, takes x_path and x_user.
        """
        if isinstance(x, (list, tuple)):
            # x_ts is usually ignored in simple baselines unless we extract features
            # Here we follow the logic: flatten(path) + user
            # Input convention: (x_path, x_user, *ts_groups)
            x_path, x_user = x[0], x[1]
        else:
            # Fallback if x is already single tensor
            return x.detach().cpu().numpy()

        B = x_path.shape[0]
        
        # Move to CPU
        x_path_np = x_path.detach().cpu().numpy()
        x_user_np = x_user.detach().cpu().numpy()
        
        # Flatten Path: (B, P, F) -> (B, P*F)
        x_path_flat = x_path_np.reshape(B, -1)
        
        # Concatenate: [Path_Flat, User]
        x_all = np.concatenate([x_path_flat, x_user_np], axis=1)
        
        return x_all

    def fit(self, X, y):
        """
        Trains the internal sklearn model.
        Args:
            X: Input tuple (x_path, x_user, *ts_groups)
            y: Tensor target (N, T)
        """
        X_flat = self._flatten_inputs(X)
        y_np = y.detach().cpu().numpy()
        
        # Check if model supports simple fit or needs multioutput handling logic inside the class
        # Most classes will initialize self.model as MultiOutputRegressor if needed
        self.model.fit(X_flat, y_np)

    def predict(self, X) -> np.ndarray:
        """
        Predicts using the internal sklearn model.
        Args:
            X: Tuple of tensors
        Returns:
            Predictions as numpy array (N, T) or Tensor equivalent
        """
        X_flat = self._flatten_inputs(X)
        preds = self.model.predict(X_flat)
        return preds

    def __call__(self, X):
        """
        Mimics PyTorch forward pass for the pipeline evaluation loops.
        Returns torch.Tensor to obey pipeline contract (loss calculation uses torch tensors).
        """
        preds_np = self.predict(X)
        # Assuming X contains tensors, use that device or CPU
        if isinstance(X, (list, tuple)) and isinstance(X[0], torch.Tensor):
            device = X[0].device 
        else:
            device = torch.device("cpu")
            
        return torch.from_numpy(preds_np).float().to(device)

    @classmethod
    def print_summary(cls, X, y):
        print(f"\n--- Sklearn Model Summary: {cls.__name__} ---")
        print("Scikit-Learn based baseline.")
        print("Input: Flattened (Path Features + User Features)")
        print(f"Output Targets: {y.shape[1]}")
        print("---------------------------------------")

    def to(self, device):
        # Sklearn models stay on CPU. This method handles the call from Trainer 
        # but does nothing for the model itself.
        return self

    def train(self):
        # Sklearn models don't have train/eval modes
        pass

    def eval(self):
        # Sklearn models don't have train/eval modes
        pass

    def update_running_mean(self, y):
        pass

    def parameters(self):
        """Returns an empty iterator to stay compatible with the pipeline's
        parameter counting logic (which calls model.parameters())."""
        return iter([])
