import torch.nn as nn
import numpy as np
from abc import abstractmethod
from typing import Dict, Any, Tuple
from torchinfo import summary

class BaseModel(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hyperparams: Dict[str, Any]):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hyperparams = hyperparams

    @property
    @abstractmethod
    def model_name(self) -> str:
        pass

    @staticmethod
    @abstractmethod
    def get_hyperparameter_space(trial) -> Dict[str, Any]:
        """Returns a dict for Optuna to sample."""
        pass

    @staticmethod
    @abstractmethod
    def get_default_parameters() -> Dict[str, Any]:
        """Returns a dict with default Model hyperparameters."""
        pass

    @classmethod
    def print_summary(cls, X: np.ndarray, y: np.ndarray) -> None:
        """
        Prints the model summary using torchinfo.
        Instantiates a dummy model using default parameters and the shapes from X.
        
        Args:
            X (np.ndarray): The input data used to determine input dimensions (N, F, T).
        """
        default_params = cls.get_default_parameters()
        
        # Infer dimensions from X: (Samples, Features, Time)
        input_dim = X.shape[1]
        time_len = X.shape[2]
        output_dim = y.shape[1] if len(y.shape) > 1 else 1
        
        # Instantiate the specific child class (cls)
        dummy_model = cls(input_dim, output_dim, default_params)

        batch_size = default_params.get('batch_size', 32)
        
        # Input shape for Conv1d models is (Batch, Features, Time)
        input_size = (batch_size, input_dim, time_len)

        print(f"\n--- Model Summary: {dummy_model.model_name} ---")
        print(f"Input Specification:")
        print(f"  - Batch Size:     {batch_size}")
        print(f"  - Timeseries Len: {time_len}")
        print(f"  - Features:       {input_dim}")
        print(f"  - Tensor Shape:   {input_size}")
        try:
            summary(dummy_model, input_size=input_size)
        except Exception as e:
            print(f"Failed to generate summary: {e}")
        print("---------------------------------------")

    @abstractmethod
    def forward(self, x: Any) -> Any:
        pass