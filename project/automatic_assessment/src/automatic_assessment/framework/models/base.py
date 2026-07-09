import torch
import torch.nn as nn
import numpy as np
from abc import abstractmethod
from typing import Dict, Any, Tuple, List, Union
from torchinfo import summary

class BaseModel(nn.Module):
    def __init__(self, input_dims: Union[int, List[Tuple[int, ...]]], output_dim: int, hyperparams: Dict[str, Any]):
        super().__init__()
        self.input_dims = input_dims
        self.output_dim = output_dim
        self.hyperparams = hyperparams

    @classmethod
    def get_input_dims(cls, X: Any) -> List[Tuple[int, ...]]:
        """
        Infers the input dimension object from the data (e.g. shapes of tensors in tuple).
        Returns a structure matching X but with shapes instead of data.
        """
        if isinstance(X, (list, tuple)):
            return [x.shape for x in X]
        return [X.shape]

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
    def print_summary(cls, X: Tuple[np.ndarray, ...], y: np.ndarray) -> None:
        """
        Prints the model summary using torchinfo.
        """
        default_params = cls.get_default_parameters()
        
        # Calculate dims using the class method
        input_dims = cls.get_input_dims(X)
        output_dim = y.shape[1] if len(y.shape) > 1 else 1
        batch_size = default_params.get('batch_size', 32)
        
        # Instantiate model with calculated dims
        dummy_model = cls(input_dims, output_dim, default_params)

        print(f"\n--- Model Summary: {dummy_model.model_name} ---")
        print(f"Input Dims Structure: {input_dims}")
        
        # Create Dummy Data from input_dims for torchinfo
        # input_dims is [(N, P, F, T), (N, P, Fp), (N, Fu)]
        # We replace N with batch_size
        dummy_input = []
        for shape in input_dims:
            # Construct shape: (batch_size, *shape[1:])
            dummy_shape = (batch_size,) + shape[1:]
            dummy_input.append(torch.zeros(dummy_shape))
            
        try:
            # Pass input_data as a list of args. Argument 0 is the input list
            # (x_path, x_user, g0_x, g0_mask, ...).
            # Models expect a single list argument containing the tensors
            summary(dummy_model, input_data=[dummy_input])
        except Exception as e:
            print(f"Failed to generate summary: {e}")
            import traceback
            traceback.print_exc()
        print("---------------------------------------")

    @abstractmethod
    def forward(self, x: Any) -> Any:
        pass