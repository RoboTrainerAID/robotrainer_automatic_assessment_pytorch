import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List

from .base import BaseModel


class ModelTemplate(BaseModel):
    model_name = "ModelTemplate"

    def __init__(self, input_dims, output_dim, hyperparams):
        super().__init__(input_dims, output_dim, hyperparams)
        
        # X tuple: (x_ts, x_path, x_user)
        #     1. x_ts: Time-Series Dataset
        #         - Shape (variable): (n_samples, n_paths, n_timeseries, variable_length)
        #         - Values: (25, 20, 32, variable_timesteps)
        #     2. x_path: Path-Level Dataset
        #         - Shape: (n_samples, n_paths, n_path_features)
        #         - Values: (25, 20, 73)
        #     3. x_user: User-Level Dataset
        #         - Shape: (n_samples, n_user_features)
        #         - Values: (25, 2)

        # y: Targets
        #     - Shape: (n_samples, n_targets)
        #     - Values: (25, 14)
        
        ts_shape = input_dims[0]
        self.n_paths = ts_shape[1]
        self.n_ts = ts_shape[2]
        self.max_timesteps = ts_shape[3]
        
        path_shape = input_dims[1]
        self.f_path = path_shape[2]
        
        user_shape = input_dims[2]
        self.f_user = user_shape[1]
        
        

    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        x_ts, x_path, x_user = x[0], x[1], x[2]
        batch_size = x_ts.shape[0]
        
        
        return prediction
    
    @staticmethod
    def get_hyperparameter_space(trial) -> Dict[str, Any]:
        return {}

    @staticmethod
    def get_default_parameters() -> Dict[str, Any]:
        return {}