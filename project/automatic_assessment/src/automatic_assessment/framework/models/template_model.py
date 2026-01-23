import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List
from .base import BaseModel


class HierarchicalTimeseriesNetwork(BaseModel):
    model_name = "Hierarchical_Timeseries_Network"

    def __init__(self, input_dims, output_dim, hyperparams):
        super().__init__(input_dims, output_dim, hyperparams)
        
        # Dimensions based on summary example:
        # ts_shape: (Batch, 20, 20, 79)
        # path_shape: (Batch, 20, 110)
        # user_shape: (Batch, 2)
        
        ts_shape = input_dims[0]
        self.n_paths = ts_shape[1]
        self.f_ts = ts_shape[2]      # 20 (timeseries)
        
        path_shape = input_dims[1]
        self.f_path = path_shape[2]  # 110 (path related features)
        
        user_shape = input_dims[2]
        self.f_user = user_shape[1]  # 2 (Demographics)
        
        

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