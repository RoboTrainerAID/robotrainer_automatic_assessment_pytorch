import torch
import torch.nn as nn
from typing import Dict, Any
from .base import BaseModel

class HierarchicalCNN(BaseModel):
    model_name = "Hierarchical_Path_CNN"

    def __init__(self, input_dim: int, output_dim: int, hyperparams: Dict[str, Any]):
        super().__init__(input_dim, output_dim, hyperparams)
        
        # Architecture Constraints based on dataset structure
        # Features (Paths * Timeseries) = 20 * 20 = 400
        self.n_paths = hyperparams.get("n_paths", 20)
        
        # Calculate features per path (e.g., 400 / 20 = 20)
        if input_dim % self.n_paths != 0:
            raise ValueError(f"Input dim {input_dim} must be divisible by n_paths {self.n_paths}")
            
        self.feats_per_path = input_dim // self.n_paths
        
        path_hidden_dim = hyperparams.get("path_hidden_dim", 32)
        dropout = hyperparams.get("dropout", 0.2)
        
        # ---------------------------------------------------------
        # 1. Temporal Extraction (Depthwise Conv over Time)
        # ---------------------------------------------------------
        # Input: (Batch, Input_Dim=400, Time=79)
        # Goal: Capture timeseries information per channel independently.
        # We use groups=input_dim to learn a separate filter for each valid feature/path combination.
        # This keeps the 400 channels separated but condenses time.
        
        self.temporal_block = nn.Sequential(
            nn.Conv1d(input_dim, input_dim, kernel_size=5, padding=2, groups=input_dim),
            nn.BatchNorm1d(input_dim),
            nn.ReLU(),
            nn.Conv1d(input_dim, input_dim, kernel_size=5, padding=2, groups=input_dim),
            nn.BatchNorm1d(input_dim),
            nn.ReLU(),
            # Condense time dimension completely (79 -> 1)
            nn.AdaptiveAvgPool1d(1) 
        )
        # Output after block: (Batch, 400, 1) -> Squeeze -> (Batch, 400)
        
        
        # ---------------------------------------------------------
        # 2. Intra-Path Feature Extraction
        # ---------------------------------------------------------
        # "Kernel steps through feature dimension... captures all features from one path"
        # We transform the (Batch, 400) vector into (Batch, 1, 400)
        # We want to convolve over the length 400.
        # - Kernel Size = feats_per_path (20) -> Captures one full path's features.
        # - Stride = feats_per_path (20) -> Jumps directly to the next path.
        # - Filters: We learn `path_hidden_dim` different ways to summarize a path.
        
        self.path_feature_extractor = nn.Sequential(
            nn.Conv1d(in_channels=1, 
                      out_channels=path_hidden_dim, 
                      kernel_size=self.feats_per_path, 
                      stride=self.feats_per_path),
            nn.BatchNorm1d(path_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        # Calculation: (400 - 20) / 20 + 1 = 19 + 1 = 20 steps.
        # Output shape: (Batch, path_hidden_dim, n_paths=20)
        
        
        # ---------------------------------------------------------
        # 3. Inter-Path Aggregation
        # ---------------------------------------------------------
        # "Done again for the paths"
        # Now we process the sequence of 20 paths to find global user patterns.
        
        self.path_aggregator = nn.Sequential(
            nn.Conv1d(path_hidden_dim, path_hidden_dim * 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(path_hidden_dim * 2),
            nn.ReLU(),
            # Condense path dimension (20 -> 1)
            nn.AdaptiveAvgPool1d(1) 
        )
        # Output shape: (Batch, path_hidden_dim * 2, 1) -> Squeeze -> (Batch, 64)
        
        
        # ---------------------------------------------------------
        # 4. Final MLP (Regression)
        # ---------------------------------------------------------
        self.regressor = nn.Sequential(
            nn.Linear(path_hidden_dim * 2, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (Batch, Features=400, Time=79)
        
        # 1. Temporal Condensation
        # Output: (Batch, 400, 1)
        x = self.temporal_block(x)
        
        # Remove time dim: (Batch, 400)
        x = x.squeeze(-1)
        
        # 2. Intra-Path processing
        # Reshape to (Batch, 1, 400) to treat features as a linear sequence
        x = x.unsqueeze(1)
        
        # Output: (Batch, path_hidden_dim, n_paths=20)
        # Each position in the last dim corresponds to one path's condensed embedding
        x = self.path_feature_extractor(x)
        
        # 3. Inter-Path Aggregation
        # Output: (Batch, path_hidden_dim*2, 1)
        x = self.path_aggregator(x)
        x = x.squeeze(-1)
        
        # 4. Regression
        return self.regressor(x)

    @staticmethod
    def get_hyperparameter_space(trial) -> Dict[str, Any]:
        return {
            "n_paths": 20, # Fixed structure assumption
            "path_hidden_dim": trial.suggest_categorical("path_hidden_dim", [16, 32, 64]),
            "dropout": trial.suggest_float("dropout", 0.1, 0.5),
            "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-3, 1, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [4, 8, 16])
        }
    
    @staticmethod
    def get_default_parameters() -> Dict[str, Any]:
        return {
            "lr": 0.001,
            "path_hidden_dim": 32,
            "dropout": 0.3,
            "weight_decay": 0.05,
            "n_paths": 20,
            "batch_size": 16
        }