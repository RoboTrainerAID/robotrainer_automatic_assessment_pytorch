import numpy as np
import torch
import torch.nn as nn
from typing import Any, Dict, List, Tuple
from automatic_assessment.framework.models.sklearn.sklearn_base import SklearnBaseModel

class SklearnTrainer:
    """
    Trainer adapter for Scikit-Learn based models ensuring compatibility 
    with the existing pipeline's Trainer interface.
    """
    def __init__(self, model_class: type, input_dims: List[int], output_dim: int, hyperparams: Dict[str, Any]):
        self.model_class = model_class
        self.hyperparams = hyperparams
        self.model = self.model_class(input_dims, output_dim, hyperparams)
        # Interface parity with Trainer (sklearn models fit in one "epoch")
        self.last_best_epoch = 1
        
    def train_model(self, X_train: tuple, y_train: torch.Tensor, epochs: int = 1, X_val: tuple = None, y_val: torch.Tensor = None) -> List[float]:
        """
        Trains the Sklearn model. 
        'epochs' is ignored as these are non-iterative (or internally iterative) solvers.
        Returns a dummy loss history.
        """
        # Fit the model
        self.model.fit(X_train, y_train)
        
        # Return a single dummy loss value or training loss if possible
        # Since sklearn fit usually doesn't return history in the same way, we return [0.0]
        return [0.0]

    def evaluate_model(self, X_val: tuple, y_val: torch.Tensor) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Evaluates the model on validation data.
        Returns:
            val_loss (float): MSE Loss
            preds (np.ndarray): Predictions
            actuals (np.ndarray): Ground truth
        """
        # Predict
        preds = self.model.predict(X_val)
        
        # Ensure y_val is numpy
        if isinstance(y_val, torch.Tensor):
            y_val_np = y_val.detach().cpu().numpy()
        else:
            y_val_np = y_val
            
        # Calculate MSE Loss manually since we don't have a torch criterion
        diff = preds - y_val_np
        val_loss = float(np.mean(diff**2))

        return val_loss, preds, y_val_np

    def train_model_and_evaluate_every_epoch(self, X_train, y_train, epochs, X_val, y_val, early_stopping_patience=None):
        """
        Mimics the epoch-by-epoch training and evaluation.
        For Sklearn, we just fit once and evaluate once, but return a history 
        structure that looks like the deep learning one for compatibility.
        """
        self.model.fit(X_train, y_train)
        
        val_loss, preds, actuals = self.evaluate_model(X_val, y_val)
        
        # Create a "fake" history with 1 epoch
        history = {
            'train_loss': [0.0],
            'val_loss': [val_loss]
        }
        return history

    def cleanup(self):
        """No GPU cleanup needed for Sklearn."""
        pass
