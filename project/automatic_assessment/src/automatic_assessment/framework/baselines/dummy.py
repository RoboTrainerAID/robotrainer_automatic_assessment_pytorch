import numpy as np
import torch
import torch.nn as nn

class DummyBaseline:
    """
    Simple baseline that predicts the mean of the training targets.
    Used for comparison against trained models.
    """
    def __init__(self):
        self.criterion = nn.HuberLoss(delta=1.0)
    
    def run(self, y_train: torch.Tensor, y_test: torch.Tensor) -> tuple[np.ndarray, float]:
        """
        Fits on training data, predicts on test data, and calculates loss.
        
        Args:
            y_train (torch.Tensor): Training targets.
            y_test (torch.Tensor): Test targets.
            
        Returns:
            tuple[np.ndarray, float]: (predictions, loss)
        """
        # Fit: Calculate mean across samples (dim 0)
        mean = torch.mean(y_train, dim=0)
        
        # Predict
        n_samples = y_test.shape[0]
        preds = mean.unsqueeze(0).expand(n_samples, -1)
        
        # Evaluate
        loss = self.criterion(preds, y_test).item()
        
        return preds.numpy(), loss
