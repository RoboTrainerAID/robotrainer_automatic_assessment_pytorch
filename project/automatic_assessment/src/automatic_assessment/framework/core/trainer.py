import torch
import torch.nn as nn
import numpy as np

from automatic_assessment.framework.data.data_utils import get_dataloader
from automatic_assessment.framework.models.base import BaseModel

class Trainer:
    def __init__(self, model_class, input_dims: list, output_dim: int, params: dict):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: BaseModel = model_class(input_dims=input_dims, output_dim=output_dim, hyperparams=params).to(self.device)
        self.params = params
        self.criterion = nn.HuberLoss(delta=1.0)

        # Improvement 1: AdamW with high weight decay for small datasets
        wd = params.get('weight_decay')
        lr = params.get('lr')
        
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=lr, 
            weight_decay=wd,
            betas=(0.9, 0.999)
        )
        
        # Improvement 2: Scheduler
        self.scheduler_class = torch.optim.lr_scheduler.CosineAnnealingLR
        self.eta_min = params.get('eta_min', 1e-7)

    def train_model(self, X_train: tuple, y_train: torch.Tensor, epochs: int) -> None:
        train_loader = get_dataloader(*X_train, y_train, batch_size=self.params.get('batch_size', 16))
        scheduler = self.scheduler_class(self.optimizer, T_max=epochs, eta_min=self.eta_min)
        
        for _ in range(epochs):
            self.train_epoch(train_loader)
            scheduler.step()

    def train_model_and_evaluate_every_epoch(self, X_train: tuple, y_train: torch.Tensor, epochs: int, X_val: tuple, y_val: torch.Tensor) -> dict:
        train_loader = get_dataloader(*X_train, y_train, batch_size=self.params.get('batch_size', 16))
        val_loader = get_dataloader(*X_val, y_val, batch_size=len(y_val), shuffle=False)
        scheduler = self.scheduler_class(self.optimizer, T_max=epochs, eta_min=self.eta_min)

        history = {'train_loss': [], 'val_loss': []}

        for _ in range(epochs):
            train_loss = self.train_epoch(train_loader)
            history['train_loss'].append(train_loss)
            
            val_loss, _, _ = self.evaluate(val_loader)
            history['val_loss'].append(val_loss)
            
            scheduler.step()
        
        return history

    def evaluate_model(self, X: tuple, y: torch.Tensor) -> tuple[float, np.ndarray, np.ndarray]:
        loader = get_dataloader(*X, y, batch_size=len(y), shuffle=False)
        return self.evaluate(loader)

    def train_epoch(self, loader) -> float:
        self.model.train()
        losses = []
        for batch in loader:
            # Inputs: (x_ts, x_path, x_user, y)
            inputs = batch[:-1] 
            y = batch[-1]
            
            inputs = [x.to(self.device) for x in inputs]
            y = y.to(self.device)
            
            self.optimizer.zero_grad()
            # Pass list of inputs to model
            pred = self.model(inputs)
            loss = self.criterion(pred, y)
            loss.backward()
            self.optimizer.step()
            losses.append(loss.item())
        return float(np.mean(losses))

    def evaluate(self, loader) -> tuple[float, np.ndarray, np.ndarray]:
        self.model.eval()
        losses = []
        preds, actuals = [], []
        with torch.no_grad():
            for batch in loader:
                inputs = batch[:-1]
                y = batch[-1]
                
                inputs = [x.to(self.device) for x in inputs]
                y = y.to(self.device)
                
                pred = self.model(inputs)
                loss = self.criterion(pred, y)
                losses.append(loss.item())
                preds.append(pred.cpu().numpy())
                actuals.append(y.cpu().numpy())
        return float(np.mean(losses)), np.concatenate(preds), np.concatenate(actuals)