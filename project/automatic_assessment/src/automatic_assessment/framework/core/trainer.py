import torch
import torch.nn as nn
import numpy as np

from automatic_assessment.framework.data.data_utils import get_dataloader

class Trainer:
    def __init__(self, model_class, input_dim: int, output_dim: int, params: dict):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model_class(input_dim=input_dim, output_dim=output_dim, hyperparams=params).to(self.device)
        self.params = params
        self.criterion = nn.HuberLoss(delta=1.0)

        # Improvement 1: AdamW with high weight decay for small datasets
        wd = params.get('weight_decay', 0.05)
        lr = params.get('lr', 1e-3)
        
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=lr, 
            weight_decay=wd,
            betas=(0.9, 0.999)
        )
        
        # Improvement 2: Scheduler
        # Reduces LR if validation loss stops improving
        # self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #     self.optimizer, 
        #     mode='min', 
        #     factor=0.5, 
        #     patience=10
        # )

    def train_model(self, X_train: torch.Tensor, y_train: torch.Tensor, epochs: int) -> None:
        train_loader = get_dataloader(X_train, y_train, batch_size=self.params.get('batch_size', 16))
        for _ in range(epochs):
            self.train_epoch(train_loader)

    def train_model_and_evaluate_every_epoch(self, X_train: torch.Tensor, y_train: torch.Tensor, epochs: int, X_val: torch.Tensor, y_val: torch.Tensor) -> dict:
        train_loader = get_dataloader(X_train, y_train, batch_size=self.params.get('batch_size', 16))
        val_loader = get_dataloader(X_val, y_val, batch_size=len(X_val), shuffle=False)

        history = {'train_loss': [], 'val_loss': []}

        for _ in range(epochs):
            train_loss = self.train_epoch(train_loader)
            history['train_loss'].append(train_loss)
            
            val_loss, _, _ = self.evaluate(val_loader)
            history['val_loss'].append(val_loss)
        
        return history

    def evaluate_model(self, X: torch.Tensor, y: torch.Tensor) -> tuple[float, np.ndarray, np.ndarray]:
        loader = get_dataloader(X, y, batch_size=len(X), shuffle=False)
        return self.evaluate(loader)

    def train_epoch(self, loader) -> float:
        self.model.train()
        losses = []
        for X, y in loader:
            X, y = X.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            pred = self.model(X)
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
            for X, y in loader:
                X, y = X.to(self.device), y.to(self.device)
                pred = self.model(X)
                loss = self.criterion(pred, y)
                losses.append(loss.item())
                preds.append(pred.cpu().numpy())
                actuals.append(y.cpu().numpy())
        return float(np.mean(losses)), np.concatenate(preds), np.concatenate(actuals)