import copy
import gc

import numpy as np
import torch
import torch.nn as nn

from automatic_assessment.framework.data.data_utils import get_dataloader
from automatic_assessment.framework.models.base import BaseModel


def _is_cuda_oom(err: BaseException) -> bool:
    """True only for genuine CUDA out-of-memory errors (never for other RuntimeErrors)."""
    oom_type = getattr(torch.cuda, "OutOfMemoryError", None)
    if oom_type is not None and isinstance(err, oom_type):
        return True
    return isinstance(err, RuntimeError) and "out of memory" in str(err).lower()


class Trainer:
    def __init__(self, model_class, input_dims: list, output_dim: int, params: dict):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_class = model_class
        self.input_dims = input_dims
        self.output_dim = output_dim
        self.params = params
        self.criterion = nn.HuberLoss(delta=1.0)

        # Scheduler config
        self.scheduler_class = torch.optim.lr_scheduler.CosineAnnealingLR
        self.eta_min = params.get('eta_min', 1e-6)

        # Epoch of the best validation loss from the last
        # train_model_and_evaluate_every_epoch call (1-indexed)
        self.last_best_epoch = None

        self._build()

    def _build(self):
        """(Re-)creates model and optimizer from scratch (fresh initialization)."""
        self.model: BaseModel = self.model_class(
            input_dims=self.input_dims, output_dim=self.output_dim, hyperparams=self.params
        ).to(self.device)

        # AdamW with decoupled weight decay for small datasets
        wd = self.params.get('weight_decay', 1e-4)
        lr = self.params.get('lr', 1e-3)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=wd,
            betas=(0.9, 0.999)
        )

    def _to_device(self, data):
        if isinstance(data, (list, tuple)):
            return [x.to(self.device, non_blocking=True) for x in data]
        return data.to(self.device, non_blocking=True)

    def _training_loss(self, pred, y):
        """
        Criterion loss plus an optional model-provided penalty
        (e.g. ElasticNetModel.regularization_loss). Returns
        (total_loss_for_backward, criterion_loss_for_logging).
        """
        base = self.criterion(pred, y)
        reg_fn = getattr(self.model, "regularization_loss", None)
        total = base + reg_fn() if callable(reg_fn) else base
        return total, base

    def train_model(self, X_train: tuple, y_train: torch.Tensor, epochs: int) -> None:
        # Optimization: Move entire dataset to GPU once for small datasets.
        try:
            self._train_model_gpu_resident(X_train, y_train, epochs)
        except RuntimeError as err:
            if not _is_cuda_oom(err):
                raise  # real bugs (shape/device errors) must surface, not be retried
            print(
                "WARNING: CUDA out of memory during GPU-resident training. "
                "Discarding the partially trained model and restarting from "
                "scratch with batch-wise DataLoader training."
            )
            torch.cuda.empty_cache()
            self._build()  # clean restart: fresh weights AND fresh optimizer state
            train_loader = get_dataloader(*X_train, y_train, batch_size=self.params.get('batch_size', 16))
            scheduler = self.scheduler_class(self.optimizer, T_max=epochs, eta_min=self.eta_min)
            for _ in range(epochs):
                self.train_epoch(train_loader)
                scheduler.step()

    def _train_model_gpu_resident(self, X_train: tuple, y_train: torch.Tensor, epochs: int) -> None:
        X_train_gpu = self._to_device(X_train)
        y_train_gpu = self._to_device(y_train)

        scheduler = self.scheduler_class(self.optimizer, T_max=epochs, eta_min=self.eta_min)
        batch_size = self.params.get('batch_size', 16)
        n_samples = y_train.shape[0]

        self.model.train()

        for _ in range(epochs):
            # Simple permutation for shuffle
            indices = torch.randperm(n_samples, device=self.device)

            for start_idx in range(0, n_samples, batch_size):
                batch_idx = indices[start_idx:start_idx + batch_size]

                # Slice on GPU
                batch_inputs = [x[batch_idx] for x in X_train_gpu]
                batch_y = y_train_gpu[batch_idx]

                if hasattr(self.model, "update_running_mean"):
                    self.model.update_running_mean(batch_y)

                self.optimizer.zero_grad()
                pred = self.model(batch_inputs)
                loss, _ = self._training_loss(pred, batch_y)

                if loss.requires_grad:
                    loss.backward()
                    self.optimizer.step()

            scheduler.step()

    def train_model_and_evaluate_every_epoch(self, X_train: tuple, y_train: torch.Tensor, epochs: int,
                                             X_val: tuple, y_val: torch.Tensor,
                                             early_stopping_patience: int = None) -> dict:
        """
        Trains with per-epoch validation.

        If early_stopping_patience is set, training stops after `patience`
        non-improving epochs AND the weights of the best epoch are restored,
        so subsequent evaluate_model() calls score the best model, not the
        last (over-trained) one. The best epoch is stored in
        self.last_best_epoch (1-indexed) for epoch-budget transfer to the
        final training run.
        """
        train_loader = get_dataloader(*X_train, y_train, batch_size=self.params.get('batch_size', 16))
        val_loader = get_dataloader(*X_val, y_val, batch_size=len(y_val), shuffle=False)
        scheduler = self.scheduler_class(self.optimizer, T_max=epochs, eta_min=self.eta_min)

        history = {'train_loss': [], 'val_loss': []}
        best_val_loss = float('inf')
        best_epoch = 0
        best_state = None
        patience_counter = 0

        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(train_loader)
            history['train_loss'].append(train_loss)

            val_loss, _, _ = self.evaluate(val_loader)
            history['val_loss'].append(val_loss)

            scheduler.step()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                best_state = copy.deepcopy(self.model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if early_stopping_patience is not None and patience_counter >= early_stopping_patience:
                    break

        self.last_best_epoch = best_epoch if best_epoch > 0 else len(history['train_loss'])

        # Restore the best weights when early stopping is active. Without
        # early stopping the caller asked for a fixed number of epochs, so
        # the final-epoch weights are kept (unchanged semantics).
        if early_stopping_patience is not None and best_state is not None:
            self.model.load_state_dict(best_state)

        return history

    def evaluate_model(self, X: tuple, y: torch.Tensor) -> tuple[float, np.ndarray, np.ndarray]:
        # Optimization: Full batch evaluation on GPU if possible
        self.model.eval()
        try:
            with torch.no_grad():
                X_gpu = self._to_device(X)
                y_gpu = self._to_device(y)

                pred = self.model(X_gpu)
                loss = self.criterion(pred, y_gpu).item()

                return loss, pred.cpu().numpy(), y_gpu.cpu().numpy()
        except RuntimeError as err:
            if not _is_cuda_oom(err):
                raise  # surface real errors instead of silently re-routing
            print("WARNING: CUDA out of memory during full-batch evaluation. Falling back to batch-wise evaluation.")
            torch.cuda.empty_cache()
            loader = get_dataloader(*X, y, batch_size=max(1, len(y) // 4), shuffle=False)
            return self.evaluate(loader)

    def train_epoch(self, loader) -> float:
        self.model.train()
        losses = []
        for batch in loader:
            # Inputs: (x_path, x_user, g0_x, g0_mask, ..., y)
            inputs = batch[:-1]
            y = batch[-1]

            # --- DATA MOVING TO VRAM ---
            # The dataset resides in system RAM (from loader).
            # It is moved to VRAM (GPU) only here, for the current batch.
            inputs = [x.to(self.device) for x in inputs]
            y = y.to(self.device)

            # Special hook for models like DummyMeanRegressor that need manual updates
            if hasattr(self.model, "update_running_mean"):
                self.model.update_running_mean(y)

            self.optimizer.zero_grad()
            # Pass list of inputs to model
            pred = self.model(inputs)
            loss, base_loss = self._training_loss(pred, y)

            # This check is preventing crashes for analytical baselines like DummyMeanRegressor.
            if loss.requires_grad:
                loss.backward()
                self.optimizer.step()

            # Log the pure criterion loss so train/val curves stay comparable
            losses.append(base_loss.item())
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

    def cleanup(self):
        """Explicitly release resources to avoid VRAM leaks."""
        # 1. Clear Optimizer state (momentum buffers live on GPU)
        if hasattr(self, 'optimizer') and self.optimizer is not None:
            self.optimizer.zero_grad(set_to_none=True)
            del self.optimizer

        # 2. Clear model and attached custom GPU tensors
        if hasattr(self, 'model') and self.model is not None:
            if hasattr(self.model, '__dict__'):
                for key in list(self.model.__dict__.keys()):
                    if isinstance(self.model.__dict__[key], torch.Tensor):
                        self.model.__dict__[key] = None

            self.model.cpu()
            del self.model

        if hasattr(self, 'criterion'):
            del self.criterion

        if hasattr(self, 'scheduler_class'):
            del self.scheduler_class

        self.model = None
        self.optimizer = None
        self.criterion = None
        self.scheduler_class = None

        gc.collect()
        torch.cuda.empty_cache()
