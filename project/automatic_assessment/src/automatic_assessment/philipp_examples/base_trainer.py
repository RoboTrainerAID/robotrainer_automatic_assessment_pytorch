import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from hydra.utils import instantiate

import wandb

from loguru import logger
from typing import Optional

from config.config import (
    TrainConfig,
    WandbConfig,
)
from src.utils.checkpoint_manager import CheckpointManager
from src.utils.early_stopping import EarlyStopping


class Trainer:
    """
    Base trainer class for model training and evaluation.

    """

    train_loader: DataLoader
    val_loader: DataLoader
    model: nn.Module
    optimizer: optim.Optimizer

    def __init__(self, train_config: TrainConfig, wandb_config: WandbConfig) -> None:
        self.train_config: TrainConfig = train_config
        self.wandb_config: WandbConfig = wandb_config

        self.device: torch.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        logger.info(f"Using device: {self.device}")

        # Initialize AMP scaler for mixed precision training
        self.use_amp = torch.cuda.is_available() and train_config.use_amp
        if self.use_amp:
            self.scaler: Optional[GradScaler] = GradScaler("cuda")
            logger.info("Using Automatic Mixed Precision (AMP) for training.")
        else:
            self.scaler = None
            logger.info("AMP not enabled; using standard precision.")

        self._setup_training()
        self._setup_checkpointing()
        self._setup_wandb()

    def _setup_training(self) -> None:
        """Setup optimizer and scheduler using Hydra instantiate."""
        # Instantiate optimizer using the configuration
        self.optimizer = instantiate(
            self.train_config.optimizer, params=self.model.parameters()
        )
        logger.info(f"Initialized optimizer: {type(self.optimizer).__name__}")

        # Instantiate main scheduler if configured
        self.main_scheduler = None
        if self.train_config.scheduler._target_ is not None:
            self.main_scheduler = instantiate(
                self.train_config.scheduler, optimizer=self.optimizer
            )
            logger.info(
                f"Initialized main scheduler: {type(self.main_scheduler).__name__}"
            )
        else:
            logger.info("No scheduler configured")

        # warmup
        self.warmup_epochs = int(
            self.train_config.num_epochs * self.train_config.warmup_ratio
        )
        if self.warmup_epochs > 0:
            self.warmup_scheduler = optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=0.1,  # start at 10% of the initial lr
                end_factor=1.0,
                total_iters=self.warmup_epochs,
            )
            logger.info(f"Using warmup for {self.warmup_epochs} epochs.")
        else:
            self.warmup_scheduler = None
            logger.info("No learning rate warmup.")

        # Early Stopping
        self.early_stopping = EarlyStopping(
            patience=self.train_config.early_stopping.patience,
            min_delta=self.train_config.early_stopping.min_delta,
            mode=self.train_config.early_stopping.mode,
        )

    def _setup_checkpointing(self) -> None:
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=self.wandb_config.save_dir,
            save_best=True,
            save_last=True,
            metric_name="val_loss",
            mode="min",
            save_top_k=2,  # keep top k models based on the metric
        )

    def _setup_wandb(self) -> None:
        wandb.init(
            project=self.wandb_config.project,
            config=dict(self.train_config.__dict__),
            dir=self.wandb_config.save_dir,
        )

    def _step_scheduler(self, epoch: int) -> None:
        """Step the appropriate scheduler based on current epoch."""
        if self.warmup_scheduler is not None and epoch < self.warmup_epochs:
            self.warmup_scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            logger.info(
                f"Warmup step - Epoch {epoch + 1}/{self.warmup_epochs}, LR: {current_lr:.6f}"
            )
        else:
            # After warmup, use main scheduler
            if self.main_scheduler is not None:
                self.main_scheduler.step()
                current_lr = self.optimizer.param_groups[0]['lr']
                logger.info(
                    f"Main scheduler step - Epoch {epoch + 1}, LR: {current_lr:.6f}"
                )

    def get_current_scheduler(self, epoch: int):
        """Get the current scheduler based on epoch."""
        if self.warmup_scheduler is not None and epoch < self.warmup_epochs:
            return self.warmup_scheduler
        return self.main_scheduler
