    

class MPNTrainer(Trainer):
    # Type hints for Pylance to avoid false positives
    model: MaskPredictionNetwork
    scaler: torch.cuda.amp.GradScaler  # Always initialized when use_amp=True

    def __init__(
        self,
        config: SIRIUSConfig,
    ) -> None:
        self.config = config
        self._pre_init(train_config=config.train, wandb_config=config.wandb)
        self._setup_model()
        self._setup_data_loaders()  # Must be called before super().__init__

        super().__init__(config.train, config.wandb)

    def _pre_init(
        self,
        train_config: TrainConfig,
        wandb_config: WandbConfig,
    ) -> None:
        """Pre-initialization to set up required attributes before model setup."""
        self.train_config = train_config
        self.wandb_config = wandb_config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _setup_model(self) -> None:
        """Set up the Mask Prediction Network model."""

        self.model = MaskPredictionNetwork(
            architecture_description=self.config.architecture,
            mask_description=self.config.mask,
            device=self.device,
        ).to(self.device)

        self.model.target_encoder.load_state_dict(
            self.model.context_encoder.state_dict(), strict=False
        )

        logger.info("Target encoder initialized with context encoder weights.")

        # Freeze target encoder (no gradients)
        for param in self.model.target_encoder.parameters():
            param.requires_grad = False

    
    # ....
    # Viel unnötiges Zeug
    # ....

    def train_step(self, x: torch.Tensor, ema_momentum: float) -> tuple[float, dict]:
        """Execute one training step (per batch).

        Args:
            x: Input batch
            ema_momentum: EMA momentum for target encoder update (calculated per epoch)

        Returns:
            Tuple of (loss_value, metrics_dict)
        """
        # Forward pass with autocast for mixed precision
        with torch.amp.autocast('cuda', enabled=self.use_amp):
            predictions, context_embeddings, target_embeddings = self.model(x)
            target_mask: torch.Tensor = self.model.get_target_mask(scale_idx=-1)
            loss = self.compute_loss(
                predictions=predictions,
                targets=target_embeddings,
                target_mask=target_mask,
            )

        # Backward pass and optimizer step
        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()

        # Update target encoder with EMA (CNN-JEPA: per-batch update)
        # Note: momentum is calculated per epoch but applied per batch
        self.update_target_encoder(momentum=ema_momentum)

        metrics = {"loss": loss.item(), "lr": self.optimizer.param_groups[0]["lr"]}

        return loss.item(), metrics

    def train_epoch(self, momentum: float) -> dict[str, float]:
        """Train for one epoch with the given EMA momentum.

        Args:
            momentum: EMA momentum value for target encoder updates

        Returns:
            Dictionary of training metrics
        """
        self.model.train()

        total_loss = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(self.train_loader):
            # Move batch to device
            batch = batch.to(self.device)

            loss, metrics = self.train_step(batch, momentum)
            total_loss += loss
            num_batches += 1

            if (batch_idx + 1) % 10 == 0:  # Log every 10 batches
                logger.info(
                    f"Batch {batch_idx + 1}/{len(self.train_loader)} - Loss: {loss:.4f} - LR: {metrics['lr']:.6f}"
                )
            avg_loss = total_loss / num_batches

        return {
            "train_loss": avg_loss,
            "ema_momentum": momentum,  # Log momentum value for monitoring
        }

    @torch.no_grad()
    def validate(self, epoch: int = 0) -> dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.amp.autocast('cuda', enabled=self.use_amp):
            for batch in self.val_loader:
                batch = batch.to(self.device)
                predictions, context_embeddings, target_embeddings = self.model(batch)
                target_mask: torch.Tensor = self.model.get_target_mask(scale_idx=-1)
                loss = self.compute_loss(
                    predictions=predictions,
                    targets=target_embeddings,
                    target_mask=target_mask,
                )
                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / num_batches

        logger.info(f"Validation Epoch {epoch}: Loss: {avg_loss:.4f}")

        return {"val_loss": avg_loss}

    def train(self):
        """Main Train loop."""
        logger.info("Starting pretraining process.")

        for epoch in range(self.train_config.num_epochs):
            logger.info(f"Epoch {epoch + 1}/{self.train_config.num_epochs}")

            # Calculate momentum for this epoch (CNN-JEPA style: 0.996 -> 1.0)
            current_momentum = self.get_ema_momentum(epoch)

            train_metrics = self.train_epoch(momentum=current_momentum)
            val_metrics = self.validate(epoch=epoch)

            self._step_scheduler(epoch=epoch)

            monitor_metric = val_metrics.get("val_loss", 0.0)
            should_stop = self.early_stopping(
                current_metric=monitor_metric, model=self.model
            )

            if should_stop:
                logger.info(
                    f"Early stopping triggered at epoch {epoch + 1}. Stopping training."
                )

                # Restore best weights
                self.early_stopping.restore_best_weights_to_model(model=self.model)
                wandb.log(
                    data={
                        "early_stopping/stopped_epoch": epoch + 1,
                        "early_stopping/best_metric": self.early_stopping.get_best_metric(),
                        "early_stopping/patience": self.early_stopping.patience,
                    }
                )
                break

            all_metrics = {
                **train_metrics,
                **val_metrics,
                "epoch": epoch + 1,
                "early_stopping/best_metric": self.early_stopping.get_best_metric(),
                "early_stopping/patience_counter": self.early_stopping.wait,
            }

            # Log metrics to WandB
            wandb.log(all_metrics)

            logger.info(
                f"Train Loss: {train_metrics['train_loss']:.4f}, "
                f"Val Loss: {val_metrics['val_loss']:.4f}, "
            )

            # Save checkpoint
            current_scheduler = self.get_current_scheduler(epoch)
            self.checkpoint_manager.save_checkpoint(
                model=self.model,
                optimizer=self.optimizer,
                scheduler=current_scheduler,
                epoch=epoch + 1,
                metrics=val_metrics,
                additional_info={
                    "train_metrics": train_metrics,
                    "config": self.config.__dict__,
                    "current_momentum": current_momentum,
                },
            )
        logger.info("Training complete.")

        # Save final checkpoint and log checkpoint information
        # Note: We use the current epoch number, which might be less than num_epochs if early stopping was triggered
        current_epoch = min(epoch + 1, self.train_config.num_epochs)
        final_val_metrics = self.validate(epoch=current_epoch)
        final_scheduler = self.get_current_scheduler(current_epoch - 1)

        # Add early stopping information to final metrics
        final_val_metrics.update(
            {
                "early_stopping/final_best_metric": self.early_stopping.get_best_metric(),
                "early_stopping/stopped_early": current_epoch
                < self.train_config.num_epochs,
                "early_stopping/final_epoch": current_epoch,
            }
        )

        self.checkpoint_manager.save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=final_scheduler,
            epoch=current_epoch,
            metrics=final_val_metrics,
            additional_info={
                "training_completed": True,
                "final_metrics": final_val_metrics,
                "config": self.config.__dict__,
            },
        )

        checkpoint_info = self.checkpoint_manager.list_checkpoints()

        logger.info("Final checkpoint information:")

        if checkpoint_info["best_checkpoints"]:
            logger.info("Best Checkpoints:")
            for ckpt in checkpoint_info["best_checkpoints"]:
                logger.info(
                    f"  - {ckpt['path']} ({ckpt['metric_name']}: {ckpt['metric_value']:.4f})"
                )
        best_checkpoint = self.checkpoint_manager.get_best_checkpoint()
        if best_checkpoint:
            logger.info(f"Best checkpoint saved at: {best_checkpoint}")

        last_checkpoint = self.checkpoint_manager.get_latest_checkpoint()
        if last_checkpoint:
            logger.info(f"Last checkpoint saved at: {last_checkpoint}")

        wandb.finish()
 