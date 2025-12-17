"""Early stopping utilities for training."""

import numpy as np
from loguru import logger
from typing import Optional, Any


class EarlyStopping:
    """Early stopping to stop training when a monitored metric has stopped improving."""

    def __init__(
        self,
        patience: int = 7,
        min_delta: float = 0.0,
        mode: str = "min",
        restore_best_weights: bool = True,
    ):
        """
        Initialize early stopping.

        Args:
            patience: Number of epochs with no improvement after which training will be stopped
            min_delta: Minimum change in the monitored quantity to qualify as an improvement
            monitor: Quantity to be monitored
            mode: One of 'min' or 'max'. In 'min' mode, training will stop when the quantity
                  monitored has stopped decreasing; in 'max' mode it will stop when the
                  quantity monitored has stopped increasing
            restore_best_weights: Whether to restore model weights from the epoch with the best value
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best_weights = restore_best_weights

        self.wait = 0
        self.stopped_epoch = 0
        self.best_weights = None

        if mode == "min":
            self.monitor_op = np.less
            self.min_delta *= -1
            self.best = np.inf
        elif mode == "max":
            self.monitor_op = np.greater
            self.min_delta *= 1
            self.best = -np.inf
        else:
            raise ValueError(f"Mode {mode} is unknown, please use 'min' or 'max'")

        logger.info(
            f"EarlyStopping initialized: patience={patience}, "
            f"min_delta={abs(self.min_delta)}, mode={mode}"
        )

    def __call__(self, current_metric: float, model: Optional[Any] = None) -> bool:
        """
        Check if training should stop early.

        Args:
            current_metric: Current value of the monitored metric
            model: Model to save weights from (if restore_best_weights is True)

        Returns:
            True if training should stop, False otherwise
        """
        if self.monitor_op(current_metric - self.min_delta, self.best):
            self.best = current_metric
            self.wait = 0

            # Save best weights if requested and model is provided
            if self.restore_best_weights and model is not None:
                import copy

                if hasattr(model, 'state_dict'):
                    self.best_weights = copy.deepcopy(model.state_dict())

            logger.info(f"EarlyStopping: new best loss: {current_metric:.6f}")
        else:
            self.wait += 1
            logger.info(
                f"EarlyStopping: no improvement for {self.wait}/{self.patience} epochs "
                f"(best loss: {self.best:.6f}, current: {current_metric:.6f})"
            )

            if self.wait >= self.patience:
                self.stopped_epoch = self.wait
                logger.info(
                    f"EarlyStopping: stopping training after {self.patience} epochs "
                    f"without improvement. Best loss: {self.best:.6f}"
                )
                return True

        return False

    def restore_best_weights_to_model(self, model: Any) -> None:
        """Restore the best weights to the model."""
        if self.best_weights is not None and self.restore_best_weights:
            if hasattr(model, 'load_state_dict'):
                model.load_state_dict(self.best_weights)
                logger.info(
                    f"EarlyStopping: restored best weights with loss: {self.best:.6f}"
                )
            else:
                logger.warning(
                    "EarlyStopping: model does not have load_state_dict method"
                )
        else:
            logger.warning("EarlyStopping: no best weights to restore")

    def get_best_metric(self) -> float:
        """Get the best metric value seen so far."""
        return self.best

    def reset(self) -> None:
        """Reset the early stopping state."""
        self.wait = 0
        self.stopped_epoch = 0
        self.best_weights = None

        if self.mode == "min":
            self.best = np.inf
        else:
            self.best = -np.inf

        logger.info("EarlyStopping: state reset")
